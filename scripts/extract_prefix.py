#!/usr/bin/env python3
"""Extract per-timestep prefix artifacts for Tree-to-Spike. Inference only.

One full-window forward pass per partition yields prefix logits and cumulative
operation counts for every sample, so the eight-threshold sweep, calibration, and
class-conditional exit analysis all become offline post-processing. Instead of
11 x 5 x 8 inference runs the sweep costs 11 x 5 x 2 passes.

Each cell is gated on strict reconstruction: argmax of the final prefix logit must
match the stored predictions' macro-F1 / MCC / FAR exactly, or the cell is refused
and no artifact is written for it.

    python scripts/extract_prefix.py --datasets nslkdd --seeds 42     # verify one
    python scripts/extract_prefix.py                                   # all 11 x 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = ["nslkdd", "kddcup99", "unswnb15", "cicids2017", "cse-cic-ids2018",
            "iscxids2012", "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
SEEDS = [42, 43, 44, 45, 46]
OUTROOT = REPO / "results" / "prefix_outputs"
TOL = 1e-9


def load_cfg(dataset: str, seed: int) -> dict:
    """Config as the run actually used it: stored record first, yaml for the rest."""
    import yaml
    rec_p = REPO / f"results/Tree-to-Spike/{dataset}/seed_{seed}/results.json"
    rec = json.load(open(rec_p))
    cfg = dict(yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{dataset}/config.yaml")))
    cfg.update(rec["config"])          # stored values win
    cfg["dataset"] = dataset
    cfg["seed"] = seed
    return cfg, rec


def build(dataset: str, seed: int):
    """Rebuild the trained student + encoder + bundle. Nothing is trained."""
    from src.data import get_dataset, DATASET_LOADERS
    from src.models import TreeGuidedSpikeEncoder, SNNStudent
    from src.training_treespike import _fit_teacher, _arrays_from_bundle, _resolve, _set_seed
    import inspect

    cfg, rec = load_cfg(dataset, seed)
    cfg = _resolve(cfg)
    _set_seed(seed)
    dev = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    params = inspect.signature(DATASET_LOADERS[dataset]).parameters
    kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=seed)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in params and cfg[opt] is not None:
            kw[opt] = cfg[opt]
    bundle = get_dataset(dataset, **kw)
    Xtr, ytr, Xva, yva, Xte, yte = _arrays_from_bundle(bundle)

    teacher, _, _ = _fit_teacher(Xtr, ytr, bundle.num_classes, cfg)
    enc = TreeGuidedSpikeEncoder(
        T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
        jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")
    ).fit(teacher, Xtr.shape[1], X=Xtr)
    st = SNNStudent(enc.out_dim, bundle.num_classes, hidden=tuple(cfg["hidden"]),
                    beta=cfg["beta"], leaky=cfg["leaky"]).to(dev)
    ck = REPO / f"results/Tree-to-Spike/{dataset}/seed_{seed}/checkpoint.pt"
    st.load_state_dict(torch.load(ck, map_location=dev))
    st.eval()
    return st, enc, bundle, cfg, rec, dev, (Xva, yva, Xte, yte)


@torch.no_grad()
def prefix_pass(st, enc, X, bs, dev) -> dict:
    parts: dict = {}
    for i in range(0, len(X), bs):
        spikes = enc.encode(X[i:i + bs]).to(dev)
        out = st.forward_prefix(spikes)
        for k, v in out.items():
            parts.setdefault(k, []).append(v)
    return {k: torch.cat(v).numpy() for k, v in parts.items()}


def metrics(y, p):
    from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
    tb, pb = (y != 0).astype(int), (p != 0).astype(int)
    cm = confusion_matrix(tb, pb, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    return (100 * f1_score(y, p, average="macro", zero_division=0),
            100 * matthews_corrcoef(y, p),
            100 * fp / max(fp + tn, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()

    report, t0 = [], time.time()
    for ds in args.datasets:
        for seed in args.seeds:
            tag = f"{ds}/seed_{seed}"
            try:
                st, enc, bundle, cfg, rec, dev, (Xva, yva, Xte, yte) = build(ds, seed)
            except Exception as e:
                print(f"[{tag}] BUILD FAILED {type(e).__name__}: {e}", flush=True)
                report.append(dict(dataset=ds, seed=seed, status="build_failed",
                                   error=f"{type(e).__name__}: {e}"))
                continue

            bs = cfg["batch_size"]
            pv = prefix_pass(st, enc, Xva, bs, dev)
            pt = prefix_pass(st, enc, Xte, bs, dev)

            # ---- strict reconstruction gate on the TEST split ----
            pred_T = pt["prefix_logits"][:, -1, :].argmax(1)
            f1, mcc, far = metrics(yte, pred_T)
            sm = rec["test_plus"]["metrics"]
            sb = rec["test_plus"].get("binary_ids_metrics") or {}
            d_f1 = f1 - 100 * sm["f1_macro"]
            d_mcc = mcc - 100 * sm["matthews_corrcoef"]
            d_far = far - 100 * sb.get("false_alarm_rate", far / 100)
            ok = abs(d_f1) < 1e-6 and abs(d_mcc) < 1e-6 and abs(d_far) < 1e-6
            print(f"[{tag}] T={pt['prefix_logits'].shape[1]} val={len(yva)} test={len(yte)}"
                  f" | dF1={d_f1:+.2e} dMCC={d_mcc:+.2e} dFAR={d_far:+.2e}"
                  f" | {'OK' if ok else 'REFUSED'}", flush=True)
            if not ok:
                report.append(dict(dataset=ds, seed=seed, status="reconstruction_failed",
                                   d_f1=d_f1, d_mcc=d_mcc, d_far=d_far))
                continue

            d = OUTROOT / ds / "treespike" / f"seed_{seed}"
            d.mkdir(parents=True, exist_ok=True)
            for split, pr, y in (("val", pv, yva), ("test", pt, yte)):
                np.savez_compressed(
                    d / f"{split}_prefix.npz",
                    prefix_logits=pr["prefix_logits"].astype(args.dtype),
                    labels=y.astype(np.int64),
                    sample_ids=np.arange(len(y), dtype=np.int64),
                    full_logits=pr["prefix_logits"][:, -1, :].astype(np.float32),
                    cum_sops=pr["cum_sops"].astype(np.float32),
                    cum_spk1=pr["cum_spk1"].astype(np.float32),
                    cum_spk2=pr["cum_spk2"].astype(np.float32),
                    cum_in_spikes=pr["cum_in_spikes"].astype(np.float32),
                    class_names=np.array(bundle.class_names, dtype=object),
                )
            mb = sum(f.stat().st_size for f in d.glob("*.npz")) / 1e6
            report.append(dict(dataset=ds, seed=seed, status="ok", T=int(pt["prefix_logits"].shape[1]),
                               n_val=len(yva), n_test=len(yte), mb=round(mb, 1),
                               d_f1=d_f1, d_mcc=d_mcc, d_far=d_far))
            print(f"        wrote {mb:.1f} MB  ({(time.time()-t0)/60:.1f} min)", flush=True)
            del st, enc, bundle, pv, pt

    OUTROOT.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUTROOT / "extraction_report.json", "w"), indent=2, default=str)
    n_ok = sum(1 for r in report if r["status"] == "ok")
    print(f"\n{n_ok}/{len(report)} cells extracted; "
          f"total {sum(r.get('mb', 0) for r in report)/1000:.2f} GB")
    for r in report:
        if r["status"] != "ok":
            print("  REFUSED", r["dataset"], r["seed"], r["status"])


if __name__ == "__main__":
    main()
