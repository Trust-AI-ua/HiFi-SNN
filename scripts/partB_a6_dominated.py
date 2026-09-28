#!/usr/bin/env python3
"""A6 (Part B amendment 1): per-class results with and without cross-partition duplicates,
for the classes B5 found dominated (> 50% of test rows duplicated in training), from the
stored models only (no training).

Cells: CIC-IDS2017 probe, CSE-CIC-IDS2018 r2l and DoS, KDDCup99 DoS; seeds 42-46; the four
models. Per seed, the duplicate mask is recomputed exactly as in B5 on the stored runs'
split (split guard). Predictions come from the stored models:
  Tree-to-Spike  argmax of the stored prefix outputs' full-window logits (deployed model)
  DNN-Tree       stored checkpoint (deployed int8-grid weights), encoder rebuilt from the
                 guarded teacher refit
  Vanilla SNN    stored checkpoint (scripts/eval_m3_fp32_models._vanilla)
  TASNN          stored checkpoint (eval_m3_fp32_models._tasnn; samples at inference, so the
                 reproduction is checked within 0.2 macro-F1 points and the gap recorded)
Guard: each model's recomputed test macro-F1 equals its stored value (TASNN: within 0.2),
and every model's test labels equal the Tree-to-Spike split's, in order.
Reported per class: precision, recall, F1 on the full test set and on the test set without
the rows duplicated in training; overall macro-F1 both ways.

    python scripts/partB_a6_dominated.py
Output: results/journal_partB/A6/<run_id>/{per_class.csv,guards.csv}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

CELLS = {"cicids2017": ["probe"], "cse-cic-ids2018": ["r2l", "dos"], "kddcup99": ["dos"]}
TASNN_TOL = 0.2


def dup_mask(Xtr, Xte):
    htr = pd.util.hash_pandas_object(pd.DataFrame(Xtr), index=False).to_numpy()
    hte = pd.util.hash_pandas_object(pd.DataFrame(Xte), index=False).to_numpy()
    rep = pd.Series(np.arange(len(htr))).groupby(htr).first()
    hit = np.isin(hte, rep.index.to_numpy())
    idx = np.where(hit)[0]
    same = (Xtr[rep.reindex(hte[idx]).to_numpy()] == Xte[idx]).all(axis=1)
    m = np.zeros(len(Xte), bool)
    m[idx[same]] = True
    return m


def dnn_preds(ds, seed, b, arrays, teacher):
    import yaml
    import src.training_dnn as tdnn
    from src.models import TreeGuidedSpikeEncoder, DNNStudent
    cfg = yaml.safe_load(open(C.REPO / f"experiments/DNN-Tree/{ds}/config.yaml"))
    Xtr, Xte = arrays[0], arrays[4]
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                 jitter=cfg["jitter"]).fit(teacher, Xtr.shape[1])
    m = DNNStudent(enc.out_dim, b.num_classes, hidden=tuple(cfg["hidden"]), clip_relu=cfg.get("clip_relu"))
    m.load_state_dict(torch.load(C.REPO / f"results/DNN-Tree/{ds}/seed_{seed}/checkpoint.pt", map_location="cpu"))
    m.eval()
    out = {}
    for dev in ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",):
        with torch.no_grad():
            out[dev] = tdnn._forward_batches(m.to(dev), enc, Xte, cfg["batch_size"], torch.device(dev))[0]
    return out


def main():
    import eval_m3_fp32_models as M3
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    C.no_thread_caps()
    st = C.stamp("A6")
    out = C.run_dir(st)
    rows, guards = [], []
    for ds, classes in CELLS.items():
        for seed in C.SEEDS:
            b, arrays = C.load_bundle(ds, seed)
            Xtr, yte, Xte = arrays[0], arrays[5], arrays[4]
            dm = dup_mask(Xtr, Xte)
            teacher, _ = C.fit_teacher_guarded(ds, seed, b, arrays)
            preds = {}
            z = np.load(C.REPO / f"results/prefix_outputs/{ds}/treespike/seed_{seed}/test_prefix.npz", allow_pickle=True)
            if not np.array_equal(z["labels"], yte):
                sys.exit(f"{ds}/{seed}: prefix labels are not the stored split's test labels")
            preds["Tree-to-Spike"] = z["full_logits"].argmax(1)
            stored = {"Tree-to-Spike": C.stored_record(ds, seed)["test_plus"]["metrics"]["f1_macro"],
                      "DNN-Tree": json.loads((C.REPO / f"results/DNN-Tree/{ds}/seed_{seed}/results.json").read_text())["test_plus"]["metrics"]["f1_macro"]}
            dnn = dnn_preds(ds, seed, b, arrays, teacher)
            f1 = lambda p: float(f1_score(yte, p, average="macro", zero_division=0))
            ok_dev = [d for d, p in dnn.items() if f1(p) == stored["DNN-Tree"]]
            preds["DNN-Tree"] = dnn[ok_dev[0] if ok_dev else "cuda" if "cuda" in dnn else "cpu"]
            for name, builder in (("Vanilla SNN", M3._vanilla), ("TASNN", M3._tasnn)):
                m, run, names, rd = builder(ds, seed)
                y, p, _ = run(m)
                if not np.array_equal(y, yte):
                    sys.exit(f"{name} {ds}/{seed}: test labels differ from the Tree-to-Spike split")
                preds[name] = p
                stored[name] = json.loads((rd / "results.json").read_text())["test_plus"]["metrics"]["f1_macro"]
            for name, p in preds.items():
                gap = 100 * (f1(p) - stored[name])
                ok = abs(gap) <= (TASNN_TOL if name == "TASNN" else 1e-9)
                guards.append(dict(dataset=ds, seed=seed, model=name, recomputed_f1=100 * f1(p), stored_f1=100 * stored[name],
                                   gap=gap, ok=ok, dnn_device=(ok_dev[0] if ok_dev else "none") if name == "DNN-Tree" else ""))
                if not ok:
                    sys.exit(f"reproduction guard failed: {name} {ds}/{seed} gap {gap:.4f}")
                for cls in classes:
                    c = b.class_names.index(cls)
                    for subset, keep in (("full", np.ones(len(yte), bool)), ("without_duplicates", ~dm)):
                        pr, rc, fs, sup = precision_recall_fscore_support(yte[keep], p[keep], labels=[c], zero_division=0)
                        rows.append(dict(dataset=ds, seed=seed, model=name, cls=cls, subset=subset, support=int(sup[0]),
                                         precision=100 * pr[0], recall=100 * rc[0], f1=100 * fs[0],
                                         macro_f1=100 * float(f1_score(yte[keep], p[keep], average="macro", zero_division=0)),
                                         dup_share_of_class=100 * float(dm[yte == c].mean())))
            print(f"{ds} {seed}: guards ok; dup rows {100*dm.mean():.1f}% of test", flush=True)
            pd.DataFrame(rows).to_csv(out / "per_class.csv", index=False)
            pd.DataFrame(guards).to_csv(out / "guards.csv", index=False)
    print(pd.DataFrame(rows).groupby(["dataset", "cls", "model", "subset"])[["recall", "f1"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
