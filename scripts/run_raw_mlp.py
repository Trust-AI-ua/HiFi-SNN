#!/usr/bin/env python3
"""Raw-feature MLP deployment baseline: F_raw -> 64 -> 64 -> K.

Answers the question the DNN-Tree twin cannot: is the compact dense model
competitive because of the tree-derived representation, or would an equally small
raw-feature MLP do just as well?

Deliberately reuses ``DNNStudent`` -- the identical dense topology used by the
DNN-Tree twin -- so the ONLY difference is the input representation: raw
preprocessed features instead of tree-guided encoder channels. Parameter counting
and ``macs_per_sample`` therefore stay directly comparable.

Trained with plain cross-entropy: no teacher, no distillation, no encoder. This is
a standalone baseline, not a distilled twin, so a win for DNN-Tree cannot be
attributed to KD that the baseline was denied. Optimiser and schedule follow the
brief (Adam, lr 1e-3, batch 128) and the epoch budget matches DNN-Tree. Data
partitions, categorical vocabulary and scaling come from the same loaders, so
nothing about the split differs.

    python scripts/run_raw_mlp.py --datasets nslkdd --seeds 42
    python scripts/run_raw_mlp.py                       # all 11 x 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = ["nslkdd", "kddcup99", "unswnb15", "cicids2017", "cse-cic-ids2018",
            "iscxids2012", "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
SEEDS = [42, 43, 44, 45, 46]
MODEL_NAME = "RawMLP"
LR = 1e-3
BATCH = 128
HIDDEN = (64, 64)


def loader_kwargs(dataset: str, seed: int) -> tuple:
    """Mirror the DNN-Tree run's dataset options so partitions match exactly."""
    import yaml
    import inspect
    from src.data import DATASET_LOADERS
    cfg = yaml.safe_load(open(REPO / f"experiments/DNN-Tree/{dataset}/config.yaml"))
    params = inspect.signature(DATASET_LOADERS[dataset]).parameters
    kw = dict(batch_size=BATCH, val_fraction=cfg.get("val_fraction", 0.1), seed=seed)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in params and cfg[opt] is not None:
            kw[opt] = cfg[opt]
    return kw, cfg


def run_one(dataset: str, seed: int, force: bool = False) -> dict | None:
    from src.data import get_dataset
    from src.models import DNNStudent
    from src.metrics import (compute_metrics, compute_binary_ids_metrics,
                             count_parameters, clear_eval_context)
    from src.training_treespike import _arrays_from_bundle, _set_seed

    rd = REPO / f"results/{MODEL_NAME}/{dataset}/seed_{seed}"
    if (rd / "results.json").exists() and not force:
        return json.load(open(rd / "results.json"))
    rd.mkdir(parents=True, exist_ok=True)

    clear_eval_context()
    _set_seed(seed)
    kw, cfg = loader_kwargs(dataset, seed)
    epochs = int(cfg.get("epochs", 10))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = get_dataset(dataset, **kw)
    Xtr, ytr, Xva, yva, Xte, yte = _arrays_from_bundle(bundle)
    F_raw, K = Xtr.shape[1], bundle.num_classes

    model = DNNStudent(F_raw, K, hidden=HIDDEN, clip_relu=None).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    N = len(Xtr_t)
    curve, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        run, nb = 0.0, 0
        for i in range(0, N, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss.item()); nb += 1
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(Xva), 4096):
                vp.append(model(torch.from_numpy(Xva[i:i + 4096]).to(dev)).argmax(1).cpu())
            vacc = float((torch.cat(vp).numpy() == yva).mean())
        curve.append({"epoch": ep + 1, "train_loss": run / max(nb, 1), "val_accuracy": vacc})
        print(f"    ep{ep+1}/{epochs} loss={run/max(nb,1):.4f} val_acc={vacc:.4f}", flush=True)
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        preds, scores = [], []
        for i in range(0, len(Xte), 4096):
            lg = model(torch.from_numpy(Xte[i:i + 4096]).to(dev))
            preds.append(lg.argmax(1).cpu().numpy()); scores.append(lg.float().cpu().numpy())
    pred = np.concatenate(preds); score = np.concatenate(scores, axis=0)

    m = compute_metrics(yte, pred, y_score=score, class_names=bundle.class_names)
    bm = compute_binary_ids_metrics(yte, pred, normal_class_index=0)
    rec = {
        "model": MODEL_NAME, "dataset": dataset, "neuron": "ReLU", "encoding": "raw",
        "config": {"hidden": list(HIDDEN), "lr": LR, "batch_size": BATCH,
                   "epochs": epochs, "seed": seed, "loss": "cross_entropy",
                   "distillation": False, "encoder": None,
                   "val_fraction": kw["val_fraction"], "device": str(dev)},
        "data": {"num_features": F_raw, "num_classes": K,
                 "class_names": bundle.class_names,
                 "class_counts_train": bundle.class_counts_train,
                 "splits": {"train": len(Xtr), "val": len(Xva), "test_plus": len(Xte)}},
        "model_size": {**count_parameters(model),
                       "macs_per_sample": model.macs_per_sample()},
        "training": {"train_time_s": train_s, "training_curve": curve},
        "test_plus": {"n_samples": int(len(yte)), "metrics": m,
                      "binary_ids_metrics": bm},
        "test_minus_21": None,
    }
    json.dump(rec, open(rd / "results.json", "w"), indent=2, default=str)
    torch.save(model.state_dict(), rd / "checkpoint.pt")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0, done = time.time(), 0
    todo = [(d, s) for d in args.datasets for s in args.seeds]
    for ds, seed in todo:
        try:
            rec = run_one(ds, seed, args.force)
            m = rec["test_plus"]["metrics"]
            b = rec["test_plus"]["binary_ids_metrics"]
            done += 1
            print(f"[{done}/{len(todo)}] {ds}/seed{seed} F={rec['data']['num_features']} "
                  f"f1m={100*m['f1_macro']:.3f} mcc={100*m['matthews_corrcoef']:.3f} "
                  f"FAR={100*b['false_alarm_rate']:.3f} "
                  f"params={rec['model_size']['trainable_parameters']} "
                  f"| {(time.time()-t0)/60:.1f} min", flush=True)
        except Exception as e:
            print(f"[FAIL] {ds}/seed{seed}: {type(e).__name__}: {e}", flush=True)
    print(f"finished in {(time.time()-t0)/60:.1f} min")
