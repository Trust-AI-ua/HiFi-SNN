#!/usr/bin/env python3
"""int8-grid weight projection vs floating-point, for Raw MLP and DNN-Tree.

Inference only, nothing retrained. The projection is the one already used in the
manuscript, W_q = s * clip(round(W/s), -127, 127) with s = max|W|/127
(`src.models.quantize_int8_`).

IMPORTANT: this reports whether each checkpoint was ALREADY on the int8 grid
before we touch it. Both Tree-to-Spike and DNN-Tree run with `quantize: true`,
which applies the projection before evaluation and saves the checkpoint
afterwards -- so their published metrics are already int8-grid and their
floating-point weights no longer exist on disk. Rather than assume that, each
checkpoint is probed: re-projecting weights already on the grid is a no-op, so a
max relative weight change of 0 proves it.

Terminology: these are int8-GRID WEIGHTS WITH FLOATING-POINT EXECUTION. Not
integer inference.

    python scripts/eval_int8_grid.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

DATASETS = ["nslkdd", "kddcup99", "unswnb15", "cicids2017", "cse-cic-ids2018",
            "iscxids2012", "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
SEEDS = [42, 43, 44, 45, 46]
OUT = REPO / "results" / "deployment"


def grid_distance(model) -> float:
    """Max relative weight change under re-projection. 0 => already on the grid."""
    import torch.nn as nn
    worst = 0.0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                w = m.weight.data
                s = w.abs().max() / 127.0 + 1e-12
                wq = torch.round(w / s).clamp(-127, 127) * s
                denom = w.abs().max().clamp(min=1e-12)
                worst = max(worst, float((wq - w).abs().max() / denom))
    return worst


def metrics_of(y, pred, score, names):
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    m = compute_metrics(y, pred, y_score=score, class_names=names)
    b = compute_binary_ids_metrics(y, pred, normal_class_index=0)
    return (100 * m["f1_macro"], 100 * m["matthews_corrcoef"],
            100 * b["false_alarm_rate"], 100 * b["detection_rate"])


def eval_raw_mlp(dataset, seed):
    from src.data import get_dataset
    from src.models import DNNStudent, quantize_int8_
    from src.training_treespike import _arrays_from_bundle
    from run_raw_mlp import loader_kwargs, HIDDEN

    ck = REPO / f"results/RawMLP/{dataset}/seed_{seed}/checkpoint.pt"
    if not ck.exists():
        return None
    kw, _ = loader_kwargs(dataset, seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = get_dataset(dataset, **kw)
    _, _, _, _, Xte, yte = _arrays_from_bundle(b)
    m = DNNStudent(b.num_features, b.num_classes, hidden=HIDDEN).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev)); m.eval()

    def run(mod):
        ps, sc = [], []
        with torch.no_grad():
            for i in range(0, len(Xte), 4096):
                lg = mod(torch.from_numpy(Xte[i:i + 4096]).to(dev))
                ps.append(lg.argmax(1).cpu().numpy()); sc.append(lg.float().cpu().numpy())
        return np.concatenate(ps), np.concatenate(sc, axis=0)

    d0 = grid_distance(m)
    p, s = run(m); fp = metrics_of(yte, p, s, b.class_names)
    mq = quantize_int8_(copy.deepcopy(m))
    p, s = run(mq); q8 = metrics_of(yte, p, s, b.class_names)
    return fp, q8, d0, b.num_features


def eval_dnn_tree(dataset, seed):
    from src.data import get_dataset, DATASET_LOADERS
    from src.models import TreeGuidedSpikeEncoder, DNNStudent, quantize_int8_
    from src.training_treespike import _fit_teacher, _arrays_from_bundle, _set_seed
    from src.training_dnn import _resolve, _forward_batches
    import yaml, inspect

    ck = REPO / f"results/DNN-Tree/{dataset}/seed_{seed}/checkpoint.pt"
    rec_p = REPO / f"results/DNN-Tree/{dataset}/seed_{seed}/results.json"
    if not ck.exists():
        return None
    cfg = yaml.safe_load(open(REPO / f"experiments/DNN-Tree/{dataset}/config.yaml"))
    cfg.update(json.load(open(rec_p))["config"])
    cfg["dataset"] = dataset; cfg["seed"] = seed
    cfg = _resolve(cfg)
    _set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = inspect.signature(DATASET_LOADERS[dataset]).parameters
    kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=seed)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in params and cfg[opt] is not None:
            kw[opt] = cfg[opt]
    b = get_dataset(dataset, **kw)
    Xtr, ytr, _, _, Xte, yte = _arrays_from_bundle(b)
    teacher, _, _ = _fit_teacher(Xtr, ytr, b.num_classes, cfg)
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
                                 prune_quantile=cfg["prune_quantile"],
                                 jitter=cfg["jitter"]).fit(teacher, Xtr.shape[1])
    m = DNNStudent(enc.out_dim, b.num_classes, hidden=tuple(cfg["hidden"]),
                   clip_relu=cfg["clip_relu"]).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev)); m.eval()

    d0 = grid_distance(m)
    with torch.no_grad():
        p, s = _forward_batches(m, enc, Xte, cfg["batch_size"], dev)
    fp = metrics_of(yte, p, s, b.class_names)
    mq = quantize_int8_(copy.deepcopy(m))
    with torch.no_grad():
        p, s = _forward_batches(mq, enc, Xte, cfg["batch_size"], dev)
    q8 = metrics_of(yte, p, s, b.class_names)
    return fp, q8, d0, enc.out_dim


if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--models", nargs="+", default=["rawmlp", "dnntree"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.time()
    for model in args.models:
        fn = eval_raw_mlp if model == "rawmlp" else eval_dnn_tree
        for ds in args.datasets:
            for seed in args.seeds:
                try:
                    r = fn(ds, seed)
                except Exception as e:
                    print(f"[FAIL] {model}/{ds}/{seed}: {type(e).__name__}: {e}", flush=True)
                    continue
                if r is None:
                    print(f"[skip] {model}/{ds}/{seed} no checkpoint", flush=True)
                    continue
                fp, q8, d0, in_dim = r
                rows.append(dict(model=model, dataset=ds, seed=seed, in_dim=in_dim,
                                 already_on_int8_grid=bool(d0 < 1e-9), grid_distance=d0,
                                 fp_macro_f1=fp[0], int8_macro_f1=q8[0],
                                 d_macro_f1=q8[0] - fp[0],
                                 fp_mcc=fp[1], int8_mcc=q8[1], d_mcc=q8[1] - fp[1],
                                 fp_FAR=fp[2], int8_FAR=q8[2], d_FAR=q8[2] - fp[2],
                                 fp_DR=fp[3], int8_DR=q8[3]))
                print(f"[{len(rows)}] {model}/{ds}/s{seed} on_grid={d0<1e-9} "
                      f"f1 {fp[0]:.3f} -> {q8[0]:.3f} ({q8[0]-fp[0]:+.3f}) "
                      f"| {(time.time()-t0)/60:.1f}m", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "int8_grid.csv", index=False)
    print("\n=== already on the int8 grid before we projected? ===")
    print(df.groupby("model")["already_on_int8_grid"].agg(["sum", "count"]).to_string())
    print("\n=== FP -> int8-grid (mean over datasets+seeds) ===")
    print(df.groupby("model")[["fp_macro_f1", "int8_macro_f1", "d_macro_f1",
                               "d_mcc", "d_FAR"]].mean().round(4).to_string())
    print("\n=== per dataset, Raw MLP ===")
    r = df[df.model == "rawmlp"]
    if len(r):
        print(r.groupby("dataset")[["fp_macro_f1", "int8_macro_f1", "d_macro_f1"]]
                .mean().round(3).to_string())
    print(f"\nwrote {OUT}/int8_grid.csv")
