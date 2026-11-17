#!/usr/bin/env python3
"""B2, stage 1 (CPU): the per-packet teacher (results/preregistration/2026-11-17_neuro_stream.md).

XGBoost with the production Edge-IIoTset teacher settings (src/training_treespike._fit_teacher:
200 trees, depth 6, learning rate 0.1, subsample 0.9, colsample 0.8, hist, n_jobs 4,
inverse-frequency class weights) on the prefix vectors of the training flows at every packet
up to the flow's length. Each prefix is weighted by class weight / flow length, so each flow
counts once. Thread variables must be unset.

Writes results/neuro/b2/seed_<S>/teacher.json (the booster) and, outside git,
dataset/edge-iiotset/_stream/b2/seed_<S>_teacher.npz with the teacher's probabilities and
margins after every packet (1..32) for the train, validation and test flows.

    python scripts/stream_b2_teacher.py <seed>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.edgeiiot_stream import load_stream  # noqa: E402

OUT = REPO / "results/neuro/b2"
BIG = REPO / "dataset/edge-iiotset/_stream/b2"


def prefix_rows(X, L):
    """All (flow, packet) prefixes up to each flow's length: rows [sum L, F], flow index, packet index."""
    K = X.shape[1]
    mask = np.arange(K)[None, :] < L[:, None]
    f, k = np.nonzero(mask)
    return X[f, k], f, k


def main():
    seed = int(sys.argv[1])
    if any(os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise SystemExit("thread variables are set")
    from xgboost import XGBClassifier
    s = load_stream(seed)
    K = len(s.class_names)
    t0 = time.time()
    R, f, k = prefix_rows(s.X["train"], s.length["train"])
    y = s.y["train"][f]
    if not np.array_equal(np.unique(s.y["train"]), np.arange(K)):
        raise SystemExit("a class is missing from training")
    counts = np.bincount(s.y["train"], minlength=K).astype(np.float64)      # flows, not prefixes
    cls_w = counts.sum() / (counts + 1e-6)
    w = cls_w[y] / s.length["train"][f]
    cols = [f"f{i}" for i in range(R.shape[1])]
    teacher = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
                            importance_type="gain", tree_method="hist", n_jobs=4, eval_metric="mlogloss",
                            random_state=seed)
    teacher.fit(pd.DataFrame(R, columns=cols), y, sample_weight=w)
    fit_s = time.time() - t0
    od = OUT / f"seed_{seed}"
    od.mkdir(parents=True, exist_ok=True)
    teacher.save_model(od / "teacher.json")
    BIG.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "val", "test"):
        X = s.X[split]
        n, P, F = X.shape
        flat = pd.DataFrame(X.reshape(n * P, F), columns=cols)
        out[f"{split}_proba"] = teacher.predict_proba(flat).reshape(n, P, K).astype(np.float32)
        out[f"{split}_margin"] = teacher.predict(flat, output_margin=True).reshape(n, P, K).astype(np.float32)
    np.savez(BIG / f"seed_{seed}_teacher.npz", **out)
    from sklearn.metrics import f1_score
    rep = dict(seed=seed, n_prefix_rows=int(len(R)), fit_seconds=round(fit_s, 1), class_weights=cls_w.tolist(),
               importances=dict(zip(s.features, map(float, teacher.feature_importances_))))
    for split in ("val", "test"):
        L = s.length[split]
        last = out[f"{split}_proba"][np.arange(len(L)), L - 1].argmax(1)
        rep[f"{split}_macro_f1_at_flow_end"] = 100 * f1_score(s.y[split], last, average="macro")
        rep[f"{split}_macro_f1_after_packet_1"] = 100 * f1_score(s.y[split], out[f"{split}_proba"][:, 0].argmax(1),
                                                                  average="macro")
    (od / "teacher_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("importances", "class_weights")}), flush=True)


if __name__ == "__main__":
    main()
