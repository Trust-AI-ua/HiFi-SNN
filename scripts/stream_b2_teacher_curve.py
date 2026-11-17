#!/usr/bin/env python3
"""B2: the per-packet XGBoost teacher's macro-F1 against packets seen, k = 1..32, seeds 42 to 46
(the baseline curve for the go/no-go figure; results/preregistration/2026-11-17_neuro_stream.md).

At packet k each flow is scored on its state after min(k, length) packets: a flow that has ended
keeps its last decision. 'live' rows score only the flows still running at packet k (length >= k).
The flow-end score uses the stream feature set (19 incremental features from the pcaps); it is not
comparable with the CSV-feature Edge-IIoTset results.

    .venv/bin/python scripts/stream_b2_teacher_curve.py
Output: results/neuro/b2/teacher_curve.csv (per seed), teacher_curve_mean.csv (mean, SD over seeds)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BIG = REPO / "dataset/edge-iiotset/_stream/b2"
OUT = REPO / "results/neuro/b2"
SEEDS = range(42, 47)


def main():
    from sklearn.metrics import f1_score
    from src.data.edgeiiot_stream import load_stream
    rows = []
    for seed in SEEDS:
        s = load_stream(seed)
        t = np.load(BIG / f"seed_{seed}_teacher.npz")
        for split in ("val", "test"):
            P, y, L = t[f"{split}_proba"], s.y[split], s.length[split]
            n, K32, _ = P.shape
            ar = np.arange(n)
            for k in range(1, K32 + 1):
                pred = P[ar, np.minimum(k, L) - 1].argmax(1)
                live = L >= k
                rows.append(dict(seed=seed, split=split, k=k, macro_f1=100 * f1_score(y, pred, average="macro"),
                                 accuracy=100 * float((pred == y).mean()), live_share=100 * float(live.mean()),
                                 live_classes=int(len(np.unique(y[live]))),
                                 macro_f1_live=100 * f1_score(y[live], pred[live], average="macro") if live.any() else None))
            end = P[ar, L - 1].argmax(1)
            rows.append(dict(seed=seed, split=split, k="end", macro_f1=100 * f1_score(y, end, average="macro"),
                             accuracy=100 * float((end == y).mean()), live_share=None, live_classes=None, macro_f1_live=None))
        te = [r for r in rows if r["seed"] == seed and r["split"] == "test"]
        print(f"test F1 k=1 {te[0]['macro_f1']:.2f}, k=4 {te[3]['macro_f1']:.2f}, k=32 {te[31]['macro_f1']:.2f}, "
              f"end {te[32]['macro_f1']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "teacher_curve.csv", index=False)
    df["k"] = df["k"].astype(str)
    g = df.groupby(["split", "k"], sort=False)
    m = g[["macro_f1", "accuracy", "live_share", "macro_f1_live"]].agg(["mean", "std"])
    m.columns = [f"{a}_{b}" for a, b in m.columns]
    m.insert(0, "n_seeds", g.size())
    m.reset_index().to_csv(OUT / "teacher_curve_mean.csv", index=False)
    print(m.loc["test"].round(2).to_string())


if __name__ == "__main__":
    main()
