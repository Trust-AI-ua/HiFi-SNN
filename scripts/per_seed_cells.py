#!/usr/bin/env python3
"""Rebuild the 44-cell per-seed table (11 datasets x 4 models x 5 seeds).

Reads the headline runs' results.json and writes the file m9_stats.py consumes.
Refuses to write unless every cell has five seeds, so a partly re-run grid cannot
silently produce a 43-cell analysis. Adds `empty_classes`, since the reported
macro-F1 averages only the classes present in the test split.

    python scripts/per_seed_cells.py
Output: results/diagnostics/per_seed_all_cells.csv
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/diagnostics/per_seed_all_cells.csv"
DATASETS = ["kddcup99", "nslkdd", "ctu13", "iscxids2012", "cicids2017", "cse-cic-ids2018",
            "unswnb15", "ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
PATTERNS = {"vanilla": "results/{d}/SNN_LeakyParallel/latency/seed_*/results.json",
            "treespike": "results/Tree-to-Spike/{d}/seed_*/results.json",
            "dnn": "results/DNN-Tree/{d}/seed_*/results.json",
            "tasnn": "results/TASNN/{d}/seed_*/results.json"}


def main():
    rows = []
    for d in DATASETS:
        for model, pat in PATTERNS.items():
            for f in sorted(glob.glob(str(REPO / pat.format(d=d)))):
                m = json.load(open(f))["test_plus"]["metrics"]
                cm = np.array(m["confusion_matrix"])
                support = cm.sum(1)
                names = m["class_names"]
                per = np.array([m["per_class"][n]["f1"] for n in names])
                rows.append(dict(
                    dataset=d, model=model, seed=int(f.split("seed_")[1][:2]),
                    f1=m["f1_macro"] * 100,
                    f1_present=per[support > 0].mean() * 100,
                    accuracy=m["accuracy"] * 100,
                    balanced_accuracy=m["balanced_accuracy"] * 100,
                    mcc=m["matthews_corrcoef"] * 100,
                    empty_classes=int((support == 0).sum())))
    df = pd.DataFrame(rows)
    counts = df.groupby(["dataset", "model"]).size()
    bad = counts[counts != 5]
    if len(bad) or len(counts) != 44:
        raise SystemExit(f"expected 44 cells x 5 seeds; got {len(counts)} cells, "
                         f"irregular: {bad.to_dict()}")
    df.sort_values(["dataset", "model", "seed"]).to_csv(OUT, index=False)
    print(f"[wrote] {OUT}: {len(df)} rows, {len(counts)} cells")
    print(df.groupby(["dataset", "model"]).f1.mean().unstack().round(2).to_string())


if __name__ == "__main__":
    main()
