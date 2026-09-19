#!/usr/bin/env python3
"""Per-layer weight-scale ratio before and after the dataset repair (M6/M3).

The DNN-Tree instability on CSE-CIC-IDS2018 was traced to one first-layer weight
on `Src Port`, the capture-day artifact: with max|W|/p99|W| of 12.7-15.9 the
per-tensor scale s = max|W|/127 crushes the rest of fc1 (up to 50% of weights
round to zero). If removing the artifact removes the outlier, the ratio should
fall into the 1.2-2.8 band the other layers show.

Ratios are computed from the deployed (int8-grid) checkpoints: the projection
preserves max|W| exactly and moves p99 by at most half a step, so the ratio is
unaffected to within max/254.

    python scripts/fc1_ratio.py [--models DNN-Tree Tree-to-Spike] [--dataset cse-cic-ids2018]
Output: results/diagnostics/fc1_ratio_repair.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/diagnostics/fc1_ratio_repair.csv"


def stats(ck: Path):
    sd = torch.load(ck, map_location="cpu")
    rows = []
    for name, w in sd.items():
        if not (name.endswith("weight") and w.dim() == 2):
            continue
        a = w.abs().float()
        mx, p99 = a.max().item(), a.flatten().quantile(0.99).item()
        s = mx / 127.0
        rows.append(dict(layer=name.split(".")[0], max_abs=mx, p99_abs=p99,
                         max_over_p99=mx / max(p99, 1e-12),
                         frac_rounded_to_zero=float((a < s / 2).float().mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["DNN-Tree", "Tree-to-Spike"])
    ap.add_argument("--dataset", default="cse-cic-ids2018")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    a = ap.parse_args()
    rows = []
    for arm, base in (("pre_repair", REPO / "results/pre_repair"), ("repaired", REPO / "results")):
        for m in a.models:
            for s in a.seeds:
                ck = base / m / a.dataset / f"seed_{s}" / "checkpoint.pt"
                if not ck.exists():
                    continue
                for r in stats(ck):
                    rows.append(dict(arm=arm, model=m, dataset=a.dataset, seed=s, **r))
    if not rows:
        raise SystemExit("no checkpoints found")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    piv = df.pivot_table(index=["model", "layer"], columns="arm",
                         values="max_over_p99", aggfunc=["min", "max"]).round(2)
    print(piv.to_string())
    fc1 = df[df.layer == "fc1"]
    print("\nfc1 max/p99 by seed:")
    print(fc1.pivot_table(index=["model", "seed"], columns="arm",
                          values=["max_over_p99", "frac_rounded_to_zero"]).round(3).to_string())
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
