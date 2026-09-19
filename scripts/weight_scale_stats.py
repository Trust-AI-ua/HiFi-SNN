#!/usr/bin/env python3
"""Per-layer weight-scale statistics for the int8 projection (M3 mechanism).

For every Linear weight in the fp32 checkpoints saved by the M3 reruns
(results/diagnostics/val_curves/<model>/<dataset>/runs/seed_<S>/checkpoint_fp32.pt),
records max|W|, p99|W|, their ratio, the fraction of weights the per-tensor
projection s = max|W|/127 rounds to zero, and the input channel holding max|W|
(fc1 only: position within the encoder's importance-ordered channel list).

    python scripts/weight_scale_stats.py
Output: results/diagnostics/weight_scale_stats.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/diagnostics/val_curves"
OUT = REPO / "results/diagnostics/weight_scale_stats.csv"


def main():
    rows = []
    for ck in sorted(ROOT.glob("*/*/runs/seed_*/checkpoint_fp32.pt")):
        model, dataset = ck.parts[-5], ck.parts[-4]
        seed = int(ck.parent.name.split("_")[1])
        sd = torch.load(ck, map_location="cpu")
        for name, w in sd.items():
            if not (name.endswith("weight") and w.dim() == 2):
                continue
            a = w.abs().float()
            mx = a.max().item()
            p99 = a.flatten().quantile(0.99).item()
            s = mx / 127.0
            r, c = divmod(int(a.argmax()), a.shape[1])
            rows.append(dict(model=model, dataset=dataset, seed=seed, layer=name.split(".")[0],
                             shape=f"{w.shape[0]}x{w.shape[1]}", max_abs=mx, p99_abs=p99,
                             max_over_p99=mx / max(p99, 1e-12),
                             frac_rounded_to_zero=float((a < s / 2).float().mean()),
                             argmax_in_channel=c, in_channels=a.shape[1],
                             argmax_channel_rel=c / max(a.shape[1] - 1, 1)))
    if not rows:
        sys.exit("no fp32 checkpoints found")
    df = pd.DataFrame(rows).sort_values(["model", "dataset", "seed", "layer"])
    df.to_csv(OUT, index=False)
    print(f"[wrote] {OUT} ({len(df)} rows)")
    print(df.groupby(["model", "layer"]).max_over_p99.describe().round(2).to_string())


if __name__ == "__main__":
    main()
