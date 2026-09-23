#!/usr/bin/env python3
"""M5: how many retained features violate the TTFS resolution bound T >= |E_j| + 1.

Read directly from the encoder tables saved by the repaired M3 grid
(results/diagnostics/val_curves/Tree-to-Spike/<ds>/runs/seed_<S>/encoder_channels.json).
Each channel records its feature and ``n_edges`` = |E_j|, the number of distinct
teacher split thresholds on that feature. A retained feature violates the bound
when |E_j| >= T, i.e. its thresholds need more distinct spike times than the
window has. No refit.

Supersedes results/Journal_results/encoder_edge_stats.json (2026-07-30) and
results/edge_stats_all.json (2026-09-09), which both predate the leakage repair.

    python scripts/m5_edges.py
Output: results/tables/tab_m5_edges.{csv,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/diagnostics/val_curves/Tree-to-Spike"
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]


def main():
    rows = []
    for f in sorted(ROOT.glob("*/runs/seed_*/encoder_channels.json")):
        ds = f.parts[-4]
        seed = int(f.parent.name.split("_")[1])
        T = int(yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))["T"])
        ch = json.loads(f.read_text())["channels"]
        E = {}
        for c in ch:
            E[int(c["feature"])] = int(c["n_edges"])        # identical across a feature's channels
        e = np.array(list(E.values()))
        rows.append(dict(dataset=ds, seed=seed, T=T, retained=len(e), violating=int((e >= T).sum()),
                         frac=float((e >= T).mean()), median_E=float(np.median(e)), max_E=int(e.max()),
                         at_or_below_1=int((e <= 1).sum())))
    d = pd.DataFrame(rows)
    assert (d.groupby("dataset").seed.nunique() == 5).all()
    g = d.groupby("dataset").agg(retained=("retained", "mean"), violating=("violating", "mean"),
                                 frac=("frac", "mean"), frac_sd=("frac", "std"),
                                 median_E=("median_E", "mean"), max_E=("max_E", "max"))
    g["iot"] = g.index.isin(IOT)
    g = g.sort_values(["iot", "frac"], ascending=[False, False])
    g.to_csv(REPO / "results/tables/tab_m5_edges.csv")
    print(g.round(3).to_string())
    for lab, sub in (("IoT four", d[d.dataset.isin(IOT)]), ("all eleven", d)):
        print(f"{lab}: {sub.violating.sum()} of {sub.retained.sum()} retained features violate "
              f"T >= |E_j|+1 at T = 20 ({100*sub.violating.sum()/sub.retained.sum():.1f}%); "
              f"per-cell mean {100*sub.frac.mean():.1f}%")
    md = ["| Dataset | Retained features | Violating | Share violating (± SD over seeds) | Median |E_j| | Max |E_j| |",
          "|---|---|---|---|---|---|"]
    for ds, r in g.iterrows():
        md.append(f"| {ds}{' (IoT)' if r.iot else ''} | {r.retained:.1f} | {r.violating:.1f} | "
                  f"{100*r.frac:.1f}% ± {100*r.frac_sd:.1f} | {r.median_E:.0f} | {int(r.max_E)} |")
    iot = d[d.dataset.isin(IOT)]
    md.append(f"\nIoT four, pooled: {iot.violating.sum()} of {iot.retained.sum()} retained features "
              f"({100*iot.violating.sum()/iot.retained.sum():.1f}%) violate T >= |E_j| + 1 at T = 20. "
              "Five seeds per dataset, read from the saved encoder tables.")
    (REPO / "results/tables/tab_m5_edges.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
