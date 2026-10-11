#!/usr/bin/env python3
"""Arithmetic energy on the IoT four, with every system counted the same way.

Each system's synaptic operations, all layers, five seeds:

  Tree-to-Spike   stored synops (hidden layers) + encoder_channels * hidden[0]
  DNN-Tree        C*h0 + h0*h1 + h1*K, the dense twin's MACs, all three layers
  Vanilla SNN     its own recorded synops, which ALREADY counts both its layers
                  (see results/tables/vanilla_sops_double_count.md: the published
                  tables add num_features * hidden on top of this, which counts
                  its input layer twice and assumes every feature fires)

Priced at the Horowitz 45 nm 32-bit figures: 4.6 pJ per MAC, 0.9 pJ per AC. This
is ARITHMETIC ONLY. The memory-inclusive band is a separate model with its own
assumptions, and the band against the compact ensemble has been dropped.

    python scripts/iot_arithmetic.py
Output: results/tables/tab_iot_arithmetic.{csv,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
SEEDS = [42, 43, 44, 45, 46]
E_MAC, E_AC = 4.6, 0.9


def main():
    rows = []
    for ds in IOT:
        for s in SEEDS:
            rec = {}
            f = REPO / f"results/Tree-to-Spike/{ds}/seed_{s}/results.json"
            if f.exists():
                r = json.loads(f.read_text())
                h = r["config"]["hidden"]
                h0, h1 = (h[0], h[1]) if isinstance(h, (list, tuple)) else (64, 64)
                C = r["data"]["encoder_channels"]
                K = r["data"].get("num_classes")
                rec["t2s_sops"] = r["test_plus"]["spike_stats"]["synops_per_sample"] + C * h0
                rec["dnn_macs"] = C * h0 + h0 * h1 + h1 * K
            g = REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_{s}/results.json"
            if g.exists():
                v = json.loads(g.read_text())["test_plus"]["spike_stats"].get("synops_per_sample")
                rec["vanilla_sops"] = v
            if rec:
                rows.append(dict(dataset=ds, seed=s, **rec))
    d = pd.DataFrame(rows)
    d["E_t2s"] = d.t2s_sops * E_AC
    d["E_dnn"] = d.dnn_macs * E_MAC
    d["E_vanilla"] = d.vanilla_sops * E_AC
    g = d.groupby("dataset")
    out = pd.DataFrame({
        "T2S SOPs": g.t2s_sops.mean(), "DNN MACs": g.dnn_macs.mean(),
        "Vanilla SOPs": g.vanilla_sops.mean(),
        "T2S pJ": g.E_t2s.mean(), "DNN pJ": g.E_dnn.mean(), "Vanilla pJ": g.E_vanilla.mean()})
    out["DNN / T2S"] = out["DNN pJ"] / out["T2S pJ"]
    out["Vanilla / T2S"] = out["Vanilla pJ"] / out["T2S pJ"]
    out = out.reindex(IOT)
    print("five-seed means, IoT four, arithmetic only")
    print(out.round(2).to_string())
    print(f"\nratio of means, DNN-Tree / Tree-to-Spike: "
          f"{out['DNN pJ'].mean()/out['T2S pJ'].mean():.2f}x")
    print(f"ratio of means, Vanilla SNN / Tree-to-Spike: "
          f"{out['Vanilla pJ'].mean()/out['T2S pJ'].mean():.2f}x "
          f"(below 1 means the conventional SNN does less arithmetic)")
    out.to_csv(REPO / "results/tables/tab_iot_arithmetic.csv")
    cols = ["dataset"] + list(out.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for ds, r in out.iterrows():
        md.append("| " + ds + " | " + " | ".join(f"{v:,.2f}" for v in r) + " |")
    (REPO / "results/tables/tab_iot_arithmetic.md").write_text("\n".join(md) + "\n")
    print(f"\n[wrote] {REPO/'results/tables/tab_iot_arithmetic.csv'} and .md")


if __name__ == "__main__":
    main()
