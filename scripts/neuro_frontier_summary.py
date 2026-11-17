#!/usr/bin/env python3
"""Energy-accuracy frontier summary (exploratory; results/preregistration/2026-11-17_neuro_stream.md,
amendment 5). Runs in .venv.

Rows per dataset: Tree-to-Spike full window and with C1-gate early exit and the fixed-point Vanilla
SNN (A3, results/neuro/a3_sanafe/energy_sanafe.csv, base mapping), and the Tree-to-Spike variants
(results/neuro/frontier/<ds>_<tag>.json). Energy and latency: SANA-FE per inference on the
2,000-flow sample, mean and SD (calibration caveat about 10%). Accuracy: test macro-F1 of the
simulated network (bit-exact emulation of Lava's arithmetic on the whole test split;
results/neuro/a2_lava/lava_fulltest.csv). Host encoder comparisons are listed separately.

Dominance (amendment 5): a Tree-to-Spike row dominates the fixed-point Vanilla on a dataset if
its simulated macro-F1 is higher and its mean energy is no higher. Pareto: no other row of the
dataset has both energy <= and macro-F1 >= with one strict.

    .venv/bin/python scripts/neuro_frontier_summary.py
Output: results/neuro/frontier/frontier.csv, FRONTIER.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FR = REPO / "results/neuro/frontier"
DS = ["toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
NAMES = {"t2s_full": "T2S, full window", "t2s_exit": "T2S, early exit (C1 gate)", "vanilla_fixed": "Vanilla, fixed point",
         "T10": "T2S variant: T = 10", "uniform": "T2S variant: uniform allocation", "q40": "T2S variant: prune q = 0.40",
         "M2": "T2S variant: pop_max 2", "prune_only": "T2S variant: pop_max 1 (prune q = 0.20)"}


def main():
    e = pd.read_csv(REPO / "results/neuro/a3_sanafe/energy_sanafe.csv")
    e = e[e.mapping == "base"]
    ft = pd.read_csv(REPO / "results/neuro/a2_lava/lava_fulltest.csv")
    fmap = {"t2s_full": "t2s", "t2s_exit": "t2s_exit", "vanilla_fixed": "vanilla"}
    rows = []
    for ds in DS:
        for v in ("t2s_full", "t2s_exit", "vanilla_fixed"):
            r = e[(e.dataset == ds) & (e.variant == v)].iloc[0]
            f = ft[(ft.dataset == ds) & (ft.model == fmap[v])].iloc[0]
            ch = int(np.load(REPO / f"results/neuro/inputs/{ds}_seed42.npz")["W1"].shape[1]) if v != "vanilla_fixed" else None
            rows.append(dict(dataset=ds, row=v, channels=ch, T=None, mean_steps=r.mean_steps,
                             energy_nJ_mean=r.energy_nJ_mean, energy_nJ_sd=r.energy_nJ_sd, latency_us_mean=r.latency_us_mean,
                             latency_us_sd=r.latency_us_sd, simulated_macro_f1=f.lava_arith_macro_f1,
                             host_encoder_comparisons=r.host_encoder_comparisons, sanafe_match_rate=r.reference_match_rate))
        for p in sorted(FR.glob(f"{ds}_*.json")):
            j = json.loads(p.read_text())
            rows.append(dict(dataset=ds, row=j["variant"], channels=j["channels"], T=j["T"], mean_steps=float(j["T"]),
                             energy_nJ_mean=j["energy_nJ_mean"], energy_nJ_sd=j["energy_nJ_sd"],
                             latency_us_mean=j["latency_us_mean"], latency_us_sd=j["latency_us_sd"],
                             simulated_macro_f1=j["simulated_macro_f1_full_test"],
                             host_encoder_comparisons=j["host_encoder_comparisons"], sanafe_match_rate=j["sanafe_match_rate"]))
    df = pd.DataFrame(rows)
    df["pareto"] = False; df["dominates_vanilla"] = None
    for ds in DS:
        d = df[df.dataset == ds]
        van = d[d.row == "vanilla_fixed"].iloc[0]
        for i, r in d.iterrows():
            others = d.drop(i)
            df.loc[i, "pareto"] = not ((others.energy_nJ_mean <= r.energy_nJ_mean) & (others.simulated_macro_f1 >= r.simulated_macro_f1)
                                       & ((others.energy_nJ_mean < r.energy_nJ_mean) | (others.simulated_macro_f1 > r.simulated_macro_f1))).any()
            if r.row != "vanilla_fixed":
                df.loc[i, "dominates_vanilla"] = bool(r.simulated_macro_f1 > van.simulated_macro_f1 and r.energy_nJ_mean <= van.energy_nJ_mean)
    df.to_csv(FR / "frontier.csv", index=False)
    md = ["# Energy-accuracy frontier (exploratory, amendment 5)", "",
          "SANA-FE energy and latency per inference (2,000-flow sample, mean ± SD; about 10% calibration caveat), and "
          "test macro-F1 of the simulated network (bit-exact emulation of Lava's arithmetic, whole test split). "
          "Host encoder comparisons are not in the energy.", ""]
    for ds in DS:
        d = df[df.dataset == ds].sort_values("energy_nJ_mean")
        md += [f"## {ds}", "", "| Network | Channels | Steps | Energy (nJ) | Latency (µs) | Macro-F1 (simulated) | Encoder comparisons | Pareto | Dominates Vanilla |",
               "|---|--:|--:|--:|--:|--:|--:|---|---|"]
        for _, r in d.iterrows():
            md.append(f"| {NAMES.get(r.row, r.row)} | {'' if pd.isna(r.channels) else int(r.channels)} | {r.mean_steps:.1f} | "
                      f"{r.energy_nJ_mean:.0f} ± {r.energy_nJ_sd:.0f} | {r.latency_us_mean:.1f} ± {r.latency_us_sd:.1f} | "
                      f"{r.simulated_macro_f1:.2f} | {'' if pd.isna(r.host_encoder_comparisons) else int(r.host_encoder_comparisons)} | "
                      f"{'yes' if r.pareto else ''} | {'' if r.dominates_vanilla is None else ('**yes**' if r.dominates_vanilla else 'no')} |")
        md.append("")
    (FR / "FRONTIER.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
