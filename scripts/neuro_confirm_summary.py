#!/usr/bin/env python3
"""Amendment 6 summary: five-seed confirmation of the frontier dominance results. Runs in .venv.

Comparisons (seed by seed, each against the fixed-point Vanilla SNN of the same seed):
  Edge-IIoTset  Tree-to-Spike T = 10 (window-sweep variant)
  CICIoMT2024   Tree-to-Spike with C1-gate early exit (full window where a seed's gate disables exit)
Dominance (amendment 5's rule, on five-seed means): simulated test macro-F1 higher and mean SANA-FE
energy per inference no higher. Per-seed wins are reported. Context rows (descriptive): the other
Tree-to-Spike rows run for the same seeds. Seed 42 is rerun and checked against the frontier/A3 values.

    .venv/bin/python scripts/neuro_confirm_summary.py
Output: results/neuro/confirm/confirm.csv, CONFIRM.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
C = REPO / "results/neuro/confirm"
SEEDS = list(range(42, 47))
TESTS = {"edgeiiot": "T10", "ciciomt2024": "t2s_exit"}
NAMES = {"T10": "T2S, T = 10", "t2s_exit": "T2S, early exit (C1 gate)", "t2s_full": "T2S, full window (T = 20)",
         "vanilla_fixed": "Vanilla, fixed point"}


def seed42_check(df):
    out = []
    a3 = pd.read_csv(REPO / "results/neuro/a3_sanafe/energy_sanafe.csv")
    ft = pd.read_csv(REPO / "results/neuro/a2_lava/lava_fulltest.csv")
    for _, r in df[df.seed == 42].iterrows():
        if r.row == "T10":
            j = json.loads((REPO / f"results/neuro/frontier/{r.dataset}_T10.json").read_text())
            ref_e, ref_f = j["energy_nJ_mean"], j["simulated_macro_f1_full_test"]
        else:
            e = a3[(a3.dataset == r.dataset) & (a3.variant == r.row) & (a3.mapping == "base")]
            f = ft[(ft.dataset == r.dataset) & (ft.model == {"t2s_full": "t2s", "t2s_exit": "t2s_exit", "vanilla_fixed": "vanilla"}[r.row])]
            if not len(e) or not len(f):
                continue
            ref_e, ref_f = float(e.energy_nJ_mean.iloc[0]), float(f.lava_arith_macro_f1.iloc[0])
        out.append(f"{r.dataset} {r.row}: energy {r.energy_nJ_mean:.4f} vs {ref_e:.4f}, macro-F1 {r.simulated_macro_f1_full_test:.4f} "
                   f"vs {ref_f:.4f}: {'identical' if abs(r.energy_nJ_mean - ref_e) < 1e-6 and abs(r.simulated_macro_f1_full_test - ref_f) < 1e-9 else 'DIFFERENT'}")
    return out


def main():
    df = pd.DataFrame([json.loads(p.read_text()) for p in sorted(C.glob("*_seed*_*.json"))])
    df.to_csv(C / "confirm.csv", index=False)
    md = ["# Amendment 6: five-seed confirmation of the frontier dominance results", "",
          "SANA-FE energy and latency per inference (each seed's 2,000-flow sample; about 10% calibration caveat) and test "
          "macro-F1 of the simulated network (bit-exact emulation of Lava's arithmetic, whole test split). "
          "Each Tree-to-Spike row is compared with the fixed-point Vanilla SNN of the same seed.", ""]
    for ds, test in TESTS.items():
        d = df[df.dataset == ds]
        md += [f"## {ds}: {NAMES[test]} against the fixed-point Vanilla", "",
               "| Seed | " + " | ".join(f"{NAMES[r]} energy (nJ) / macro-F1" for r in (test, "vanilla_fixed")) + " | Win |", "|--:|--:|--:|---|"]
        wins, have = 0, []
        for s in SEEDS:
            a = d[(d.seed == s) & (d.row == test)]; v = d[(d.seed == s) & (d.row == "vanilla_fixed")]
            if not len(a) or not len(v):
                md.append(f"| {s} | pending | pending | |"); continue
            a, v = a.iloc[0], v.iloc[0]
            w = bool(a.simulated_macro_f1_full_test > v.simulated_macro_f1_full_test and a.energy_nJ_mean <= v.energy_nJ_mean)
            wins += w; have.append(s)
            gate = "" if test != "t2s_exit" else (" (gate disabled: full window)" if pd.isna(a.get("exit_gate")) else f" (gate {a.exit_gate})")
            md.append(f"| {s} | {a.energy_nJ_mean:.0f} / {a.simulated_macro_f1_full_test:.2f}{gate} | "
                      f"{v.energy_nJ_mean:.0f} / {v.simulated_macro_f1_full_test:.2f} | {'yes' if w else 'no'} |")
        if len(have) == len(SEEDS):
            g = lambda r, k: d[(d.row == r) & d.seed.isin(have)][k]
            ea, ev = g(test, "energy_nJ_mean"), g("vanilla_fixed", "energy_nJ_mean")
            fa, fv = g(test, "simulated_macro_f1_full_test"), g("vanilla_fixed", "simulated_macro_f1_full_test")
            dom = fa.mean() > fv.mean() and ea.mean() <= ev.mean()
            md += ["", f"Five-seed means: {NAMES[test]} {ea.mean():.0f} ± {ea.std(ddof=1):.0f} nJ, {fa.mean():.2f} ± {fa.std(ddof=1):.2f}; "
                   f"Vanilla {ev.mean():.0f} ± {ev.std(ddof=1):.0f} nJ, {fv.mean():.2f} ± {fv.std(ddof=1):.2f}. "
                   f"Energy ratio {ea.mean() / ev.mean():.2f}; macro-F1 difference {fa.mean() - fv.mean():+.2f}.", "",
                   f"**Dominance on five-seed means: {'yes' if dom else 'no'}.** Per-seed wins: {wins} of {len(have)}.", ""]
        else:
            md += ["", f"**Incomplete: {len(have)} of {len(SEEDS)} seeds.**", ""]
        ctx = d[~d.row.isin([test, "vanilla_fixed"])]
        if len(ctx):
            md += ["Context (descriptive), mean over the seeds run:", "", "| Row | Seeds | Energy (nJ) | Latency (µs) | Macro-F1 |", "|---|--:|--:|--:|--:|"]
            for r, g in ctx.groupby("row"):
                md.append(f"| {NAMES.get(r, r)} | {len(g)} | {g.energy_nJ_mean.mean():.0f} | {g.latency_us_mean.mean():.1f} | "
                          f"{g.simulated_macro_f1_full_test.mean():.2f} |")
            md.append("")
    md += ["## Seed 42 rerun against the frontier and A3 values", ""] + [f"- {x}" for x in seed42_check(df)] + [""]
    (C / "CONFIRM.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
