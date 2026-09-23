#!/usr/bin/env python3
"""The four-step decomposition between the conventional SNN and Tree-to-Spike.

Each step changes one thing, and the arms either exist already or come from
scripts/run_m8_arm.py:

    1. Vanilla SNN (fp32)   -> M8 arm (fp32)    student architecture and training
    2. M8 arm (fp32)        -> M8 arm (int8)    weight precision
    3. M8 arm (int8)        -> no-KD variant    encoder (raw latency -> tree-guided TTFS)
    4. no-KD variant        -> Tree-to-Spike    teacher-posterior supervision

CAVEAT, stated because it is not removable without another 20 runs: step 1 also
changes the time window, because the M8 arm runs at the Tree-to-Spike T = 20
while the conventional SNN baseline runs at its own T = 25. Steps 2 to 4 each
change exactly one thing.

At N = 4 datasets this reports the consistency of the SIGN across datasets, not
significance.

    python scripts/m8_decomposition.py
Output: results/tables/tab_m8_decomposition.{csv,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
SEEDS = [42, 43, 44, 45, 46]


def f1_of(path: Path, key="test_plus"):
    if not path.exists():
        return np.nan
    d = json.loads(path.read_text())
    return 100 * d[key]["metrics"]["f1_macro"]


def collect() -> pd.DataFrame:
    rows = []
    for ds in IOT:
        for s in SEEDS:
            m8 = REPO / f"results/encoder_ablation/{ds}/m8_raw/seed_{s}/test_fp32_int8.json"
            m8d = json.loads(m8.read_text()) if m8.exists() else {}
            rows.append(dict(
                dataset=ds, seed=s,
                vanilla=f1_of(REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_{s}/results.json"),
                arm_fp32=m8d.get("fp_macro_f1", np.nan),
                arm_int8=m8d.get("int8_macro_f1", np.nan),
                nokd=f1_of(REPO / f"results/encoder_ablation/{ds}/nokd/seed_{s}/results.json"),
                t2s=f1_of(REPO / f"results/Tree-to-Spike/{ds}/seed_{s}/results.json")))
    return pd.DataFrame(rows)


STEPS = [("1 architecture+training (and T 25->20)", "vanilla", "arm_fp32"),
         ("2 weight precision (fp32 -> int8)", "arm_fp32", "arm_int8"),
         ("3 encoder (raw latency -> tree-guided TTFS)", "arm_int8", "nokd"),
         ("4 teacher-posterior supervision", "nokd", "t2s")]


def main():
    r = collect()
    out = REPO / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "tab_m8_decomposition_per_seed.csv", index=False)
    have = r.dropna(subset=["arm_fp32"])
    if len(have) < len(r):
        print(f"[partial] {len(have)}/{len(r)} cells have the M8 arm; "
              f"missing {sorted(set(map(tuple, r[r.arm_fp32.isna()][['dataset','seed']].values)))}")

    per = r.groupby("dataset").mean(numeric_only=True).drop(columns="seed")
    sd = r.groupby("dataset").std(numeric_only=True).drop(columns="seed")
    print("\nfive-seed mean macro-F1 per arm")
    print(per.round(2).to_string())
    print("\n(SD)")
    print(sd.round(2).to_string())

    tab = []
    for name, a, b in STEPS:
        d = per[b] - per[a]
        row = {"step": name, **{ds: round(float(d[ds]), 2) for ds in IOT if ds in d.index}}
        vals = [d[ds] for ds in IOT if ds in d.index and not np.isnan(d[ds])]
        row["IoT mean"] = round(float(np.mean(vals)), 2) if vals else np.nan
        row["sign consistent"] = ("yes, all " + ("+" if vals[0] > 0 else "-")
                                  if vals and all(np.sign(v) == np.sign(vals[0]) for v in vals)
                                  else "no")
        tab.append(row)
    t = pd.DataFrame(tab)
    print("\nfour-step decomposition, macro-F1 points (N = 4 datasets, sign consistency)")
    print(t.to_string(index=False))
    t.to_csv(out / "tab_m8_decomposition.csv", index=False)
    cols = list(t.columns)
    md = ["| " + " | ".join(cols) + " |",
          "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in t.values]
    (out / "tab_m8_decomposition.md").write_text("\n".join(md) + "\n")
    tot = per["t2s"] - per["vanilla"]
    print("\ntotal Vanilla SNN -> Tree-to-Spike:")
    print(tot.round(2).to_string())
    print(f"\n[wrote] {out/'tab_m8_decomposition.csv'} and .md")


if __name__ == "__main__":
    main()
