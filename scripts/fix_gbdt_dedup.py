#!/usr/bin/env python3
"""One-off: drop rows the keyed merge failed to supersede in gbdt_compression.csv.

`gain_prune` holds "none" and numbers, so a re-read yields the string "0.9" where
the writing process held the float 0.9; the two never matched and the pre-repair
rows for a re-run dataset survived alongside the new ones. scripts/gbdt_compression.py
now normalises the column; this repairs the file written before that fix by keeping,
for each key, the row with the newest run_id (rows without one are pre-repair).

    python scripts/fix_gbdt_dedup.py [--dry-run]
"""
import argparse
from pathlib import Path
import pandas as pd

P = Path(__file__).resolve().parents[1] / "results/deployment/gbdt_compression.csv"
KEY = ["dataset", "n_classes", "n_estimators", "max_depth", "gain_prune", "leaf_bits"]

ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
d = pd.read_csv(P)
d["gain_prune"] = d["gain_prune"].astype(str)
if "run_id" not in d:
    d["run_id"] = pd.NA
before = len(d)
d = d.sort_values("run_id", na_position="first").drop_duplicates(subset=KEY, keep="last")
print(f"{before} -> {len(d)} rows; per dataset: {d.groupby('dataset').size().to_dict()}")
stale = d[d.dataset.isin(["cse-cic-ids2018", "edgeiiot"]) & d.run_id.isna()]
if len(stale):
    print(f"WARNING: {len(stale)} rows for a re-run dataset still lack a run_id")
if not a.dry_run:
    d.to_csv(P, index=False)
    print(f"[wrote] {P}")
