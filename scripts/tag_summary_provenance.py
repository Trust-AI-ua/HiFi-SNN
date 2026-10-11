#!/usr/bin/env python3
"""Flag every row of the summary CSVs with its provenance (never strip rows).

After the leakage repair the summary rows live in two places:
  results/archive/pre_repair/summary_snapshot/summary*.csv  pre-repair state (snapshot
      taken before any re-run finished; identical to results/tables/ at the time)
  results/summary*.csv                             rows written by the re-runs
      (the experiment configs still name the pre-move path, so the re-runs wrote
      fresh files there, containing post-repair rows only)

This merges them into results/tables/summary*.csv with a `row_provenance` column:
  unaffected           dataset not touched by the repair
  pre_repair           a headline configuration on CSE-CIC-IDS2018 / Edge-IIoTset,
                       trained on the contaminated corpus
  phase1_contaminated  a Phase-1 (neuron x encoding) configuration on a
                       contaminated corpus; never reported
  post_repair          re-run on the repaired corpus
Readers must filter, e.g. keep {"unaffected", "post_repair"} for current numbers
(use src.results_io.read_summary). Idempotent: re-running rebuilds from the two
sources, so it can be repeated after further re-runs.

    python scripts/tag_summary_provenance.py [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "results/archive/pre_repair/summary_snapshot"
NEW = REPO / "results"
OUT = REPO / "results/tables"
AFFECTED = {"cse-cic-ids2018", "edgeiiot"}
FILES = ["summary.csv", "summary_dnn.csv", "summary_tasnn.csv", "summary_treespike.csv",
         "summary_snn_efficiency.csv"]


def headline(row, fname):
    """True for the configuration the paper reports (vs Phase-1 extras)."""
    if fname in ("summary.csv", "summary_snn_efficiency.csv"):
        return row.get("neuron") == "LeakyParallel" and row.get("encoding") == "latency"
    return True


def tag_old(df, fname):
    def one(r):
        if r["dataset"] not in AFFECTED:
            return "unaffected"
        return "pre_repair" if headline(r, fname) else "phase1_contaminated"
    return df.apply(one, axis=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for f in FILES:
        snap = SNAP / f
        if not snap.exists():
            print(f"[skip] no snapshot for {f}")
            continue
        old = pd.read_csv(snap)
        old["row_provenance"] = tag_old(old, f)
        parts = [old]
        new_p = NEW / f
        if new_p.exists():
            new = pd.read_csv(new_p)
            bad = new[~new.dataset.isin(AFFECTED)]
            if len(bad):
                raise SystemExit(f"{new_p}: {len(bad)} rows for unaffected datasets; "
                                 "unexpected, refusing to merge")
            new["row_provenance"] = "post_repair"
            parts.append(new)
        out = pd.concat(parts, ignore_index=True)
        counts = out.row_provenance.value_counts().to_dict()
        print(f"{f}: {len(out)} rows {counts}")
        if not a.dry_run:
            out.to_csv(OUT / f, index=False)
    if a.dry_run:
        print("(dry run: nothing written)")


if __name__ == "__main__":
    main()
