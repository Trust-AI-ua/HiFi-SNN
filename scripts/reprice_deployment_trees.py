#!/usr/bin/env python3
"""Correct the leaf pricing of the GBDT rows in the deployment measurement.

``pack_booster`` priced one value per leaf PER CLASS. A multiclass XGBoost
ensemble is K trees per round, each scoring its own class, so a leaf holds one
margin. The measurement is otherwise untouched: bytes are an exact function of
the tree structure, which the run already recorded as ``n_trees`` and
``n_nodes``, so the correction needs no refit and no re-timing.

Guard: the OLD price is reconstructed from those two counts and must reproduce
the stored ``model_bytes`` exactly before anything is rewritten. If it does not,
the stored number did not come from the formula being corrected, and the row is
left alone and reported.

    python scripts/reprice_deployment_trees.py [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results/deployment/deployment_measurements.csv"
LEAF_BITS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    header = [l for l in CSV.read_text().splitlines() if l.startswith("#")]
    d = pd.read_csv(CSV, comment="#")
    lb = LEAF_BITS // 8
    bad, changed = [], []
    for i, r in d[d.system.str.contains("gbdt")].iterrows():
        leaves = (r.n_nodes + r.n_trees) / 2          # binary trees: leaves = internals + trees
        internals = leaves - r.n_trees
        if leaves != int(leaves) or internals <= 0:
            bad.append((r.dataset, r.system, "node counts not a binary forest")); continue
        leaves, internals = int(leaves), int(internals)
        residual = r.model_bytes - 10 * internals     # 2 B feature + 4 B float32 thr + 4 B children
        per_leaf_old, rem = divmod(residual, leaves * lb)
        if rem or per_leaf_old < 1:
            bad.append((r.dataset, r.system, f"old price not reproduced (residual {residual})"))
            continue
        new_model = 10 * internals + leaves * lb
        changed.append(dict(dataset=r.dataset, system=r.system, per_leaf_old=int(per_leaf_old),
                            internals=internals, leaves=leaves,
                            old_bytes=int(r.model_bytes), new_bytes=int(new_model),
                            ratio=new_model / r.model_bytes))
        d.loc[i, "total_bytes"] = int(r.total_bytes - r.model_bytes + new_model)
        d.loc[i, "model_bytes"] = int(new_model)
    ch = pd.DataFrame(changed)
    print(ch.to_string(index=False) if len(ch) else "nothing to change")
    for b in bad:
        print(f"  [left alone] {b[0]} {b[1]}: {b[2]}")
    if a.dry_run:
        print("\n[dry run] nothing written")
        return
    if len(ch) == 0 or (ch.per_leaf_old == 1).all():
        # Already priced one value per leaf (a run made after src/serialize.py was
        # fixed). Rewriting would only re-serialize floats and stamp a header
        # claiming a reprice that did not happen.
        print("\n[no-op] every row already one value per leaf; nothing written")
        return
    shutil.copy2(CSV, CSV.with_suffix(".csv.pre_leaf_reprice"))
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    header.append(f"# leaf_reprice={stamp}-{sha} source=scripts/reprice_deployment_trees.py "
                  f"rows={len(ch)} note=one-value-per-leaf; timings and accuracies unchanged")
    CSV.write_text("\n".join(header) + "\n" + d.to_csv(index=False))
    print(f"\n[rewrote] {CSV}  (previous kept at {CSV.name}.pre_leaf_reprice)")


if __name__ == "__main__":
    main()
