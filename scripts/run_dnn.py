#!/usr/bin/env python3
"""Run the DNN-Tree (iso-architecture ANN twin of Tree-to-Spike) for one dataset
over one or more seeds. Mirrors scripts/run_treespike.py: dataset is a required
positional arg; each seed writes results/DNN-Tree/<dataset>/seed_<S>/.
Skip-guard: runs with an existing results.json are skipped unless --force.

    python scripts/run_dnn.py kddcup99
    python scripts/run_dnn.py kddcup99 --seeds 42 43 44 45 46
    python scripts/run_dnn.py cse-cic-ids2018 --seeds 42 --force
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DNN_DIR = REPO_ROOT / "experiments" / "DNN-Tree"


def available_datasets():
    if not DNN_DIR.exists():
        return []
    return sorted(d.name for d in DNN_DIR.iterdir() if d.is_dir() and (d / "run.py").exists())


def status_for(dataset, seed):
    rj = REPO_ROOT / "results" / "DNN-Tree" / dataset / f"seed_{seed}" / "results.json"
    if not rj.exists():
        return "pending"
    try:
        r = json.load(open(rj)); m = r["test_plus"]["metrics"]; c = r["test_plus"]["compute"]
        return f"done acc={m['accuracy']:.3f} f1m={m['f1_macro']:.3f} MACs={c['macs_per_sample']}"
    except Exception:
        return "done (unparseable)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_py = DNN_DIR / args.dataset / "run.py"
    if not run_py.exists():
        sys.exit(f"error: no DNN-Tree script at {run_py}\navailable: {available_datasets()}")

    print(f"model        : DNN-Tree (iso-architecture ANN twin of Tree-to-Spike)")
    print(f"dataset      : {args.dataset}\nseeds        : {args.seeds}\nforce        : {args.force}\n")
    for s in args.seeds:
        print(f"  seed {s:>4}  {status_for(args.dataset, s)}")
    if args.dry_run:
        return

    py = sys.executable; t0 = time.time(); failures = []
    for i, s in enumerate(args.seeds, 1):
        env = dict(os.environ); env["HISNN_SEED"] = str(s)
        if args.force:
            env["HISNN_FORCE"] = "1"
        print(f"\n[{i}/{len(args.seeds)}] DNN-Tree/{args.dataset} seed={s}")
        t = time.time()
        try:
            subprocess.run([py, str(run_py)], check=True, cwd=REPO_ROOT, env=env)
            print(f"  ok in {time.time()-t:.1f}s — {status_for(args.dataset, s)}")
        except subprocess.CalledProcessError as ex:
            print(f"  FAIL in {time.time()-t:.1f}s (exit {ex.returncode})")
            failures.append((s, ex.returncode))
    print("\n" + "=" * 60)
    print(f"DNN-Tree/{args.dataset} complete in {(time.time()-t0)/60:.1f} min   "
          f"{len(args.seeds)-len(failures)}/{len(args.seeds)} ok, {len(failures)} failed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
