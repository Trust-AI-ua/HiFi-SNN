#!/usr/bin/env python3
"""Status of the attribution-ladder queue (scripts/run_ladder.py).

Prints whether the job is alive, a completion matrix of arms x seeds, any
failures, and an ETA derived from observed run times. Safe to run at any time;
reads only.

    python scripts/ladder_status.py
    python scripts/ladder_status.py --results   # + current macro-F1 per arm
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# Arm list is imported from the driver so the two can never drift apart.
from scripts.run_ladder import ARMS as _ARM_CFG, DATASETS, SEEDS  # noqa: E402

ARMS = list(_ARM_CFG)


def alive() -> str:
    try:
        out = subprocess.run(["pgrep", "-af", "run_ladder.py"], capture_output=True,
                             text=True).stdout.strip()
        rows = [l for l in out.splitlines() if "pgrep" not in l]
        return f"RUNNING (pid {rows[0].split()[0]})" if rows else "NOT RUNNING"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="store_true", help="also show macro-F1 per arm")
    ap.add_argument("--arms", nargs="+", default=ARMS,
                    help="restrict the view to these arms (e.g. raw_full tree_ce)")
    args = ap.parse_args()
    arms = [a for a in ARMS if a in args.arms]
    total = len(arms) * len(DATASETS) * len(SEEDS)

    done = {}
    for f in glob.glob(str(REPO / "results/ladder/*/*/seed_*/results.json")):
        p = Path(f).parts
        done[(p[-3], p[-4], int(p[-2].split("_")[1]))] = f   # (arm, dataset, seed)

    done = {k: v for k, v in done.items() if k[0] in arms}
    n = len(done)
    print(f"job: {alive()}")
    print(f"progress: {n}/{total} runs ({100*n/total:.1f}%)\n")

    print(f"{'arm':22s} " + " ".join(f"{s:>4d}" for s in SEEDS) + "   complete")
    for a in arms:
        cells, tot = [], 0
        for s in SEEDS:
            c = sum(1 for d in DATASETS if (a, d, s) in done)
            tot += c
            cells.append("  ok" if c == len(DATASETS) else (f"{c}/{len(DATASETS)}" if c else "   ."))
        flag = "  <-- COMPLETE" if tot == len(DATASETS) * len(SEEDS) else ""
        print(f"{a:22s} " + " ".join(f"{c:>4s}" for c in cells) + f"   {tot:2d}/20{flag}")

    # failures + ETA from the logs
    fails, times = [], []
    for lg in sorted(glob.glob(str(REPO / "results/ladder*.log"))):
        txt = Path(lg).read_text(errors="ignore")
        fails += re.findall(r"^\[FAIL\].*$", txt, re.M)
        for m in re.finditer(r"elapsed ([\d.]+) h", txt):
            times.append(float(m.group(1)))
    print(f"\nfailures: {len(fails)}")
    for f in fails[:5]:
        print(f"  {f}")

    rt = [json.load(open(f))["training"]["train_time_s"] for f in list(done.values())[:40]]
    if rt:
        per = np.median(rt) / 3600
        print(f"\nmedian run time: {per*60:.0f} min")
        print(f"remaining: {total-n} runs ~ {(total-n)*per:.0f} h "
              f"(~{(total-n)*per/24:.1f} days at current rate)")

    if args.results:
        print(f"\n{'arm':22s} {'macro-F1':>9s} {'n':>3s}")
        for a in arms:
            v = [json.load(open(p))["test_plus"]["metrics"]["f1_macro"] * 100
                 for (aa, _, _), p in done.items() if aa == a]
            if v:
                print(f"{a:22s} {np.mean(v):9.2f} {len(v):3d}")


if __name__ == "__main__":
    main()
