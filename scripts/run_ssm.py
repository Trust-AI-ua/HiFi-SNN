#!/usr/bin/env python3
"""Run the Spiking-SSM hybrid (Design A) for one dataset over one or more seeds.

Mirrors ``scripts/run_sweep.py`` / ``run_sxlstm.py`` / ``run_griffin.py``: the
dataset is a required positional argument, so each invocation targets a single
dataset and datasets can be run in parallel in separate tmux sessions.

Each seed writes to ``results/Spiking-SSM/<dataset>/seed_<S>/``; the summary
``results/summary_ssm.csv`` keys on (model, dataset, seed) for dedup. Runs whose
``results.json`` already exists are skipped unless ``--force`` is passed.

Usage
-----
    python scripts/run_ssm.py kddcup99
    python scripts/run_ssm.py kddcup99 --seeds 42 43 44 45 46
    python scripts/run_ssm.py kddcup99 --seeds 44 45 46 --force
    python scripts/run_ssm.py kddcup99 --dry-run
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
SSM_DIR = REPO_ROOT / "experiments" / "Spiking-SSM"


def available_datasets():
    if not SSM_DIR.exists():
        return []
    return sorted(d.name for d in SSM_DIR.iterdir()
                  if d.is_dir() and (d / "run.py").exists())


def status_for(dataset: str, seed: int) -> str:
    rj = REPO_ROOT / "results" / "Spiking-SSM" / dataset / f"seed_{seed}" / "results.json"
    if not rj.exists():
        return "pending"
    try:
        r = json.load(open(rj))
        m = r["test_plus"]["metrics"]
        return (f"done acc={m['accuracy']:.3f} f1m={m['f1_macro']:.3f} "
                f"MCC={m['matthews_corrcoef']:.3f}")
    except Exception:
        return "done (unparseable)"


def gpu_preflight():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print("---- GPU pre-flight ----")
    print(f"CUDA_VISIBLE_DEVICES = {visible if visible is not None else '(unset)'}")
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        for line in out.strip().splitlines():
            print("  " + line)
    except FileNotFoundError:
        print("nvidia-smi not found — assuming CPU-only.")
    except Exception as e:
        print(f"  warn: nvidia-smi probe failed ({e}) — proceeding.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset name (e.g. kddcup99); one per invocation")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="one or more seeds (default: 42)")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if results.json exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan, run nothing")
    args = ap.parse_args()

    run_py = SSM_DIR / args.dataset / "run.py"
    if not run_py.exists():
        sys.exit(f"error: no Spiking-SSM script at {run_py}\n"
                 f"available datasets: {available_datasets()}")

    total_jobs = len(args.seeds)
    gpu_preflight()
    print(f"model         : Spiking-SSM (selective spiking state-space)")
    print(f"dataset       : {args.dataset}")
    print(f"seeds         : {args.seeds}")
    print(f"total jobs    : {total_jobs}")
    print(f"force re-run  : {args.force}")
    print()
    print(f"{'#':>3}  {'dataset':14}  {'seed':>4}  status (pre-run)")
    print("-" * 70)
    for i, s in enumerate(args.seeds, 1):
        print(f"{i:3d}  {args.dataset:14}  {s:>4}  {status_for(args.dataset, s)}")
    print()
    if args.dry_run:
        return

    py = sys.executable
    overall_start = time.time()
    failures = []
    for i, s in enumerate(args.seeds, 1):
        env = dict(os.environ)
        env["HISNN_SEED"] = str(s)
        if args.force:
            env["HISNN_FORCE"] = "1"
        print(f"\n[{i}/{total_jobs}] Spiking-SSM/{args.dataset} seed={s}")
        t0 = time.time()
        try:
            subprocess.run([py, str(run_py)], check=True, cwd=REPO_ROOT, env=env)
            print(f"  ok in {time.time()-t0:.1f}s — {status_for(args.dataset, s)}")
        except subprocess.CalledProcessError as ex:
            print(f"  FAIL in {time.time()-t0:.1f}s (exit {ex.returncode})")
            failures.append((s, ex.returncode))

    total = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"Spiking-SSM/{args.dataset} complete in {total/60:.1f} min   "
          f"{total_jobs - len(failures)}/{total_jobs} ok, {len(failures)} failed")
    if failures:
        for s, code in failures:
            print(f"  FAIL  Spiking-SSM/{args.dataset}  seed={s}  exit={code}")
        sys.exit(1)


if __name__ == "__main__":
    main()
