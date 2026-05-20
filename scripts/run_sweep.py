#!/usr/bin/env python3
"""Run the Phase-1 SNN sweep for one dataset.

Discovers every ``experiments/<dataset>/SNN_<Neuron>/<encoding>/run.py``,
runs each as a subprocess, optionally over multiple seeds, and prints one
status line per (combo, seed). Skips runs whose ``results.json`` already exists
unless ``--force`` is passed.

Multi-seed
----------
``--seeds 42 43 44`` runs the whole grid three times. Each seed writes its
results under ``results/<dataset>/SNN_<N>/<enc>/seed_<S>/`` — different seeds
never overwrite each other. ``results/summary.csv`` keys on
``(dataset, neuron, encoding, seed)`` for dedup.

GPU pre-flight
--------------
The orchestrator prints the visible CUDA devices + current utilisation up
front so you can decide whether to wait for an idle GPU. Pin to a specific
GPU with ``CUDA_VISIBLE_DEVICES=N python scripts/run_sweep.py ...``.

Usage
-----
    python scripts/run_sweep.py nslkdd                          # 27 runs, seed=42
    python scripts/run_sweep.py nslkdd --seeds 42 43 44         # 27 × 3 = 81 runs
    python scripts/run_sweep.py nslkdd --only Leaky --encodings rate
    python scripts/run_sweep.py nslkdd --force
    python scripts/run_sweep.py nslkdd --dry-run
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

NEURONS = ["Alpha", "Lapicque", "Leaky", "LeakyParallel",
           "RLeaky", "RSynaptic", "SConv2dLSTM", "SLSTM", "Synaptic"]
ENCODINGS = ["rate", "latency", "delta"]


# ---------------------------------------------------------------- GPU pre-flight
def gpu_preflight() -> None:
    """Print visible CUDA devices and their utilisation/memory. Soft: any
    failure prints a hint and proceeds (e.g. CPU-only host or no nvidia-smi)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print("---- GPU pre-flight ----")
    print(f"CUDA_VISIBLE_DEVICES = {visible if visible is not None else '(unset)'}")
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        print(f"{'idx':>3}  {'name':24}  {'util%':>6}  {'mem_used_MB':>12}  {'mem_total_MB':>13}")
        for line in out.strip().splitlines():
            idx, name, util, used, total = [p.strip() for p in line.split(",")]
            print(f"{idx:>3}  {name:24}  {util:>6}  {used:>12}  {total:>13}")
        print("Tip: pin to a specific GPU with  CUDA_VISIBLE_DEVICES=<idx>  before this script.")
    except FileNotFoundError:
        print("nvidia-smi not found — assuming CPU-only.")
    except Exception as e:
        print(f"  warn: nvidia-smi probe failed ({e}) — proceeding.")
    print()


# ---------------------------------------------------------------- discovery + status
def discover(dataset: str, only: list[str], encodings: list[str]):
    base = REPO_ROOT / "experiments" / dataset
    if not base.exists():
        sys.exit(f"error: {base} does not exist")
    runs = []
    for n in NEURONS:
        if only and n.lower() not in {o.lower() for o in only}:
            continue
        for e in encodings:
            run_py = base / f"SNN_{n}" / e / "run.py"
            if run_py.exists():
                runs.append((n, e, run_py))
    return runs


def status_for(dataset: str, neuron: str, encoding: str, seed: int) -> str:
    rj = (REPO_ROOT / "results" / dataset / f"SNN_{neuron}" / encoding
          / f"seed_{seed}" / "results.json")
    if not rj.exists():
        return "pending"
    try:
        with open(rj) as f:
            r = json.load(f)
        m = r["test_plus"]["metrics"]
        return (f"done acc={m['accuracy']:.3f} f1m={m['f1_macro']:.3f} "
                f"MCC={m['matthews_corrcoef']:.3f}")
    except Exception:
        return "done (unparseable)"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset name (e.g. nslkdd)")
    ap.add_argument("--only", nargs="+", default=[],
                    help="restrict to neurons whose name exactly matches (case-insensitive)")
    ap.add_argument("--encodings", nargs="+", default=ENCODINGS,
                    choices=ENCODINGS, help="restrict to specific encodings")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="one or more seeds (default: 42). Each seed writes its own "
                         "results/<dataset>/SNN_<N>/<enc>/seed_<S>/ directory.")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if results.json exists (overrides skip-guard)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan, run nothing")
    args = ap.parse_args()

    runs = discover(args.dataset, args.only, args.encodings)
    if not runs:
        sys.exit("no matching runs found")

    total_jobs = len(runs) * len(args.seeds)
    gpu_preflight()
    print(f"dataset       : {args.dataset}")
    print(f"combinations  : {len(runs)} (neuron × encoding)")
    print(f"seeds         : {args.seeds}")
    print(f"total jobs    : {total_jobs}")
    print(f"force re-run  : {args.force}")
    print()
    print(f"{'#':>3}  {'neuron':14}  {'enc':8}  {'seed':>4}  status (pre-run)")
    print("-" * 88)
    i = 0
    for s in args.seeds:
        for n, e, _ in runs:
            i += 1
            print(f"{i:3d}  {n:14}  {e:8}  {s:>4}  {status_for(args.dataset, n, e, s)}")
    print()
    if args.dry_run:
        return

    py = sys.executable
    overall_start = time.time()
    failures: list[tuple[str, str, int, str]] = []
    i = 0
    for s in args.seeds:
        env = dict(os.environ)
        env["HISNN_SEED"] = str(s)
        if args.force:
            env["HISNN_FORCE"] = "1"
        for n, e, run_py in runs:
            i += 1
            label = f"SNN_{n}/{e} seed={s}"
            print(f"\n[{i}/{total_jobs}] {label}")
            t0 = time.time()
            try:
                subprocess.run([py, str(run_py)], check=True, cwd=REPO_ROOT, env=env)
                dt = time.time() - t0
                print(f"  ok in {dt:.1f}s — {status_for(args.dataset, n, e, s)}")
            except subprocess.CalledProcessError as ex:
                dt = time.time() - t0
                print(f"  FAIL in {dt:.1f}s (exit {ex.returncode})")
                failures.append((n, e, s, str(ex.returncode)))

    total = time.time() - overall_start
    print("\n" + "=" * 88)
    print(f"sweep complete in {total/60:.1f} min   "
          f"{total_jobs - len(failures)}/{total_jobs} ok, {len(failures)} failed")
    if failures:
        for n, e, s, code in failures:
            print(f"  FAIL  SNN_{n}/{e}  seed={s}  exit={code}")
        sys.exit(1)


if __name__ == "__main__":
    main()
