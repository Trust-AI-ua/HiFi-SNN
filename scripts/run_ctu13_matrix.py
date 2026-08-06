#!/usr/bin/env python3
"""CTU-13 model matrix under the three aggregation/partition conditions.

Matrix A -- architecture comparison, combined features
    {gbdt, vanilla, treespike, dnn, tasnn}
  x {legacy_random, causal_random, causal_scenario}
  x 4 folds x seeds 42-46                                        = 300 cells

Matrix B -- feature-block control
    {gbdt, treespike, dnn} x {base, combined}
  x {causal_random, causal_scenario}
  x 4 folds x seeds 42-46                                        = 240 cells

Matrix B is a 240-cell factorial, but its `combined` half (3 x 2 x 4 x 5 = 120
cells) is the same computation as the corresponding Matrix A cells -- identical
model, condition, feature block, fold and seed. Result directories are keyed by
the full tuple, so those 120 are served by the existing skip-guard rather than
trained twice. Matrix B therefore adds 120 new runs and the union is

    300 + 120 = 420 distinct runs.

Nothing is approximated: one run serves both matrices.

Folds are the statistical unit; seeds are optimisation variability within a fold.
The random partitions are keyed on the fold, so every model and seed sees an
identical partition per (condition, fold).

    python scripts/run_ctu13_matrix.py --dry-run
    python scripts/run_ctu13_matrix.py --matrix both --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FOLDS = [0, 1, 2, 3]
SEEDS = [42, 43, 44, 45, 46]
CONDITIONS = ["legacy_random", "causal_random", "causal_scenario"]

A_MODELS = ["gbdt", "vanilla", "treespike", "dnn", "tasnn"]
A_CONDITIONS = CONDITIONS
A_FEATURES = ["combined"]

B_MODELS = ["gbdt", "treespike", "dnn"]
B_CONDITIONS = ["causal_random", "causal_scenario"]
B_FEATURES = ["base", "combined"]

_CFG_SRC = {"treespike": "Tree-to-Spike/toniot", "dnn": "DNN-Tree/toniot",
            "tasnn": "TASNN/toniot", "gbdt": "Tree-to-Spike/toniot",
            "vanilla": "Tree-to-Spike/toniot"}


def result_dir(model: str, condition: str, feature_set: str, fold: int) -> Path:
    return REPO / f"results/ctu13/matrix/{condition}/{feature_set}/{model}/fold{fold}"


def build_cfg(model: str, condition: str, feature_set: str, fold: int,
              seed: int) -> dict:
    """Base config from the TON-IoT experiment; only the dataset name changes, so
    architecture and optimisation stay identical across conditions."""
    cfg = yaml.safe_load(open(REPO / f"experiments/{_CFG_SRC[model]}/config.yaml"))
    if model == "vanilla":      # Vanilla SNN keeps its own architecture
        cfg = {"dataset": "toniot", "num_steps": 25, "hidden_size": 128,
               "neuron": "LeakyParallel", "encoding": "latency",
               "batch_size": 128, "epochs": 10, "lr": 1e-3}
    rd = result_dir(model, condition, feature_set, fold)
    cfg.update(dataset=f"ctu13_{condition}_f{fold}_{feature_set}", seed=seed,
               results_dir=str(rd), summary_csv=str(rd / f"seed_{seed}" / "summary.csv"))
    cfg.pop("label_mode", None)          # condition fixes the label space
    cfg.pop("subsample_fraction", None)
    return cfg


def cells(matrix: str) -> list:
    out = []
    if matrix in ("A", "both"):
        out += [(m, c, fs, f, s) for m in A_MODELS for c in A_CONDITIONS
                for fs in A_FEATURES for f in FOLDS for s in SEEDS]
    if matrix in ("B", "both"):
        out += [(m, c, fs, f, s) for m in B_MODELS for c in B_CONDITIONS
                for fs in B_FEATURES for f in FOLDS for s in SEEDS]
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="both", choices=["A", "B", "both"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--single", nargs=5,
                    metavar=("MODEL", "CONDITION", "FEATURESET", "FOLD", "SEED"))
    args = ap.parse_args()

    if args.single:
        model, condition, feature_set = args.single[0], args.single[1], args.single[2]
        fold, seed = int(args.single[3]), int(args.single[4])
        if model == "treespike":
            from src.training_treespike import run_experiment
        elif model == "dnn":
            from src.training_dnn import run_experiment
        elif model == "vanilla":
            from src.training import run_experiment
        elif model == "tasnn":
            from src.training_tasnn import run_experiment
        else:
            from scripts.run_gbdt_leakage import run_experiment
        rec = run_experiment(build_cfg(model, condition, feature_set, fold, seed))
        m = rec["test_plus"]["metrics"]
        b = rec["test_plus"].get("binary_ids_metrics") or {}
        pg = m.get("per_group", {}).get("groups", {})
        print(f"[done] {model}/{condition}/{feature_set}/fold{fold}/seed_{seed} "
              f"f1m={m['f1_macro']:.4f} mcc={m['matthews_corrcoef']:.4f} "
              f"DR={b.get('detection_rate', 0):.4f} FAR={b.get('false_alarm_rate', 0):.4f}",
              flush=True)
        for g, blk in pg.items():
            print(f"        scenario {g}: f1m={blk['macro_f1']:.4f} "
                  f"DR={blk['DR']:.4f} FAR={blk['FAR']:.4f} n={blk['n']}", flush=True)
        return

    todo_all = cells(args.matrix)
    if args.models:
        todo_all = [c for c in todo_all if c[0] in args.models]
    if args.folds:
        todo_all = [c for c in todo_all if c[3] in args.folds]
    if args.seeds:
        todo_all = [c for c in todo_all if c[4] in args.seeds]
    todo = [c for c in todo_all
            if not (result_dir(c[0], c[1], c[2], c[3]) / f"seed_{c[4]}"
                    / "results.json").exists()]
    print(f"[ctu13-matrix] {len(todo_all)} distinct cells, {len(todo)} pending "
          f"({len(todo_all)-len(todo)} already done)", flush=True)
    if args.dry_run:
        for t in todo[:40]:
            print("  would run", t)
        if len(todo) > 40:
            print(f"  ... and {len(todo)-40} more")
        return

    t0, n = time.time(), [0]
    lock = threading.Lock()

    def one(job):
        model, condition, feature_set, fold, seed = job
        cmd = [sys.executable, str(Path(__file__).resolve()), "--single",
               model, condition, feature_set, str(fold), str(seed)]
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", str(max(1, 8 // max(args.workers, 1))))
        for attempt in (1, 2):
            r = subprocess.run(cmd, cwd=str(REPO), env=env)
            if r.returncode == 0:
                break
            print(f"[FAIL] {job}: exit {r.returncode}"
                  f"{' (retry)' if attempt == 1 else ' (giving up)'}", flush=True)
            time.sleep(10)
        with lock:
            n[0] += 1
            el = time.time() - t0
            print(f"[ctu13-matrix {n[0]}/{len(todo)}] {model}/{condition}/"
                  f"{feature_set}/f{fold}/s{seed} | elapsed {el/3600:.2f} h "
                  f"| eta {el/n[0]*(len(todo)-n[0])/3600:.1f} h", flush=True)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, todo))
    print(f"[ctu13-matrix] finished in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
