#!/usr/bin/env python3
"""Leakage-resistant evaluation (round-2 reviewer item 1).

Runs the comparator suite under three protocols so that the partition effect is
separable from the change in task granularity:

  toniot                      10-class, random flow-level   (standard benchmark,
                                                             already in results/)
  toniot_flow_binary          binary,   random flow-level   (CONTROL)
  toniot_hostdisjoint_binary  binary,   source-IP-disjoint  (PRIMARY)

The key comparison is binary-flow vs binary-host-disjoint: identical rows,
identical global de-duplication, identical encoding pipeline, only the partition
differs.

Folds, not seeds, are the statistical unit. Seeds measure optimisation
variability within a fold; the four folds are the independent test populations.
Aggregate predictions and metrics across folds -- never model parameters, whose
input dimension differs by fold because vocabularies are fitted per training
partition.

    python scripts/run_leakage.py --dry-run
    python scripts/run_leakage.py --workers 2
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

PROTOCOLS = ["toniot_flow_binary", "toniot_hostdisjoint_binary"]
FOLDS = [0, 1, 2, 3]              # outer partitions (frozen manifests)
SEEDS = [42, 43, 44, 45, 46]      # optimization seeds, INDEPENDENT of the fold
# Fold and seed were previously the same parameter, which gave one run per cell and
# conflated endpoint-composition shift with optimization noise. They are now crossed.
# Full comparator matrix. Reporting only Tree-to-Spike and its dense twin would
# invite the reading that the protocol was applied to a favourable pairing.
MODELS = ["treespike", "dnn", "vanilla", "tasnn", "gbdt"]


_CFG_SRC = {"treespike": "Tree-to-Spike/toniot", "dnn": "DNN-Tree/toniot",
            "tasnn": "TASNN/toniot", "gbdt": "Tree-to-Spike/toniot",
            "vanilla": "Tree-to-Spike/toniot"}


def build_cfg(model: str, protocol: str, fold: int, seed: int) -> dict:
    """Base config comes from the TON-IoT experiment; only the dataset changes.

    Every model reads the SAME frozen fold manifest via the dataset name, so no
    trainer reconstructs a partition of its own.
    """
    cfg = yaml.safe_load(open(REPO / f"experiments/{_CFG_SRC[model]}/config.yaml"))
    if model == "vanilla":     # Vanilla SNN keeps its own architecture (T=25, 128 hidden)
        cfg = {"dataset": "toniot", "num_steps": 25, "hidden_size": 128,
               "neuron": "LeakyParallel", "encoding": "latency",
               "batch_size": 128, "epochs": 10, "lr": 1e-3}
    rd = REPO / f"results/archive/leakage/{protocol}/{model}/fold{fold}"
    cfg.update(dataset=f"{protocol}_f{fold}", seed=seed, results_dir=str(rd),
               summary_csv=str(rd / f"seed_{seed}" / "summary.csv"))
    cfg.pop("label_mode", None)          # protocol fixes the label space
    cfg.pop("subsample_fraction", None)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=PROTOCOLS, choices=PROTOCOLS)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--single", nargs=4, metavar=("MODEL", "PROTOCOL", "FOLD", "SEED"))
    args = ap.parse_args()

    if args.single:
        model, protocol = args.single[0], args.single[1]
        fold, seed = int(args.single[2]), int(args.single[3])
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
        rec = run_experiment(build_cfg(model, protocol, fold, seed))
        m = rec["test_plus"]["metrics"]
        b = rec["test_plus"].get("binary_ids_metrics") or {}
        print(f"[done] {model}/{protocol}/fold{fold}/seed_{seed} f1m={m['f1_macro']:.4f} "
              f"mcc={m['matthews_corrcoef']:.4f} DR={b.get('detection_rate',0):.4f} "
              f"FAR={b.get('false_alarm_rate',0):.4f}", flush=True)
        return

    todo = [(m, p, f, s) for p in args.protocols for m in args.models
            for f in args.folds for s in args.seeds
            if not (REPO / f"results/archive/leakage/{p}/{m}/fold{f}/seed_{s}/results.json").exists()]
    print(f"[leakage] {len(todo)} runs pending", flush=True)
    if args.dry_run:
        for t in todo:
            print("  would run", t)
        return

    t0, n = time.time(), [0]
    lock = threading.Lock()

    def one(job):
        model, protocol, fold, seed = job
        cmd = [sys.executable, str(Path(__file__).resolve()), "--single",
               model, protocol, str(fold), str(seed)]
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
            print(f"[leakage {n[0]}/{len(todo)}] {model}/{protocol}/f{fold}/s{seed} "
                  f"| elapsed {el/3600:.2f} h "
                  f"| eta {el/n[0]*(len(todo)-n[0])/3600:.1f} h", flush=True)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, todo))
    print(f"[leakage] finished in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
