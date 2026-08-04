#!/usr/bin/env python3
"""Attribution / component-isolation ladder for the TNNLS revision.

Answers reviewer concerns 1 (causal contribution of the tree-guided encoder),
8 (ablation combines multiple mechanisms), 10 (class-weighted student control)
and 5 (FP32 vs int8-grid predictive effect).

Every arm holds the 64-64 LIF student, T=20, fast-sigmoid surrogate, Adam
(2e-3, 10 epochs, batch 128), mean-membrane readout and the early-exit
mechanism constant; only the listed keys change. Arms already present in
results/Tree-to-Spike (full) and results/encoder_ablation (quantile / nokd /
uniform / all) are NOT re-run -- this script only fills the gaps.

Scope: the four primary IoT/IIoT/IoMT datasets. Arms are ordered by evidentiary
value so that stopping early still leaves the most important controls complete.

Writes results/ladder/<dataset>/<arm>/seed_<S>/results.json (+ checkpoint.pt)
and results/ladder/summary.csv. Existing results.json are skipped, so the
script is restartable.

Usage
-----
    python scripts/run_ladder.py                 # everything, priority order
    python scripts/run_ladder.py --arms raw prune_only
    python scripts/run_ladder.py --datasets toniot --seeds 42
    python scripts/run_ladder.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
SEEDS = [42, 43, 44, 45, 46]

# Ordered by evidentiary value: arm 1 (the reviewer's requested control) first,
# then the encoder-mechanism decomposition, then the loss-term decomposition.
ARMS = {
    # --- encoder x objective 2x2 (round-2 reviewer item 2) --------------------
    # The four cells below cross {raw, tree} encoder with {CE-only, full}
    # objective, so the encoder main effect, objective main effect and their
    # interaction are all identified. "raw" and the deployed model supply the
    # other two cells; only raw_full and tree_ce were missing.
    #   raw     = raw encoder,  CE only            (cell 1,1)
    #   tree_ce = tree encoder, CE only            (cell 2,1)  NEW
    #   raw_full= raw encoder,  full objective     (cell 1,2)  NEW
    #   full    = tree encoder, full objective     (cell 2,2)  = deployed
    "raw": dict(encoder_mode="raw", a_kd=0.0, a_rank=0.0, a_fp=0.0, a_sparse=0.0),
    "raw_full": dict(encoder_mode="raw"),
    "tree_ce": dict(a_kd=0.0, a_rank=0.0, a_fp=0.0, a_sparse=0.0),

    # --- encoder mechanism: pruning vs importance allocation, separated ---
    "tree_only":  dict(encoder_mode="tree", prune_quantile=0.0, pop_max=1),
    "prune_only": dict(encoder_mode="tree", prune_quantile=0.20, pop_max=1),
    "alloc_only": dict(encoder_mode="tree", prune_quantile=0.0, pop_max=3),

    # --- objective: KD and ranking separated (nokd = both off, already run) ---
    "kd_only":   dict(a_rank=0.0),
    "rank_only": dict(a_kd=0.0),

    # --- remaining loss terms ---
    "no_sparse": dict(a_sparse=0.0),
    "no_fp":     dict(a_fp=0.0),

    # --- concern 10: imbalance-policy transfer vs knowledge transfer ---
    "class_weighted":    dict(class_weighted=True),
    "class_weighted_nokd": dict(class_weighted=True, a_kd=0.0, a_rank=0.0),

    # --- concern 5: predictive effect of the int8 weight grid ---
    "fp32": dict(quantize=False),
}
ORDER = list(ARMS)


def build_cfg(ds: str, arm: str, seed: int) -> dict:
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    cfg.update(ARMS[arm])
    cfg.update(seed=seed,
               results_dir=str(REPO / f"results/ladder/{ds}/{arm}"),
               summary_csv=str(REPO / "results/ladder/summary.csv"))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--single", nargs=3, metavar=("ARM", "DATASET", "SEED"),
                    help="internal: run exactly one experiment in this process")
    args = ap.parse_args()

    # --- worker mode: one experiment, then exit -------------------------------
    # The parent invokes this via subprocess so that a hard crash (e.g. a C++
    # abort from the CUDA caching allocator, which no Python except: can catch)
    # kills only the single run instead of the whole queue.
    if args.single:
        from src.training_treespike import run_experiment
        arm, ds, seed = args.single[0], args.single[1], int(args.single[2])
        rec = run_experiment(build_cfg(ds, arm, seed))
        m = rec["test_plus"]["metrics"]
        print(f"[done] {arm}/{ds}/seed_{seed} acc={m['accuracy']:.4f} "
              f"f1m={m['f1_macro']:.4f} mcc={m['matthews_corrcoef']:.4f} "
              f"channels={rec['data']['encoder_channels']}", flush=True)
        return

    todo = [(a, ds, s) for a in args.arms for ds in args.datasets for s in args.seeds
            if not (REPO / f"results/ladder/{ds}/{a}/seed_{s}/results.json").exists()]
    print(f"[ladder] {len(todo)} runs pending "
          f"({len(args.arms)} arms x {len(args.datasets)} datasets x {len(args.seeds)} seeds)",
          flush=True)
    if args.dry_run:
        for a, ds, s in todo:
            print(f"  would run {a}/{ds}/seed_{s}  -> {ARMS[a]}")
        return

    t0 = time.time()
    for i, (arm, ds, seed) in enumerate(todo, 1):
        el = time.time() - t0
        eta = (el / max(i - 1, 1)) * (len(todo) - i + 1) / 3600 if i > 1 else float("nan")
        print(f"\n{'='*70}\n[ladder {i}/{len(todo)}] {arm} / {ds} / seed {seed} "
              f"| elapsed {el/3600:.2f} h | eta {eta:.1f} h\n{'='*70}", flush=True)
        # Each run executes in its own process: a segfault or C++ abort inside
        # torch cannot take the queue down with it, and CUDA state is reset
        # between runs. Retried once, since the allocator fault we hit was
        # transient rather than deterministic.
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--single", arm, ds, str(seed)]
        for attempt in (1, 2):
            r = subprocess.run(cmd, cwd=str(REPO))
            if r.returncode == 0:
                break
            print(f"[FAIL] {arm}/{ds}/seed_{seed}: exit {r.returncode}"
                  f"{' (retrying once)' if attempt == 1 else ' (giving up, moving on)'}",
                  flush=True)
            time.sleep(10)

    print(f"\n[ladder] finished in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
