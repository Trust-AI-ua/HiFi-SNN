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

    # --- round-2 item 7: clean class-weighting x distillation 2x2 -------------
    # Ranking is held FIXED at its deployed value in all four cells, so only
    # class weighting and KD vary. The other three cells already exist:
    #   CW off / KD on  = full        CW off / KD off = rank_only
    #   CW on  / KD on  = class_weighted
    "cw_nokd_rankfixed": dict(class_weighted=True, a_kd=0.0),

    # --- round-2 item 9: is the 10-epoch schedule the instability? -----------
    "epochs30": dict(epochs=30),

    # --- round-2 item 8: one-factor-at-a-time sensitivity around the deployed
    # operating point (T=20, q=0.20, M_max=3, delta=0.12, tau=2.0, margin=0.05,
    # a_kd=1.0, a_rank=0.3, a_sparse=1e-3, a_fp=0.5). T is prioritised because
    # the temporal-sufficiency analysis shows heavy level collision at T=20.
    "T10": dict(T=10), "T40": dict(T=40), "T80": dict(T=80),
    "q10": dict(prune_quantile=0.10), "q40": dict(prune_quantile=0.40),
    "M2": dict(pop_max=2), "M5": dict(pop_max=5),
    "d000": dict(jitter=0.0), "d006": dict(jitter=0.06), "d024": dict(jitter=0.24),
    "tau1": dict(tau=1.0), "tau4": dict(tau=4.0),
    "akd05": dict(a_kd=0.5), "akd20": dict(a_kd=2.0),
    "arank01": dict(a_rank=0.1), "arank06": dict(a_rank=0.6),
    "asparse1e4": dict(a_sparse=1e-4), "asparse1e2": dict(a_sparse=1e-2),
    "afp025": dict(a_fp=0.25), "afp100": dict(a_fp=1.0),
}
ORDER = list(ARMS)


def build_cfg(ds: str, arm: str, seed: int) -> dict:
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    cfg.update(ARMS[arm])
    # The summary CSV is rewritten wholesale by _append_summary, so a shared path
    # would be corrupted by concurrent workers. Give every run its own file inside
    # its own result directory; results.json remains the source of truth for all
    # analysis and the per-run CSVs can be concatenated afterwards if wanted.
    cfg.update(seed=seed,
               results_dir=str(REPO / f"results/ladder/{ds}/{arm}"),
               summary_csv=str(REPO / f"results/ladder/{ds}/{arm}/seed_{seed}/summary.csv"))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent runs (CPU-bound: keep workers x 4 <= cores)")
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
    done_n = [0]
    lock = threading.Lock()

    def one(job):
        arm, ds, seed = job
        # Each run executes in its own process: a segfault or C++ abort inside
        # torch cannot take the queue down with it, and CUDA state is reset
        # between runs. Retried once, since the allocator fault we hit was
        # transient rather than deterministic.
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--single", arm, ds, str(seed)]
        env = dict(os.environ)
        # Keep total CPU demand bounded: N workers x this many threads <= cores.
        env.setdefault("OMP_NUM_THREADS", str(max(1, 16 // max(args.workers, 1))))
        for attempt in (1, 2):
            r = subprocess.run(cmd, cwd=str(REPO), env=env)
            if r.returncode == 0:
                break
            print(f"[FAIL] {arm}/{ds}/seed_{seed}: exit {r.returncode}"
                  f"{' (retrying once)' if attempt == 1 else ' (giving up, moving on)'}",
                  flush=True)
            time.sleep(10)
        with lock:
            done_n[0] += 1
            el = time.time() - t0
            rate = el / max(done_n[0], 1)
            print(f"[ladder {done_n[0]}/{len(todo)}] finished {arm}/{ds}/seed_{seed} "
                  f"| elapsed {el/3600:.2f} h | eta {rate*(len(todo)-done_n[0])/3600:.1f} h",
                  flush=True)

    if args.workers > 1:
        print(f"[ladder] running {args.workers} concurrent workers", flush=True)
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, todo))
    else:
        for job in todo:
            print(f"\n{'='*70}\n[ladder] {job[0]} / {job[1]} / seed {job[2]}\n{'='*70}",
                  flush=True)
            one(job)

    print(f"\n[ladder] finished in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
