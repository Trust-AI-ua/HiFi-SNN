#!/usr/bin/env python3
"""Run ONE encoder-ablation cell (dataset, arm, seed), so the arms can run in a
worker pool instead of encoder_ablation.py's sequential in-process loop.

Same config construction as encoder_ablation.run_variants (production config +
ARM_CFG override + seed, same results_dir), so the outputs are interchangeable
and `python scripts/encoder_ablation.py --plot` aggregates them unchanged.

Two differences from that loop. Each cell writes its own summary row under
results/encoder_ablation/summaries/, because parallel writers appending to one
shared summary.csv would race. And a file lock (src/joblock.py) lets a
supplemental pool share the job list with an already-running pool without both
starting the same cell.

    python scripts/run_encoder_variant.py cse-cic-ids2018 quantile 42
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from encoder_ablation import ARM_CFG  # noqa: E402
from src.joblock import claim  # noqa: E402


def main():
    ds, mode, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if mode not in ARM_CFG:
        sys.exit(f"unknown arm {mode!r}; expected one of {sorted(ARM_CFG)}")
    rd = REPO / f"results/encoder_ablation/{ds}/{mode}"
    if (rd / f"seed_{seed}" / "results.json").exists():
        print(f"[skip] {ds}/{mode}/seed_{seed}")
        return
    lock = REPO / f"results/encoder_ablation/.locks/{ds}_{mode}_{seed}.lock"
    with claim(lock) as got:
        if not got:
            print(f"[busy] {ds}/{mode}/seed_{seed} claimed by another worker")
            return
        _run(ds, mode, seed, rd)


def _run(ds, mode, seed, rd):
    from src.training_treespike import run_experiment
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    cfg.update(ARM_CFG[mode])
    summ = REPO / "results/encoder_ablation/summaries"
    summ.mkdir(parents=True, exist_ok=True)
    cfg.update(seed=seed, results_dir=str(rd),
               summary_csv=str(summ / f"{ds}_{mode}_{seed}.csv"))
    print(f"[ablation] {ds}/{mode}/seed_{seed}", flush=True)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
