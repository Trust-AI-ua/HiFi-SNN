#!/usr/bin/env python3
"""B6 CICIoT2023 block (amendment A1): one run per (arm, seed), 20 epochs, validation selection.

12 arms, seeds 42-46, all under one rule (src/selection.py): validation macro-F1 after every
epoch, the best epoch's weights kept (ties: earliest) and used for everything downstream;
fp32 / per-tensor / per-channel test scores and the fp32 checkpoint of the selected epoch are
recorded. Each arm is otherwise its production configuration:

  tree, quantile, uniform, all, nokd, entropy, entropy_tree   Tree-to-Spike trainer (+ ARM_CFG)
  raw        Tree-to-Spike trainer, raw_vanilla encoder, no-KD loss (the M8 arm)
  dnn        DNN-Tree trainer;  dnn_tq  DNN-Tree trainer with spike-time input (E1)
  vanilla    Vanilla SNN trainer;  tasnn  TASNN trainer

Runs only after scripts/partB_block_guard.py has passed (its guard.json is required).

    python scripts/partB_b6_block.py <arm> <seed>
Output: results/journal_partB/B6/block/<arm>/ciciot2023/seed_<S>/results.json (+ checkpoints)
"""
from __future__ import annotations

import glob
import importlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402
from encoder_ablation import ARM_CFG  # noqa: E402

DS = "ciciot2023"
EPOCHS = 20
T2S_ARMS = {"tree": {}, **{k: v for k, v in ARM_CFG.items()},
            "raw": {"encoder_mode": "raw_vanilla", "a_kd": 0.0, "a_rank": 0.0}}
ARMS = list(T2S_ARMS) + ["dnn", "dnn_tq", "vanilla", "tasnn"]


def guard_passed() -> bool:
    for g in sorted(glob.glob(str(C.REPO / "results/journal_partB/B6_block_guard/*/guard.json"))):
        if json.loads(Path(g).read_text()).get("ok"):
            return True
    return False


def config(arm, seed):
    if arm in T2S_ARMS:
        cfg = C.t2s_config(DS, seed); cfg.update(T2S_ARMS[arm]); mod = "src.training_treespike"
    elif arm in ("dnn", "dnn_tq"):
        cfg = yaml.safe_load(open(C.REPO / f"experiments/DNN-Tree/{DS}/config.yaml")); mod = "src.training_dnn"
        if arm == "dnn_tq":
            cfg.update(dense_input="spike_time")
    elif arm == "vanilla":
        cfg = yaml.safe_load(open(C.REPO / f"configs/{DS}/SNN_LeakyParallel/latency/config.yaml")); mod = "src.training"
    else:
        cfg = yaml.safe_load(open(C.REPO / f"experiments/TASNN/{DS}/config.yaml")); mod = "src.training_tasnn"
    return cfg, mod


def main():
    arm, seed = sys.argv[1], int(sys.argv[2])
    if arm not in ARMS:
        sys.exit(f"unknown arm {arm}; expected one of {ARMS}")
    C.no_thread_caps()
    if not guard_passed():
        sys.exit("the block reproduction guard has not passed; refusing to run")
    rd = C.REPO / f"results/journal_partB/B6/block/{arm}/{DS}"
    if (rd / f"seed_{seed}" / "results.json").exists():
        print(f"[skip] {arm}/{seed}"); return
    from src.joblock import claim
    with claim(C.REPO / f"results/journal_partB/B6/block/.locks/{arm}_{seed}.lock") as got:
        if not got:
            print(f"[busy] {arm}/{seed}"); return
        cfg, mod = config(arm, seed)
        cfg.update(seed=seed, epochs=EPOCHS, select_on_val=True, record_precisions=True, force=True,
                   results_dir=str(rd), summary_csv=str(rd / "summary.csv"))
        rec = importlib.import_module(mod).run_experiment(cfg)
        st = C.stamp("B6_block")
        p = rd / f"seed_{seed}" / "results.json"
        r = json.loads(p.read_text())
        r["part_b"] = dict(stamp=st, arm=arm, protocol="A1: 20 epochs, validation macro-F1 selection")
        p.write_text(json.dumps(r, indent=2))
        sel = rec["training"]["selection"]
        print(f"[done] {arm}/{seed}: selected epoch {sel['selected_epoch']} (val macro-F1 {sel['best_val_macro_f1']:.2f}) "
              f"test macro-F1 {100*rec['test_plus']['metrics']['f1_macro']:.2f}", flush=True)


if __name__ == "__main__":
    main()
