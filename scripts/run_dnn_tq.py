#!/usr/bin/env python3
"""E1: matched-information dense twin (pre-registered 2026-09-24).

One cell per (dataset, seed): DNN-Tree with dense_input="spike_time", i.e. trained and
evaluated on the levels exactly as the spike train carries them, with fp32 test metrics
captured before the int8 projection and per-channel int8 scored on a copy of the fp32
weights (amendment 2026-09-24: E1 covers fp32, per-tensor and per-channel). Everything else is the production DNN-Tree config.

  python scripts/run_dnn_tq.py <dataset> <seed>
  python scripts/run_dnn_tq.py --guard          # G2: production input must reproduce
                                                  # stored DNN-Tree TON-IoT seed 42 (int8),
                                                  # the M3 fp32 capture and the per-channel
                                                  # int8 score of tab_perchannel.csv, exactly
Output: results/DNN-TQ/<dataset>/seed_<S>/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.joblock import claim  # noqa: E402


def cfg_for(ds, seed, dense_input, results_dir, summary):
    cfg = yaml.safe_load(open(REPO / f"experiments/DNN-Tree/{ds}/config.yaml"))
    cfg.update(seed=seed, dense_input=dense_input, record_fp32=True, force=True,
               results_dir=str(results_dir), summary_csv=str(summary))
    return cfg


def guard():
    import src.training_dnn as tdnn
    ds, seed = "toniot", 42
    rd = REPO / "results/diagnostics/dnn_tq_guard"
    rec = tdnn.run_experiment(cfg_for(ds, seed, "level", rd, rd / "summary.csv"))
    got_int8 = 100 * rec["test_plus"]["metrics"]["f1_macro"]
    got_fp = 100 * rec["test_plus"]["fp32_before_projection"]["metrics"]["f1_macro"]
    want_int8 = 100 * json.loads((REPO / f"results/DNN-Tree/{ds}/seed_{seed}/results.json").read_text())["test_plus"]["metrics"]["f1_macro"]
    want_fp = json.loads((REPO / f"results/diagnostics/val_curves/DNN-Tree/{ds}/runs/seed_{seed}/test_fp32_int8.json").read_text())["fp_macro_f1"]
    got_pc = 100 * rec["test_plus"]["fp32_before_projection"]["per_channel_int8"]["metrics"]["f1_macro"]
    want_pc = json.loads((REPO / f"results/diagnostics/perchannel/DNN-Tree_{ds}_seed{seed}.json").read_text())["per_channel_f1"]
    ok = abs(got_int8 - want_int8) < 1e-6 and abs(got_fp - want_fp) < 1e-6 and abs(got_pc - want_pc) < 1e-6
    msg = (f"G2 int8 {got_int8:.6f} vs stored {want_int8:.6f} | fp32 {got_fp:.6f} vs M3 {want_fp:.6f} "
           f"| per-channel {got_pc:.6f} vs perchannel_check {want_pc:.6f} -> {'PASS' if ok else 'FAIL'}")
    print(msg, flush=True)
    (rd / "G2.txt").write_text(msg + "\n")
    sys.exit(0 if ok else 1)


def main():
    if sys.argv[1] == "--guard":
        guard()
    import src.training_dnn as tdnn
    ds, seed = sys.argv[1], int(sys.argv[2])
    rd = REPO / f"results/DNN-TQ/{ds}"
    if (rd / f"seed_{seed}" / "results.json").exists():
        print(f"[skip] {ds}/seed_{seed}")
        return
    with claim(REPO / f"results/DNN-TQ/.locks/{ds}_{seed}.lock") as got:
        if not got:
            print(f"[busy] {ds}/seed_{seed}")
            return
        summ = REPO / "results/DNN-TQ/summaries"
        summ.mkdir(parents=True, exist_ok=True)
        rec = tdnn.run_experiment(cfg_for(ds, seed, "spike_time", rd, summ / f"{ds}_{seed}.csv"))
        tp = rec["test_plus"]
        print(f"[E1] {ds}/seed_{seed} fp32={100*tp['fp32_before_projection']['metrics']['f1_macro']:.2f} "
              f"int8={100*tp['metrics']['f1_macro']:.2f} "
              f"per-channel={100*tp['fp32_before_projection']['per_channel_int8']['metrics']['f1_macro']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
