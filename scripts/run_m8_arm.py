#!/usr/bin/env python3
"""M8 control arm: raw-feature latency encoding into the Tree-to-Spike student.

One cell per (dataset, seed). The arm differs from the ``no-KD`` variant in the
ENCODER ALONE: the conventional SNN's own latency code (floor, epsilon 0.01, one
channel per raw feature, no pruning, no importance budget, no jitter) replaces the
tree-guided TTFS encoder. Student, window, quantization, early-exit selection and
evaluation are the production ones, and the loss is the no-KD loss
(a_kd = a_rank = 0), so the comparison isolates the encoder.

Together with the existing arms this decomposes the gap between Tree-to-Spike and
the conventional SNN into four steps, each changing one thing:

    Vanilla SNN (fp32)  -> this arm (fp32)    student architecture and training
    this arm fp32       -> this arm int8      weight precision
    this arm int8       -> no-KD variant      encoder
    no-KD variant       -> Tree-to-Spike      teacher-posterior supervision

fp32 weights are captured immediately before the int8 projection and scored on
test, so the precision step cannot hide inside the architecture step.

    python scripts/run_m8_arm.py ciciot2023 42
Output: results/encoder_ablation/<dataset>/m8_raw/seed_<S>/ plus test_fp32_int8.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.training_treespike as tts  # noqa: E402
from src.joblock import claim  # noqa: E402

ARM = "m8_raw"
ARM_CFG = {"encoder_mode": "raw_vanilla", "a_kd": 0.0, "a_rank": 0.0}


def main():
    ds, seed = sys.argv[1], int(sys.argv[2])
    rd = REPO / f"results/encoder_ablation/{ds}/{ARM}"
    if (rd / f"seed_{seed}" / "results.json").exists():
        print(f"[skip] {ds}/{ARM}/seed_{seed}")
        return
    lock = REPO / f"results/encoder_ablation/.locks/{ds}_{ARM}_{seed}.lock"
    with claim(lock) as got:
        if not got:
            print(f"[busy] {ds}/{ARM}/seed_{seed}")
            return
        run(ds, seed, rd)


def run(ds, seed, rd):
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    cfg.update(ARM_CFG)
    summ = REPO / "results/encoder_ablation/summaries"
    summ.mkdir(parents=True, exist_ok=True)
    cfg.update(seed=seed, results_dir=str(rd), force=True,
               summary_csv=str(summ / f"{ds}_{ARM}_{seed}.csv"))

    captured = {}
    orig_arrays = tts._arrays_from_bundle
    orig_fwd = tts._encode_forward_in_batches

    def arrays_hook(bundle):
        a = orig_arrays(bundle)
        captured.update(Xval=a[2], Xte=a[4], yte=a[5], bundle=bundle)
        return a

    def fwd_hook(student, encoder, X, bs, device):
        res = orig_fwd(student, encoder, X, bs, device)
        if X is captured.get("Xval"):          # end of an epoch: keep the fp32 student
            captured["fp32"] = copy.deepcopy(student)
            captured["enc"], captured["bs"], captured["dev"] = encoder, bs, device
        return res

    tts._arrays_from_bundle, tts._encode_forward_in_batches = arrays_hook, fwd_hook
    try:
        rec = tts.run_experiment(cfg)
    finally:
        tts._arrays_from_bundle, tts._encode_forward_in_batches = orig_arrays, orig_fwd

    fm, enc = captured["fp32"], captured["enc"]
    torch.save(fm.state_dict(), rd / f"seed_{seed}" / "checkpoint_fp32.pt")
    pred, score = orig_fwd(fm, enc, captured["Xte"], captured["bs"], captured["dev"])[:2]
    yte, names = captured["yte"], captured["bundle"].class_names
    m = compute_metrics(yte, pred, y_score=score, class_names=names)
    b = compute_binary_ids_metrics(yte, pred, normal_class_index=0)
    qm = rec["test_plus"]["metrics"]; qb = rec["test_plus"]["binary_ids_metrics"]
    out = {"arm": ARM, "dataset": ds, "seed": seed,
           "encoder": "raw-feature latency (vanilla Eq. 3, eps 0.01), one channel per feature",
           "fp_macro_f1": 100 * m["f1_macro"], "int8_macro_f1": 100 * qm["f1_macro"],
           "fp_mcc": 100 * m["matthews_corrcoef"], "int8_mcc": 100 * qm["matthews_corrcoef"],
           "fp_FAR": 100 * b["false_alarm_rate"], "int8_FAR": 100 * qb["false_alarm_rate"],
           "fp_DR": 100 * b["detection_rate"], "int8_DR": 100 * qb["detection_rate"],
           "trainable_parameters": rec["model_size"]["trainable_parameters"],
           "encoder_channels": rec["data"]["encoder_channels"],
           "input_features": rec["data"]["num_features"]}
    json.dump(out, open(rd / f"seed_{seed}" / "test_fp32_int8.json", "w"), indent=1)
    print(f"[m8] {ds}/seed_{seed} fp32={out['fp_macro_f1']:.2f} int8={out['int8_macro_f1']:.2f} "
          f"channels={out['encoder_channels']} params={out['trainable_parameters']}", flush=True)


if __name__ == "__main__":
    main()
