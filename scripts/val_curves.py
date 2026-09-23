#!/usr/bin/env python3
"""Per-epoch validation macro-F1 curves for DNN-Tree / Tree-to-Spike (diagnostic).

Runs the paper's own ``run_experiment`` unchanged (config from
experiments/<model>/<dataset>/config.yaml, same splits, same preprocessing) and
only *observes* the per-epoch validation pass the training loop already makes:
the loop's val-helper is wrapped so the predictions it computes are also scored
with macro-F1 exactly as src.metrics does (sklearn default labels = classes
present in y_true or y_pred). No extra RNG draws, no selection: the test split
is used only by the unchanged end-of-run evaluation, which serves as a
reproduction check against the stored results.json.

Also logs, per epoch, macro-F1 of an int8 fake-quantized copy (the reported
numbers are post-quantization; the training loop's val pass is fp32).

M3: the final fp32 student (after the last epoch, before the pipeline's own
int8 projection) is kept, saved as checkpoint_fp32.pt, and scored on test with
the same metric code; the int8 side is the unchanged pipeline's own test result.
Written to runs/seed_<S>/test_fp32_int8.json.

Encoder side-file runs/seed_<S>/encoder_channels.json: per channel, the source
feature, population offset, the teacher's gain importance and its rank, and the
channel's activity on (up to 200k) training rows (fraction of nonzero level,
mean, std). Channels are in raw feature-index order, not importance order.

Outputs (never touches results/<model>/...):
  results/diagnostics/val_curves/<model>/<dataset>/seed_<S>.csv
  results/diagnostics/val_curves/<model>/<dataset>/runs/seed_<S>/results.json

    python scripts/val_curves.py DNN-Tree cse-cic-ids2018 --seed 42
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models import quantize_int8_  # noqa: E402
import src.training_treespike as tts  # noqa: E402
import src.training_dnn as tdnn  # noqa: E402


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def _dump_encoder(captured, path):
    import json
    enc, Xtr = captured["enc"], captured["Xtr"]
    imp = np.asarray(captured["teacher"].feature_importances_, dtype=np.float64)
    rank = (-imp).argsort().argsort()          # 0 = most important feature
    rng = np.random.default_rng(0)             # local RNG: no effect on the run
    idx = rng.choice(len(Xtr), size=min(len(Xtr), 200_000), replace=False)
    L = enc._level_channels(Xtr[np.sort(idx)])
    chans = [dict(channel=c, feature=int(j), offset=float(o),
                  importance=float(imp[j]), importance_rank=int(rank[j]),
                  frac_nonzero=float((L[:, c] > 0).mean()),
                  mean=float(L[:, c].mean()), std=float(L[:, c].std()),
                  n_edges=int(len(enc.edges_.get(j, []))))
             for c, (j, o) in enumerate(enc.channels_)]
    json.dump({"n_features": int(len(imp)), "channels": chans}, open(path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["DNN-Tree", "Tree-to-Spike"])
    ap.add_argument("dataset")
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()

    out = REPO / "results/diagnostics/val_curves" / a.model / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    if (out / "runs" / f"seed_{a.seed}" / "test_fp32_int8.json").exists():
        print(f"[skip] {a.model}/{a.dataset}/seed_{a.seed} already captured")
        return
    from src.joblock import claim
    lock = REPO / f"results/diagnostics/val_curves/.locks/{a.model}_{a.dataset}_{a.seed}.lock"
    _claim = claim(lock)
    if not _claim.__enter__():
        print(f"[busy] {a.model}/{a.dataset}/seed_{a.seed} claimed by another worker")
        return
    cfg = yaml.safe_load(open(REPO / f"experiments/{a.model}/{a.dataset}/config.yaml"))
    cfg.update(seed=a.seed, force=True, results_dir=str(out / "runs"),
               summary_csv=str(out / "runs" / "summary_scratch.csv"))

    captured = {}
    rows = []

    orig_arrays = tts._arrays_from_bundle
    orig_fit = tts._fit_teacher

    def fit_hook(Xtr, ytr, num_classes, cfg_):
        res = orig_fit(Xtr, ytr, num_classes, cfg_)
        captured["teacher"], captured["Xtr"] = res[0], Xtr
        return res

    def arrays_hook(bundle):
        arrs = orig_arrays(bundle)
        captured["Xval"], captured["yval"] = arrs[2], arrs[3]
        captured["Xte"], captured["yte"], captured["bundle"] = arrs[4], arrs[5], bundle
        return arrs

    def record(student, fp_pred, run_fn, test_fn):
        yval = captured["yval"]
        captured["fp32"] = copy.deepcopy(student)   # last call = final fp32 student
        captured["test_fn"] = test_fn
        q = quantize_int8_(copy.deepcopy(student))
        q_pred = run_fn(q)
        ep = len(rows) + 1
        rows.append(dict(seed=a.seed, epoch=ep, val_macro_f1=macro_f1(yval, fp_pred) * 100,
                         val_macro_f1_int8=macro_f1(yval, q_pred) * 100,
                         val_accuracy=float((fp_pred == yval).mean()) * 100))
        print(f"[curve] seed={a.seed} epoch={ep} val_f1={rows[-1]['val_macro_f1']:.2f} "
              f"int8={rows[-1]['val_macro_f1_int8']:.2f}", flush=True)

    if a.model == "DNN-Tree":
        tdnn._arrays_from_bundle = arrays_hook
        tdnn._fit_teacher = fit_hook
        orig = tdnn._forward_batches

        def fwd_hook(model, encoder, X, bs, device):
            res = orig(model, encoder, X, bs, device)
            captured["enc"] = encoder
            if X is captured.get("Xval") and model.training is False and len(rows) < cfg["epochs"]:
                record(model, res[0], lambda m: orig(m, encoder, X, bs, device)[0],
                       lambda m, Xt: orig(m, encoder, Xt, bs, device)[:2])
                model.eval()
            return res
        tdnn._forward_batches = fwd_hook
        rec = tdnn.run_experiment(cfg)
    else:
        tts._arrays_from_bundle = arrays_hook
        tts._fit_teacher = fit_hook
        orig = tts._encode_forward_in_batches

        def fwd_hook(student, encoder, X, bs, device):
            res = orig(student, encoder, X, bs, device)
            captured["enc"] = encoder
            if X is captured.get("Xval") and len(rows) < cfg["epochs"]:
                record(student, res[0], lambda m: orig(m, encoder, X, bs, device)[0],
                       lambda m, Xt: orig(m, encoder, Xt, bs, device)[:2])
                student.eval()
            return res
        tts._encode_forward_in_batches = fwd_hook
        rec = tts.run_experiment(cfg)

    assert len(rows) == cfg["epochs"], f"captured {len(rows)} epochs"
    stored = REPO / f"results/{a.model}/{a.dataset}/seed_{a.seed}/results.json"
    import json
    ref = json.load(open(stored))["test_plus"]["metrics"]["f1_macro"] * 100
    new = rec["test_plus"]["metrics"]["f1_macro"] * 100
    for r in rows:
        r["test_f1_rerun"] = round(new, 4) if r["epoch"] == cfg["epochs"] else ""
        r["test_f1_stored"] = round(ref, 4) if r["epoch"] == cfg["epochs"] else ""
    with open(out / f"seed_{a.seed}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[repro] seed={a.seed} test macro-F1 rerun={new:.2f} stored={ref:.2f}")

    from src.metrics import compute_metrics, compute_binary_ids_metrics
    fm = captured["fp32"]
    torch.save(fm.state_dict(), out / "runs" / f"seed_{a.seed}" / "checkpoint_fp32.pt")
    yte = captured["yte"]
    pred, score = captured["test_fn"](fm, captured["Xte"])
    m = compute_metrics(yte, pred, y_score=score, class_names=captured["bundle"].class_names)
    b = compute_binary_ids_metrics(yte, pred, normal_class_index=0)
    qm = rec["test_plus"]["metrics"]; qb = rec["test_plus"]["binary_ids_metrics"]
    res = {"model": a.model, "dataset": a.dataset, "seed": a.seed,
           "fp_macro_f1": 100 * m["f1_macro"], "int8_macro_f1": new,
           "fp_mcc": 100 * m["matthews_corrcoef"], "int8_mcc": 100 * qm["matthews_corrcoef"],
           "fp_FAR": 100 * b["false_alarm_rate"], "int8_FAR": 100 * qb["false_alarm_rate"],
           "fp_DR": 100 * b["detection_rate"], "int8_DR": 100 * qb["detection_rate"],
           "stored_int8_macro_f1": ref, "reproduced": abs(new - ref) < 1e-6}
    json.dump(res, open(out / "runs" / f"seed_{a.seed}" / "test_fp32_int8.json", "w"), indent=2)
    _dump_encoder(captured, out / "runs" / f"seed_{a.seed}" / "encoder_channels.json")
    print(f"[m3] seed={a.seed} fp32={res['fp_macro_f1']:.2f} int8={new:.2f}")
    _claim.__exit__(None, None, None)


if __name__ == "__main__":
    main()
