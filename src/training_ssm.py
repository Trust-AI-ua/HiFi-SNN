"""Shared training/eval loop for the Spiking-SSM hybrid (Design A).

Self-contained, like ``src/training_griffin.py`` and ``src/training_sxlstm.py``,
but adapted for the sequence model:

* Data comes from ``src.data.sequence_windows.get_sequence_dataset`` (consecutive
  -record windows, ``[B, L, F]``) rather than the tabular loaders.
* The model is ``src.models.SpikingSSM`` and is trained multi-task:
      loss = CE(classification)
             + a_pred  * mean next-window MSE (self-supervised predictor head)
             + a_sparse * spike rate            (energy regulariser)
* Final inference uses the model's confidence-triggered early exit, and we also
  record a self-supervised anomaly-score separation (benign vs attack) for
  reference.
* Results go to a *separate* summary file (``results/summary_ssm.csv``) with a
  ``model`` column, so SSM rows never mix with SNN / S-xLSTM / Spiking-Griffin.

Same multi-seed mechanics as the other scripts: ``seed_<N>`` sub-directory,
skip-guard, ``HISNN_SEED`` / ``HISNN_FORCE`` env overrides, and the same
``results.json`` metric schema (+ an early-exit block + a self-supervised block).

config keys (with defaults):
    dataset            "kddcup99"
    seq_len            16     window length L
    stride             16     window stride (16 = non-overlapping)
    d_model            32
    d_state            8
    n_blocks           2
    beta               0.9    LIF membrane decay
    a_pred             0.3     next-window prediction loss weight
    a_sparse           1e-3    spike-sparsity loss weight
    theta_exit         0.9     early-exit confidence threshold
    theta_uncertain    0.5     low-confidence flag threshold
    batch_size         128
    epochs             10
    lr                 2e-3
    weight_decay       0.0
    class_weighted     True   inverse-frequency CE (Design A's native choice)
    val_fraction       0.1
    seed               42
    device             None   -> auto
    results_dir        required
    summary_csv        results/summary_ssm.csv
    force              False
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.sequence_windows import get_sequence_dataset
from src.metrics import (compute_metrics, compute_binary_ids_metrics,
                          compute_spike_stats, count_parameters)
from src.models import SpikingSSM

MODEL_NAME = "Spiking-SSM"
ENCODING_NAME = "spiking-frontend"   # learned spiking front-end (no separate encoder)
NEURON_NAME = "Leaky"                # snn.Leaky with fast-sigmoid surrogate

DEFAULTS = {
    "seq_len": 16,
    "stride": 16,
    "d_model": 32,
    "d_state": 8,
    "n_blocks": 2,
    "beta": 0.9,
    "a_pred": 0.3,
    "a_sparse": 1e-3,
    "theta_exit": 0.9,
    "theta_uncertain": 0.5,
    "batch_size": 128,
    "epochs": 10,
    "lr": 2e-3,
    "weight_decay": 0.0,
    "class_weighted": True,
    "val_fraction": 0.1,
    "seed": 42,
    "device": None,
    "summary_csv": "results/summary_ssm.csv",
    "force": False,
}

SUMMARY_HEADER = [
    "model", "dataset", "neuron", "encoding",
    "accuracy", "balanced_accuracy",
    "f1_macro", "f1_weighted",
    "precision_macro", "recall_macro",
    "roc_auc_weighted", "matthews_corrcoef", "cohen_kappa",
    "detection_rate", "false_alarm_rate",
    "binary_accuracy", "binary_f1",
    "early_exit_accuracy", "early_exit_f1_macro", "mean_exit_step", "uncertain_frac",
    "ssl_score_benign", "ssl_score_attack",
    "energy_spikes_per_sample", "spike_sparsity",
    "trainable_parameters",
    "train_time_s", "inference_ms_per_sample",
    "epochs", "seq_len", "stride", "d_model", "d_state", "n_blocks", "beta",
    "a_pred", "a_sparse", "batch_size", "lr", "class_weighted", "seed",
]


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve(cfg: Dict) -> Dict:
    merged = {**DEFAULTS, **cfg}
    if merged.get("device") is None:
        merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if os.environ.get("HISNN_FORCE", "").lower() in {"1", "true", "yes"}:
        merged["force"] = True
    if "HISNN_SEED" in os.environ:
        try:
            merged["seed"] = int(os.environ["HISNN_SEED"])
        except ValueError:
            pass
    return merged


def _class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    w = counts.sum() / (num_classes * counts)
    return w / w.mean()


def _train_one_epoch(model, loader, optim, loss_fn, cfg, device):
    """Multi-task: CE + a_pred * next-window MSE + a_sparse * spike rate."""
    model.train()
    running = 0.0
    n_batches = 0
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        logits, preds, total_spikes, spike_sites = model(X)
        ce = loss_fn(logits, y)
        if len(preds) > 0:
            pred_loss = torch.stack(
                [F.mse_loss(preds[t], X[:, t + 1]) for t in range(len(preds))]).mean()
        else:
            pred_loss = X.new_zeros(())
        spike_rate = total_spikes / max(int(spike_sites), 1)
        loss = ce + cfg["a_pred"] * pred_loss + cfg["a_sparse"] * spike_rate
        optim.zero_grad(); loss.backward(); optim.step()
        running += float(loss.item())
        n_batches += 1
    return running / max(n_batches, 1)


@torch.no_grad()
def _eval_val(model, loader, loss_fn, cfg, device):
    """Lightweight validation: standard-forward accuracy + multi-task loss."""
    model.eval()
    correct = total = 0
    running = 0.0
    n_batches = 0
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        logits, preds, total_spikes, spike_sites = model(X)
        ce = loss_fn(logits, y)
        if len(preds) > 0:
            pred_loss = torch.stack(
                [F.mse_loss(preds[t], X[:, t + 1]) for t in range(len(preds))]).mean()
        else:
            pred_loss = X.new_zeros(())
        spike_rate = total_spikes / max(int(spike_sites), 1)
        running += float((ce + cfg["a_pred"] * pred_loss + cfg["a_sparse"] * spike_rate).item())
        n_batches += 1
        correct += int((logits.argmax(1) == y).sum().item())
        total += y.shape[0]
    return running / max(n_batches, 1), correct / max(total, 1)


@torch.no_grad()
def _eval_test(model, loader, cfg, device):
    """Final test eval. Headline predictions/scores come from the standard
    forward pass (so ROC-AUC is available, consistent with the other models);
    the confidence-triggered early exit is evaluated as an auxiliary block in
    the same loop. The self-supervised anomaly score is derived from the
    forward pass's next-window predictions (no extra backbone pass).

    Two backbone passes per batch (standard forward + early-exit), matching the
    S-xLSTM / Spiking-Griffin eval.
    """
    model.eval()
    all_pred, all_true, all_score = [], [], []
    all_ee_pred, all_exit = [], []
    ssl_scores = []
    n_uncertain = 0
    total_spikes = 0.0
    total_sites = 0
    n_samples = 0
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
    start = time.time()
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        # --- headline: standard forward (logits -> argmax + scores for AUC) ---
        logits, preds_ssl, tspk, tsites = model(X)
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_score.append(logits.float().cpu().numpy())
        all_true.append(y.cpu().numpy())
        total_spikes += float(tspk.item())
        total_sites += int(tsites)
        n_samples += X.shape[0]
        # --- self-supervised score from the same forward's next-window preds ---
        if len(preds_ssl) > 0:
            errs = torch.stack([
                F.mse_loss(preds_ssl[t], X[:, t + 1], reduction="none").mean(-1)
                for t in range(len(preds_ssl))], dim=0).mean(0)
            ssl_scores.append(errs.cpu().numpy())
        # --- auxiliary: confidence-triggered early exit ---
        ee_pred, exit_step, uncertain, _, _ = model.predict_early_exit(
            X, theta_exit=cfg["theta_exit"], theta_uncertain=cfg["theta_uncertain"])
        all_ee_pred.append(ee_pred.cpu().numpy())
        all_exit.append(exit_step.cpu().numpy())
        n_uncertain += int(uncertain.sum().item())
    if cuda:
        torch.cuda.synchronize()
    infer_time_s = time.time() - start
    return dict(
        y_pred=np.concatenate(all_pred),
        y_true=np.concatenate(all_true),
        y_score=np.concatenate(all_score, axis=0),
        ee_pred=np.concatenate(all_ee_pred),
        exit_step=np.concatenate(all_exit),
        ssl_score=(np.concatenate(ssl_scores) if ssl_scores else None),
        n_uncertain=n_uncertain,
        total_spikes=total_spikes, total_sites=total_sites,
        n_samples=n_samples, infer_time_s=infer_time_s,
    )


def _build_record(eval_out: Dict, class_names: List[str], seq_len: int) -> Dict:
    # Headline metrics from the standard forward pass (y_score -> ROC-AUC).
    m = compute_metrics(eval_out["y_true"], eval_out["y_pred"],
                        y_score=eval_out["y_score"], class_names=class_names)
    bin_m = compute_binary_ids_metrics(eval_out["y_true"], eval_out["y_pred"],
                                        normal_class_index=0)
    sp = compute_spike_stats(eval_out["total_spikes"], eval_out["n_samples"],
                              eval_out["total_sites"])
    # Early-exit auxiliary metrics (its own accuracy/F1 vs the standard headline).
    ee_m = compute_metrics(eval_out["y_true"], eval_out["ee_pred"],
                           y_score=None, class_names=class_names)
    # SSL anomaly-score separation: mean score on benign (class 0) vs attack rows.
    yt = eval_out["y_true"]; score = eval_out["ssl_score"]
    if score is not None:
        benign = yt == 0
        ssl_benign = float(score[benign].mean()) if benign.any() else None
        ssl_attack = float(score[~benign].mean()) if (~benign).any() else None
    else:
        ssl_benign = ssl_attack = None
    return {
        "n_samples": int(eval_out["n_samples"]),
        "metrics": m,
        "binary_ids_metrics": bin_m,
        "spike_stats": sp,
        "early_exit": {
            "accuracy": ee_m["accuracy"],
            "f1_macro": ee_m["f1_macro"],
            "mean_exit_step": float(np.mean(eval_out["exit_step"])),
            "seq_len": seq_len,
            "uncertain_frac": float(eval_out["n_uncertain"]) / max(eval_out["n_samples"], 1),
        },
        "self_supervised": {
            "score_benign": ssl_benign,
            "score_attack": ssl_attack,
        },
        "inference_time_s": float(eval_out["infer_time_s"]),
        "inference_ms_per_sample": float(eval_out["infer_time_s"] * 1000.0) / max(eval_out["n_samples"], 1),
    }


def run_experiment(cfg: Dict) -> Dict:
    cfg = _resolve(cfg)
    base_results_dir = Path(cfg["results_dir"])
    results_dir = base_results_dir / f"seed_{cfg['seed']}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_json = results_dir / "results.json"
    if results_json.exists() and not cfg.get("force", False):
        with open(results_json) as f:
            existing = json.load(f)
        print(f"[skip] {MODEL_NAME}/{cfg['dataset']}/seed_{cfg['seed']} "
              f"already has results.json — pass force=True to re-run.")
        return existing

    _set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    # forward optional loader knobs (no-op for kddcup99/nslkdd official splits)
    seq_extra = {k: cfg[k] for k in ("subsample_fraction", "test_fraction", "label_mode") if k in cfg}
    bundle = get_sequence_dataset(
        cfg["dataset"],
        seq_len=cfg["seq_len"], stride=cfg["stride"],
        batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"],
        seed=cfg["seed"], **seq_extra,
    )

    model = SpikingSSM(
        n_features=bundle.num_features,
        n_classes=bundle.num_classes,
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_blocks=cfg["n_blocks"],
        beta=cfg["beta"],
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg["weight_decay"])
    if cfg["class_weighted"]:
        y_train = bundle.train_loader.dataset.tensors[1].long()
        weight = _class_weights(y_train, bundle.num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
    else:
        loss_fn = nn.CrossEntropyLoss()

    training_curve = []
    train_start = time.time()
    for epoch in range(cfg["epochs"]):
        train_loss = _train_one_epoch(model, bundle.train_loader, optim, loss_fn, cfg, device)
        val_loss, val_acc = _eval_val(model, bundle.val_loader, loss_fn, cfg, device)
        training_curve.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "val_accuracy": val_acc,
        })
        print(f"[epoch {epoch+1}/{cfg['epochs']}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    train_time_s = time.time() - train_start

    test_eval = _eval_test(model, bundle.test_loader, cfg, device)
    test_record = _build_record(test_eval, bundle.class_names, cfg["seq_len"])

    param_info = count_parameters(model)
    hw = {"device": str(device)}
    if device.type == "cuda":
        try:
            idx = torch.cuda.current_device()
            hw["gpu_name"] = torch.cuda.get_device_name(idx)
            hw["cuda_index"] = idx
            hw["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        except Exception:
            pass
    hw["hostname"] = os.uname().nodename if hasattr(os, "uname") else ""

    record = {
        "model": MODEL_NAME,
        "dataset": cfg["dataset"],
        "neuron": NEURON_NAME,
        "encoding": ENCODING_NAME,
        "config": {k: cfg[k] for k in
                   ["seq_len", "stride", "d_model", "d_state", "n_blocks", "beta",
                    "a_pred", "a_sparse", "theta_exit", "theta_uncertain",
                    "batch_size", "epochs", "lr", "weight_decay", "class_weighted",
                    "val_fraction", "seed", "device"]},
        "hardware": hw,
        "data": {
            "num_features": bundle.num_features,
            "num_classes": bundle.num_classes,
            "class_names": bundle.class_names,
            "class_counts_train": bundle.class_counts_train,
            "seq_len": bundle.seq_len,
            "splits": {
                "train": len(bundle.train_loader.dataset),
                "val": len(bundle.val_loader.dataset),
                "test_plus": len(bundle.test_loader.dataset),
                "test_minus_21": None,
            },
        },
        "model_size": param_info,
        "training": {
            "train_time_s": train_time_s,
            "training_curve": training_curve,
        },
        "test_plus": test_record,
        "test_minus_21": None,
    }

    with open(results_dir / "results.json", "w") as f:
        json.dump(record, f, indent=2)
    torch.save(model.state_dict(), results_dir / "checkpoint.pt")
    _append_summary(cfg["summary_csv"], cfg, record)
    return record


def _append_summary(csv_path: str, cfg: Dict, record: Dict):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if p.exists() and p.stat().st_size > 0:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            if set(reader.fieldnames or []) == set(SUMMARY_HEADER):
                existing_rows = [
                    r for r in reader
                    if not (r.get("model") == MODEL_NAME
                            and r.get("dataset") == cfg["dataset"]
                            and str(r.get("seed")) == str(cfg["seed"]))
                ]
    tp = record["test_plus"]
    tp_m = tp["metrics"]
    tp_bin = tp["binary_ids_metrics"]
    ee = tp["early_exit"]
    ssl = tp["self_supervised"]
    row = {
        "model": MODEL_NAME,
        "dataset": cfg["dataset"],
        "neuron": NEURON_NAME,
        "encoding": ENCODING_NAME,
        "accuracy": tp_m["accuracy"],
        "balanced_accuracy": tp_m["balanced_accuracy"],
        "f1_macro": tp_m["f1_macro"],
        "f1_weighted": tp_m["f1_weighted"],
        "precision_macro": tp_m["precision_macro"],
        "recall_macro": tp_m["recall_macro"],
        "roc_auc_weighted": tp_m["roc_auc_weighted"],
        "matthews_corrcoef": tp_m["matthews_corrcoef"],
        "cohen_kappa": tp_m["cohen_kappa"],
        "detection_rate": tp_bin["detection_rate"],
        "false_alarm_rate": tp_bin["false_alarm_rate"],
        "binary_accuracy": tp_bin["binary_accuracy"],
        "binary_f1": tp_bin["binary_f1"],
        "early_exit_accuracy": ee["accuracy"],
        "early_exit_f1_macro": ee["f1_macro"],
        "mean_exit_step": ee["mean_exit_step"],
        "uncertain_frac": ee["uncertain_frac"],
        "ssl_score_benign": ssl["score_benign"],
        "ssl_score_attack": ssl["score_attack"],
        "energy_spikes_per_sample": tp["spike_stats"]["energy_spikes_per_sample"],
        "spike_sparsity": tp["spike_stats"]["spike_sparsity"],
        "trainable_parameters": record["model_size"]["trainable_parameters"],
        "train_time_s": record["training"]["train_time_s"],
        "inference_ms_per_sample": tp["inference_ms_per_sample"],
        "epochs": cfg["epochs"],
        "seq_len": cfg["seq_len"],
        "stride": cfg["stride"],
        "d_model": cfg["d_model"],
        "d_state": cfg["d_state"],
        "n_blocks": cfg["n_blocks"],
        "beta": cfg["beta"],
        "a_pred": cfg["a_pred"],
        "a_sparse": cfg["a_sparse"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "class_weighted": cfg["class_weighted"],
        "seed": cfg["seed"],
    }
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        w.writeheader()
        for r in existing_rows:
            w.writerow(r)
        w.writerow(row)
