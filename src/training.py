"""Shared training/eval loop used by every Phase-1 experiment script.

A single ``run_experiment(config)`` call is all an experiment script does.

Pipeline per run
----------------
1. Build dataset bundle: train / val / test (KDDTest+) / test21 (KDDTest-21).
2. Train for ``epochs`` epochs on train; after each epoch evaluate val_loss /
   val_acc and append to the training curve.
3. Final evaluation on **both** test sets — KDDTest+ and KDDTest-21 — using the
   shared ``compute_metrics`` (full metric set: accuracy, balanced accuracy,
   F1 macro/weighted, per-class P/R/F1, MCC, Cohen's Kappa, ROC-AUC weighted,
   confusion matrix).
4. Spike accounting for the energy proxy + sparsity, plus model parameter
   count and inference timing.
5. Write ``results.json``, ``checkpoint.pt`` and append one row to
   ``results/summary.csv``.

config keys (with defaults):
    dataset           "nslkdd"
    neuron            "Leaky"   one of the 9 snntorch neurons
    encoding          "rate"    "rate" | "latency" | "delta"
    num_steps         25
    hidden_size       128
    batch_size        256
    epochs            5
    lr                1e-3
    val_fraction      0.1
    seed              0
    device            None      → auto (cuda if available else cpu)
    results_dir       required  → where to write results.json / checkpoint.pt
    summary_csv       results/summary.csv
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

from src.data import get_dataset
from src.encoding import get_encoder
from src.metrics import (compute_metrics, compute_binary_ids_metrics,
                          compute_spike_stats, count_parameters)
from src.models import SNN


DEFAULTS = {
    "num_steps": 25,
    "hidden_size": 128,
    # batch_size=128 + epochs=10 + lr=1e-3 + Adam matches the dominant
    # NSL-KDD baseline (Attention-CNN-LSTM, Nature 2025) and Zhou 2020 (SNN)
    # so headline numbers are directly comparable to prior work.
    "batch_size": 128,
    "epochs": 10,
    "lr": 1e-3,
    "val_fraction": 0.1,
    "seed": 42,            # fixed for reproducibility — overridable per-run for multi-seed sweeps
    "device": None,
    "summary_csv": "results/tables/summary.csv",
    "force": False,        # if True, re-run and overwrite results.json/checkpoint.pt even if present
}

SUMMARY_HEADER = [
    "dataset", "neuron", "encoding",
    # Headline test metrics (KDDTest+) — 5-class
    "accuracy", "balanced_accuracy",
    "f1_macro", "f1_weighted",
    "precision_macro", "recall_macro",
    "roc_auc_weighted", "matthews_corrcoef", "cohen_kappa",
    # IDS-canonical binary metrics on KDDTest+ (normal vs attack)
    "detection_rate", "false_alarm_rate",
    "binary_accuracy", "binary_f1",
    # Test-21 standard split (when available) — both 5-class headline + binary
    "test21_accuracy", "test21_f1_macro", "test21_matthews_corrcoef",
    "test21_detection_rate", "test21_false_alarm_rate",
    # Efficiency / cost
    "energy_spikes_per_sample", "spike_sparsity",
    "trainable_parameters",
    "train_time_s", "inference_ms_per_sample",
    # Provenance
    "epochs", "num_steps", "hidden_size", "batch_size", "lr", "seed",
]


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN (small perf cost; required for bit-identical reruns)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve(cfg: Dict) -> Dict:
    merged = {**DEFAULTS, **cfg}
    if merged.get("device") is None:
        merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # Orchestrator env-var overrides — keep the 27 configs untouched while
    # sweeping seeds or force-rerunning.
    if os.environ.get("HISNN_FORCE", "").lower() in {"1", "true", "yes"}:
        merged["force"] = True
    if "HISNN_SEED" in os.environ:
        try:
            merged["seed"] = int(os.environ["HISNN_SEED"])
        except ValueError:
            pass
    return merged


def _eval(model, loader, encoder, num_steps, device, loss_fn=None):
    """Single-pass evaluation. Returns predictions, targets, scores, mean loss,
    spike count, spike sites, sample count, and wall-clock inference time.

    Timing brackets ``torch.cuda.synchronize()`` so it reflects compute, not
    async queue depth — important when the GPU is shared with another process.
    """
    model.eval()
    all_pred, all_true, all_score = [], [], []
    total_spikes = 0.0
    total_sites = 0
    n_samples = 0
    running_loss = 0.0
    n_batches = 0
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device); y = y.to(device)
            spikes = encoder(X, num_steps)
            logits, spike_count, spike_sites = model(spikes)
            if loss_fn is not None:
                running_loss += float(loss_fn(logits, y).item())
                n_batches += 1
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_score.append(logits.float().cpu().numpy())
            all_true.append(y.cpu().numpy())
            total_spikes += float(spike_count.item())
            total_sites += int(spike_sites)
            n_samples += X.shape[0]
    if cuda:
        torch.cuda.synchronize()
    infer_time_s = time.time() - start
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score, axis=0)
    mean_loss = running_loss / max(n_batches, 1) if loss_fn is not None else None
    return dict(y_pred=y_pred, y_true=y_true, y_score=y_score,
                total_spikes=total_spikes, total_sites=total_sites,
                n_samples=n_samples, infer_time_s=infer_time_s, mean_loss=mean_loss)


def _build_record_for_split(eval_out: Dict, class_names: List[str]) -> Dict:
    m = compute_metrics(eval_out["y_true"], eval_out["y_pred"],
                        y_score=eval_out["y_score"], class_names=class_names)
    # IDS-canonical binary metrics — class index 0 is "normal" by convention.
    bin_m = compute_binary_ids_metrics(eval_out["y_true"], eval_out["y_pred"],
                                        normal_class_index=0)
    sp = compute_spike_stats(eval_out["total_spikes"], eval_out["n_samples"],
                              eval_out["total_sites"])
    return {
        "n_samples": int(eval_out["n_samples"]),
        "metrics": m,
        "binary_ids_metrics": bin_m,
        "spike_stats": sp,
        "inference_time_s": float(eval_out["infer_time_s"]),
        "inference_ms_per_sample": float(eval_out["infer_time_s"] * 1000.0) / max(eval_out["n_samples"], 1),
    }


def run_experiment(cfg: Dict) -> Dict:
    cfg = _resolve(cfg)
    # Multi-seed-safe leaf path:  results/{dataset}/SNN_{N}/{enc}/seed_{S}/
    # so re-running with a different seed never overwrites a prior run.
    base_results_dir = Path(cfg["results_dir"])
    results_dir = base_results_dir / f"seed_{cfg['seed']}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------- skip if already done (unless force=True) ----------
    results_json = results_dir / "results.json"
    if results_json.exists() and not cfg.get("force", False):
        with open(results_json) as f:
            existing = json.load(f)
        print(f"[skip] {cfg['dataset']}/{cfg['neuron']}/{cfg['encoding']} "
              f"already has results.json — pass force=True to re-run.")
        return existing

    _set_seed(cfg["seed"])

    device = torch.device(cfg["device"])
    # Forward dataset-loader-specific kwargs the config sets — but only those
    # the target loader's signature accepts (NSL-KDD and KDDCup99 don't have
    # subsample_fraction/test_fraction; CIC-IDS2017 and CSE-CIC-IDS2018 do).
    import inspect
    from src.data import DATASET_LOADERS
    loader_params = inspect.signature(DATASET_LOADERS[cfg["dataset"]]).parameters
    dataset_kwargs = dict(
        batch_size=cfg["batch_size"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
    )
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in loader_params:
            dataset_kwargs[opt] = cfg[opt]
    bundle = get_dataset(cfg["dataset"], **dataset_kwargs)
    encoder = get_encoder(cfg["encoding"])
    num_steps = cfg["num_steps"]

    model = SNN(
        num_features=bundle.num_features,
        hidden_size=cfg["hidden_size"],
        num_classes=bundle.num_classes,
        neuron_name=cfg["neuron"],
        num_steps=num_steps,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    # ---------- training ----------
    training_curve = []  # list of {epoch, train_loss, val_loss, val_accuracy}
    from src.selection import ValSelector, precision_scores
    sel = ValSelector(cfg.get("select_on_val", False))   # Part B A1; off by default
    train_start = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for X, y in bundle.train_loader:
            X = X.to(device); y = y.to(device)
            spikes = encoder(X, num_steps)
            logits, _, _ = model(spikes)
            loss = loss_fn(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            running_loss += float(loss.item())
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        val_eval = _eval(model, bundle.val_loader, encoder, num_steps, device, loss_fn=loss_fn)
        val_loss = val_eval["mean_loss"]
        val_acc = float((val_eval["y_pred"] == val_eval["y_true"]).mean())
        training_curve.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })
        vf1 = sel.update(epoch + 1, val_eval["y_true"], val_eval["y_pred"], model)
        if vf1 is not None:
            training_curve[-1]["val_macro_f1"] = vf1
        print(f"[epoch {epoch+1}/{cfg['epochs']}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    train_time_s = time.time() - train_start
    selection = sel.finalize(model)            # selected checkpoint for everything below
    precisions = None
    if cfg.get("record_precisions"):
        def _score(m):
            e = _eval(m, bundle.test_loader, encoder, num_steps, device)
            return e["y_pred"], e.get("y_score")
        precisions = precision_scores(model, _score, bundle.test_loader.dataset.tensors[1].numpy().astype(np.int64),
                                      bundle.class_names)

    # ---------- final evaluation on both test sets ----------
    test_eval = _eval(model, bundle.test_loader, encoder, num_steps, device, loss_fn=loss_fn)
    test_record = _build_record_for_split(test_eval, bundle.class_names)
    test_record["test_loss"] = test_eval["mean_loss"]

    test21_record: Optional[Dict] = None
    if bundle.test21_loader is not None:
        test21_eval = _eval(model, bundle.test21_loader, encoder, num_steps, device, loss_fn=loss_fn)
        test21_record = _build_record_for_split(test21_eval, bundle.class_names)
        test21_record["test_loss"] = test21_eval["mean_loss"]

    # ---------- compile and persist ----------
    param_info = count_parameters(model)
    # Hardware provenance so train_time_s rows can be filtered by GPU later.
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
        "dataset": cfg["dataset"],
        "neuron": cfg["neuron"],
        "encoding": cfg["encoding"],
        "config": {k: cfg[k] for k in
                   ["num_steps", "hidden_size", "batch_size", "epochs", "lr",
                    "val_fraction", "seed", "device"]},
        # The exact keyword arguments passed to get_dataset: subsample fraction, split,
        # label granularity and seed. Earlier runs did not record them, so the config
        # files were their only record (see results/diagnostics/loader_provenance.csv).
        "loader_kwargs": dict(dataset_kwargs),
        "hardware": hw,
        "data": {
            "num_features": bundle.num_features,
            "num_classes": bundle.num_classes,
            "class_names": bundle.class_names,
            "class_counts_train": bundle.class_counts_train,
            "splits": {
                "train": len(bundle.train_loader.dataset),
                "val": len(bundle.val_loader.dataset),
                "test_plus": len(bundle.test_loader.dataset),
                "test_minus_21": (len(bundle.test21_loader.dataset)
                                  if bundle.test21_loader is not None else None),
            },
        },
        "model_size": param_info,
        "training": {
            "train_time_s": train_time_s,
            "training_curve": training_curve,
        },
        "test_plus": test_record,
        "test_minus_21": test21_record,
    }

    if selection.get("enabled"):
        record["training"]["selection"] = selection
    if precisions is not None:
        record["test_plus"]["precisions"] = precisions
    with open(results_dir / "results.json", "w") as f:
        json.dump(record, f, indent=2)
    torch.save(model.state_dict(), results_dir / "checkpoint.pt")
    _append_summary(cfg["summary_csv"], cfg, record)
    return record


def _append_summary(csv_path: str, cfg: Dict, record: Dict):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Read any existing rows and drop the previous entry for this
    # (dataset, neuron, encoding, seed) — that way a re-run replaces rather
    # than duplicates. This makes summary.csv stable under repeated invocations.
    existing_rows = []
    if p.exists() and p.stat().st_size > 0:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            if set(reader.fieldnames or []) == set(SUMMARY_HEADER):
                existing_rows = [
                    r for r in reader
                    if not (r.get("dataset") == cfg["dataset"]
                            and r.get("neuron") == cfg["neuron"]
                            and r.get("encoding") == cfg["encoding"]
                            and str(r.get("seed")) == str(cfg["seed"]))
                ]
            else:
                # Schema mismatch (stale file) — drop and start fresh
                existing_rows = []
    new = True
    tp = record["test_plus"]
    tp_m = tp["metrics"]
    tp_bin = tp["binary_ids_metrics"]
    t21 = record.get("test_minus_21") or {}
    t21_m = t21.get("metrics", {}) if t21 else {}
    t21_bin = t21.get("binary_ids_metrics", {}) if t21 else {}
    row = {
        "dataset": cfg["dataset"],
        "neuron": cfg["neuron"],
        "encoding": cfg["encoding"],
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
        "test21_accuracy": t21_m.get("accuracy"),
        "test21_f1_macro": t21_m.get("f1_macro"),
        "test21_matthews_corrcoef": t21_m.get("matthews_corrcoef"),
        "test21_detection_rate": t21_bin.get("detection_rate") if t21_bin else None,
        "test21_false_alarm_rate": t21_bin.get("false_alarm_rate") if t21_bin else None,
        "energy_spikes_per_sample": tp["spike_stats"]["energy_spikes_per_sample"],
        "spike_sparsity": tp["spike_stats"]["spike_sparsity"],
        "trainable_parameters": record["model_size"]["trainable_parameters"],
        "train_time_s": record["training"]["train_time_s"],
        "inference_ms_per_sample": tp["inference_ms_per_sample"],
        "epochs": cfg["epochs"],
        "num_steps": cfg["num_steps"],
        "hidden_size": cfg["hidden_size"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "seed": cfg["seed"],
    }
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        w.writeheader()
        for r in existing_rows:
            w.writerow(r)
        w.writerow(row)
