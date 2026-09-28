"""Training / evaluation for TASNN (Transformer-Augmented Spiking NN) baseline.

Competitor model (`src/models/tasnn.py`) trained on the *same* data footing as
every other model here: shared loaders (one-hot + log1p + MinMax→[0,1], no
resampling), shared latency spike encoding, shared metrics/energy/early-exit.
See the fairness note in `src/models/tasnn.py` — TASNN's privileged preprocessing
(PAAN / Port-Role embeddings / CPSMOT-Tomek rebalancing) is deliberately omitted.

Results go to a *separate* summary file (`results/summary_tasnn.csv`) so the
competitor never pollutes the Vanilla-SNN / hybrid summaries. Mirrors
`src/training_sxlstm.py` exactly except for the model and its config keys.

Config keys (with defaults):
    num_steps          25     latency time window T
    d_model            64     transformer / spiking width
    n_heads            4      FTE attention heads
    n_layers           2      FTE transformer layers
    ff                 128    FTE feed-forward width
    beta               0.9    LIF membrane decay
    batch_size         128
    epochs             10
    lr                 1e-3
    weight_decay       1e-4
    class_weighted     False  inverse-frequency CE weights (off, matches our models)
    early_exit_conf    0.9
    val_fraction       0.1
    seed               42
    summary_csv        results/summary_tasnn.csv
"""
from __future__ import annotations

import csv
import inspect
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.data import get_dataset, DATASET_LOADERS
from src.metrics import (compute_metrics, compute_binary_ids_metrics,
                          compute_spike_stats, count_parameters)
from src.models.tasnn import TASNN, compute_total_loss

MODEL_NAME = "TASNN"
NEURON_NAME = "LIF+lateral"        # custom LIF core (Eq. 16-17), not LeakyParallel
ENCODING_NAME = "MASE"             # TTFS + population coding (Eq. 10-11)

DEFAULTS = {
    # --- spike window + transformer (paper defaults: d_model=128, heads=4, layers=3) ---
    "num_steps": 16,               # T (paper)
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "ff": 256,
    "dropout": 0.1,
    # --- spike path ---
    "pop_M": 3,                    # population-coding receptive fields / feature
    "stc_channels": 64,
    "stc_kernel": 3,
    "lif_hidden": [128, 64],
    "lif_beta": 0.9,
    "lif_v_th": 1.0,
    "lif_gamma": 0.1,              # lateral-inhibition coefficient (Eq. 16)
    "surrogate_beta": 5.0,
    "target_rate": 0.1,            # homeostatic r* (Eq. 29)
    # --- loss suite weights (Eq. 32, paper defaults) ---
    "w_tr": 1.0, "w_snn": 1.0, "w_kd": 0.5,
    "w_homeo": 0.1, "w_energy": 1e-3, "w_gate": 1e-3,
    "focal_gamma": 2.0, "kd_tau": 3.0,
    # --- training ---
    "batch_size": 128,
    "epochs": 10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "class_weighted": False,       # focal alpha (class rebalancing) OFF -> same footing as our models
    "early_exit_conf": 0.9,
    "val_fraction": 0.1,
    "seed": 42,
    "device": None,
    "summary_csv": "results/tables/summary_tasnn.csv",
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
    "early_exit_accuracy", "early_exit_f1_macro", "mean_exit_step",
    "test21_accuracy", "test21_f1_macro", "test21_matthews_corrcoef",
    "test21_detection_rate", "test21_false_alarm_rate",
    "energy_spikes_per_sample", "spike_sparsity",
    "trainable_parameters",
    "train_time_s", "inference_ms_per_sample",
    "epochs", "num_steps", "d_model", "n_heads", "n_layers", "ff",
    "stc_channels", "lif_beta",
    "batch_size", "lr", "weight_decay", "class_weighted", "seed",
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


def _eval(model, loader, device, loss_fn=None, early_exit_conf=None):
    model.eval()
    all_pred, all_true, all_score = [], [], []
    ee_pred, ee_exit = [], []
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
            out = model(X)
            logits = TASNN.ensemble_logits(out)          # combine the two heads
            if loss_fn is not None:
                running_loss += float(loss_fn(out, y).item())
                n_batches += 1
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_score.append(logits.float().cpu().numpy())
            all_true.append(y.cpu().numpy())
            total_spikes += float(out["total_spikes"].item())
            total_sites += int(out["spike_sites"])
            n_samples += X.shape[0]
            if early_exit_conf is not None:
                p, e, _, _ = model.predict_early_exit(X, conf_threshold=early_exit_conf)
                ee_pred.append(p.cpu().numpy())
                ee_exit.append(e.cpu().numpy())
    if cuda:
        torch.cuda.synchronize()
    infer_time_s = time.time() - start
    out = dict(
        y_pred=np.concatenate(all_pred),
        y_true=np.concatenate(all_true),
        y_score=np.concatenate(all_score, axis=0),
        total_spikes=total_spikes, total_sites=total_sites,
        n_samples=n_samples, infer_time_s=infer_time_s,
        mean_loss=(running_loss / max(n_batches, 1) if loss_fn is not None else None),
    )
    if early_exit_conf is not None:
        out["ee_pred"] = np.concatenate(ee_pred)
        out["ee_exit"] = np.concatenate(ee_exit)
    return out


def _build_record_for_split(eval_out: Dict, class_names: List[str]) -> Dict:
    m = compute_metrics(eval_out["y_true"], eval_out["y_pred"],
                        y_score=eval_out["y_score"], class_names=class_names)
    bin_m = compute_binary_ids_metrics(eval_out["y_true"], eval_out["y_pred"],
                                        normal_class_index=0)
    sp = compute_spike_stats(eval_out["total_spikes"], eval_out["n_samples"],
                              eval_out["total_sites"])
    rec = {
        "n_samples": int(eval_out["n_samples"]),
        "metrics": m,
        "binary_ids_metrics": bin_m,
        "spike_stats": sp,
        "inference_time_s": float(eval_out["infer_time_s"]),
        "inference_ms_per_sample": float(eval_out["infer_time_s"] * 1000.0) / max(eval_out["n_samples"], 1),
    }
    if "ee_pred" in eval_out:
        ee = compute_metrics(eval_out["y_true"], eval_out["ee_pred"],
                             y_score=None, class_names=class_names)
        rec["early_exit"] = {
            "accuracy": ee["accuracy"],
            "f1_macro": ee["f1_macro"],
            "mean_exit_step": float(np.mean(eval_out["ee_exit"])),
        }
    return rec


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

    model = TASNN(
        num_features=bundle.num_features,
        num_classes=bundle.num_classes,
        num_steps=cfg["num_steps"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        ff=cfg["ff"], dropout=cfg["dropout"],
        pop_M=cfg["pop_M"], stc_channels=cfg["stc_channels"],
        stc_kernel=cfg["stc_kernel"], lif_hidden=tuple(cfg["lif_hidden"]),
        lif_beta=cfg["lif_beta"], lif_v_th=cfg["lif_v_th"],
        lif_gamma=cfg["lif_gamma"], surrogate_beta=cfg["surrogate_beta"],
        target_rate=cfg["target_rate"],
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg["weight_decay"])

    # focal-loss class weights (alpha): OFF unless class_weighted -> same footing
    # as our other models (no class-rebalancing advantage for the competitor).
    alpha = None
    if cfg["class_weighted"]:
        y_train = bundle.train_loader.dataset.tensors[1].long()
        alpha = _class_weights(y_train, bundle.num_classes).to(device)

    def loss_fn(out, y):
        return compute_total_loss(
            out, y, w_tr=cfg["w_tr"], w_snn=cfg["w_snn"], w_kd=cfg["w_kd"],
            w_homeo=cfg["w_homeo"], w_energy=cfg["w_energy"], w_gate=cfg["w_gate"],
            gamma=cfg["focal_gamma"], alpha=alpha, tau=cfg["kd_tau"],
            target_rate=cfg["target_rate"])

    # ---------- training ----------
    training_curve = []
    from src.selection import ValSelector, precision_scores
    sel = ValSelector(cfg.get("select_on_val", False))   # Part B A1; off by default
    train_start = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for X, y in bundle.train_loader:
            X = X.to(device); y = y.to(device)
            out = model(X)
            loss = loss_fn(out, y)
            optim.zero_grad(); loss.backward(); optim.step()
            running_loss += float(loss.item())
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        val_eval = _eval(model, bundle.val_loader, device, loss_fn=loss_fn)
        val_loss = val_eval["mean_loss"]
        val_acc = float((val_eval["y_pred"] == val_eval["y_true"]).mean())
        training_curve.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "val_accuracy": val_acc,
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
            e = _eval(m, bundle.test_loader, device)
            return e["y_pred"], e.get("y_score")
        precisions = precision_scores(model, _score, _eval(model, bundle.test_loader, device)["y_true"],
                                      bundle.class_names)

    # ---------- final evaluation (with early-exit on the headline test set) ----------
    test_eval = _eval(model, bundle.test_loader, device, loss_fn=loss_fn,
                      early_exit_conf=cfg["early_exit_conf"])
    test_record = _build_record_for_split(test_eval, bundle.class_names)
    test_record["test_loss"] = test_eval["mean_loss"]

    test21_record: Optional[Dict] = None
    if bundle.test21_loader is not None:
        test21_eval = _eval(model, bundle.test21_loader, device, loss_fn=loss_fn)
        test21_record = _build_record_for_split(test21_eval, bundle.class_names)
        test21_record["test_loss"] = test21_eval["mean_loss"]

    # ---------- provenance + persist ----------
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
                   ["num_steps", "d_model", "n_heads", "n_layers", "ff", "dropout",
                    "pop_M", "stc_channels", "stc_kernel", "lif_hidden", "lif_beta",
                    "lif_v_th", "lif_gamma", "surrogate_beta", "target_rate",
                    "w_tr", "w_snn", "w_kd", "w_homeo", "w_energy", "w_gate",
                    "focal_gamma", "kd_tau", "batch_size", "epochs", "lr",
                    "weight_decay", "class_weighted", "early_exit_conf",
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
    ee = tp.get("early_exit", {})
    t21 = record.get("test_minus_21") or {}
    t21_m = t21.get("metrics", {}) if t21 else {}
    t21_bin = t21.get("binary_ids_metrics", {}) if t21 else {}
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
        "early_exit_accuracy": ee.get("accuracy"),
        "early_exit_f1_macro": ee.get("f1_macro"),
        "mean_exit_step": ee.get("mean_exit_step"),
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
        "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"],
        "n_layers": cfg["n_layers"],
        "ff": cfg["ff"],
        "stc_channels": cfg["stc_channels"],
        "lif_beta": cfg["lif_beta"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "class_weighted": cfg["class_weighted"],
        "seed": cfg["seed"],
    }
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        w.writeheader()
        for r in existing_rows:
            w.writerow(r)
        w.writerow(row)
