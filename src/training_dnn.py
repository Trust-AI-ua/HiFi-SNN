"""Training / eval for the iso-architecture DNN twin of Tree-to-Spike (DNN-Tree).

Fair SNN-vs-DNN baseline. This reuses the *exact same* pipeline as
``src/training_treespike.py`` — same cost-sensitive XGBoost teacher, the same
``TreeGuidedSpikeEncoder``, and the same distillation objective — so the only
difference between the SNN student and this DNN student is the computational
paradigm (dense one-shot MACs vs spike-triggered SOPs over T steps), not the
data, the teacher, the representation, or the model capacity.

  * The DNN consumes ``encoder.encode_dense(X)`` ([B, out_dim] bin levels), the
    dense analog of the spike train the SNN student sees -> identical ``in_dim``.
  * ``DNNStudent`` has the *exact same* Linear topology / parameter count as
    ``SNNStudent`` (LIF neurons add no weights).
  * Loss = CE + KD + ranking + FP-aware  (the SNN's spike-sparsity term is
    dropped — there are no spikes; ``a_sparse=0``).
  * Energy is reported as Horowitz MACs (N_MAC x 4.6 pJ) and batch-1 latency, for
    a direct comparison against the SNN student's SOPs (N_AC x 0.9 pJ).
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.data import get_dataset, DATASET_LOADERS
from src.metrics import (compute_metrics, compute_binary_ids_metrics, count_parameters)
from src.models import TreeGuidedSpikeEncoder, DNNStudent, quantize_int8_
from src.models.tree_to_spike import quantize_int8_per_channel_
from src.training_treespike import (_fit_teacher, _teacher_proba_global,
                                    _arrays_from_bundle, distillation_loss)

MODEL_NAME = "DNN-Tree"
ENCODING_NAME = "tree-guided-dense"
E_MAC_PJ = 4.6   # Horowitz 45nm: 32-bit FP MAC

DEFAULTS = {
    "T": 20, "pop_max": 3, "prune_quantile": 0.20, "jitter": 0.12,   # encoder (== Tree-to-Spike)
    "hidden": [64, 64], "clip_relu": None,                            # DNN student
    "teacher_n_estimators": 200, "teacher_max_depth": 6, "teacher_lr": 0.1,
    "tau": 2.0, "a_kd": 1.0, "a_rank": 0.3, "a_fp": 0.5,             # distillation (no a_sparse)
    "epochs": 10, "batch_size": 128, "lr": 2e-3, "weight_decay": 0.0,
    "val_fraction": 0.1, "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "results_dir": "results/DNN-Tree",
    "summary_csv": "results/tables/summary_dnn.csv",
    "latency_samples": 1000,   # cap for the batch-1 latency probe on large test sets
    "quantize": False,         # int8 fake-quant before eval (matched precision vs the SNN student)
}

SUMMARY_HEADER = [
    "model", "dataset", "encoding",
    "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
    "precision_macro", "recall_macro", "roc_auc_weighted", "matthews_corrcoef",
    "cohen_kappa", "detection_rate", "false_alarm_rate", "binary_accuracy", "binary_f1",
    "teacher_accuracy", "teacher_f1_macro", "encoder_channels",
    "macs_per_sample", "energy_pj_horowitz", "trainable_parameters",
    "train_time_s", "inference_ms_per_sample_batch1", "inference_ms_per_sample_batched",
    "epochs", "hidden", "clip_relu", "teacher_n_estimators", "teacher_max_depth",
    "batch_size", "lr", "seed",
]


def _set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve(cfg: Dict) -> Dict:
    merged = {**DEFAULTS, **cfg}
    if os.environ.get("HISNN_FORCE", "").lower() in {"1", "true", "yes"}:
        merged["force"] = True
    if "HISNN_SEED" in os.environ:
        try:
            merged["seed"] = int(os.environ["HISNN_SEED"])
        except ValueError:
            pass
    return merged


@torch.no_grad()
def _forward_batches(model, encoder, X, bs, device):
    model.eval()
    preds, scores = [], []
    for i in range(0, len(X), bs):
        xb = encoder.encode_dense(X[i:i + bs]).to(device)
        logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
        scores.append(logits.float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(scores, axis=0)


@torch.no_grad()
def _measure_latency(model, encoder, X, device, n_cap):
    """Batch-1 (edge-realistic) and batched inference latency, ms/sample."""
    model.eval()
    Xs = X[:min(len(X), n_cap)]
    dense = encoder.encode_dense(Xs).to(device)
    cuda = device.type == "cuda"
    # warmup
    _ = model(dense[:1])
    if cuda: torch.cuda.synchronize()
    t0 = time.time()
    for i in range(dense.shape[0]):
        _ = model(dense[i:i + 1])
    if cuda: torch.cuda.synchronize()
    bs1 = (time.time() - t0) * 1000.0 / dense.shape[0]
    if cuda: torch.cuda.synchronize()
    t0 = time.time()
    _ = model(dense)
    if cuda: torch.cuda.synchronize()
    batched = (time.time() - t0) * 1000.0 / dense.shape[0]
    return bs1, batched


def run_experiment(cfg: Dict) -> Dict:
    cfg = _resolve(cfg)
    base = Path(cfg["results_dir"]); results_dir = base / f"seed_{cfg['seed']}"
    results_dir.mkdir(parents=True, exist_ok=True)
    rj = results_dir / "results.json"
    if rj.exists() and not cfg.get("force", False):
        print(f"[skip] {MODEL_NAME}/{cfg['dataset']}/seed_{cfg['seed']} has results.json")
        return json.load(open(rj))

    _set_seed(cfg["seed"])
    device = torch.device(cfg["device"])

    # ---------- data (identical to Tree-to-Spike) ----------
    import inspect
    lp = inspect.signature(DATASET_LOADERS[cfg["dataset"]]).parameters
    dk = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=cfg["seed"])
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in lp:
            dk[opt] = cfg[opt]
    bundle = get_dataset(cfg["dataset"], **dk)
    num_classes = bundle.num_classes
    Xtr, ytr, Xval, yval, Xte, yte = _arrays_from_bundle(bundle)

    train_start = time.time()
    # ---------- same teacher + same tree-guided encoder ----------
    teacher, Xtr_df, teacher_classes = _fit_teacher(Xtr, ytr, num_classes, cfg)
    encoder = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
                                     prune_quantile=cfg["prune_quantile"],
                                     jitter=cfg["jitter"]).fit(teacher, Xtr.shape[1])
    # E1 (matched information): "spike_time" feeds the dense twin the levels exactly as
    # the spike train carries them (see TreeGuidedSpikeEncoder.encode_dense). Default
    # "level" is the production input, unchanged.
    dense_input = cfg.get("dense_input", "level")
    if dense_input not in ("level", "spike_time"):
        raise ValueError(f"dense_input must be 'level' or 'spike_time', got {dense_input!r}")
    encoder.dense_time_quantized = dense_input == "spike_time"
    print(f"[encoder] {encoder.out_dim} channels from {Xtr.shape[1]} features (in_dim shared with SNN)")

    # ---------- DNN student (iso-architecture twin) ----------
    student = DNNStudent(encoder.out_dim, num_classes, hidden=tuple(cfg["hidden"]),
                         clip_relu=cfg["clip_relu"]).to(device)
    optim = torch.optim.Adam(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    teacher_proba_all = torch.tensor(
        _teacher_proba_global(teacher, Xtr_df, teacher_classes, num_classes), dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    N, bs = len(Xtr), cfg["batch_size"]
    zero_rate = torch.zeros((), device=device)
    curve = []
    for ep in range(cfg["epochs"]):
        student.train()
        perm = torch.randperm(N); running = 0.0; nb = 0; parts = {}
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb = encoder.encode_dense(Xtr[idx.numpy()]).to(device)
            logits = student(xb)
            loss, parts = distillation_loss(
                logits, yt[idx].to(device), teacher_proba_all[idx].to(device),
                zero_rate, tau=cfg["tau"], a_kd=cfg["a_kd"], a_rank=cfg["a_rank"],
                a_sparse=0.0, a_fp=cfg["a_fp"])
            optim.zero_grad(); loss.backward(); optim.step()
            running += float(loss.item()); nb += 1
        vp, _ = _forward_batches(student, encoder, Xval, bs, device)
        val_acc = float((vp == yval).mean())
        curve.append({"epoch": ep + 1, "train_loss": running / max(nb, 1), "val_accuracy": val_acc})
        print(f"[epoch {ep+1}/{cfg['epochs']}] loss={running/max(nb,1):.4f} "
              f"ce={parts.get('ce',0):.3f} kd={parts.get('kd',0):.3f} val_acc={val_acc:.4f}")
    train_time_s = time.time() - train_start

    # ---------- optional: fp32 test metrics, captured BEFORE the int8 projection ----------
    fp32_record = None
    if cfg.get("record_fp32"):
        fp_pred, fp_score = _forward_batches(student, encoder, Xte, bs, device)
        fp32_record = {"metrics": compute_metrics(yte, fp_pred, y_score=fp_score,
                                                  class_names=bundle.class_names),
                       "binary_ids_metrics": compute_binary_ids_metrics(yte, fp_pred, normal_class_index=0)}
        # per-channel int8 of the same fp32 weights (one scale per output row), on a copy,
        # so E1 covers the three precisions of tab_perchannel.csv (fp32, per-tensor, per-channel)
        import copy
        pc = quantize_int8_per_channel_(copy.deepcopy(student)).eval()
        pc_pred, pc_score = _forward_batches(pc, encoder, Xte, bs, device)
        fp32_record["per_channel_int8"] = {
            "metrics": compute_metrics(yte, pc_pred, y_score=pc_score, class_names=bundle.class_names),
            "binary_ids_metrics": compute_binary_ids_metrics(yte, pc_pred, normal_class_index=0)}
        del pc

    # ---------- deploy: int8 fake-quant (matched precision vs the int8 SNN student) ----------
    if cfg.get("quantize"):
        quantize_int8_(student)

    # ---------- eval (one-shot) ----------
    cuda = device.type == "cuda"
    if cuda: torch.cuda.synchronize()
    y_pred, y_score = _forward_batches(student, encoder, Xte, bs, device)
    m = compute_metrics(yte, y_pred, y_score=y_score, class_names=bundle.class_names)
    bin_m = compute_binary_ids_metrics(yte, y_pred, normal_class_index=0)
    bs1_ms, batched_ms = _measure_latency(student, encoder, Xte, device, cfg["latency_samples"])

    import pandas as pd
    Xte_df = pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])])
    t_pred = teacher_classes[teacher.predict(Xte_df)]
    t_m = compute_metrics(yte, t_pred, y_score=None, class_names=bundle.class_names)

    macs = student.macs_per_sample()
    energy_pj = macs * E_MAC_PJ
    test_record = {
        "n_samples": int(len(yte)),
        "metrics": m,
        "binary_ids_metrics": bin_m,
        "compute": {"macs_per_sample": macs, "energy_pj_horowitz": energy_pj,
                    "e_mac_pj": E_MAC_PJ, "paradigm": "dense one-shot ANN (T=1)",
                    "quantized_int8": bool(cfg.get("quantize"))},
        "teacher_upper_bound": {"accuracy": t_m["accuracy"], "f1_macro": t_m["f1_macro"],
                                "matthews_corrcoef": t_m["matthews_corrcoef"]},
        "inference_ms_per_sample_batch1": bs1_ms,
        "inference_ms_per_sample_batched": batched_ms,
    }
    if fp32_record is not None:
        test_record["fp32_before_projection"] = fp32_record
    test_record["dense_input"] = dense_input
    param_info = count_parameters(student)
    hw = {"device": str(device)}
    if cuda:
        try:
            i = torch.cuda.current_device(); hw["gpu_name"] = torch.cuda.get_device_name(i)
        except Exception:
            pass
    hw["hostname"] = os.uname().nodename if hasattr(os, "uname") else ""

    record = {
        "model": MODEL_NAME, "dataset": cfg["dataset"], "neuron": "ReLU",
        "encoding": ENCODING_NAME,
        "config": {k: cfg[k] for k in
                   ["T", "pop_max", "prune_quantile", "jitter", "hidden", "clip_relu",
                    "teacher_n_estimators", "teacher_max_depth", "teacher_lr",
                    "tau", "a_kd", "a_rank", "a_fp", "epochs", "batch_size", "lr",
                    "weight_decay", "quantize", "val_fraction", "seed", "device"]},
        # The exact keyword arguments passed to get_dataset: subsample fraction, split,
        # label granularity and seed. Earlier runs did not record them, so the config
        # files were their only record (see results/diagnostics/loader_provenance.csv).
        "loader_kwargs": dict(dk),
        "hardware": hw,
        "data": {"num_features": bundle.num_features, "num_classes": bundle.num_classes,
                 "class_names": bundle.class_names, "class_counts_train": bundle.class_counts_train,
                 "encoder_channels": int(encoder.out_dim)},
        "model_size": param_info,
        "training": {"train_time_s": train_time_s, "training_curve": curve},
        "test_plus": test_record, "test_minus_21": None,
    }
    with open(rj, "w") as f:
        json.dump(record, f, indent=2)
    torch.save(student.state_dict(), results_dir / "checkpoint.pt")
    _append_summary(cfg["summary_csv"], cfg, record)
    return record


def _append_summary(path, cfg, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tp = record["test_plus"]; m = tp["metrics"]; b = tp["binary_ids_metrics"]
    c = tp["compute"]; tub = tp["teacher_upper_bound"]
    row = {
        "model": MODEL_NAME, "dataset": cfg["dataset"], "encoding": ENCODING_NAME,
        "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
        "f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"],
        "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"],
        "roc_auc_weighted": m.get("roc_auc_weighted"), "matthews_corrcoef": m["matthews_corrcoef"],
        "cohen_kappa": m["cohen_kappa"], "detection_rate": b["detection_rate"],
        "false_alarm_rate": b["false_alarm_rate"], "binary_accuracy": b["binary_accuracy"],
        "binary_f1": b["binary_f1"], "teacher_accuracy": tub["accuracy"],
        "teacher_f1_macro": tub["f1_macro"], "encoder_channels": record["data"]["encoder_channels"],
        "macs_per_sample": c["macs_per_sample"], "energy_pj_horowitz": c["energy_pj_horowitz"],
        "trainable_parameters": record["model_size"]["trainable_parameters"],
        "train_time_s": record["training"]["train_time_s"],
        "inference_ms_per_sample_batch1": tp["inference_ms_per_sample_batch1"],
        "inference_ms_per_sample_batched": tp["inference_ms_per_sample_batched"],
        "epochs": cfg["epochs"], "hidden": str(cfg["hidden"]), "clip_relu": cfg["clip_relu"],
        "teacher_n_estimators": cfg["teacher_n_estimators"], "teacher_max_depth": cfg["teacher_max_depth"],
        "batch_size": cfg["batch_size"], "lr": cfg["lr"], "seed": cfg["seed"],
    }
    prior = []
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as f:
            rd = csv.DictReader(f)
            if set(rd.fieldnames or []) == set(SUMMARY_HEADER):
                prior = [r for r in rd if not (r.get("model") == MODEL_NAME
                         and r.get("dataset") == cfg["dataset"] and str(r.get("seed")) == str(cfg["seed"]))]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        w.writeheader()
        for r in prior:
            w.writerow(r)
        w.writerow(row)
