"""Shared training/eval loop for Tree-to-Spike (Design C).

Self-contained, like the other ``src/training_*.py`` modules, but with Design C's
train-time teacher + encoder fitting folded in:

  1. Extract the (already preprocessed, MinMax'd) tabular arrays from the shared
     KDDCup99 loader.
  2. Fit a cost-sensitive XGBoost teacher (sample weights up-weight rare classes).
  3. Fit the ``TreeGuidedSpikeEncoder`` from the teacher (split thresholds -> spike
     bins, feature importance -> per-feature channel budget + pruning).
  4. Train the compact ``SNNStudent`` by distillation:
         loss = CE + a_kd*KD + a_rank*ranking + a_sparse*spike_rate + a_fp*FP-aware
  5. (deploy) int8-quantize the student.
  6. Evaluate the SNN-only student with confidence-triggered early exit; report
     the shared metric set (+ AUC), an early-exit block, a SynOps energy proxy,
     and the teacher's upper bound for reference.

The tree is **not** run at inference — only the encoder + SNN student. Results
go to a separate ``results/summary_treespike.csv`` with a ``model`` column.
Same multi-seed mechanics as the other scripts (``seed_<N>`` dir, skip-guard,
``HISNN_SEED`` / ``HISNN_FORCE`` env overrides).
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import get_dataset, DATASET_LOADERS
from src.metrics import (compute_metrics, compute_binary_ids_metrics,
                          compute_spike_stats, count_parameters)
from src.models import TreeGuidedSpikeEncoder, SNNStudent, quantize_int8_

MODEL_NAME = "Tree-to-Spike"
ENCODING_NAME = "tree-guided-latency"

DEFAULTS = {
    # encoder
    "T": 20,
    "pop_max": 3,
    "prune_quantile": 0.20,
    "jitter": 0.12,
    "encoder_mode": "tree",   # "tree" | "uniform" | "all" (encoder ablation, §1)
    # student
    "hidden": [64, 64],
    "beta": 0.9,
    "leaky": True,            # False -> non-leaky IF ablation
    # teacher (XGBoost)
    "teacher_n_estimators": 200,
    "teacher_max_depth": 6,
    "teacher_lr": 0.1,
    # distillation loss weights
    "tau": 2.0,
    "a_kd": 1.0,
    "a_rank": 0.3,
    "a_sparse": 1e-3,
    "a_fp": 0.5,
    # training
    "epochs": 10,
    "batch_size": 128,
    "lr": 2e-3,
    "weight_decay": 0.0,
    # deploy
    "quantize": True,
    "theta_exit": 0.9,
    "theta_uncertain": 0.5,
    # plumbing
    "val_fraction": 0.1,
    "seed": 42,
    "device": None,
    "summary_csv": "results/summary_treespike.csv",
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
    "teacher_accuracy", "teacher_f1_macro",
    "encoder_channels",
    "energy_spikes_per_sample", "spike_sparsity", "synops_per_sample",
    "trainable_parameters",
    "train_time_s", "inference_ms_per_sample",
    "epochs", "T", "pop_max", "hidden", "beta", "leaky", "quantize",
    "teacher_n_estimators", "teacher_max_depth",
    "batch_size", "lr", "seed",
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


def distillation_loss(logits, y, teacher_proba, hidden_rate, benign_class=0,
                      tau=2.0, a_kd=1.0, a_rank=0.3, a_sparse=1e-3, a_fp=0.5):
    """CE + KD + ranking + spike-sparsity + FP-aware (see Design C reference)."""
    eps = 1e-8
    ce = F.cross_entropy(logits, y)

    # KD: soft cross-entropy from the (temperature-sharpened) teacher distribution
    t = teacher_proba.clamp_min(eps) ** (1.0 / tau)
    t = t / t.sum(-1, keepdim=True)
    kd = -(t * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    # ranking: preserve the teacher's ordering of attack-likelihood
    s_attack = 1.0 - F.softmax(logits, dim=-1)[:, benign_class]
    t_attack = 1.0 - teacher_proba[:, benign_class]
    perm = torch.randperm(logits.size(0), device=logits.device)
    sign = torch.sign(t_attack - t_attack[perm])
    rank = F.relu(-sign * (s_attack - s_attack[perm]) + 0.05).mean()

    sparse = hidden_rate  # push toward low firing (energy)

    # FP-aware: be confident on true-benign flows (avoid alert fatigue)
    benign = y == benign_class
    fp = (1.0 - F.softmax(logits, dim=-1)[benign, benign_class]).mean() if benign.any() \
        else torch.zeros((), device=logits.device)

    total = ce + a_kd * kd + a_rank * rank + a_sparse * sparse + a_fp * fp
    # .item() (not float()) on grad tensors avoids a per-batch detach warning.
    sparse_v = sparse.item() if torch.is_tensor(sparse) else float(sparse)
    return total, {"ce": ce.item(), "kd": kd.item(), "rank": rank.item(),
                   "sparse": sparse_v, "fp": (fp.item() if torch.is_tensor(fp) else float(fp))}


def _arrays_from_bundle(bundle):
    """Pull the (preprocessed) tabular arrays out of the shared loader's tensors."""
    Xtr = bundle.train_loader.dataset.tensors[0].numpy()
    ytr = bundle.train_loader.dataset.tensors[1].numpy().astype(np.int64)
    Xval = bundle.val_loader.dataset.tensors[0].numpy()
    yval = bundle.val_loader.dataset.tensors[1].numpy().astype(np.int64)
    Xte = bundle.test_loader.dataset.tensors[0].numpy()
    yte = bundle.test_loader.dataset.tensors[1].numpy().astype(np.int64)
    return Xtr, ytr, Xval, yval, Xte, yte


def _fit_teacher(Xtr, ytr, num_classes, cfg):
    """Fit the cost-sensitive XGBoost teacher.

    XGBoost's sklearn wrapper requires the training labels to be a contiguous
    0..k-1 set, but some datasets do not exhibit every global class (e.g.
    cse-cic-ids2018 has no `probe`, so ytr is {0,1,3,4}). We therefore train on
    locally-remapped labels and return ``local_to_global`` so the caller can
    scatter the teacher's [N, k] probabilities/predictions back into the global
    ``num_classes`` space the student is trained against.
    """
    from xgboost import XGBClassifier
    import pandas as pd
    present = np.unique(ytr)                          # global ids actually present
    remap = np.full(num_classes, -1, dtype=np.int64)
    remap[present] = np.arange(len(present))          # global -> local (contiguous)
    ytr_local = remap[ytr]
    counts = np.bincount(ytr_local, minlength=len(present)).astype(np.float32)
    cls_w = counts.sum() / (counts + 1e-6)
    # cost-sensitivity strength: 1.0 = full inverse-frequency (default), 0.0 = uniform
    # (accuracy-optimal). Softening (e.g. 0.5 = sqrt) trades macro-F1 for accuracy.
    power = float(cfg.get("teacher_cost_power", 1.0))
    if power != 1.0:
        cls_w = cls_w ** power
    sample_w = cls_w[ytr_local]
    Xdf = pd.DataFrame(Xtr, columns=[f"f{i}" for i in range(Xtr.shape[1])])
    teacher = XGBClassifier(
        n_estimators=cfg["teacher_n_estimators"], max_depth=cfg["teacher_max_depth"],
        learning_rate=cfg["teacher_lr"], subsample=0.9, colsample_bytree=0.8,
        importance_type="gain", tree_method="hist", n_jobs=4,
        eval_metric="mlogloss", random_state=cfg["seed"])
    teacher.fit(Xdf, ytr_local, sample_weight=sample_w)
    return teacher, Xdf, present


def _teacher_proba_global(teacher, Xdf, local_to_global, num_classes):
    """teacher.predict_proba in the global num_classes space (absent classes -> 0)."""
    p_local = teacher.predict_proba(Xdf)
    P = np.zeros((p_local.shape[0], num_classes), dtype=np.float32)
    P[:, local_to_global] = p_local
    return P


@torch.no_grad()
def _encode_forward_in_batches(student, encoder, X, batch_size, device):
    """Standard-forward over X in batches -> (y_pred, y_score, total_spikes,
    total_sites, synops_sum, n_samples)."""
    student.eval()
    preds, scores = [], []
    total_spikes = 0.0; total_sites = 0; synops_sum = 0.0; n = 0
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        spikes = encoder.encode(xb).to(device)
        logits, tspk, tsites, synops = student(spikes)
        preds.append(logits.argmax(1).cpu().numpy())
        scores.append(logits.float().cpu().numpy())
        total_spikes += float(tspk); total_sites += int(tsites)
        synops_sum += float(synops) * xb.shape[0]; n += xb.shape[0]
    return (np.concatenate(preds), np.concatenate(scores, axis=0),
            total_spikes, total_sites, synops_sum, n)


def run_experiment(cfg: Dict) -> Dict:
    cfg = _resolve(cfg)
    neuron_name = "Leaky" if cfg["leaky"] else "IF"
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

    # ---------- data (tabular, preprocessed) ----------
    loader_params = __import__("inspect").signature(DATASET_LOADERS[cfg["dataset"]]).parameters
    dataset_kwargs = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"],
                          seed=cfg["seed"])
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in loader_params:
            dataset_kwargs[opt] = cfg[opt]
    bundle = get_dataset(cfg["dataset"], **dataset_kwargs)
    num_classes = bundle.num_classes
    Xtr, ytr, Xval, yval, Xte, yte = _arrays_from_bundle(bundle)

    train_start = time.time()

    # ---------- 1) cost-sensitive XGBoost teacher ----------
    teacher, Xtr_df, teacher_classes = _fit_teacher(Xtr, ytr, num_classes, cfg)

    # ---------- 2) tree-guided spike encoder ----------
    encoder = TreeGuidedSpikeEncoder(
        T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
        jitter=cfg["jitter"], encoder_mode=cfg["encoder_mode"]).fit(teacher, Xtr.shape[1])
    print(f"[encoder] kept {encoder.out_dim} spike channels from {Xtr.shape[1]} features")

    # ---------- 3) distillation training of the SNN student ----------
    student = SNNStudent(encoder.out_dim, num_classes, hidden=tuple(cfg["hidden"]),
                         beta=cfg["beta"], leaky=cfg["leaky"]).to(device)
    optim = torch.optim.Adam(student.parameters(), lr=cfg["lr"],
                             weight_decay=cfg["weight_decay"])
    teacher_proba_all = torch.tensor(
        _teacher_proba_global(teacher, Xtr_df, teacher_classes, num_classes),
        dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    N = len(Xtr)
    bs = cfg["batch_size"]
    training_curve = []
    for ep in range(cfg["epochs"]):
        student.train()
        perm = torch.randperm(N)
        running = 0.0; n_batches = 0; last_parts = {}
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            spikes = encoder.encode(Xtr[idx.numpy()]).to(device)
            logits, tspk, tsites, _ = student(spikes)
            hidden_rate = tspk / max(int(tsites), 1)
            loss, last_parts = distillation_loss(
                logits, yt[idx].to(device), teacher_proba_all[idx].to(device),
                hidden_rate, tau=cfg["tau"], a_kd=cfg["a_kd"], a_rank=cfg["a_rank"],
                a_sparse=cfg["a_sparse"], a_fp=cfg["a_fp"])
            optim.zero_grad(); loss.backward(); optim.step()
            running += float(loss.item()); n_batches += 1
        # val accuracy (standard forward)
        vp, _, _, _, _, _ = _encode_forward_in_batches(student, encoder, Xval, bs, device)
        val_acc = float((vp == yval).mean())
        training_curve.append({"epoch": ep + 1, "train_loss": running / max(n_batches, 1),
                               "val_accuracy": val_acc, **{f"loss_{k}": v for k, v in last_parts.items()}})
        print(f"[epoch {ep+1}/{cfg['epochs']}] loss={running/max(n_batches,1):.4f} "
              f"ce={last_parts.get('ce',0):.3f} kd={last_parts.get('kd',0):.3f} "
              f"val_acc={val_acc:.4f}")
    train_time_s = time.time() - train_start

    # ---------- 4) deploy: int8 quantize (SNN-only inference) ----------
    if cfg["quantize"]:
        quantize_int8_(student)

    # ---------- 5) final eval: headline (forward) + early-exit (aux) ----------
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    y_pred, y_score, tot_spk, tot_sites, synops_sum, n_test = _encode_forward_in_batches(
        student, encoder, Xte, bs, device)
    # early-exit auxiliary pass
    ee_pred = []; exit_steps = []; n_uncertain = 0
    with torch.no_grad():
        for i in range(0, len(Xte), bs):
            xb = Xte[i:i + bs]
            spikes = encoder.encode(xb).to(device)
            p, e, unc, _, _, _ = student.predict_early_exit(
                spikes, theta_exit=cfg["theta_exit"], theta_uncertain=cfg["theta_uncertain"])
            ee_pred.append(p.cpu().numpy()); exit_steps.append(e.cpu().numpy())
            n_uncertain += int(unc.sum().item())
    if cuda:
        torch.cuda.synchronize()
    infer_time_s = time.time() - t0
    ee_pred = np.concatenate(ee_pred); exit_steps = np.concatenate(exit_steps)

    # ---------- metrics ----------
    m = compute_metrics(yte, y_pred, y_score=y_score, class_names=bundle.class_names)
    bin_m = compute_binary_ids_metrics(yte, y_pred, normal_class_index=0)
    sp = compute_spike_stats(tot_spk, n_test, tot_sites)
    ee_m = compute_metrics(yte, ee_pred, y_score=None, class_names=bundle.class_names)
    # teacher upper bound on test
    import pandas as pd
    Xte_df = pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])])
    t_pred = teacher_classes[teacher.predict(Xte_df)]   # local -> global labels
    t_m = compute_metrics(yte, t_pred, y_score=None, class_names=bundle.class_names)

    test_record = {
        "n_samples": int(n_test),
        "metrics": m,
        "binary_ids_metrics": bin_m,
        "spike_stats": {**sp, "synops_per_sample": synops_sum / max(n_test, 1)},
        "early_exit": {
            "accuracy": ee_m["accuracy"], "f1_macro": ee_m["f1_macro"],
            "mean_exit_step": float(np.mean(exit_steps)), "max_steps": cfg["T"],
            "uncertain_frac": float(n_uncertain) / max(n_test, 1),
        },
        "teacher_upper_bound": {
            "accuracy": t_m["accuracy"], "f1_macro": t_m["f1_macro"],
            "matthews_corrcoef": t_m["matthews_corrcoef"],
        },
        "inference_ms_per_sample": float(infer_time_s * 1000.0) / max(n_test, 1),
        "inference_time_s": float(infer_time_s),
    }

    param_info = count_parameters(student)
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
        "neuron": neuron_name,
        "encoding": ENCODING_NAME,
        "config": {k: cfg[k] for k in
                   ["T", "pop_max", "prune_quantile", "jitter", "encoder_mode", "hidden", "beta",
                    "leaky", "teacher_n_estimators", "teacher_max_depth", "teacher_lr",
                    "tau", "a_kd", "a_rank", "a_sparse", "a_fp", "quantize",
                    "theta_exit", "theta_uncertain", "epochs", "batch_size", "lr",
                    "weight_decay", "val_fraction", "seed", "device"]},
        "hardware": hw,
        "data": {
            "num_features": bundle.num_features,
            "num_classes": bundle.num_classes,
            "class_names": bundle.class_names,
            "class_counts_train": bundle.class_counts_train,
            "encoder_channels": int(encoder.out_dim),
            "splits": {
                "train": len(bundle.train_loader.dataset),
                "val": len(bundle.val_loader.dataset),
                "test_plus": len(bundle.test_loader.dataset),
                "test_minus_21": None,
            },
        },
        "model_size": param_info,
        "training": {"train_time_s": train_time_s, "training_curve": training_curve},
        "test_plus": test_record,
        "test_minus_21": None,
    }

    with open(results_dir / "results.json", "w") as f:
        json.dump(record, f, indent=2)
    torch.save(student.state_dict(), results_dir / "checkpoint.pt")
    _append_summary(cfg["summary_csv"], cfg, record, neuron_name)
    return record


def _append_summary(csv_path: str, cfg: Dict, record: Dict, neuron_name: str):
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
    tp_m = tp["metrics"]; tp_bin = tp["binary_ids_metrics"]
    ee = tp["early_exit"]; tub = tp["teacher_upper_bound"]; spk = tp["spike_stats"]
    row = {
        "model": MODEL_NAME, "dataset": cfg["dataset"],
        "neuron": neuron_name, "encoding": ENCODING_NAME,
        "accuracy": tp_m["accuracy"], "balanced_accuracy": tp_m["balanced_accuracy"],
        "f1_macro": tp_m["f1_macro"], "f1_weighted": tp_m["f1_weighted"],
        "precision_macro": tp_m["precision_macro"], "recall_macro": tp_m["recall_macro"],
        "roc_auc_weighted": tp_m["roc_auc_weighted"],
        "matthews_corrcoef": tp_m["matthews_corrcoef"], "cohen_kappa": tp_m["cohen_kappa"],
        "detection_rate": tp_bin["detection_rate"], "false_alarm_rate": tp_bin["false_alarm_rate"],
        "binary_accuracy": tp_bin["binary_accuracy"], "binary_f1": tp_bin["binary_f1"],
        "early_exit_accuracy": ee["accuracy"], "early_exit_f1_macro": ee["f1_macro"],
        "mean_exit_step": ee["mean_exit_step"], "uncertain_frac": ee["uncertain_frac"],
        "teacher_accuracy": tub["accuracy"], "teacher_f1_macro": tub["f1_macro"],
        "encoder_channels": record["data"]["encoder_channels"],
        "energy_spikes_per_sample": spk["energy_spikes_per_sample"],
        "spike_sparsity": spk["spike_sparsity"],
        "synops_per_sample": spk["synops_per_sample"],
        "trainable_parameters": record["model_size"]["trainable_parameters"],
        "train_time_s": record["training"]["train_time_s"],
        "inference_ms_per_sample": tp["inference_ms_per_sample"],
        "epochs": cfg["epochs"], "T": cfg["T"], "pop_max": cfg["pop_max"],
        "hidden": "x".join(map(str, cfg["hidden"])), "beta": cfg["beta"],
        "leaky": cfg["leaky"], "quantize": cfg["quantize"],
        "teacher_n_estimators": cfg["teacher_n_estimators"],
        "teacher_max_depth": cfg["teacher_max_depth"],
        "batch_size": cfg["batch_size"], "lr": cfg["lr"], "seed": cfg["seed"],
    }
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        w.writeheader()
        for r in existing_rows:
            w.writerow(r)
        w.writerow(row)
