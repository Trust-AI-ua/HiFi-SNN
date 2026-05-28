"""Unified metrics module — every experiment script calls compute_metrics.

Produces:
  * Headline (single-number) metrics
      accuracy, balanced_accuracy
      f1_macro, f1_weighted
      precision_macro, recall_macro
      roc_auc_weighted (OVR, weighted, restricted to classes that actually appear)
      matthews_corrcoef
      cohen_kappa
  * Per-class breakdown
      precision_per_class, recall_per_class, f1_per_class, support_per_class
  * Confusion matrix + class names

`compute_spike_stats` summarises the spike record (energy proxy + sparsity).
`count_parameters` reports the trainable parameter count of an nn.Module.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_fscore_support, precision_score, recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    out: Dict = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    out["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))

    # Per-class — aligned to class_names if given, else 0..K-1
    if class_names is not None:
        labels = list(range(len(class_names)))
        names = list(class_names)
    else:
        labels = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
        names = [str(l) for l in labels]
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    out["per_class"] = {
        names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(len(names))
    }
    out["class_names"] = names

    # ROC-AUC, weighted one-vs-rest. sklearn has separate signatures for
    # binary vs multi-class: binary wants y_score as a 1-D array of positive-
    # class probabilities; multi-class wants a 2-D (n, k) matrix that sums to
    # 1 per row. We softmax-normalise the score subset so the multi-class
    # branch satisfies sklearn's sum-to-1 requirement even when every output
    # spike count was zero.
    if y_score is not None:
        try:
            present = np.unique(y_true)
            if y_score.ndim == 2 and y_score.shape[1] >= 2 and present.size >= 2:
                score_subset = y_score[:, present].astype(np.float64)
                shift = score_subset - score_subset.max(axis=1, keepdims=True)
                e = np.exp(shift)
                prob = e / e.sum(axis=1, keepdims=True)
                if present.size == 2:
                    # Binary: feed only the positive-class column (last in
                    # `present`, i.e. the higher-index class).
                    out["roc_auc_weighted"] = float(
                        roc_auc_score(y_true, prob[:, 1])
                    )
                else:
                    out["roc_auc_weighted"] = float(
                        roc_auc_score(y_true, prob, multi_class="ovr",
                                      average="weighted", labels=present)
                    )
            else:
                out["roc_auc_weighted"] = None
        except Exception as e:
            out["roc_auc_weighted"] = None
            out["roc_auc_error"] = str(e)
    else:
        out["roc_auc_weighted"] = None

    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return out


def compute_binary_ids_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normal_class_index: int = 0,
) -> Dict:
    """IDS-canonical binary metrics: Detection Rate, False Alarm Rate.

    Collapses the 5-class problem to (normal, attack) and reports the metrics
    that NSL-KDD literature emphasises as "most important" (per thinline72's
    community baseline and the general IDS evaluation convention).

      Detection Rate  DR  = TP / (TP + FN)   = recall on the "attack" class
                          = fraction of attacks correctly flagged.
      False Alarm Rate FAR = FP / (FP + TN)   = FPR on the "normal" class
                          = fraction of normal traffic wrongly flagged as attack.

    Also returns binary accuracy and F1 for cross-checking against the
    multi-class headline numbers, and the binary confusion matrix.
    """
    y_true_bin = (y_true != normal_class_index).astype(int)
    y_pred_bin = (y_pred != normal_class_index).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    binary_acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    binary_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    binary_f1 = (2 * binary_precision * detection_rate
                 / (binary_precision + detection_rate)
                 if (binary_precision + detection_rate) > 0 else 0.0)
    return {
        "binary_accuracy": float(binary_acc),
        "detection_rate": float(detection_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "binary_precision": float(binary_precision),
        "binary_f1": float(binary_f1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "binary_confusion_matrix": cm.tolist(),
    }


def compute_spike_stats(total_spikes: float, n_samples: int,
                         total_sites: Optional[int]) -> Dict:
    """Energy proxy + sparsity from accumulated spike counts.

    total_sites is the total number of (timestep × neuron × batch_sample)
    sites that fired any spike at all *could* fire; sparsity =
    1 - (total_spikes / total_sites). If total_sites is unknown, sparsity is
    None.
    """
    out = {
        "energy_spikes_per_sample": float(total_spikes) / max(int(n_samples), 1),
        "total_spikes": float(total_spikes),
    }
    if total_sites and total_sites > 0:
        density = float(total_spikes) / float(total_sites)
        out["spike_density"] = density
        out["spike_sparsity"] = 1.0 - density
    else:
        out["spike_density"] = None
        out["spike_sparsity"] = None
    return out


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}
