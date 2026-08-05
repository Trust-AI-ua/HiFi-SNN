#!/usr/bin/env python3
"""Cost-sensitive GBDT teacher evaluated under the frozen leakage manifests.

The teacher is the most informative comparator for the partition question: it
shows whether source-endpoint shift degrades the underlying tree model, the
compiled encoder + SNN student, the dense student, or all of them alike. It
consumes exactly the same fold manifest, train-only preprocessing and binary
labels as every other model.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def run_experiment(cfg: dict) -> dict:
    import sys; sys.path.insert(0, str(REPO))
    from src.data import get_dataset
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    from src.training_treespike import _arrays_from_bundle, _fit_teacher
    import pandas as pd

    rd = Path(cfg["results_dir"]) / f"seed_{cfg['seed']}"
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "results.json").exists() and not cfg.get("force"):
        return json.load(open(rd / "results.json"))

    b = get_dataset(cfg["dataset"], batch_size=cfg.get("batch_size", 128),
                    seed=cfg["seed"])
    Xtr, ytr, Xva, yva, Xte, yte = _arrays_from_bundle(b)
    t0 = time.time()
    teacher, _, present = _fit_teacher(Xtr, ytr, b.num_classes, cfg)
    train_s = time.time() - t0
    Xte_df = pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])])
    pred = present[teacher.predict(Xte_df)]

    m = compute_metrics(yte, pred, y_score=None, class_names=b.class_names)
    bm = compute_binary_ids_metrics(yte, pred, normal_class_index=0)
    booster = teacher.get_booster()
    tdf = booster.trees_to_dataframe()
    rec = {"model": "GBDT", "dataset": cfg["dataset"], "config": dict(cfg),
           "data": {"num_features": b.num_features, "num_classes": b.num_classes,
                    "class_names": b.class_names, "encoder_channels": 0},
           "model_size": {"trainable_parameters": int(len(tdf)),
                          "n_trees": int(tdf["Tree"].nunique()),
                          "n_nodes": int(len(tdf)),
                          "max_depth": int(cfg.get("teacher_max_depth", 6))},
           "training": {"train_time_s": train_s, "training_curve": []},
           "test_plus": {"n_samples": int(len(yte)), "metrics": m,
                         "binary_ids_metrics": bm,
                         "spike_stats": {"synops_per_sample": 0.0, "spike_sparsity": 0.0},
                         "inference_ms_per_sample": 0.0},
           "test_minus_21": None}
    json.dump(rec, open(rd / "results.json", "w"), indent=2, default=str)
    return rec
