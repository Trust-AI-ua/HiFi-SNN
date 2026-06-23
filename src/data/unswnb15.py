"""UNSW-NB15 loader — modern IDS benchmark (Moustafa & Slay, 2015).

Uses the official partitioned files `UNSW_NB15_training-set.csv` (175,341 rows) and
`UNSW_NB15_testing-set.csv` (82,332 rows). 45 columns = `id` + 41 features +
`attack_cat` (10-class label) + `label` (binary).

Preprocessing mirrors the other loaders here (esp. `nslkdd.py`), so it slots into the
same SNN pipeline:
  * Drop `id` (identifier) and `label` (the binary label — kept as `attack_cat` would
    leak it). Label = `attack_cat`, with `Normal` mapped to class index 0 so
    `compute_binary_ids_metrics(normal_class_index=0)` works.
  * One-hot `{proto, service, state}` over the **train+test union** (so feature dims
    match across splits; `proto` alone has ~133 categories).
  * `log1p` the heavy-tailed numerics (bytes/loads/rates/jitter), then MinMax to [0, 1]
    fit on the train split only.
  * Official split honoured: test = the testing-set file; a stratified 90/10 split of
    the training-set gives train/val (val never touches the scaler fit).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.nslkdd import DatasetBundle

CATEGORICAL = ["proto", "service", "state"]
DROP_COLUMNS = ["id", "label"]
LABEL_COLUMN = "attack_cat"


def load_unswnb15(
    data_dir: str = "dataset/unsw-nb15",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    subsample_fraction: float = 1.0,
    num_workers: int = 0,
    seed: int = 0,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    # utf-8-sig strips the BOM on the `id` header
    train_df = pd.read_csv(data_dir / "UNSW_NB15_training-set.csv", encoding="utf-8-sig")
    test_df = pd.read_csv(data_dir / "UNSW_NB15_testing-set.csv", encoding="utf-8-sig")

    # ---------- labels: Normal = 0, then attacks (sorted, deterministic) ----------
    class_names = ["Normal"] + sorted(set(train_df[LABEL_COLUMN].astype(str)) - {"Normal"})
    idx = {c: i for i, c in enumerate(class_names)}
    y_train_full = train_df[LABEL_COLUMN].astype(str).map(idx).to_numpy(np.int64)
    y_test = test_df[LABEL_COLUMN].astype(str).map(idx).to_numpy(np.int64)

    drop = DROP_COLUMNS + [LABEL_COLUMN]
    train_df = train_df.drop(columns=[c for c in drop if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop if c in test_df.columns])

    # ---------- log1p numerics + one-hot categoricals over the union ----------
    n_train = len(train_df)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    num_cols = [c for c in combined.columns if c not in CATEGORICAL]
    combined[num_cols] = combined[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    combined[num_cols] = np.log1p(combined[num_cols].clip(lower=0).astype(np.float64))
    combined = pd.get_dummies(combined, columns=CATEGORICAL, dtype=np.float32)
    feat = combined.to_numpy(dtype=np.float32)
    X_train_full, X_test = feat[:n_train], feat[n_train:]

    # ---------- stratified 90/10 train/val from the training set ----------
    strat = y_train_full if np.bincount(y_train_full).min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_fraction,
        random_state=seed, stratify=strat)

    if subsample_fraction < 1.0:  # optional throttle (applied to the train split)
        rng = np.random.RandomState(seed)
        k = rng.choice(len(X_train), int(len(X_train) * subsample_fraction), replace=False)
        X_train, y_train = X_train[k], y_train[k]

    # ---------- MinMax (fit on train only) ----------
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = np.clip(scaler.transform(X_val), 0.0, 1.0).astype(np.float32)
    X_test = np.clip(scaler.transform(X_test), 0.0, 1.0).astype(np.float32)

    def _loader(Xs, ys, shuffle):
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys.astype(np.int64)))
        g = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=g, drop_last=False)

    train_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    train_counts = {class_names[int(k)]: int(v) for k, v in train_counts.items()}

    return DatasetBundle(
        train_loader=_loader(X_train, y_train, True),
        val_loader=_loader(X_val, y_val, False),
        test_loader=_loader(X_test, y_test, False),
        test21_loader=None,
        num_features=X_train.shape[1],
        num_classes=len(class_names),
        class_names=class_names,
        class_counts_train=train_counts,
    )
