"""Consecutive-record sequence windowing for the Spiking-SSM (Design A).

Design A is a sequence model that consumes flow-window sequences ``[B, L, F]``,
whereas the rest of the framework is tabular per-flow ``[B, F]``. KDDCup99's
standard 41-feature format carries no source-IP / destination-IP / timestamp
columns, so genuine *per-host* windows are not possible. Instead we build
**consecutive-record sliding windows** over the (temporally ordered) KDDCup99
files: window i = rows [i, i+L), and the window label is the class of its last
row. This is the standard way to apply a sequence model to KDDCup99.

Preprocessing is identical to the tabular loader (``src/data/kddcup99.py``):
log1p on heavy-tailed numerics, one-hot of the three categoricals against the
train+test union, and MinMax to [0, 1] fit on train rows. The official KDDCup99
split is preserved (train = ``kddcup.data_10_percent_corrected``, test =
``corrected/corrected``); windows are built **within** each file so a window
never straddles the train/test boundary, and the train windows are then split
90/10 (stratified on the window label) into train/val.

Only KDDCup99 is wired up for now; ``SEQ_DATASET_LOADERS`` is the extension
point for additional datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.kddcup99 import KDDCUP99_COLUMNS, _read_split as _read_kdd_split
from src.data.nslkdd import (
    CATEGORICAL, LOG1P_COLUMNS, CLASS_NAMES, _label_to_class_idx,
)


@dataclass
class SeqDatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    test21_loader: Optional[DataLoader]
    num_features: int
    num_classes: int
    class_names: List[str]
    class_counts_train: dict
    seq_len: int


def _make_windows(X: np.ndarray, y: np.ndarray, seq_len: int, stride: int):
    """Build consecutive-record windows from order-preserving rows.

    X: [N, F] (temporal order), y: [N]. Returns Xw [Nw, L, F] and yw [Nw],
    where yw[i] is the class of the *last* row in window i.
    """
    n = X.shape[0]
    if n < seq_len:
        raise ValueError(f"only {n} rows for seq_len={seq_len}")
    starts = np.arange(0, n - seq_len + 1, stride)
    # gather index matrix [Nw, L] then fancy-index -> [Nw, L, F]
    idx = starts[:, None] + np.arange(seq_len)[None, :]
    Xw = X[idx]                                  # [Nw, L, F]
    yw = y[starts + seq_len - 1]                 # label = last row in each window
    return Xw.astype(np.float32), yw.astype(np.int64)


def load_kddcup99_sequences(
    data_dir: str = "dataset/kddcup99",
    seq_len: int = 16,
    stride: int = 16,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 0,
) -> SeqDatasetBundle:
    data_dir = Path(data_dir)
    train_path = data_dir / "kddcup.data_10_percent_corrected"
    test_path = data_dir / "corrected" / "corrected"
    if not test_path.exists():
        test_path = data_dir / "corrected"

    train_df = _read_kdd_split(train_path)
    test_df = _read_kdd_split(test_path)

    # ---------- labels -> 5-class (order preserved) ----------
    y_train_rows = train_df["label"].map(_label_to_class_idx).to_numpy().astype(np.int64)
    y_test_rows = test_df["label"].map(_label_to_class_idx).to_numpy().astype(np.int64)

    # ---------- log1p on heavy-tailed numerics ----------
    for col in LOG1P_COLUMNS:
        train_df[col] = np.log1p(train_df[col].clip(lower=0).astype(np.float64))
        test_df[col] = np.log1p(test_df[col].clip(lower=0).astype(np.float64))

    # ---------- one-hot over union (keeps row order via _split marker) ----------
    n_train = len(train_df)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = pd.get_dummies(combined, columns=CATEGORICAL, dtype=np.float32)
    feat = combined.drop(columns=["label"]).to_numpy(dtype=np.float32)
    X_train_rows = feat[:n_train]
    X_test_rows = feat[n_train:]

    # ---------- MinMax (fit on train rows) ----------
    scaler = MinMaxScaler()
    X_train_rows = scaler.fit_transform(X_train_rows).astype(np.float32)
    X_test_rows = np.clip(scaler.transform(X_test_rows), 0.0, 1.0).astype(np.float32)

    # ---------- windows within each file (no boundary crossing) ----------
    Xw_train, yw_train = _make_windows(X_train_rows, y_train_rows, seq_len, stride)
    Xw_test, yw_test = _make_windows(X_test_rows, y_test_rows, seq_len, stride)

    # ---------- stratified 90/10 split of TRAIN windows -> train/val ----------
    # (stratify falls back gracefully if a class has too few windows)
    strat = yw_train if len(np.unique(yw_train)) > 1 and \
        np.bincount(yw_train).min() >= 2 else None
    Xtr, Xval, ytr, yval = train_test_split(
        Xw_train, yw_train, test_size=val_fraction, random_state=seed, stratify=strat)

    def _loader(Xs, ys, shuffle):
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys.astype(np.int64)))
        g = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=g, drop_last=False)

    train_loader = _loader(Xtr, ytr, True)
    val_loader = _loader(Xval, yval, False)
    test_loader = _loader(Xw_test, yw_test, False)

    counts = dict(zip(*np.unique(ytr, return_counts=True)))
    counts = {CLASS_NAMES[int(k)]: int(v) for k, v in counts.items()}

    return SeqDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test21_loader=None,
        num_features=Xw_train.shape[2],
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        class_counts_train=counts,
        seq_len=seq_len,
    )


SEQ_DATASET_LOADERS = {
    "kddcup99": load_kddcup99_sequences,
}


def get_sequence_dataset(name: str, **kwargs) -> SeqDatasetBundle:
    if name not in SEQ_DATASET_LOADERS:
        raise ValueError(f"No sequence loader for {name!r}. "
                         f"Available: {list(SEQ_DATASET_LOADERS)}")
    return SEQ_DATASET_LOADERS[name](**kwargs)
