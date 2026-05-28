"""ISCXIDS2012 loader.

The UNB CIC ISCXIDS2012 dataset ships with **binary** labels only
({Normal, Attack}) — no multi-class attack taxonomy. So this is the one
Phase-1 dataset where we run a true binary problem; CLASS_NAMES = ["normal",
"attack"] and num_classes = 2 propagate through to the model.

Schema handling
---------------
21 raw columns:
* **Drop** (high-cardinality strings / payload / metadata that don't
  generalise): generated, sourcePayloadAsBase64, sourcePayloadAsUTF,
  destinationPayloadAsBase64, destinationPayloadAsUTF, source, destination,
  startDateTime, stopDateTime.
* **Numeric** (kept as-is, then log1p): totalSourceBytes, totalDestinationBytes,
  totalSourcePackets, totalDestinationPackets, sourcePort, destinationPort.
* **One-hot** (categorical, fit over the *union* of all 6 files so train/test
  feature dimensions agree): appName (~100 values), direction (4), protocolName
  (5), sourceTCPFlagsDescription (~20), destinationTCPFlagsDescription (~20).
* TCP-flag descriptions can be NaN — filled as the literal "missing" before
  one-hot.

After preprocessing: ~150 features × ~2.07 M rows.

Disk cache
----------
First load (~7 min: parse 6 CSVs, fillna, one-hot the cats, log1p, concat)
writes ``dataset/iscxids2012/_preprocessed_cache.npz`` (~1 GB compressed).
Subsequent loads ~10 s.

Train / val / test
------------------
ISCXIDS2012 ships no canonical train/test split. We use:
  * Stratified 70/30 train_pool / test
  * 90/10 train / val from the pool
Same pattern as CIC-IDS2017 / CSE-CIC-IDS2018.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.nslkdd import DatasetBundle

CLASS_NAMES = ["normal", "attack"]
CLASS_TO_IDX = {"normal": 0, "attack": 1}

ISCX_FILES = [
    "TestbedSatJun12Flows.csv",
    "TestbedSunJun13Flows.csv",
    "TestbedMonJun14Flows.csv",
    "TestbedTueJun15Flows.csv",
    "TestbedWedJun16Flows.csv",
    "TestbedThuJun17Flows.csv",
]
CSV_SUBDIR = Path("iscxids2012-master") / "data" / "CSV"

DROP_COLS = [
    "generated",
    "sourcePayloadAsBase64", "sourcePayloadAsUTF",
    "destinationPayloadAsBase64", "destinationPayloadAsUTF",
    "source", "destination",
    "startDateTime", "stopDateTime",
]
NUMERIC_COLS = [
    "totalSourceBytes", "totalDestinationBytes",
    "totalSourcePackets", "totalDestinationPackets",
    "sourcePort", "destinationPort",
]
CATEGORICAL_COLS = [
    "appName", "direction", "protocolName",
    "sourceTCPFlagsDescription", "destinationTCPFlagsDescription",
]


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def _build_full_arrays(data_dir: Path):
    """Heavy CSV → numpy step. Result cached; subsequent loads <10 s."""
    cache_path = data_dir / "_preprocessed_cache.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        return z["X"].astype(np.float32), z["y"].astype(np.int64)

    csv_root = data_dir / CSV_SUBDIR
    if not csv_root.exists():
        raise FileNotFoundError(
            f"missing extracted ISCXIDS2012 CSVs at {csv_root} — extract via:\n"
            f"  tar -xzf {data_dir}/iscxids2012-master.tar.gz "
            f"iscxids2012-master/data/CSV/ -C {data_dir}/"
        )

    frames = []
    for fname in ISCX_FILES:
        p = csv_root / fname
        if not p.exists():
            raise FileNotFoundError(f"missing CSV: {p}")
        frames.append(_read_one(p))
    df = pd.concat(frames, ignore_index=True)

    # ---------- label → binary (raw labels are "Normal" / "Attack", capitalised) ----------
    df["Label"] = df["Label"].astype(str).str.strip().str.lower()
    df = df[df["Label"].isin(CLASS_TO_IDX)].copy()
    df["y"] = df["Label"].map(CLASS_TO_IDX).astype(np.int64)

    # ---------- drop irrelevant columns ----------
    drop = [c for c in DROP_COLS + ["Label"] if c in df.columns]
    df = df.drop(columns=drop)

    # ---------- fill NaN in categoricals so one-hot keeps the row ----------
    for c in CATEGORICAL_COLS:
        df[c] = df[c].fillna("missing").astype(str)

    # ---------- numeric: coerce to float, fill Inf/NaN with 0 ----------
    df[NUMERIC_COLS] = (
        df[NUMERIC_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    # ---------- one-hot the categoricals over the union ----------
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dtype=np.float32)

    y = df["y"].to_numpy()
    X_df = df.drop(columns=["y"])

    # ---------- log1p on numeric columns (heavy-tailed) ----------
    for c in NUMERIC_COLS:
        if c in X_df.columns:
            X_df[c] = np.log1p(X_df[c].clip(lower=0).astype(np.float64))

    X = X_df.to_numpy(dtype=np.float32)

    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def load_iscxids2012(
    data_dir: str = "dataset/iscxids2012",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.3,
    subsample_fraction: float = 1.0,
    num_workers: int = 0,
    seed: int = 0,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    X, y = _build_full_arrays(data_dir)

    if subsample_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n_keep = int(len(X) * subsample_fraction)
        idx = rng.choice(len(X), n_keep, replace=False)
        X, y = X[idx], y[idx]

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool, y_train_pool,
        test_size=val_fraction, random_state=seed, stratify=y_train_pool,
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = np.clip(scaler.transform(X_val), 0.0, 1.0).astype(np.float32)
    X_test = np.clip(scaler.transform(X_test), 0.0, 1.0).astype(np.float32)

    def _loader(Xs, ys, shuffle: bool):
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys.astype(np.int64)))
        g = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=g, drop_last=False)

    train_loader = _loader(X_train, y_train, shuffle=True)
    val_loader = _loader(X_val, y_val, shuffle=False)
    test_loader = _loader(X_test, y_test, shuffle=False)

    train_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    train_counts = {CLASS_NAMES[int(k)]: int(v) for k, v in train_counts.items()}

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test21_loader=None,
        num_features=X_train.shape[1],
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        class_counts_train=train_counts,
    )
