"""Edge-IIoTset loader — IoT/IIoT intrusion detection (Ferrag et al., IEEE Access 2022).

Uses the official "Selected dataset for ML and DL" file
`DNN-EdgeIIoT-dataset.csv` (~2.22 M rows, 63 columns = 61 features + `Attack_label`
binary + `Attack_type` 15-class) and applies the authors' published preprocessing
recipe, adapted to this project's `DatasetBundle` interface.

Recipe (Ferrag et al., faithfully followed):
  * Drop 15 identifier / leakage / payload columns (IPs, timestamps, raw payloads,
    full URIs, ephemeral ports) — these would let a model cheat or memorise hosts.
  * `dropna(how="any")` + `drop_duplicates`.
  * Label = `Attack_type` (15 classes: Normal + 14 attacks); `Attack_label` (the
    redundant binary label) is dropped so it cannot leak.

Encoding (matches the recipe, just auto-detected):
  * One-hot every *object*-dtype column over the full table. Across the full file these
    are exactly the recipe's 7 categoricals — `http.request.method`, `http.referer`,
    `http.request.version`, `dns.qry.name.len` (mislabelled: holds DNS query *names*),
    `mqtt.conack.flags`, `mqtt.protoname`, `mqtt.topic` — yielding ~56 dummy columns, so
    this faithfully reproduces the recipe's dummy encoding without hard-coding names.
  * All remaining numerics get `log1p` (heavy-tailed counts/lengths) then MinMax to
    [0, 1] (fit on train), matching every other loader here so the spike encoders behave.
  * Net feature dim ≈ 95 (≈39 numeric + ≈56 one-hot).

`Normal` is class index 0 so `compute_binary_ids_metrics(normal_class_index=0)` works.
First load parses the 1.2 GB CSV (~1-2 min) and caches `_preprocessed_cache.npz`.
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

CSV_REL = "Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

DROP_COLUMNS = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
]
LABEL_COLUMNS = ["Attack_label", "Attack_type"]


def _build_full_arrays(data_dir: Path):
    """Parse + preprocess the DNN CSV. Cached as `_preprocessed_cache.npz`.

    Returns (X float32 [N, F], y int64 [N], class_names list). The cache stores the
    class-name order so labels are stable across runs.
    """
    cache = data_dir / "_preprocessed_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (z["X"].astype(np.float32), z["y"].astype(np.int64),
                [str(c) for c in z["classes"]])

    csv = data_dir / CSV_REL
    if not csv.exists():
        raise FileNotFoundError(f"missing Edge-IIoTset DNN CSV: {csv}")
    df = pd.read_csv(csv, low_memory=False)

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.dropna(axis=0, how="any").drop_duplicates()

    y_raw = df["Attack_type"].astype(str).str.strip().to_numpy()
    df = df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])

    # genuinely-categorical (string) columns -> one-hot; everything else -> numeric
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in df.columns if c not in obj_cols]

    Xnum = df[num_cols].apply(pd.to_numeric, errors="coerce")
    Xnum = Xnum.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xnum = np.log1p(np.clip(Xnum.to_numpy(), 0.0, None)).astype(np.float32)

    if obj_cols:
        Xcat = pd.get_dummies(df[obj_cols].astype(str), dummy_na=False).astype(np.float32)
        X = np.concatenate([Xnum, Xcat.to_numpy(dtype=np.float32)], axis=1).astype(np.float32)
    else:
        X = Xnum

    # labels: Normal = 0, then the attacks (sorted, deterministic)
    classes = ["Normal"] + sorted(set(y_raw) - {"Normal"})
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[v] for v in y_raw], dtype=np.int64)

    print(f"  Edge-IIoTset: X={X.shape} ({len(num_cols)} numeric + "
          f"{X.shape[1]-len(num_cols)} one-hot), {len(classes)} classes, "
          f"{len(y)} rows after dropna/dedup")
    np.savez_compressed(cache, X=X, y=y, classes=np.array(classes, dtype=object))
    return X, y, classes


def load_edgeiiot(
    data_dir: str = "dataset/Edge-IIoTset",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.3,
    subsample_fraction: float = 1.0,
    num_workers: int = 0,
    seed: int = 0,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    X, y, class_names = _build_full_arrays(data_dir)

    if subsample_fraction < 1.0:
        rng = np.random.RandomState(seed)
        keep = rng.choice(len(X), int(len(X) * subsample_fraction), replace=False)
        X, y = X[keep], y[keep]

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool, y_train_pool, test_size=val_fraction,
        random_state=seed, stratify=y_train_pool)

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
