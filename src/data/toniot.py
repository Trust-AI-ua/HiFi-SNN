"""TON-IoT (Network) loader — UNSW ToN_IoT Bro/Zeek flow IDS benchmark.

Uses the official `Train_Test_Network.csv` (211,043 rows, 44 columns = 42 Zeek/Bro
features + `label` binary + `type` 10-class). Single balanced file (no predefined
split), so we make a stratified split.

Label = `type`: 10 classes — `normal` + backdoor, ddos, dos, injection, mitm, password,
ransomware, scanning, xss. `normal` is class index 0 (so DR/FAR work). The binary `label`
column is dropped (derive binary via `label_mode`). TON-IoT has no coarser attack-family
grouping in the literature, so `category` == `full` == the 10-class `type`.

Preprocessing matches the other loaders:
  * Drop network identifiers / free-text content fields (`src_ip`, `dst_ip`, `dns_query`,
    `http_uri`, `http_user_agent`, `ssl_subject`, `ssl_issuer`) — these memorise the specific
    testbed (e.g. the OWASP-Shepherd cert appears only in web attacks) and would leak.
  * One-hot the remaining low-cardinality categoricals (`proto`, `service`, `conn_state`,
    dns/ssl/http flags, `weird_*`) over the full table.
  * log1p heavy-tailed numerics (bytes/pkts/ports/durations) then MinMax to [0,1] (train fit).
  * Stratified 80/20 test + 90/10 val.

Source: Moustafa, "A new distributed architecture for evaluating AI-based security systems
... Network TON_IoT datasets," Sustainable Cities and Society, 2021.
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
from src.data._label_modes import apply_label_mode

CSV_NAME = "train_test_network.csv"
BENIGN_FINE = "normal"
DROP_IDENT = ["src_ip", "dst_ip", "dns_query", "http_uri", "http_user_agent",
              "ssl_subject", "ssl_issuer"]
LABEL_COLUMNS = ["type", "label"]   # `type` is kept as the label; binary `label` dropped


def _category(t: str) -> str:
    """TON-IoT has no coarser attack-family grouping -> category == the 10-class type."""
    return t


def _build_full_arrays(data_dir: Path):
    """Parse + preprocess. Cached as `_preprocessed_cache.npz`.
    Returns (X float32 [N, F], y_fine int64 [N], fine_names) with `normal` at index 0."""
    cache = data_dir / "_preprocessed_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (z["X"].astype(np.float32), z["y"].astype(np.int64),
                [str(c) for c in z["classes"]])

    csv = data_dir / CSV_NAME
    if not csv.exists():
        raise FileNotFoundError(f"missing TON-IoT CSV: {csv}")
    df = pd.read_csv(csv, encoding="utf-8-sig", low_memory=False)   # utf-8-sig strips BOM

    y_raw = df["type"].astype(str).str.strip().to_numpy()
    df = df.drop(columns=[c for c in DROP_IDENT + LABEL_COLUMNS if c in df.columns])

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

    fine_names = [BENIGN_FINE] + sorted(set(y_raw) - {BENIGN_FINE})
    idx = {c: i for i, c in enumerate(fine_names)}
    y = np.array([idx[v] for v in y_raw], dtype=np.int64)

    print(f"  TON-IoT: X={X.shape} ({len(num_cols)} numeric + {X.shape[1]-len(num_cols)} "
          f"one-hot), {len(fine_names)} classes, {len(y)} rows")
    np.savez_compressed(cache, X=X, y=y, classes=np.array(fine_names, dtype=object))
    return X, y, fine_names


def load_toniot(
    data_dir: str = "dataset/ton-iot",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    subsample_fraction: float = 1.0,
    label_mode: str = "full",   # "binary" | "full"/"category" (10-class type)
    num_workers: int = 0,
    seed: int = 0,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    X, y_fine, fine_names = _build_full_arrays(data_dir)
    y, class_names = apply_label_mode(label_mode, y_fine, fine_names, _category, BENIGN_FINE)

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
