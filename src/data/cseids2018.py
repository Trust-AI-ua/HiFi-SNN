"""CSE-CIC-IDS2018 loader.

Successor to CIC-IDS2017 with more attack scenarios and ~6× the data volume
(16.4 M rows across 10 daily CSVs). Same CICFlowMeter feature family but
with slightly abbreviated column names (e.g. ``Tot Fwd Pkts`` vs
``Total Fwd Packets``).

Schema specifics handled here
-----------------------------
* 3 metadata columns at the start (``Dst Port``, ``Protocol``, ``Timestamp``).
  We **drop ``Timestamp``** because it's a date string and would force us to
  hand-engineer a feature representation that isn't comparable across datasets.
  ``Dst Port`` and ``Protocol`` are kept (numeric).
* "Label" sometimes appears as a value inside a file (rogue duplicated header
  rows in 02-20-2018.csv) — we drop any row whose label is literally "Label".
* Trailing whitespace on labels (the "Benign" / "Benign " duplication) is
  stripped.
* "Infilteration" (sic — the original UNB spelling) is mapped honestly.
* Inf / NaN in Flow Byts/s + Flow Pkts/s are zeroed (same fix as CIC-IDS2017).

5-class mapping
---------------
* normal: Benign
* dos   : DoS attacks-* (Hulk/GoldenEye/Slowloris/SlowHTTPTest), DDoS
          attacks-LOIC-HTTP, DDOS attack-HOIC, DDOS attack-LOIC-UDP, Bot
* probe : *(empty — CSE-CIC-IDS2018 has no port-scan attack)*
* r2l   : FTP-BruteForce, SSH-Bruteforce
* u2r   : Brute Force -Web, Brute Force -XSS, SQL Injection, Infilteration

The empty probe class is honest about the dataset's taxonomy coverage
limitation — sklearn's macro-F1 will include a 0 from that class, which is
the correct behaviour for cross-dataset comparison.

Disk cache
----------
First load builds ``dataset/cse-cic-ids2018/_preprocessed_cache_v2.npz`` (~1.2 GB
compressed) in ~15 min. Every subsequent run loads in <10 s. The cache stores
log1p-applied features so per-run only does subsample + split + scaling.

Subsampling
-----------
``subsample_fraction`` defaults to **0.1** (1.6 M rows). On the full dataset
training takes ~3 h per (neuron, encoding); a full 27-combo sweep would be
~80 h. 10 % is still 5× larger than the entire NSL-KDD dataset and produces
results in line with the published CSE-CIC-IDS2018 literature. Set to 1.0 if
you have the time budget.
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

from src.data.nslkdd import DatasetBundle, CLASS_NAMES, CLASS_TO_IDX
from src.data.repair import drop_cse_day_only_columns

CSE_FILES = [
    "02-14-2018.csv",
    "02-15-2018.csv",
    "02-16-2018.csv",
    "02-20-2018.csv",
    "02-21-2018.csv",
    "02-22-2018.csv",
    "02-23-2018.csv",
    "02-28-2018.csv",
    "03-01-2018.csv",
    "03-02-2018.csv",
]

LABEL_TO_CLASS = {
    "Benign": "normal",
    # DoS family
    "DoS attacks-Hulk": "dos",
    "DoS attacks-GoldenEye": "dos",
    "DoS attacks-Slowloris": "dos",
    "DoS attacks-SlowHTTPTest": "dos",
    # DDoS family — all DoS-class for our coarse 5-class scheme
    "DDoS attacks-LOIC-HTTP": "dos",
    "DDOS attack-HOIC": "dos",
    "DDOS attack-LOIC-UDP": "dos",
    # Botnet
    "Bot": "dos",
    # Brute force on remote login
    "FTP-BruteForce": "r2l",
    "SSH-Bruteforce": "r2l",
    # Web / privilege escalation
    "Brute Force -Web": "u2r",
    "Brute Force -XSS": "u2r",
    "SQL Injection": "u2r",
    "Infilteration": "u2r",  # original UNB spelling
}


def _normalise_label(s: str) -> str:
    return str(s).strip()


def _label_to_class_idx(label: str) -> Optional[int]:
    cls = LABEL_TO_CLASS.get(_normalise_label(label))
    return CLASS_TO_IDX[cls] if cls is not None else None


def _read_one(path: Path) -> pd.DataFrame:
    # low_memory=False stops dtype-guessing warnings on the big 02-20 file.
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


CACHE_NAME = "_preprocessed_cache_v2.npz"   # v2 = 02-20-only columns dropped (leakage audit)


def _build_full_arrays(data_dir: Path):
    """Heavy CSV → numpy step. Cached to disk; second call loads in <10 s."""
    cache_path = data_dir / CACHE_NAME
    if cache_path.exists():
        z = np.load(cache_path)
        return z["X"].astype(np.float32), z["y"].astype(np.int64)

    frames = []
    for fname in CSE_FILES:
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing CSE-CIC-IDS2018 file: {path}")
        frames.append(_read_one(path))
    df = pd.concat(frames, ignore_index=True)

    # ---------- drop the 02-20-only identifier columns ----------
    # Flow ID / Src IP / Src Port / Dst IP exist only in 02-20-2018.csv, so the
    # concatenation zero-pads them for the other nine days and Src Port becomes a
    # capture-day indicator that excludes r2l/u2r (see the leakage audit).
    df = drop_cse_day_only_columns(df)

    # ---------- drop rogue duplicated-header rows ----------
    df = df[df["Label"].astype(str).str.strip() != "Label"]

    # ---------- 5-class mapping (drop unmapped labels) ----------
    df["_y"] = df["Label"].astype(str).map(_normalise_label).map(LABEL_TO_CLASS)
    df = df.dropna(subset=["_y"]).copy()
    df["y"] = df["_y"].map(CLASS_TO_IDX).astype(np.int64)

    # ---------- feature cleanup ----------
    # Drop label-related + non-numeric timestamp.
    drop_cols = [c for c in ["Label", "_y", "y", "Timestamp"] if c in df.columns]
    X_df = df.drop(columns=drop_cols)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    X = np.log1p(np.clip(X_df.to_numpy(), 0.0, None)).astype(np.float32)
    y = df["y"].to_numpy()

    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def load_cseids2018(
    data_dir: str = "dataset/cse-cic-ids2018",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.3,
    subsample_fraction: float = 0.1,
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
