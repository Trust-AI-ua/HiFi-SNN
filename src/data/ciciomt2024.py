"""CICIoMT2024 loader — Internet of Medical Things IDS (CIC, 2024), WiFi-MQTT part.

The CIC release ships per-attack CSVs split into `csv/train/` (51 files) and
`csv/test/` (21 files); **the label is the filename**, not a column. Each file has the
same 45 CICFlowMeter-style numeric features (`Header_Length` … `Weight`).

We use the dataset's own taxonomy — the 18 attack types grouped into the **5 attack
classes** from the paper (DDoS, DoS, Recon, MQTT, Spoofing) plus Benign → **6 classes**,
with `Benign` = index 0 so `compute_binary_ids_metrics(normal_class_index=0)` works:
  Benign_*                -> Benign
  ARP_Spoofing_*          -> Spoofing
  MQTT-*                  -> MQTT
  Recon-*                 -> Recon
  TCP_IP-DDoS-*           -> DDoS
  TCP_IP-DoS-*            -> DoS

Preprocessing matches the other numeric loaders: log1p heavy-tailed features then MinMax
to [0,1] (fit on train). The CIC train/test split is honoured (test = the test files; a
stratified 90/10 split of the train files gives train/val). `subsample_fraction` throttles
both splits (the full train set is ~7.16 M rows). First load parses ~2.2 GB of CSVs and
caches `_preprocessed_cache.npz`.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.nslkdd import DatasetBundle
from src.data._label_modes import apply_label_mode

BENIGN_FINE = "Benign"


def _attack_type(fname: str) -> str:
    """Fine (19-class) label from filename: strip _train/_test, .pcap.csv, and the
    numeric shard suffix (TCP_IP-DDoS-ICMP1 -> TCP_IP-DDoS-ICMP)."""
    b = os.path.basename(fname).replace(".pcap.csv", "")
    b = re.sub(r"_(train|test)$", "", b)
    b = re.sub(r"\d+$", "", b)
    return b


def _category(t: str) -> str:
    """Fine attack type -> the paper's 6-class grouping (5 families + Benign)."""
    if t == "Benign":              return "Benign"
    if t.startswith("ARP_Spoofing"):  return "Spoofing"
    if t.startswith("MQTT"):       return "MQTT"
    if t.startswith("Recon"):      return "Recon"
    if t.startswith("TCP_IP-DDoS"):   return "DDoS"
    if t.startswith("TCP_IP-DoS"):    return "DoS"
    raise ValueError(f"unmapped CICIoMT2024 type: {t}")


def _read_split(glob_pat: str):
    """Concatenate all CSVs matching glob_pat; label = fine attack type (filename).
    Returns (X log1p'd float32 [N, F], y_raw str [N])."""
    frames, labels = [], []
    for f in sorted(glob.glob(glob_pat)):
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
        labels.append(np.full(len(df), _attack_type(f), dtype=object))
    if not frames:
        raise FileNotFoundError(f"no CSVs at {glob_pat}")
    big = pd.concat(frames, ignore_index=True)
    y_raw = np.concatenate(labels)
    X = big.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = np.log1p(np.clip(X.to_numpy(), 0.0, None)).astype(np.float32)
    return X, y_raw


def _build_full_arrays(data_dir: Path):
    cache = data_dir / "_preprocessed_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (z["Xtr"].astype(np.float32), z["ytr"].astype(np.int64),
                z["Xte"].astype(np.float32), z["yte"].astype(np.int64),
                [str(c) for c in z["classes"]])

    Xtr, ytr_raw = _read_split(str(data_dir / "csv" / "train" / "*.csv"))
    Xte, yte_raw = _read_split(str(data_dir / "csv" / "test" / "*.csv"))

    # cache the FINEST (19-class) labels (union of train+test) so any mode derives later
    fine_names = [BENIGN_FINE] + sorted((set(ytr_raw) | set(yte_raw)) - {BENIGN_FINE})
    idx = {c: i for i, c in enumerate(fine_names)}
    ytr = np.array([idx[v] for v in ytr_raw], dtype=np.int64)
    yte = np.array([idx[v] for v in yte_raw], dtype=np.int64)

    print(f"  CICIoMT2024: train X={Xtr.shape} test X={Xte.shape}, {len(fine_names)} fine classes")
    np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                        classes=np.array(fine_names, dtype=object))
    return Xtr, ytr, Xte, yte, fine_names


def load_ciciomt2024(
    data_dir: str = "dataset/cic-iomt-2024",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    subsample_fraction: float = 1.0,
    label_mode: str = "category",  # "binary" | "category" (6) | "full" (19)
    num_workers: int = 0,
    seed: int = 0,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    X_train_full, ytr_fine, X_test, yte_fine, fine_names = _build_full_arrays(data_dir)
    y_train_full, class_names = apply_label_mode(label_mode, ytr_fine, fine_names, _category, BENIGN_FINE)
    y_test, _ = apply_label_mode(label_mode, yte_fine, fine_names, _category, BENIGN_FINE)

    if subsample_fraction < 1.0:
        rng = np.random.RandomState(seed)
        ktr = rng.choice(len(X_train_full), int(len(X_train_full) * subsample_fraction), replace=False)
        kte = rng.choice(len(X_test), int(len(X_test) * subsample_fraction), replace=False)
        X_train_full, y_train_full = X_train_full[ktr], y_train_full[ktr]
        X_test, y_test = X_test[kte], y_test[kte]

    strat = y_train_full if np.bincount(y_train_full).min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_fraction,
        random_state=seed, stratify=strat)

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
