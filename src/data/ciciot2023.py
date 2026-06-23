"""CICIoT2023 loader — large-scale IoT IDS benchmark (CIC, 2023).

The dataset ships as 169 Spark-part CSVs under `csv/` (`part-00000…00168`), one logical
table of ~46.69 M rows × 47 columns = 46 CICFlowMeter-style numeric features + a string
`label` column with 34 fine-grained attack types.

We use the paper's coarse taxonomy — the 34 types grouped into **8 classes** (Benign +
the 7 attack families), with `Benign` = index 0 so
`compute_binary_ids_metrics(normal_class_index=0)` works:
  BenignTraffic                          -> Benign
  DDoS-*                                 -> DDoS
  DoS-*                                  -> DoS
  Mirai-*                                -> Mirai
  Recon-*, VulnerabilityScan             -> Recon
  DNS_Spoofing, MITM-ArpSpoofing         -> Spoofing
  BrowserHijacking/CommandInjection/Sql  -> Web
  Injection/XSS/Uploading/Backdoor       -> Web
  DictionaryBruteForce                   -> BruteForce

Preprocessing matches the other numeric loaders: log1p heavy-tailed features then MinMax
to [0,1] (fit on train), stratified 70/30 test + 90/10 val (no predefined split exists).
`subsample_fraction` is important here — the full set is 46.7 M rows. First load parses
~14 GB of CSVs (~5-10 min, several GB RAM) and caches `_preprocessed_cache.npz`.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.nslkdd import DatasetBundle
from src.data._label_modes import apply_label_mode

BENIGN_FINE = "BenignTraffic"
_WEB = {"BrowserHijacking", "CommandInjection", "SqlInjection", "XSS",
        "Uploading_Attack", "Backdoor_Malware"}
_SPOOF = {"DNS_Spoofing", "MITM-ArpSpoofing"}


def _family(lbl: str) -> str:
    if lbl == "BenignTraffic":           return "Benign"
    if lbl.startswith("DDoS-"):          return "DDoS"
    if lbl.startswith("DoS-"):           return "DoS"
    if lbl.startswith("Mirai-"):         return "Mirai"
    if lbl.startswith("Recon-") or lbl == "VulnerabilityScan":  return "Recon"
    if lbl in _SPOOF:                    return "Spoofing"
    if lbl == "DictionaryBruteForce":    return "BruteForce"
    if lbl in _WEB:                      return "Web"
    raise ValueError(f"unmapped CICIoT2023 label: {lbl!r}")


def _build_full_arrays(data_dir: Path):
    """Read all shards -> (X log1p'd float32 [N, 46], y int64 [N], class_names).
    Cached as `_preprocessed_cache.npz`. Reads shards one at a time and downcasts to
    float32 immediately to keep peak memory bounded."""
    cache = data_dir / "_preprocessed_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (z["X"].astype(np.float32), z["y"].astype(np.int64),
                [str(c) for c in z["classes"]])

    files = sorted(glob.glob(str(data_dir / "csv" / "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSV shards in {data_dir/'csv'}")

    Xs, ys_raw = [], []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        lbl = df["label"].astype(str).to_numpy()
        feats = df.drop(columns=["label"]).apply(pd.to_numeric, errors="coerce")
        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xs.append(np.log1p(np.clip(feats.to_numpy(), 0.0, None)).astype(np.float32))
        ys_raw.append(lbl)
    X = np.concatenate(Xs); del Xs
    y_raw = np.concatenate(ys_raw)

    # cache the FINEST (34-class) labels so any label_mode can be derived later
    fine_names = [BENIGN_FINE] + sorted(str(c) for c in set(y_raw) - {BENIGN_FINE})
    idx = {c: i for i, c in enumerate(fine_names)}
    y = np.array([idx[v] for v in y_raw], dtype=np.int64)

    print(f"  CICIoT2023: X={X.shape}, {len(fine_names)} fine classes, {len(y)} rows")
    np.savez_compressed(cache, X=X, y=y, classes=np.array(fine_names, dtype=object))
    return X, y, fine_names


def load_ciciot2023(
    data_dir: str = "dataset/cic-iot2023",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,   # paper uses 80/20
    subsample_fraction: float = 1.0,
    label_mode: str = "category",  # "binary" | "category" (8) | "full" (34)
    num_workers: int = 0,
    seed: int = 0,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    X, y_fine, fine_names = _build_full_arrays(data_dir)
    y, class_names = apply_label_mode(label_mode, y_fine, fine_names, _family, BENIGN_FINE)

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
