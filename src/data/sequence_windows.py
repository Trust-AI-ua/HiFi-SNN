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

Datasets wired up (``SEQ_DATASET_LOADERS``):
  * **kddcup99 / nslkdd** — KDD 41-feature schema carries no host/IP/time, so
    windows are L *consecutive records* within each official split file (label =
    last record's class). Preprocessing matches the tabular loaders.
  * **cicids2017** — CICFlowMeter flows have no Source IP / timestamp columns, so
    windows are L consecutive flows in file (Mon..Fri) concatenation order. There
    is no official split, so windows are split stratified 70/30 (test) + 90/10
    (val), mirroring the tabular loader; MinMax is fit on the train windows only.
  * **cse-cic-ids2018** — same record-order windowing as cicids2017 (Timestamp is
    dropped, no Source IP). Reuses the existing ``_preprocessed_cache.npz`` and
    defaults to ``subsample_fraction=0.5`` to limit load while the vanilla sweep runs.
  * **ctu13** — the raw ``.binetflow`` files DO carry SrcAddr and are time-ordered,
    so these are genuine **per-host temporal windows**: L consecutive flows from the
    same (scenario, SrcAddr) group, label = last flow. Same host-context feature
    engineering as the tabular CTU-13 loader; windows split stratified like CIC.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.kddcup99 import KDDCUP99_COLUMNS, _read_split as _read_kdd_split
from src.data.nslkdd import (
    CATEGORICAL, LOG1P_COLUMNS, CLASS_NAMES, _label_to_class_idx,
    NSLKDD_COLUMNS, _read_split as _read_nsl_split,
)
from src.data import cicids2017 as _cic
from src.data import ctu13 as _ctu
from src.data import cseids2018 as _cse
from src.data import iscxids2012 as _iscx


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


def _make_grouped_windows(X: np.ndarray, y: np.ndarray, group: np.ndarray,
                          seq_len: int, stride: int):
    """Build consecutive windows *within* each group (no cross-group straddle).

    Rows of X are assumed time-ordered within each group id. A window never
    spans two groups, so for ctu13 a window is always L flows from one host.
    """
    order = np.argsort(group, kind="stable")     # cluster groups, keep in-group order
    Xs, ys, gs = X[order], y[order], group[order]
    _, starts = np.unique(gs, return_index=True)
    bounds = list(np.sort(starts)) + [len(gs)]
    Xws, yws = [], []
    arange_L = np.arange(seq_len)
    for a, b in zip(bounds[:-1], bounds[1:]):
        n = b - a
        if n < seq_len:
            continue
        st = np.arange(0, n - seq_len + 1, stride)
        idx = a + st[:, None] + arange_L[None, :]
        Xws.append(Xs[idx])
        yws.append(ys[a + st + seq_len - 1])
    if not Xws:
        raise ValueError(f"no group had >= seq_len={seq_len} rows")
    return np.concatenate(Xws).astype(np.float32), np.concatenate(yws).astype(np.int64)


def _stratify_or_none(y: np.ndarray):
    return y if (len(np.unique(y)) > 1 and np.bincount(y).min() >= 2) else None


def _window_split_scale_loaders(
    Xw, yw, *, seq_len, val_fraction, test_fraction, subsample_fraction,
    batch_size, num_workers, seed, class_names, num_classes,
):
    """Shared path for datasets with no official split (cicids2017, ctu13).

    Windows are split stratified test/val (mirrors the tabular loaders), and the
    MinMax scaler is fit on the TRAIN windows only (rows reshaped to [N*L, F]) so
    there is no validation/test leakage.
    """
    if subsample_fraction < 1.0:
        rng = np.random.RandomState(seed)
        keep = rng.choice(len(Xw), int(len(Xw) * subsample_fraction), replace=False)
        Xw, yw = Xw[keep], yw[keep]

    idx = np.arange(len(yw))
    tr_idx, te_idx = train_test_split(
        idx, test_size=test_fraction, random_state=seed, stratify=_stratify_or_none(yw))
    tr_idx, va_idx = train_test_split(
        tr_idx, test_size=val_fraction, random_state=seed,
        stratify=_stratify_or_none(yw[tr_idx]))

    F = Xw.shape[2]
    scaler = MinMaxScaler().fit(Xw[tr_idx].reshape(-1, F))

    def _scale(a):
        return np.clip(scaler.transform(a.reshape(-1, F)), 0.0, 1.0).reshape(a.shape).astype(np.float32)

    def _loader(Xs, ys, shuffle):
        ds = TensorDataset(torch.from_numpy(_scale(Xs)), torch.from_numpy(ys.astype(np.int64)))
        g = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=g, drop_last=False)

    train_loader = _loader(Xw[tr_idx], yw[tr_idx], True)
    val_loader = _loader(Xw[va_idx], yw[va_idx], False)
    test_loader = _loader(Xw[te_idx], yw[te_idx], False)

    counts = dict(zip(*np.unique(yw[tr_idx], return_counts=True)))
    counts = {class_names[int(k)]: int(v) for k, v in counts.items()}
    return SeqDatasetBundle(
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        test21_loader=None, num_features=F, num_classes=num_classes,
        class_names=class_names, class_counts_train=counts, seq_len=seq_len,
    )


def load_nslkdd_sequences(
    data_dir: str = "dataset/nslkdd",
    seq_len: int = 16,
    stride: int = 16,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    seed: int = 0,
    **_ignore,
) -> SeqDatasetBundle:
    """NSL-KDD consecutive-record windows (same schema/preprocessing as KDDCup99,
    using the official KDDTrain+/KDDTest+ files; KDDTest-21 becomes test21)."""
    data_dir = Path(data_dir)
    train_df = _read_nsl_split(data_dir / "KDDTrain+.txt").drop(columns=["difficulty"])
    test_df = _read_nsl_split(data_dir / "KDDTest+.txt").drop(columns=["difficulty"])
    t21_path = data_dir / "KDDTest-21.txt"
    test21_df = (_read_nsl_split(t21_path).drop(columns=["difficulty"])
                 if t21_path.exists() else None)

    def _labels(df):
        return df["label"].map(_label_to_class_idx).to_numpy().astype(np.int64)

    y_train_rows, y_test_rows = _labels(train_df), _labels(test_df)
    y_t21_rows = _labels(test21_df) if test21_df is not None else None

    for df in (train_df, test_df) + ((test21_df,) if test21_df is not None else ()):
        for col in LOG1P_COLUMNS:
            df[col] = np.log1p(df[col].clip(lower=0).astype(np.float64))

    # one-hot over the union of all splits (keeps consistent feature columns)
    parts = [train_df, test_df] + ([test21_df] if test21_df is not None else [])
    sizes = [len(p) for p in parts]
    combined = pd.get_dummies(pd.concat(parts, ignore_index=True),
                              columns=CATEGORICAL, dtype=np.float32)
    feat = combined.drop(columns=["label"]).to_numpy(dtype=np.float32)
    off = np.cumsum([0] + sizes)
    X_train_rows = feat[off[0]:off[1]]
    X_test_rows = feat[off[1]:off[2]]
    X_t21_rows = feat[off[2]:off[3]] if test21_df is not None else None

    scaler = MinMaxScaler()
    X_train_rows = scaler.fit_transform(X_train_rows).astype(np.float32)
    X_test_rows = np.clip(scaler.transform(X_test_rows), 0.0, 1.0).astype(np.float32)
    if X_t21_rows is not None:
        X_t21_rows = np.clip(scaler.transform(X_t21_rows), 0.0, 1.0).astype(np.float32)

    Xw_train, yw_train = _make_windows(X_train_rows, y_train_rows, seq_len, stride)
    Xw_test, yw_test = _make_windows(X_test_rows, y_test_rows, seq_len, stride)

    strat = _stratify_or_none(yw_train)
    Xtr, Xval, ytr, yval = train_test_split(
        Xw_train, yw_train, test_size=val_fraction, random_state=seed, stratify=strat)

    def _loader(Xs, ys, shuffle):
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys.astype(np.int64)))
        g = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=g, drop_last=False)

    test21_loader = None
    if X_t21_rows is not None:
        Xw_t21, yw_t21 = _make_windows(X_t21_rows, y_t21_rows, seq_len, stride)
        test21_loader = _loader(Xw_t21, yw_t21, False)

    counts = dict(zip(*np.unique(ytr, return_counts=True)))
    counts = {CLASS_NAMES[int(k)]: int(v) for k, v in counts.items()}
    return SeqDatasetBundle(
        train_loader=_loader(Xtr, ytr, True), val_loader=_loader(Xval, yval, False),
        test_loader=_loader(Xw_test, yw_test, False), test21_loader=test21_loader,
        num_features=Xw_train.shape[2], num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES, class_counts_train=counts, seq_len=seq_len,
    )


def load_cicids2017_sequences(
    data_dir: str = "dataset/cicids2017",
    seq_len: int = 16,
    stride: int = 16,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.3,
    subsample_fraction: float = 1.0,
    num_workers: int = 0,
    seed: int = 0,
    **_ignore,
) -> SeqDatasetBundle:
    """CIC-IDS2017 consecutive-flow windows in file (Mon..Fri) order. No IP/time
    columns exist, so this is record-order windowing (a documented proxy)."""
    X_rows, y_rows = _cic._build_full_arrays(Path(data_dir))      # log1p'd, ordered
    Xw, yw = _make_windows(X_rows, y_rows, seq_len, stride)
    return _window_split_scale_loaders(
        Xw, yw, seq_len=seq_len, val_fraction=val_fraction, test_fraction=test_fraction,
        subsample_fraction=subsample_fraction, batch_size=batch_size,
        num_workers=num_workers, seed=seed, class_names=CLASS_NAMES,
        num_classes=len(CLASS_NAMES))


def _build_ctu13_seq_arrays(data_dir: Path):
    """Like ``ctu13._build_full_arrays`` but ALSO returns a per-flow group id
    ``(scenario, SrcAddr)`` so windows can be built per host in time order.
    Cached separately as ``_seq_cache.npz`` (first build re-reads the raw CSVs)."""
    cache = data_dir / "_seq_cache.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"].astype(np.float32), z["y"].astype(np.int64), z["group"].astype(np.int64)

    files = sorted(glob.glob(str(data_dir / "binetflow" / "scenario-*-*.binetflow")))
    if not files:
        raise FileNotFoundError(f"no .binetflow files in {data_dir/'binetflow'}")

    frames = []
    for si, f in enumerate(files):
        df = _ctu._read_one(Path(f))
        lbl = df["Label"].astype(str)
        df = df[lbl.str.contains("Botnet", na=False) | lbl.str.contains("Normal", na=False)].copy()
        if len(df) == 0:
            continue
        for c in ("Dur", "TotPkts", "TotBytes", "SrcBytes", "sTos", "dTos"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df["Sport"] = pd.to_numeric(df["Sport"], errors="coerce").fillna(0)
        df["Dport"] = pd.to_numeric(df["Dport"], errors="coerce").fillna(0)
        for c in ("Proto", "Dir", "State"):
            df[c] = df[c].astype(str).str.strip()
        df["y"] = df["Label"].astype(str).str.contains("Botnet", na=False).astype(np.int64)
        df = df.merge(_ctu._aggregate_per_source(df), on="SrcAddr", how="left")
        df["scenario"] = si
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)      # row order = within-scenario time order
    y = big["y"].to_numpy().astype(np.int64)
    group = pd.factorize(big["scenario"].astype(str) + "|" + big["SrcAddr"].astype(str))[0].astype(np.int64)

    keep = [c for c in (_ctu.FLOW_NUMERICS + _ctu.FLOW_CATS + _ctu.HOST_AGG_NUMERICS)
            if c in big.columns]
    Xf = pd.get_dummies(big[keep].copy(),
                        columns=[c for c in _ctu.FLOW_CATS if c in big.columns], dtype=np.float32)
    for c in _ctu.LOG1P_COLS:
        if c in Xf.columns:
            Xf[c] = np.log1p(Xf[c].clip(lower=0).astype(np.float64))
    Xf = Xf.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    X = Xf.to_numpy(dtype=np.float32)
    np.savez_compressed(cache, X=X, y=y, group=group)
    return X, y, group


def load_ctu13_sequences(
    data_dir: str = "dataset/ctu13",
    seq_len: int = 16,
    stride: int = 16,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.3,
    subsample_fraction: float = 1.0,
    num_workers: int = 0,
    seed: int = 0,
    **_ignore,
) -> SeqDatasetBundle:
    """CTU-13 genuine per-host temporal windows: L consecutive flows from the same
    (scenario, SrcAddr), label = last flow's class. Same feature engineering as the
    tabular CTU-13 loader."""
    X, y, group = _build_ctu13_seq_arrays(Path(data_dir))
    Xw, yw = _make_grouped_windows(X, y, group, seq_len, stride)
    return _window_split_scale_loaders(
        Xw, yw, seq_len=seq_len, val_fraction=val_fraction, test_fraction=test_fraction,
        subsample_fraction=subsample_fraction, batch_size=batch_size,
        num_workers=num_workers, seed=seed, class_names=_ctu.CLASS_NAMES,
        num_classes=len(_ctu.CLASS_NAMES))


def load_cseids2018_sequences(
    data_dir: str = "dataset/cse-cic-ids2018",
    seq_len: int = 16,
    stride: int = 16,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.3,
    subsample_fraction: float = 0.5,
    num_workers: int = 0,
    seed: int = 0,
    **_ignore,
) -> SeqDatasetBundle:
    """CSE-CIC-IDS2018 consecutive-flow windows in file order. Timestamp is dropped
    (date string) and there is no Source IP, so this is record-order windowing (a
    documented proxy). Reuses the existing ``_preprocessed_cache.npz`` — no raw
    re-parse, so it will not contend with a running vanilla sweep. Default
    ``subsample_fraction`` is capped at 0.5 for the same reason."""
    X_rows, y_rows = _cse._build_full_arrays(Path(data_dir))      # log1p'd, ordered
    Xw, yw = _make_windows(X_rows, y_rows, seq_len, stride)
    return _window_split_scale_loaders(
        Xw, yw, seq_len=seq_len, val_fraction=val_fraction, test_fraction=test_fraction,
        subsample_fraction=subsample_fraction, batch_size=batch_size,
        num_workers=num_workers, seed=seed, class_names=CLASS_NAMES,
        num_classes=len(CLASS_NAMES))


def load_iscxids2012_sequences(
    data_dir: str = "dataset/iscxids2012",
    seq_len: int = 16,
    stride: int = 16,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.3,
    subsample_fraction: float = 1.0,
    num_workers: int = 0,
    seed: int = 0,
    **_ignore,
) -> SeqDatasetBundle:
    """ISCX-IDS2012 consecutive-flow windows in file order (source/destination IP
    identifiers are dropped, so no per-host windows). Binary (normal/attack).
    Reuses the existing ``_preprocessed_cache.npz`` — no raw re-parse."""
    X_rows, y_rows = _iscx._build_full_arrays(Path(data_dir))     # log1p'd, ordered
    Xw, yw = _make_windows(X_rows, y_rows, seq_len, stride)
    return _window_split_scale_loaders(
        Xw, yw, seq_len=seq_len, val_fraction=val_fraction, test_fraction=test_fraction,
        subsample_fraction=subsample_fraction, batch_size=batch_size,
        num_workers=num_workers, seed=seed, class_names=_iscx.CLASS_NAMES,
        num_classes=len(_iscx.CLASS_NAMES))


SEQ_DATASET_LOADERS = {
    "kddcup99": load_kddcup99_sequences,
    "nslkdd": load_nslkdd_sequences,
    "cicids2017": load_cicids2017_sequences,
    "ctu13": load_ctu13_sequences,
    "cse-cic-ids2018": load_cseids2018_sequences,
    "iscxids2012": load_iscxids2012_sequences,
}


def get_sequence_dataset(name: str, **kwargs) -> SeqDatasetBundle:
    if name not in SEQ_DATASET_LOADERS:
        raise ValueError(f"No sequence loader for {name!r}. "
                         f"Available: {list(SEQ_DATASET_LOADERS)}")
    return SEQ_DATASET_LOADERS[name](**kwargs)
