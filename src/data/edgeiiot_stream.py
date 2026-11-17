"""Edge-IIoTset stream dataset (track B; results/preregistration/2026-11-17_neuro_stream.md).

Built by scripts/stream_extract.py and scripts/stream_build.py into
dataset/edge-iiotset/_stream/stream_K32.npz. Split by flow, stratified by class: 20% test,
then 10% of the rest for validation (the Edge-IIoTset loader convention), for a given seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PATH = Path(__file__).resolve().parents[2] / "dataset/edge-iiotset/_stream/stream_K32.npz"


@dataclass
class StreamSplit:
    X: dict          # split -> [n, K, F] float32
    length: dict     # split -> [n]
    y: dict          # split -> [n]
    idx: dict        # split -> indices into the stored arrays
    class_names: list
    features: list


def load_stream(seed: int, test_fraction: float = 0.2, val_fraction: float = 0.1, path: Path = PATH) -> StreamSplit:
    z = np.load(path)
    y = z["y"]
    i = np.arange(len(y))
    tr, te = train_test_split(i, test_size=test_fraction, random_state=seed, stratify=y)
    tr, va = train_test_split(tr, test_size=val_fraction, random_state=seed, stratify=y[tr])
    idx = dict(train=np.sort(tr), val=np.sort(va), test=np.sort(te))
    X, L = z["X"], z["length"]
    return StreamSplit(X={k: X[v] for k, v in idx.items()}, length={k: L[v] for k, v in idx.items()},
                       y={k: y[v] for k, v in idx.items()}, idx=idx, class_names=list(z["class_names"]),
                       features=list(z["features"]))


def state_after(X: np.ndarray, length: np.ndarray, k) -> np.ndarray:
    """Feature vector after packet k (1-based; 'last' = the flow's last kept packet)."""
    if k == "last":
        return X[np.arange(len(X)), length - 1]
    return X[:, k - 1]
