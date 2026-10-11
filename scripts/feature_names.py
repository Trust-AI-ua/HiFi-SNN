#!/usr/bin/env python3
"""Recover feature names for the loaders' preprocessed arrays (audit helper).

The loaders return bare arrays. For loaders that one-hot categoricals, the
column order is recovered by *replaying the loader's own code* with its disk
cache disabled and cache writes blocked, recording the columns of every
DataFrame.to_numpy() call and picking the call (or consecutive pair) whose width
matches the array. Header-only datasets (no one-hot) use the CSV header.
CTU-13 is not handled (its 37 engineered features raise no audit flags).

    python scripts/feature_names.py            # writes results/diagnostics/leakage_audit/feature_names.csv
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
D = REPO / "dataset"
OUT = REPO / "results/diagnostics/leakage_audit/feature_names.csv"


def _header(path, drop):
    return [c.strip() for c in pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
            if c.strip() not in drop]


def header_names(ds):
    from src.data import cicids2017, cseids2018
    if ds == "cicids2017":
        return _header(D / "cicids2017" / cicids2017.CIC_FILES[0], {"Label"})
    if ds == "ciciot2023":
        return _header(sorted(glob.glob(str(D / "cic-iot2023/csv/*.csv")))[0], {"label"})
    if ds == "ciciomt2024":
        return _header(sorted(glob.glob(str(D / "cic-iomt-2024/csv/train/*.csv")))[0], set())
    if ds == "cse-cic-ids2018":
        cols = []
        for f in cseids2018.CSE_FILES:
            cols += [c for c in _header(D / "cse-cic-ids2018" / f, set()) if c not in cols]
        return [c for c in cols if c not in ("Label", "Timestamp")]
    raise KeyError(ds)


def replay_names(ds, width):
    """Run the loader's array builder with cache disabled; record to_numpy columns."""
    import importlib
    mod = {"edgeiiot": "edgeiiot", "toniot": "toniot", "iscxids2012": "iscxids2012",
           "unswnb15": "unswnb15", "nslkdd": "nslkdd", "kddcup99": "kddcup99"}[ds]
    m = importlib.import_module(f"src.data.{mod}")
    calls = []
    orig_to_numpy = pd.DataFrame.to_numpy
    orig_exists = Path.exists
    orig_savez = np.savez_compressed

    def rec(self, *a, **k):
        calls.append(list(map(str, self.columns)))
        return orig_to_numpy(self, *a, **k)

    pd.DataFrame.to_numpy = rec
    Path.exists = lambda p: False if p.name == "_preprocessed_cache.npz" else orig_exists(p)
    np.savez_compressed = lambda *a, **k: None          # never overwrite the real caches
    try:
        if hasattr(m, "_build_full_arrays"):
            dd = {"edgeiiot": "edge-iiotset", "toniot": "ton-iot", "iscxids2012": "iscxids2012"}[ds]
            m._build_full_arrays(D / dd)
        else:
            loader = {"unswnb15": "load_unswnb15", "nslkdd": "load_nslkdd",
                      "kddcup99": "load_kddcup99"}[ds]
            getattr(m, loader)()
    finally:
        pd.DataFrame.to_numpy = orig_to_numpy
        Path.exists = orig_exists
        np.savez_compressed = orig_savez
    for c in calls:
        if len(c) == width:
            return c
    for a, b in zip(calls, calls[1:]):
        if len(a) + len(b) == width:
            return a + b
    raise RuntimeError(f"{ds}: no recorded frame of width {width}: {[len(c) for c in calls]}")


def main():
    feats = pd.read_csv(REPO / "results/diagnostics/leakage_audit/features.csv")
    width = feats.groupby("dataset").n_features.first().to_dict()
    rows = []
    for ds, w in width.items():
        if ds == "ctu13":
            continue
        try:
            names = header_names(ds)
        except KeyError:
            names = replay_names(ds, w)
        if len(names) != w:
            raise RuntimeError(f"{ds}: {len(names)} names for {w} features")
        rows += [dict(dataset=ds, feature=j, name=n) for j, n in enumerate(names)]
        print(f"{ds}: {w} names", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
