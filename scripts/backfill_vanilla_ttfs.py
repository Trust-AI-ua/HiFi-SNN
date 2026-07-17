#!/usr/bin/env python3
"""Backfill the Vanilla SNN's mean encoder time-to-first-spike (TTFS).

The vanilla pipeline latency-encodes the raw preprocessed features:
t = floor((1 - x) * (T - 1)), and inputs <= threshold (0.01) never spike
(src/encoding/latency.py). Mean TTFS is therefore a deterministic property of
the test features; this script rebuilds each run's deterministic split and
computes the mean spike time over SPIKING feature-channels x test samples
(sub-threshold silent channels excluded), writing idempotently into

    test_plus["spike_stats"]["mean_spike_time"]   # steps, of ttfs_T
    test_plus["spike_stats"]["ttfs_T"]            # num_steps (25)

Usage:  python scripts/backfill_vanilla_ttfs.py [--only nslkdd ...] [--force]
"""
from __future__ import annotations

import argparse
import glob
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import get_dataset, DATASET_LOADERS

DATASETS = ["kddcup99", "nslkdd", "ctu13", "iscxids2012", "cicids2017",
            "cse-cic-ids2018", "unswnb15", "ciciot2023", "ciciomt2024",
            "edgeiiot", "toniot"]
THRESHOLD = 0.01   # matches latency_encode's default


def _write_json_atomic(path, obj):
    """Crash-safe in-place update: write to a sibling tmp file, then rename."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)



def _loader_kwargs(ds, cfg, seed):
    """Loader kwargs exactly as src/training.py used them (per-run config.yaml)."""
    cfg_path = REPO / "configs" / ds / "SNN_LeakyParallel" / "latency" / "config.yaml"
    file_cfg = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {}
    params = inspect.signature(DATASET_LOADERS[ds]).parameters
    kw = dict(seed=seed)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in (file_cfg or {}) and opt in params:
            kw[opt] = file_cfg[opt]
    return kw


def backfill_dataset(ds: str, force: bool):
    rjs = sorted(glob.glob(str(REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_*/results.json")))
    for rj in rjs:
        r = json.load(open(rj))
        if (not force) and "mean_spike_time" in r["test_plus"].get("spike_stats", {}):
            print(f"[{ds}] {Path(rj).parent.name}: skip (present)"); continue
        cfg = r["config"]; seed = cfg["seed"]; T = cfg["num_steps"]
        t0 = time.time()
        bundle = get_dataset(ds, **_loader_kwargs(ds, cfg, seed))
        Xte = bundle.test_loader.dataset.tensors[0].numpy()
        Xc = np.clip(Xte, 0.0, 1.0)
        times = np.floor((1.0 - Xc) * (T - 1))
        fire = Xc > THRESHOLD                     # silent channels excluded
        tt = float(times[fire].mean()) if fire.any() else float("nan")
        r["test_plus"].setdefault("spike_stats", {})["mean_spike_time"] = tt
        r["test_plus"]["spike_stats"]["ttfs_T"] = int(T)
        _write_json_atomic(rj, r)
        print(f"[{ds}] seed_{seed}: mean TTFS = {tt:.2f}/{T} "
              f"(firing frac {fire.mean():.2f})  [{time.time()-t0:.0f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=DATASETS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for ds in args.only:
        backfill_dataset(ds, args.force)
    print(f"\nall done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
