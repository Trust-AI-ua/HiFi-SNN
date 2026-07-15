#!/usr/bin/env python3
"""Backfill the encoder's mean time-to-first-spike (TTFS) for the encoder ablation.

The tree-guided encoder emits exactly ONE spike per channel, at
t_c = round((1 - level_c) * (T - 1)); the spike TIMING is therefore a property
of the (teacher-derived) encoder and the data, not of the trained SNN weights.
This script rebuilds, per (dataset, seed), the XGBoost teacher (once) and the
three encoder variants (tree / uniform / all), computes the mean spike time
over all channels x test samples, and writes it into the matching results.json:

    test_plus["spike_stats"]["mean_spike_time"]   # mean TTFS, in steps of T
    test_plus["spike_stats"]["ttfs_T"]            # the window T (20)

Idempotent (skips records that already have mean_spike_time unless --force).

Usage:  python scripts/backfill_encoder_ttfs.py [--only nslkdd ...] [--force]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import get_dataset, DATASET_LOADERS
from src.models import TreeGuidedSpikeEncoder
from src.training_treespike import _fit_teacher, _arrays_from_bundle

DATASETS = ["kddcup99", "nslkdd", "ctu13", "iscxids2012", "cicids2017",
            "cse-cic-ids2018", "unswnb15", "ciciot2023", "ciciomt2024",
            "edgeiiot", "toniot"]
RJ = {  # arm -> results.json glob (relative, {ds}/{seed} filled in)
    "tree":     "results/Tree-to-Spike/{ds}/seed_{seed}/results.json",
    "quantile": "results/encoder_ablation/{ds}/quantile/seed_{seed}/results.json",
    "nokd":     "results/encoder_ablation/{ds}/nokd/seed_{seed}/results.json",
    "uniform":  "results/encoder_ablation/{ds}/uniform/seed_{seed}/results.json",
    "all":      "results/encoder_ablation/{ds}/all/seed_{seed}/results.json",
}
ENC_MODE = {"nokd": "tree"}   # nokd is a LOSS arm; its encoder is the tree encoder


def mean_ttfs(enc: TreeGuidedSpikeEncoder, X: np.ndarray, batch: int = 100_000):
    """Mean spike time over channels x samples (each channel fires exactly once)."""
    tot = 0.0
    n = 0
    for i in range(0, len(X), batch):
        lv = enc._level_channels(X[i:i + batch])              # [B, C] in [0, 1]
        st = np.round((1.0 - lv) * (enc.T - 1))               # spike times
        tot += float(st.sum())
        n += st.size
    return tot / max(n, 1)


def backfill_dataset(ds: str, force: bool):
    tcfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    lp = __import__("inspect").signature(DATASET_LOADERS[ds]).parameters
    seeds = sorted({Path(f).parent.name.split("_")[1]
                    for f in glob.glob(str(REPO / f"results/Tree-to-Spike/{ds}/seed_*/results.json"))})
    for seed in seeds:
        # which records still need it?
        todo = {}
        for mode, patt in RJ.items():
            p = REPO / patt.format(ds=ds, seed=seed)
            if not p.exists():
                continue
            r = json.load(open(p))
            if force or "mean_spike_time" not in r.get("test_plus", {}).get("spike_stats", {}):
                todo[mode] = (p, r)
        if not todo:
            print(f"[{ds}] seed_{seed}: skip (all present)"); continue

        t0 = time.time()
        dk = dict(seed=int(seed))
        for opt in ("subsample_fraction", "test_fraction", "label_mode"):
            if opt in tcfg and opt in lp:
                dk[opt] = tcfg[opt]
        bundle = get_dataset(ds, **dk)
        Xtr, ytr, _, _, Xte, _ = _arrays_from_bundle(bundle)
        fit_cfg = dict(tcfg); fit_cfg["seed"] = int(seed)
        teacher, _, _ = _fit_teacher(Xtr, ytr, bundle.num_classes, fit_cfg)

        for mode, (p, r) in todo.items():
            enc = TreeGuidedSpikeEncoder(
                T=tcfg["T"], pop_max=tcfg["pop_max"], prune_quantile=tcfg["prune_quantile"],
                jitter=tcfg["jitter"], encoder_mode=ENC_MODE.get(mode, mode)).fit(
                    teacher, Xtr.shape[1], X=Xtr)
            if enc.out_dim != r["data"]["encoder_channels"]:
                print(f"[{ds}] seed_{seed}/{mode}: encoder mismatch "
                      f"({enc.out_dim} vs {r['data']['encoder_channels']}) - SKIPPING")
                continue
            tt = mean_ttfs(enc, Xte)
            r["test_plus"].setdefault("spike_stats", {})["mean_spike_time"] = tt
            r["test_plus"]["spike_stats"]["ttfs_T"] = int(tcfg["T"])
            with open(p, "w") as f:
                json.dump(r, f, indent=2)
            print(f"[{ds}] seed_{seed}/{mode:8}: mean TTFS = {tt:.2f}/{tcfg['T']}", flush=True)
        print(f"[{ds}] seed_{seed}: done in {time.time()-t0:.0f}s", flush=True)


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
