#!/usr/bin/env python3
"""Measure the frozen tree-guided encoder's deployment storage per dataset.

The deployed Tree-to-Spike artifact is NOT the student weights alone: the
encoder's split-threshold tables must ship with it, because inference does a
searchsorted against them for every flow (src/models/tree_to_spike.py::_levels).
The DNN-Tree twin consumes the SAME encoder (encode_dense), so it carries the
identical table cost; the Vanilla SNN (closed-form latency code) and TASNN
(internal encoding) carry none.

Storage accounted here (seed 42, the architecture table's reference seed):
  thresholds : sum_j |E_j| over kept features, 4 bytes each (fp32)
  channel map: C entries of (feature index 2 B + latency offset 1 B)

-> results/Journal_results/encoder_storage.json  {dataset: {...}}
Run:  python scripts/measure_encoder_storage.py
"""
from __future__ import annotations
import inspect, json, sys, time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import get_dataset, DATASET_LOADERS
from src.models import TreeGuidedSpikeEncoder
from src.training_treespike import _fit_teacher, _arrays_from_bundle

DATASETS = ["kddcup99", "nslkdd", "ctu13", "iscxids2012", "cicids2017",
            "cse-cic-ids2018", "unswnb15", "ciciot2023", "ciciomt2024",
            "edgeiiot", "toniot"]
SEED = 42
B_THRESH, B_FEATIDX, B_OFFSET = 4, 2, 1
OUT = REPO / "results" / "Journal_results" / "encoder_storage.json"


def measure(ds: str) -> dict:
    tcfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    dk = dict(seed=SEED)
    for o in ("subsample_fraction", "test_fraction", "label_mode"):
        if o in tcfg and o in lp:
            dk[o] = tcfg[o]
    bundle = get_dataset(ds, **dk)
    Xtr, ytr, _, _, _, _ = _arrays_from_bundle(bundle)
    cfg = dict(tcfg); cfg["seed"] = SEED
    teacher, _, _ = _fit_teacher(Xtr, ytr, bundle.num_classes, cfg)
    enc = TreeGuidedSpikeEncoder(
        T=tcfg["T"], pop_max=tcfg["pop_max"], prune_quantile=tcfg["prune_quantile"],
        jitter=tcfg["jitter"], encoder_mode=tcfg.get("encoder_mode", "tree"),
    ).fit(teacher, Xtr.shape[1], X=Xtr)
    kept = sorted({j for j, _ in enc.channels_})
    n_edges = sum(len(enc.edges_[j]) for j in kept)
    n_ch = len(enc.channels_)
    return {"kept_features": len(kept), "n_thresholds": int(n_edges), "channels": n_ch,
            "bytes": int(n_edges * B_THRESH + n_ch * (B_FEATIDX + B_OFFSET))}


def main():
    out = json.load(open(OUT)) if OUT.exists() else {}
    t0 = time.time()
    for ds in DATASETS:
        if ds in out:
            print(f"[{ds}] skip (cached)"); continue
        m = measure(ds)
        out[ds] = m
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"[{ds}] thresholds={m['n_thresholds']:,} channels={m['channels']} "
              f"-> {m['bytes']/1024:.1f} KB", flush=True)
    print(f"\n[wrote] {OUT}  ({len(out)} datasets, {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
