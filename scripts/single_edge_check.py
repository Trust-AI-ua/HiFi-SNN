#!/usr/bin/env python3
"""Do the tree and the entropy cutter pick the same cut on single-threshold features?

For Edge-IIoTset and TON-IoT (seed 42) the teacher is refitted exactly as in
production (default thread environment), the tree and entropy encoders are built,
and for every kept feature whose tree edge count is 1, the two cuts are compared as
partitions of the feature's distinct training values. The encoder bins with
searchsorted(side="right"), so a value is above a cut iff x >= cut; two cuts are
equivalent iff (u >= a) == (u >= c) for every distinct value u.

Supports the Section V-B reading that on these datasets most features carry a single
threshold that either method finds. Seed 42 only: a spot check, not a five-seed claim.

    python scripts/single_edge_check.py
Output: results/diagnostics/single_edge_partition.csv
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/diagnostics/single_edge_partition.csv"


def main():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(v):
            sys.exit(f"{v} is set; the teacher fit depends on it (unset it to match production)")
    from src.data import get_dataset, DATASET_LOADERS
    from src.training_treespike import _fit_teacher, _arrays_from_bundle
    from src.models.tree_to_spike import TreeGuidedSpikeEncoder
    rows = []
    for ds in ("edgeiiot", "toniot"):
        cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
        lp = inspect.signature(DATASET_LOADERS[ds]).parameters
        kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=42)
        for o in ("subsample_fraction", "test_fraction", "label_mode"):
            if o in cfg and o in lp:
                kw[o] = cfg[o]
        b = get_dataset(ds, **kw)
        Xtr, ytr, _, _, _, _ = _arrays_from_bundle(b)
        teacher, _, _ = _fit_teacher(Xtr, ytr, b.num_classes, {**cfg, "seed": 42})
        mk = lambda m: TreeGuidedSpikeEncoder(
            T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"], jitter=cfg["jitter"],
            encoder_mode=m, cost_power=float(cfg.get("teacher_cost_power", 1.0))).fit(teacher, Xtr.shape[1], X=Xtr, y=ytr)
        tr, en = mk("tree"), mk("entropy")
        for j in sorted({j for j, _ in tr.channels_}):
            if len(tr.edges_[j]) != 1:
                continue
            u = np.unique(Xtr[:, j].astype(np.float32))
            a, c = float(tr.edges_[j][0]), float(en.edges_[j][0])
            rows.append(dict(dataset=ds, seed=42, feature=int(j), n_distinct=len(u), tree_cut=a, entropy_cut=c,
                             same_partition=bool(np.array_equal(u >= a, u >= c))))
    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(d.groupby("dataset").agg(single=("feature", "size"), same=("same_partition", "sum"),
                                   binary=("n_distinct", lambda s: int((s == 2).sum()))).to_string())


if __name__ == "__main__":
    main()
