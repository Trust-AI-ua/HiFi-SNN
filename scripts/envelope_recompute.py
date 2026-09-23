#!/usr/bin/env python3
"""Reprice the compression sweep under the verified packed format.

The sweep priced internal nodes at 2 B feature + 1 B int8 threshold + 4 B children.
One-byte thresholds were then measured and rejected: they cost 16.9 macro-F1 on
UNSW-NB15 and 46.4 on CICIoT2023 (scripts/verify_serialization.py), so the packed
format is 2 B feature + 4 B float32 threshold + 4 B children, leaves at 16 bits,
which reproduces the native model.

Rather than re-running the 27 h sweep, the internal-node count is recovered from
the stored byte count, which is exact for binary trees (leaves = internals +
trees):

    bytes_old = 7 I + per_leaf (I + T) (leaf_bits / 8)
 => I        = (bytes_old - per_leaf T leaf_bits/8) / (7 + per_leaf leaf_bits/8)
    bytes_new = 10 I + per_leaf (I + T) (leaf_bits / 8)

``--spot-check N`` refits N configurations and compares the recovered node counts
with the real ones, so the derivation is verified rather than assumed.

    python scripts/envelope_recompute.py [--spot-check 5]
Output: results/deployment/gbdt_compression_repriced.csv
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SRC = REPO / "results/deployment/gbdt_compression.csv"
OUT = REPO / "results/deployment/gbdt_compression_repriced.csv"
BYTES_PER_INTERNAL_OLD = 2 + 1 + 4
BYTES_PER_INTERNAL_NEW = 2 + 4 + 4


def per_leaf(n_classes):
    """Leaf values per leaf AS THE SWEEP PRICED THEM: max(K, 1), binary included.

    The inversion has to reproduce the sweep's own accounting, right or wrong,
    or the node counts it recovers are not the sweep's. Whether pricing K values
    at every leaf of a per-class ensemble is correct is a separate question,
    settled by ``leaf_pricing`` below.
    """
    return np.maximum(n_classes, 1)


def recover_internals(df):
    """Recover internal-node counts, inferring how many classes the teacher fit.

    The sweep priced leaves at the DECLARED class count but fit only the classes
    present in the training split, so on CSE-CIC-IDS2018 (five declared, four
    present) there are ``est * 4`` trees while the byte formula multiplies by 5.
    The present count is not recorded, so it is recovered by the one property it
    controls: the node count must come out an integer. It is a property of the
    DATASET, not of a row, so a candidate is accepted only if it works for every
    row of that dataset, and the largest such candidate is taken (a smaller one
    can divide evenly by coincidence). Binary boosters have one tree per round.
    """
    I = pd.Series(np.nan, index=df.index)
    trees = pd.Series(np.nan, index=df.index)
    kp_by_ds, resid = {}, 0.0
    lb = df.leaf_bits // 8
    pl = per_leaf(df.n_classes)
    for ds, g in df.groupby("dataset"):
        K = int(g.n_classes.max())
        chosen = None
        for kp in range(K, 0, -1):
            t = g.effective_trees * (kp if K > 2 else 1)
            Ig = (g.deploy_bytes - pl[g.index] * t * lb[g.index]) / \
                 (BYTES_PER_INTERNAL_OLD + pl[g.index] * lb[g.index])
            if ((Ig - Ig.round()).abs() <= 1e-6).all() and (Ig > 0).all():
                chosen = (kp, t, Ig)
                break
        if chosen is None:
            raise SystemExit(f"{ds}: no class count makes the node counts integral")
        kp, t, Ig = chosen
        kp_by_ds[ds] = kp
        I.loc[g.index], trees.loc[g.index] = Ig.round(), t
        resid = max(resid, float((Ig - Ig.round()).abs().max()))
    print("classes the teacher actually fit, inferred from the byte counts:")
    for ds, kp in sorted(kp_by_ds.items()):
        print(f"  {ds:16s} declared {int(df[df.dataset == ds].n_classes.max()):2d}  fit {kp:2d}")
    return I, trees, resid


def spot_check(df, n):
    """Refit N configurations THROUGH THE SWEEP'S OWN fit_eval and compare counts.

    An earlier version of this check refit with a local XGBClassifier and matched
    0 of 5: the sweep fits with class-balanced ``sample_weight`` and loads the
    bundle with its own keyword set, and both change the tree structure. Calling
    the sweep's function removes the chance of that drift recurring.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import inspect as _inspect
    from gbdt_compression import fit_eval
    from src.data import get_dataset, DATASET_LOADERS
    from src.training_treespike import _arrays_from_bundle
    rows = []
    sample = df.sample(min(n, len(df)), random_state=0)
    cse = df[df.dataset == "cse-cic-ids2018"]
    if len(cse) and "cse-cic-ids2018" not in set(sample.dataset):
        sample = pd.concat([cse.sample(1, random_state=0), sample.iloc[:-1]])
    cache = {}
    for r in sample.itertuples():
        if r.dataset not in cache:
            cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{r.dataset}/config.yaml"))
            params = _inspect.signature(DATASET_LOADERS[r.dataset]).parameters
            kw = dict(seed=42)                      # the sweep's own keyword set
            for opt in ("subsample_fraction", "test_fraction", "label_mode"):
                if opt in cfg and opt in params:
                    kw[opt] = cfg[opt]
            b = get_dataset(r.dataset, **kw)
            cache[r.dataset] = (*_arrays_from_bundle(b)[:2], b)
        Xtr, ytr, b = cache[r.dataset]
        _, boo, eff = fit_eval(Xtr, ytr, Xtr[:1], ytr[:1], b.num_classes,
                               int(r.n_estimators), int(r.max_depth), 42, None)
        fr = boo.trees_to_dataframe()
        real_I = int((fr.Feature != "Leaf").sum())
        rows.append(dict(dataset=r.dataset, n_estimators=r.n_estimators, max_depth=r.max_depth,
                         gain_prune=r.gain_prune, leaf_bits=r.leaf_bits,
                         recovered_internals=int(r.internals_recovered), real_internals=real_I,
                         real_leaves=int((fr.Feature == "Leaf").sum()),
                         recovered_trees=int(r.trees_total), real_trees=int(fr.Tree.nunique()),
                         match=bool(abs(int(r.internals_recovered) - real_I) <= 1)))
        print(f"  {r.dataset:16s} est={int(r.n_estimators):3d} d={int(r.max_depth)} "
              f"recovered={int(r.internals_recovered):7d} real={real_I:7d} "
              f"{'OK' if rows[-1]['match'] else 'MISMATCH'}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot-check", type=int, default=5)
    a = ap.parse_args()
    d = pd.read_csv(SRC)
    d["gain_prune"] = d["gain_prune"].astype(str)
    d = d[~((d.dataset == "edgeiiot") & (d.n_classes == 15))].copy()   # pre-repair task
    I, trees, err = recover_internals(d)
    d["internals_recovered"] = I.astype(int)
    d["trees_total"] = np.asarray(trees).astype(int)
    lb = d.leaf_bits // 8
    pl = per_leaf(d.n_classes)
    leaves = d.internals_recovered + d.trees_total
    # The sweep's own price, with only the threshold field corrected, kept so the
    # two corrections can be read separately.
    d["deploy_bytes_fp32_thresholds_sweep_leaves"] = (
        BYTES_PER_INTERNAL_NEW * d.internals_recovered + pl * leaves * lb).astype(int)
    # The verified price: one value per leaf (see src/serialize.pack_booster).
    d["deploy_bytes_packed_fp32_thresholds"] = (
        BYTES_PER_INTERNAL_NEW * d.internals_recovered + leaves * lb).astype(int)
    d["byte_ratio_new_over_old"] = (d.deploy_bytes_packed_fp32_thresholds / d.deploy_bytes)
    print(f"max non-integrality in the recovered node counts: {err:.3f}")
    print(f"size ratio: mean {d.byte_ratio_new_over_old.mean():.3f} "
          f"min {d.byte_ratio_new_over_old.min():.3f} max {d.byte_ratio_new_over_old.max():.3f}")
    for B, lab in ((128 * 1024, "128 KB"), (256 * 1024, "256 KB")):
        old = d.groupby("dataset").apply(lambda g: int((g.deploy_bytes <= B).sum()),
                                         include_groups=False)
        new = d.groupby("dataset").apply(
            lambda g: int((g.deploy_bytes_packed_fp32_thresholds <= B).sum()),
            include_groups=False)
        print(f"\nconfigurations fitting {lab} (old 1-byte -> verified format):")
        print(pd.DataFrame({"old": old, "packed_fp32": new}).to_string())
    d.to_csv(OUT, index=False)
    print(f"\n[wrote] {OUT}")
    if a.spot_check:
        print(f"\nspot-checking {a.spot_check} configurations against refitted teachers:")
        sc = spot_check(d[d.gain_prune == "none"], a.spot_check)
        sc.to_csv(REPO / "results/deployment/envelope_spotcheck.csv", index=False)
        print(f"[wrote] envelope_spotcheck.csv  matches: {int(sc.match.sum())}/{len(sc)}")


if __name__ == "__main__":
    main()
