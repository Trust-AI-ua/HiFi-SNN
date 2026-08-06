"""CTU-13 three-condition protocols as first-class datasets.

Registers the conditions the RandomForest diagnostic established, so every trainer
consumes them unchanged:

  legacy_random     global per-scenario aggregates + random flow split
  causal_random     partition-local causal aggregates + random flow split
  causal_scenario   causal aggregates + scenario-disjoint split (frozen manifest)

Two design points that are easy to get wrong:

1. `causal_scenario` legitimately uses the precomputed `causal__*` columns. Those
   are built from strictly earlier flows *within a scenario*, and a whole test
   scenario lies entirely inside the test partition, so no value crosses the
   boundary. `causal_random` cannot use them -- a random split cuts across
   scenarios -- so it recomputes aggregates inside each partition.

2. The random partition is keyed on the FOLD, never on the optimisation seed.
   Folds are the statistical unit; seeds measure optimisation variability within a
   fold. This mirrors the TON-IoT leakage protocol so the two are comparable.

Duplicates are NOT removed before partitioning. That would change the frozen
partitions. Instead the test rows whose feature hash also occurs in training are
flagged, and `compute_metrics` reports a duplicate-free sensitivity block
alongside the full result. The overlap is reported, never claimed to be zero.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.nslkdd import DatasetBundle
from src.data.ctu13_causal import HOST_COLS, METADATA_COLS, load_frozen_folds

REPO = Path(__file__).resolve().parents[2]
PQ = REPO / "dataset/ctu13/_three_condition.parquet"
FOLDS_JSON = REPO / "results/ctu13/scenario_folds.json"

BASE_COLS = ["Dur", "TotPkts", "TotBytes", "SrcBytes", "sTos", "dTos",
             "Sport", "Dport"]
LEG = [f"legacy__{c}" for c in HOST_COLS]
CAU = [f"causal__{c}" for c in HOST_COLS]
CONDITIONS = ("legacy_random", "causal_random", "causal_scenario")
FEATURE_SETS = ("base", "combined")


def _random_partition(n: int, fold: int) -> tuple:
    """Fold-keyed random split, identical to the 36-run diagnostic (rng 1000+fold)
    so the matrix and the diagnostic describe the same partition."""
    rng = np.random.default_rng(1000 + int(fold))
    perm = rng.permutation(n)
    nte, nva = int(0.24 * n), int(0.08 * n)
    return perm[nte + nva:], perm[nte:nte + nva], perm[:nte]


def load_ctu13_condition(condition: str, fold: int, feature_set: str = "combined",
                         batch_size: int = 128, seed: int = 42,
                         num_workers: int = 0, **_ignored) -> DatasetBundle:
    from src.data.leakage import encode_train_only
    from src.metrics import set_eval_context

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set {feature_set!r}; expected {FEATURE_SETS}")

    d = pd.read_parquet(PQ)
    fo = load_frozen_folds(FOLDS_JSON, d)      # verifies sizes, never re-solves
    f = fo["folds"][int(fold)]
    n = len(d)
    y = d["y"].to_numpy(np.int64)

    if condition == "causal_scenario":
        tr = np.where(d.scenario.isin(f["train_scenarios"]))[0]
        va = np.where(d.scenario.isin(f["val_scenarios"]))[0]
        te = np.where(d.scenario.isin(f["test_scenarios"]))[0]
        host_frame = d[CAU]
    else:
        tr, va, te = _random_partition(n, fold)
        if condition == "legacy_random":
            host_frame = d[LEG]
        else:
            from scripts.ctu13_stage1 import _partition_local_causal
            host_frame = _partition_local_causal(d, (tr, va, te), CAU)

    cols_note = "base" if feature_set == "base" else "base + host-context"
    feat = d[BASE_COLS] if feature_set == "base" else pd.concat(
        [d[BASE_COLS], host_frame.set_axis(CAU, axis=1)], axis=1)
    feat = feat.reset_index(drop=True)
    assert set(METADATA_COLS).isdisjoint(feat.columns), "metadata reached the features"

    # Duplicate-free sensitivity mask: test rows whose raw feature vector also
    # occurs in training. Hash-based, so it is an upper bound on true duplicates.
    h = pd.util.hash_pandas_object(feat, index=False).to_numpy()
    dupfree = ~np.asarray(pd.Index(h[te]).isin(pd.Index(h[tr])), dtype=bool)

    Xtr, Xva, Xte, meta = encode_train_only(feat, tr, va, te)

    set_eval_context(test_groups=d["scenario"].to_numpy()[te],
                     test_dupfree_mask=dupfree,
                     group_name="scenario")

    print(f"  [ctu13/{condition}/fold{fold}/{feature_set}] "
          f"rows tr={len(tr)} va={len(va)} te={len(te)} | features {meta['n_features']} "
          f"({cols_note}) | test scenarios {sorted(set(d.scenario.to_numpy()[te].tolist()))} "
          f"| dup-free excludes {int((~dupfree).sum())} "
          f"({100*(~dupfree).mean():.4f}%)", flush=True)

    def _loader(Xs, ys, shuffle):
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys.astype(np.int64)))
        g = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=g, drop_last=False)

    u, c = np.unique(y[tr], return_counts=True)
    classes = ["normal", "botnet"]
    return DatasetBundle(
        train_loader=_loader(Xtr, y[tr], True),
        val_loader=_loader(Xva, y[va], False),
        test_loader=_loader(Xte, y[te], False),
        test21_loader=None,
        num_features=Xtr.shape[1],
        num_classes=2,
        class_names=classes,
        class_counts_train={classes[int(k)]: int(v) for k, v in zip(u, c)},
    )
