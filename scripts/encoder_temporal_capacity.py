#!/usr/bin/env python3
"""Temporal-information capacity of the TTFS code at T (reviewer-requested analysis).

For each kept feature j the encoder maps a value to a bin level
    level = searchsorted(E_j, x) / |E_j|      ->  |E_j| + 1 reachable levels
and the latency code maps that level to a spike time
    t = round((1 - level) * (T - 1))          ->  at most T distinct times.

So whenever |E_j| + 1 > T, distinct split outcomes must share a spike time. This
quantifies exactly that:

  kept_features                  features retained by the encoder
  min/median/p90/max_edges       per-feature split-threshold counts |E_j|
  frac_features_T_sufficient     fraction with T >= |E_j| + 1 (no loss possible)
  level_collision_frac           1 - (distinct spike times) / (reachable levels)
  recoverable_split_outcome_frac (distinct spike times) / (reachable levels)

Inference only: refits the deterministic XGBoost teacher and the encoder. No model
is trained. Validates against the committed four-dataset artifact before extending.

    python scripts/encoder_temporal_capacity.py --validate
    python scripts/encoder_temporal_capacity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = ["nslkdd", "kddcup99", "unswnb15", "cicids2017", "cse-cic-ids2018",
            "iscxids2012", "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
PRIMARY = ["ciciomt2024", "ciciot2023", "edgeiiot", "toniot"]
REF = REPO / "results/Journal_results/encoder_edge_stats.json"
OUT = REPO / "results/Journal_results/new_run_results/encoder_temporal_capacity.json"


def stats_for(dataset: str, seed: int = 42) -> dict:
    from src.data import get_dataset, DATASET_LOADERS
    from src.models import TreeGuidedSpikeEncoder
    from src.training_treespike import (_fit_teacher, _arrays_from_bundle, _resolve,
                                        _set_seed)
    import yaml, inspect

    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{dataset}/config.yaml"))
    rp = REPO / f"results/Tree-to-Spike/{dataset}/seed_{seed}/results.json"
    if rp.exists():
        cfg.update(json.load(open(rp))["config"])
    cfg["dataset"], cfg["seed"] = dataset, seed
    cfg = _resolve(cfg)
    _set_seed(seed)

    params = inspect.signature(DATASET_LOADERS[dataset]).parameters
    kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=seed)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in params and cfg[opt] is not None:
            kw[opt] = cfg[opt]
    b = get_dataset(dataset, **kw)
    Xtr, ytr, *_ = _arrays_from_bundle(b)
    teacher, _, _ = _fit_teacher(Xtr, ytr, b.num_classes, cfg)
    enc = TreeGuidedSpikeEncoder(
        T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
        jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")
    ).fit(teacher, Xtr.shape[1], X=Xtr)

    T = int(cfg["T"])
    kept = sorted({j for j, _ in enc.channels_})
    counts, total_levels, total_distinct, n_suff = [], 0, 0, 0
    for j in kept:
        e = enc.edges_.get(j, np.array([]))
        n = int(len(e))
        counts.append(n)
        L = n + 1                                    # reachable bin levels
        if n == 0:
            distinct = 1
        else:
            lv = np.arange(L) / float(n)
            distinct = int(len(np.unique(np.round((1.0 - lv) * (T - 1)))))
        total_levels += L
        total_distinct += distinct
        if T >= L:
            n_suff += 1
    c = np.array(counts, dtype=float)
    return {
        "T": T, "kept_features": len(kept),
        "min_edges": int(c.min()) if len(c) else 0,
        "median_edges": float(np.median(c)) if len(c) else 0.0,
        "p90_edges": float(np.percentile(c, 90)) if len(c) else 0.0,
        "max_edges": int(c.max()) if len(c) else 0,
        "frac_features_T_sufficient": n_suff / max(len(kept), 1),
        "level_collision_frac": 1.0 - total_distinct / max(total_levels, 1),
        "recoverable_split_outcome_frac": total_distinct / max(total_levels, 1),
        "reachable_levels_total": int(total_levels),
        "distinct_spike_times_total": int(total_distinct),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="only recompute the 4 primary datasets and diff vs the artifact")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ref = json.load(open(REF)) if REF.exists() else {}
    targets = PRIMARY if args.validate else DATASETS
    out, worst = {}, 0.0
    for ds in targets:
        try:
            s = stats_for(ds, args.seed)
        except Exception as e:
            print(f"[FAIL] {ds}: {type(e).__name__}: {e}", flush=True)
            continue
        out[ds] = s
        tag = ""
        if ds in ref:
            keys = ["level_collision_frac", "frac_features_T_sufficient",
                    "recoverable_split_outcome_frac", "median_edges", "max_edges",
                    "kept_features"]
            d = max(abs(float(s[k]) - float(ref[ds][k])) for k in keys if k in ref[ds])
            worst = max(worst, d)
            tag = f"  [vs artifact: max abs diff {d:.2e} {'MATCH' if d < 1e-9 else 'DIFFERS'}]"
        print(f"{ds:16s} T={s['T']:2d} kept={s['kept_features']:3d} "
              f"edges med={s['median_edges']:6.1f} max={s['max_edges']:4d} | "
              f"T-sufficient {100*s['frac_features_T_sufficient']:5.1f}% | "
              f"collision {100*s['level_collision_frac']:5.1f}% | "
              f"recoverable {100*s['recoverable_split_outcome_frac']:5.1f}%{tag}", flush=True)

    if args.validate:
        print(f"\nvalidation worst abs diff vs committed artifact: {worst:.3e} "
              f"-> {'REPRODUCED' if worst < 1e-9 else 'NOT REPRODUCED'}")
    else:
        json.dump(out, open(OUT, "w"), indent=2)
        col = [v["level_collision_frac"] for v in out.values()]
        suf = [v["frac_features_T_sufficient"] for v in out.values()]
        print(f"\n{len(out)} datasets. collision {100*min(col):.1f}-{100*max(col):.1f}% "
              f"(median {100*np.median(col):.1f}%); "
              f"T-sufficient {100*min(suf):.1f}-{100*max(suf):.1f}%")
        prim = [out[d]["level_collision_frac"] for d in PRIMARY if d in out]
        print(f"primary-4 collision {100*min(prim):.1f}-{100*max(prim):.1f}%")
        print(f"wrote {OUT}")
