#!/usr/bin/env python3
"""Run 4: GBDT compression Pareto -- can any tree ensemble reach a 256 KB envelope?

Applies to the tree the same standard the Limitations section demands of the dense
baseline: quantized AND pruned, not merely capped. Four arms per dataset:

  n_estimators x max_depth grid   (structural capacity)
  int8 thresholds                 (feature-comparison precision)
  16-bit / 8-bit leaf values      (leaf arrays dominate multiclass ensembles:
                                   with K=15 on Edge-IIoTset leaves can exceed nodes)
  gain-pruned arm                 (drop trees by cumulative gain contribution --
                                   the tree-side equivalent of magnitude pruning)

Reported PER DATASET, never only as a mean: the 256 KB question can resolve
differently for CTU-13 (2 classes, ~37 features) than Edge-IIoTset (15 classes).

Deployment byte model, stated explicitly:
  internal node = 2 B feature index + 1 B int8 threshold + 4 B two uint16 children
  leaf          = K x leaf_bytes
Native XGBoost serialized size is also recorded for reference; it is larger because
it stores fp32 thresholds and bookkeeping a deployed kernel does not need.

PRE-COMMITTED REPORTING (fixed before results are seen):
  no config <=256 KB within 1.0 macro-F1 point of the full teacher
      -> the memory-envelope claim holds and is not an artifact of leaving the
         tree uncompressed
  some config fits
      -> the envelope claim falls, the deployment argument becomes substrate-only,
         and the fitting configuration is named in Limitations
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
OUT = REPO/"results"/"deployment"; OUT.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

DATASETS = ["kddcup99","nslkdd","ctu13","iscxids2012","cicids2017","cse-cic-ids2018",
            "unswnb15","ciciot2023","ciciomt2024","edgeiiot","toniot"]
GRID_EST, GRID_DEPTH, LEAF_BITS = [25,50,100,200], [3,4,6], [16,8]
BUDGET = 256*1024

def deploy_bytes(booster, n_classes, leaf_bits):
    df = booster.trees_to_dataframe()
    n_leaf = int((df.Feature == "Leaf").sum()); n_int = int(len(df) - n_leaf)
    return n_int*(2+1+4) + n_leaf*max(n_classes,1)*(leaf_bits//8)

def fit_eval(Xtr, ytr, Xte, yte, n_cls, est, depth, seed, gain_prune=None):
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score
    present = np.unique(ytr); remap = np.full(n_cls, -1, np.int64)
    remap[present] = np.arange(len(present))
    cnt = np.bincount(remap[ytr], minlength=len(present)).astype(np.float32)
    w = (cnt.sum()/(cnt+1e-6))[remap[ytr]]
    Xd = pd.DataFrame(Xtr, columns=[f"f{i}" for i in range(Xtr.shape[1])])
    m = XGBClassifier(n_estimators=est, max_depth=depth, learning_rate=0.1,
                      subsample=0.9, colsample_bytree=0.8, importance_type="gain",
                      tree_method="hist", n_jobs=8, eval_metric="mlogloss",
                      random_state=seed)
    m.fit(Xd, remap[ytr], sample_weight=w)
    if gain_prune is not None:                     # keep trees covering `gain_prune` of total gain
        df = m.get_booster().trees_to_dataframe()
        g = df[df.Feature != "Leaf"].groupby("Tree").Gain.sum().sort_values(ascending=False)
        keep = int(max(1, (g.cumsum()/g.sum() <= gain_prune).sum()))
        est = keep
        m = XGBClassifier(n_estimators=keep, max_depth=depth, learning_rate=0.1,
                          subsample=0.9, colsample_bytree=0.8, importance_type="gain",
                          tree_method="hist", n_jobs=8, eval_metric="mlogloss",
                          random_state=seed)
        m.fit(Xd, remap[ytr], sample_weight=w)
    Xe = pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])])
    pred = present[m.predict(Xe)]
    return f1_score(yte, pred, average="macro")*100, m.get_booster(), est

def _flush(rows):
    """Append-merge to disk keyed on the config identity.

    The previous version overwrote the CSV, so invoking the script for a single
    dataset destroyed the other ten. Merging on the key also makes re-runs
    idempotent and lets a corrected arm supersede an earlier one.
    """
    import os, subprocess, datetime
    new = pd.DataFrame(rows)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=REPO, text=True).strip()
    except Exception:
        sha = "unknown"
    new["run_id"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + sha
    f = OUT / "gbdt_compression.csv"
    if f.exists():
        old = pd.read_csv(f)
        new = pd.concat([old, new], ignore_index=True)
    key = ["dataset", "n_classes", "n_estimators", "max_depth", "gain_prune", "leaf_bits"]
    # gain_prune mixes "none" with numbers, so a re-read gives strings where this
    # process has floats; without normalising, 0.9 != "0.9" and stale rows survive.
    new["gain_prune"] = new["gain_prune"].astype(str)
    new = new.drop_duplicates(subset=key, keep="last")
    new.to_csv(f, index=False)
    return new


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="+", default=DATASETS); a = ap.parse_args()
    from src.data import get_dataset
    from src.training_treespike import _arrays_from_bundle
    rows = []
    for ds in a.datasets:
        try:
            # The per-dataset config's label_mode MUST be honoured. Omitting it made
            # Edge-IIoTset fall back to the loader default (category, K=6) while
            # Tree-to-Spike is evaluated on full (K=15) -- different tasks, and the
            # deployment byte model scales with K through the leaf arrays.
            import yaml as _yaml, inspect as _inspect
            from src.data import DATASET_LOADERS as _DL
            _cfg = _yaml.safe_load(open(REPO/f"experiments/Tree-to-Spike/{ds}/config.yaml"))
            _lp = _inspect.signature(_DL[ds]).parameters
            _dk = dict(seed=a.seed)
            for _o in ("subsample_fraction", "test_fraction", "label_mode"):
                if _o in _cfg and _o in _lp:
                    _dk[_o] = _cfg[_o]
            b = get_dataset(ds, **_dk)
            Xtr, ytr, _, _, Xte, yte = _arrays_from_bundle(b); K = b.num_classes
        except Exception as e:
            print(f"[skip] {ds}: {e}", flush=True); continue
        for est in GRID_EST:
            for dep in GRID_DEPTH:
                for gp in (None, 0.90):
                    try:
                        f1, boo, eff = fit_eval(Xtr, ytr, Xte, yte, K, est, dep, a.seed, gp)
                    except Exception as e:
                        print(f"[fail] {ds} {est}x{dep} gp={gp}: {e}", flush=True); continue
                    for lb in LEAF_BITS:
                        rows.append(dict(dataset=ds, n_classes=K, n_estimators=est,
                            effective_trees=eff, max_depth=dep,
                            gain_prune=("none" if gp is None else gp), leaf_bits=lb,
                            macro_f1=round(f1,3), deploy_bytes=deploy_bytes(boo,K,lb)))
                    r = rows[-1]
                    print(f"  {ds:16s} est={est:3d} d={dep} gp={r['gain_prune']:>4} "
                          f"F1={f1:6.2f} bytes={r['deploy_bytes']:8d} "
                          f"({'FITS' if r['deploy_bytes']<=BUDGET else 'over'} 256KB)", flush=True)
        _flush(rows)
    df = _flush(rows)
    print("\n=== per-dataset verdict: smallest config within 1.0 pt of that dataset's best ===")
    for ds, g in df.groupby("dataset"):
        best = g.macro_f1.max(); ok = g[g.macro_f1 >= best-1.0]
        w = ok.loc[ok.deploy_bytes.idxmin()]
        print(f"  {ds:16s} best={best:6.2f} | smallest within 1pt: {w.deploy_bytes/1024:8.1f} KB "
              f"(F1 {w.macro_f1:6.2f}, {int(w.effective_trees)}x{int(w.max_depth)}, "
              f"{int(w.leaf_bits)}b leaves) -> {'FITS 256KB' if w.deploy_bytes<=BUDGET else 'OVER'}")
