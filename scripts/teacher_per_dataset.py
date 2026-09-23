#!/usr/bin/env python3
"""Per-dataset teacher table: quality, footprint, latency, energy.

BACC was never persisted by the deployment measurement, so it is recomputed here.
Everything else is joined from artifacts already on disk rather than re-derived:

  quality    recomputed (seed 42, the deployment protocol) -- the compact model is
             evaluated as a boosting-round PREFIX of the full fit via
             iteration_range, which is exactly what validation selection chose, so
             one fit serves both arms and the two are guaranteed consistent.
  footprint  model_bytes         <- deployment_measurements.csv
  latency    p50_ms              <- deployment_measurements.csv
  energy     tree_comparisons x 0.9 pJ (Horowitz accumulate class; a tree
             comparison is a subtract-and-branch), and the ratio against
             t2s_exit's SOP-priced arithmetic energy on the SAME dataset.

Emits results/tab_teacher_per_dataset.{tex,csv}.
"""
from __future__ import annotations
import inspect, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
E_ADD = 0.9
SEED = 42
NAME = {"kddcup99":"KDDCup99","nslkdd":"NSL-KDD","ctu13":"CTU-13",
        "iscxids2012":"ISCXIDS2012","cicids2017":"CIC-IDS2017",
        "cse-cic-ids2018":"CSE-CIC-IDS2018","unswnb15":"UNSW-NB15",
        "ciciot2023":"CICIoT2023","ciciomt2024":"CICIoMT2024",
        "edgeiiot":"Edge-IIoTset","toniot":"TON-IoT"}

def evaluate(ds, sel_rounds, seed=SEED):
    from src.data import get_dataset, DATASET_LOADERS
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    from src.training_treespike import _arrays_from_bundle, _fit_teacher
    cfg = yaml.safe_load(open(REPO/f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    dk = dict(seed=seed)
    for o in ("subsample_fraction","test_fraction","label_mode"):
        if o in cfg and o in lp: dk[o] = cfg[o]
    b = get_dataset(ds, **dk)
    Xtr, ytr, _, _, Xte, yte = _arrays_from_bundle(b)
    teacher, _, present = _fit_teacher(Xtr, ytr, b.num_classes, {**cfg, "seed": seed})
    Xe = pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])])
    out = {}
    for arm, rng in (("full", None), ("compact", (0, int(sel_rounds)))):
        pred = present[teacher.predict(Xe, iteration_range=rng)] if rng \
               else present[teacher.predict(Xe)]
        m = compute_metrics(yte, pred, y_score=None, class_names=b.class_names)
        bm = compute_binary_ids_metrics(yte, pred, normal_class_index=0)
        out[arm] = dict(ACC=m["accuracy"]*100, BACC=m["balanced_accuracy"]*100,
                        F1M=m["f1_macro"]*100, MCC=m["matthews_corrcoef"]*100,
                        DR=bm["detection_rate"]*100, FAR=bm["false_alarm_rate"]*100)
    return out

if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    # Quality columns are five-seed mean +/- SD; footprint and latency stay at the
    # single seed-42 deployment run. Mixing the two aggregations in one table
    # without saying which applies per column group is what this fixes.
    _ap.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    _a = _ap.parse_args()
    meas = pd.read_csv(REPO/"results/deployment/deployment_measurements.csv", comment="#")
    meas = meas[meas.seed == SEED]
    comp = pd.read_csv(REPO/"results/deployment/compact_gbdt.csv")
    comp = comp[comp.seed == SEED].set_index("dataset")
    rows = []
    for ds in NAME:
        per_seed = {}
        for _s in _a.seeds:
            try:
                per_seed[_s] = evaluate(ds, comp.loc[ds, "selected_rounds"], seed=_s)
            except Exception as e:
                print(f"[fail] {ds} seed {_s}: {e}", flush=True)
        if not per_seed:
            continue
        q, qsd = {}, {}
        for arm in ("full", "compact"):
            vals = [v[arm] for v in per_seed.values() if arm in v]
            q[arm] = {k: float(np.mean([x[k] for x in vals])) for k in vals[0]}
            qsd[arm] = {k + "_sd": float(np.std([x[k] for x in vals], ddof=1))
                        if len(vals) > 1 else 0.0 for k in vals[0]}
        sub = meas[meas.dataset == ds].set_index("system")
        t2s_e = float(sub.loc["t2s_exit", "sops"]) * E_ADD
        for arm, sysname in (("full", "full_gbdt"), ("compact", "compact_gbdt")):
            if sysname not in sub.index: continue
            r = sub.loc[sysname]
            e = float(r.tree_comparisons) * E_ADD
            rows.append(dict(dataset=ds, name=NAME[ds], arm=arm, n_seeds=len(per_seed),
                             **q[arm], **qsd[arm],
                             bytes=int(r.model_bytes), p50_ms=float(r.p50_ms),
                             comparisons=float(r.tree_comparisons),
                             E_pJ=e, rel_E_vs_t2s=e/t2s_e if t2s_e else np.nan,
                             n_trees=int(r.n_trees), n_nodes=int(r.n_nodes)))
            print(f"  {NAME[ds]:16s} {arm:8s} BACC={q[arm]['BACC']:6.2f} "
                  f"MCC={q[arm]['MCC']:6.2f} DR={q[arm]['DR']:6.2f} "
                  f"FAR={q[arm]['FAR']:5.2f} {r.model_bytes/1024:8.1f}KB "
                  f"{r.p50_ms:6.3f}ms E={e:8.0f}pJ ({e/t2s_e:.2f}x)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(REPO/"results/tables/tab_teacher_per_dataset.csv", index=False)
    print(f"\n[wrote] results/tables/tab_teacher_per_dataset.csv ({len(d)} rows)")
