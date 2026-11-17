#!/usr/bin/env python3
"""Validation-selected compact GBDT within 128 KiB (revision 2, step 3).

Pre-registered in results/preregistration/2026-11-16_r2_integer_gbdt.md. Reruns the
48-configuration grid of scripts/gbdt_compression.py for one dataset and seed, prices each
configuration in the verified packed format, and selects on VALIDATION macro-F1 among those
within the budget (ties to fewer bytes). Test is scored once, for the selected configuration.

The grid exactly as scripts/gbdt_compression.py defines it: rounds 25/50/100/200 by depth
3/4/6, the "gain_prune 0.9" arm, 16- and 8-bit leaves, the same XGBoost hyperparameters,
cost weights and n_jobs=8. The "gain_prune 0.9" arm counts trees (rounds x classes) covering
90% of the total gain of the R-round model and refits with that many ROUNDS, so it adds rounds
rather than pruning; it is kept for comparability and labelled "more rounds" in reports.

Efficiency: configurations of one depth differ only in their number of rounds, and a boosted
model's first r rounds do not depend on how many follow (same seed, same row and column
sampling stream), so every configuration of a depth is a prefix of one fitted model. Checked
each time: the prefix of the longer model predicts exactly as the 200-round fit on the
validation split.

Leaves are quantized to the configuration's leaf width inside the model (min-max over the
prefix's leaves, as src/serialize.predict_packed does), so 8-bit configurations are scored as
the packed artifact computes them. The published sweep scored both widths unquantized.
Checked once per depth against src/serialize.predict_packed: identical leaves, and margins that
differ only by XGBoost's per-class intercept (K float32 values, at most 56 B, which the packed
price omits and predict_packed leaves out; the intercept can break an exact tie of 8-bit sums).

    python scripts/r2_gbdt_valsel.py <dataset> <seed> [--guard]
    --guard (seed 42): also score every configuration on test, natively, and compare with
                       results/deployment/gbdt_compression_repriced.csv

Output: results/journal_r2/gbdt_valsel/<dataset>/seed_<S>.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")
OUT = REPO / "results/journal_r2/gbdt_valsel"
GRID_EST, GRID_DEPTH, LEAF_BITS = [25, 50, 100, 200], [3, 4, 6], [16, 8]
BUDGET = 128 * 1024
XGB = dict(learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, importance_type="gain",
           tree_method="hist", n_jobs=8, eval_metric="mlogloss")


def load(ds, seed):
    from src.data import get_dataset, DATASET_LOADERS
    from src.training_treespike import _arrays_from_bundle
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    kw = dict(seed=seed)
    for o in ("subsample_fraction", "test_fraction", "label_mode"):
        if o in cfg and o in lp:
            kw[o] = cfg[o]
    b = get_dataset(ds, **kw)
    return b, _arrays_from_bundle(b)


def fit(Xtr, ytr_r, w, rounds, depth, seed):
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=rounds, max_depth=depth, random_state=seed, **XGB)
    m.fit(pd.DataFrame(Xtr, columns=[f"f{i}" for i in range(Xtr.shape[1])]), ytr_r, sample_weight=w)
    return m


def keep_rounds(tdf, est, k_units):
    """The sweep's gain-prune count, from the first `est` rounds of a longer model."""
    sub = tdf[(tdf.Tree < est * k_units) & (tdf.Feature != "Leaf")]
    g = sub.groupby("Tree").Gain.sum().sort_values(ascending=False)
    return int(max(1, (g.cumsum() / g.sum() <= 0.90).sum()))


def quantized_booster(model, rounds, k_units, leaf_bits):
    """Copy of the booster truncated to `rounds`, leaves quantized as the packed format does."""
    import xgboost as xgb
    js = json.loads(model.get_booster()[0:rounds].save_raw("json").decode())
    trees = js["learner"]["gradient_booster"]["model"]["trees"]
    leaf_vals = []
    for t in trees:
        lc = np.asarray(t["left_children"])
        leaf_vals.append(np.asarray(t["split_conditions"], dtype=np.float64)[lc == -1])
    allv = np.concatenate(leaf_vals)
    lo, hi = allv.min(), allv.max()
    step = (hi - lo) / (2 ** leaf_bits - 1) if hi > lo else 1.0
    for t in trees:
        lc = np.asarray(t["left_children"])
        sc = np.asarray(t["split_conditions"], dtype=np.float64)
        bw = np.asarray(t["base_weights"], dtype=np.float64)
        q = lo + np.round((sc - lo) / step) * step
        sc[lc == -1] = q[lc == -1]
        bw[lc == -1] = q[lc == -1]
        t["split_conditions"] = sc.astype(np.float32).tolist()
        t["base_weights"] = bw.astype(np.float32).tolist()
    b = xgb.Booster()
    b.load_model(bytearray(json.dumps(js).encode()))
    return b


def predict(booster, X, k_units):
    import xgboost as xgb
    p = booster.predict(xgb.DMatrix(pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])))
    return p.argmax(1) if k_units > 1 else (p >= 0.5).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("seed", type=int)
    ap.add_argument("--guard", action="store_true")
    a = ap.parse_args()
    from sklearn.metrics import f1_score
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    from src import serialize as SER
    ds, seed = a.dataset, a.seed
    out = OUT / ds / f"seed_{seed}.json"
    if out.exists() and not a.guard:
        print(f"[skip] {out}"); return
    t0 = time.time()
    b, (Xtr, ytr, Xva, yva, Xte, yte) = load(ds, seed)
    K = b.num_classes
    present = np.unique(ytr)
    remap = np.full(K, -1, np.int64); remap[present] = np.arange(len(present))
    cnt = np.bincount(remap[ytr], minlength=len(present)).astype(np.float32)
    w = (cnt.sum() / (cnt + 1e-6))[remap[ytr]]
    Kt = len(present); k_units = Kt if Kt > 2 else 1
    split = dict(n_test=int(len(yte)), class_counts_train=np.bincount(ytr, minlength=K).tolist())
    configs, guard = [], []
    pub = None
    if a.guard:
        pub = pd.read_csv(REPO / "results/deployment/gbdt_compression_repriced.csv")
        pub = pub[pub.dataset == ds].copy(); pub["gain_prune"] = pub.gain_prune.astype(str)
    for depth in GRID_DEPTH:
        m200 = fit(Xtr, remap[ytr], w, 200, depth, seed)
        tdf = m200.get_booster().trees_to_dataframe()
        keeps = {est: keep_rounds(tdf, est, k_units) for est in GRID_EST}
        need = max([200] + list(keeps.values()))
        model = m200
        if need > 200:
            model = fit(Xtr, remap[ytr], w, need, depth, seed)
            sub = Xva[:20000]
            pa = m200.get_booster().predict(__import__("xgboost").DMatrix(pd.DataFrame(sub, columns=[f"f{i}" for i in range(sub.shape[1])])))
            pb = model.get_booster()[0:200].predict(__import__("xgboost").DMatrix(pd.DataFrame(sub, columns=[f"f{i}" for i in range(sub.shape[1])])))
            if not np.array_equal(pa, pb):
                sys.exit(f"{ds}/{seed} depth {depth}: prefix of the {need}-round model differs from the 200-round fit")
        for est in GRID_EST:
            for arm, rounds in (("none", est), ("0.9", keeps[est])):
                for lb in LEAF_BITS:
                    packed = SER.pack_booster(model, rounds, leaf_bits=lb, n_classes=Kt)
                    qb = quantized_booster(model, rounds, k_units, lb)
                    pv = present[predict(qb, Xva, k_units)]
                    c = dict(n_estimators=est, max_depth=depth, gain_prune=arm, leaf_bits=lb, rounds=int(rounds),
                             bytes=int(packed["deploy_bytes"]), fits=bool(packed["deploy_bytes"] <= BUDGET),
                             val_macro_f1=100 * compute_metrics(yva, pv, class_names=b.class_names)["f1_macro"])
                    configs.append(c)
                    if a.guard and lb == 16:
                        nat = present[model.predict(pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])]),
                                                    iteration_range=(0, rounds))]
                        r = pub[(pub.n_estimators == est) & (pub.max_depth == depth) & (pub.gain_prune == arm)
                                & (pub.leaf_bits == 16)].iloc[0]
                        guard.append(dict(n_estimators=est, max_depth=depth, gain_prune=arm, rounds=int(rounds),
                                          published_rounds=int(r.effective_trees),
                                          test_sklearn_f1=100 * f1_score(yte, nat, average="macro"),
                                          published_f1=float(r.macro_f1), bytes=int(packed["deploy_bytes"]),
                                          published_bytes=int(r.deploy_bytes_packed_fp32_thresholds)))
                    print(f"  {ds} s{seed} d{depth} est{est:3d} {arm:>4} r{rounds:4d} {lb:2d}b "
                          f"{packed['deploy_bytes'] / 1024:7.1f} KiB val {c['val_macro_f1']:6.2f}", flush=True)
        # packed-prediction check, once per depth on the smallest configuration: same leaves, and
        # margins equal up to XGBoost's per-class intercept (a constant per class that
        # predict_packed omits; it can flip an exact tie of the packed sums, so labels are not compared)
        import xgboost as xgb
        pk = SER.pack_booster(model, 25, leaf_bits=8, n_classes=Kt)
        sub = Xva[:300]
        qb = quantized_booster(model, 25, k_units, 8)
        dm = xgb.DMatrix(pd.DataFrame(sub, columns=[f"f{i}" for i in range(sub.shape[1])]))
        off = qb.predict(dm, output_margin=True).reshape(len(sub), -1) - SER.predict_packed(pk, sub, Kt, leaf_bits=8, margin=True)
        same_leaves = np.array_equal(qb.predict(dm, pred_leaf=True), model.get_booster()[0:25].predict(dm, pred_leaf=True))
        if not same_leaves or np.ptp(off, axis=0).max() > 1e-4:
            sys.exit(f"{ds}/{seed} depth {depth}: quantized booster disagrees with predict_packed "
                     f"(same leaves {same_leaves}, offset spread {np.ptp(off, axis=0).max():.2e})")
    fit_ = [c for c in configs if c["fits"]]
    sel = max(fit_, key=lambda c: (c["val_macro_f1"], -c["bytes"])) if fit_ else None
    rec = dict(dataset=ds, seed=seed, split=split, budget_bytes=BUDGET, n_configs=len(configs),
               n_within_budget=len(fit_), configs=configs, guard=guard,
               preregistration="results/preregistration/2026-11-16_r2_integer_gbdt.md")
    if sel:
        m = fit(Xtr, remap[ytr], w, sel["rounds"], sel["max_depth"], seed)
        qb = quantized_booster(m, sel["rounds"], k_units, sel["leaf_bits"])
        pt = present[predict(qb, Xte, k_units)]
        met = compute_metrics(yte, pt, class_names=b.class_names)
        bm = compute_binary_ids_metrics(yte, pt, normal_class_index=0)
        rec["selected"] = dict(**sel, test_macro_f1=100 * met["f1_macro"], test_accuracy=100 * met["accuracy"],
                               test_mcc=100 * met["matthews_corrcoef"], test_far=100 * bm["false_alarm_rate"],
                               test_dr=100 * bm["detection_rate"])
    rec["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    s = rec.get("selected")
    print(f"[done] {ds} seed {seed}: {len(fit_)}/{len(configs)} within budget; selected "
          + (f"{s['n_estimators']}x{s['max_depth']} {s['gain_prune']} {s['leaf_bits']}b ({s['rounds']} rounds, "
             f"{s['bytes'] / 1024:.1f} KiB) val {s['val_macro_f1']:.2f} test {s['test_macro_f1']:.2f} FAR {s['test_far']:.2f}"
             if s else "none") + f" [{rec['seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
