#!/usr/bin/env python3
"""B1: empirical check of Proposition 1 and the encoder audit (pre-registered, Part B; no training).

For every dataset (eleven) and seed (42-46), with the split and teacher guards:
1. Build the tree encoder with every split threshold, before pruning (encoder.edges_,
   parsed by the production encoder from the booster's trees_to_dataframe dump).
2. For each test flow, a surrogate with the same level vector: for split feature j with
   k = #{tau <= x_j} (the encoder's convention, searchsorted side="right"), x'_j = tau_k
   (k >= 1) or the largest float32 below tau_1 (k = 0); features never split on -> 0.
3. Score original and surrogate through the production call (a DataFrame into
   XGBClassifier.predict_proba; float32 casting and missing-value routing are XGBoost's).
   Inputs contain no NaN (asserted). Reported: identical predicted labels, max |d proba|.
Diagnostics: the booster's exact float32 split conditions are read from its JSON model;
the number of dumped thresholds that differ from them is recorded, and the surrogate test
is repeated with the exact thresholds, so a mismatch can be attributed.

Export (one row per dataset, seed, feature): retained, |E_j|, allocated channels, T,
unique unshifted levels on test, unique occupied time bins on test, gain, dtype, comparison
convention, dumped-vs-exact threshold mismatches.
Guard: per IoT cell (and every cell with a saved table), the retained features and their
|E_j| equal the M3 saved encoder table, which reproduces 352 violating / 767 retained.

    python scripts/partB_b1_prop1.py [--datasets ...] [--seeds ...] [--workers 3]
Output: results/journal_partB/B1/<run_id>/{features.csv,cells.csv,summary.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

CONVENTION = "encoder: x >= tau counts as above (searchsorted side=right); XGBoost: x < t goes left"


def exact_thresholds(booster):
    m = json.loads(bytes(booster.save_raw("json")))
    out = {}
    for tr in m["learner"]["gradient_booster"]["model"]["trees"]:
        for j, t, lc in zip(tr["split_indices"], tr["split_conditions"], tr["left_children"]):
            if lc != -1:
                out.setdefault(int(j), set()).add(np.float32(t))
    return {j: np.array(sorted(v), dtype=np.float32) for j, v in out.items()}


def surrogate(X, edges):
    S = np.zeros_like(X, dtype=np.float32)
    for j, e in edges.items():
        k = np.searchsorted(e, X[:, j], side="right")
        lower = np.nextafter(e[0], np.float32(-np.inf), dtype=np.float32)
        S[:, j] = np.where(k == 0, lower, e[np.maximum(k - 1, 0)])
    return S


def cell(ds, seed):
    from src.models.tree_to_spike import TreeGuidedSpikeEncoder
    b, arrays = C.load_bundle(ds, seed)
    teacher, present = C.fit_teacher_guarded(ds, seed, b, arrays)
    cfg = C.t2s_config(ds, seed)
    Xte = arrays[4].astype(np.float32)
    if np.isnan(Xte).any():
        raise SystemExit(f"{ds}/{seed}: NaN in test inputs")
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                 jitter=cfg["jitter"], encoder_mode="tree").fit(teacher, Xte.shape[1])
    edges = {j: np.asarray(e, dtype=np.float32) for j, e in enc.edges_.items()}
    exact = exact_thresholds(teacher.get_booster())
    cols = [f"f{i}" for i in range(Xte.shape[1])]
    P = teacher.predict_proba(pd.DataFrame(Xte, columns=cols))
    res = {}
    for name, E in (("dump", edges), ("exact", exact)):
        Q = teacher.predict_proba(pd.DataFrame(surrogate(Xte, E), columns=cols))
        res[name] = dict(agree=float((P.argmax(1) == Q.argmax(1)).mean()), max_dproba=float(np.abs(P - Q).max()),
                         n_disagree=int((P.argmax(1) != Q.argmax(1)).sum()))
    imp = np.asarray(teacher.feature_importances_, dtype=np.float64)
    retained = {}
    for j, _ in enc.channels_:
        retained[j] = retained.get(j, 0) + 1
    T = cfg["T"]
    feats = []
    for j in range(Xte.shape[1]):
        e = edges.get(j)
        if e is not None:
            k = np.searchsorted(e, Xte[:, j], side="right")
            lv = np.unique(k).size
            bins = np.unique(np.round((1.0 - k / len(e)) * (T - 1))).size
            mism = int(len(set(e.tolist()) ^ set(exact.get(j, np.zeros(0, np.float32)).tolist())))
        else:
            lv, bins, mism = 1, 0, 0
        feats.append(dict(dataset=ds, seed=seed, feature=j, retained=j in retained, E_j=0 if e is None else len(e),
                          channels=retained.get(j, 0), T=T, unique_levels_test=lv, unique_bins_test=bins,
                          gain=imp[j], dtype=str(Xte.dtype), convention=CONVENTION, dump_vs_exact_mismatch=mism))
    # guard: retained features and |E_j| equal the saved M3 encoder table, where one exists
    tab = C.REPO / f"results/diagnostics/val_curves/Tree-to-Spike/{ds}/runs/seed_{seed}/encoder_channels.json"
    table_match = None
    if tab.exists():
        saved = {c["feature"]: c["n_edges"] for c in json.loads(tab.read_text())["channels"]}
        mine = {f["feature"]: f["E_j"] for f in feats if f["retained"]}
        table_match = saved == mine
        if ds in C.IOT and not table_match:
            raise SystemExit(f"{ds}/{seed}: retained features / |E_j| differ from the saved encoder table")
    split_feats = [f for f in feats if f["E_j"] > 0]
    pruned_gain = sum(f["gain"] for f in split_feats if not f["retained"]) / max(sum(f["gain"] for f in feats), 1e-300)
    c = dict(dataset=ds, seed=seed, n_test=len(Xte), agree_dump=res["dump"]["agree"], disagree_dump=res["dump"]["n_disagree"],
             max_dproba_dump=res["dump"]["max_dproba"], agree_exact=res["exact"]["agree"],
             disagree_exact=res["exact"]["n_disagree"], max_dproba_exact=res["exact"]["max_dproba"],
             thresholds_dump_vs_exact_mismatch=sum(f["dump_vs_exact_mismatch"] for f in feats),
             retained=sum(f["retained"] for f in feats), violating=sum(f["retained"] and f["E_j"] >= T for f in feats),
             pruned_gain_share=pruned_gain, table_match=table_match)
    return c, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=C.ALL)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(C.SEEDS))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    C.no_thread_caps()
    st = C.stamp("B1")
    out = C.run_dir(st)
    cells, feats = [], []
    jobs = [(d, s) for d in a.datasets for s in a.seeds]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fut = {ex.submit(cell, d, s): (d, s) for d, s in jobs}
        for f in as_completed(fut):
            c, fs = f.result()
            cells.append(c); feats += fs
            print(f"{c['dataset']:16s} {c['seed']} agree dump {100*c['agree_dump']:.4f}% ({c['disagree_dump']} differ) "
                  f"exact {100*c['agree_exact']:.4f}% | dumped!=exact thresholds {c['thresholds_dump_vs_exact_mismatch']} | "
                  f"retained {c['retained']} violating {c['violating']} | table {c['table_match']}", flush=True)
            pd.DataFrame(cells).sort_values(["dataset", "seed"]).to_csv(out / "cells.csv", index=False)
            pd.DataFrame(feats).sort_values(["dataset", "seed", "feature"]).to_csv(out / "features.csv", index=False)
    Cc, F = pd.DataFrame(cells), pd.DataFrame(feats)
    R = F[F.retained]
    iot = R[R.dataset.isin(C.IOT)]
    med = R.groupby(["dataset", "seed"]).E_j.median().groupby("dataset").mean()
    summary = dict(
        stamp=st, cells=len(Cc),
        agreement_dump_min=float(Cc.agree_dump.min()), agreement_exact_min=float(Cc.agree_exact.min()),
        cells_not_100pct_dump=Cc[Cc.agree_dump < 1][["dataset", "seed", "disagree_dump"]].to_dict("records"),
        cells_not_100pct_exact=Cc[Cc.agree_exact < 1][["dataset", "seed", "disagree_exact"]].to_dict("records"),
        thresholds_dump_vs_exact_mismatch_total=int(Cc.thresholds_dump_vs_exact_mismatch.sum()),
        iot_violating=int((iot.E_j >= 20).sum()), iot_retained=int(len(iot)),
        guard_352_767=bool(int((iot.E_j >= 20).sum()) == 352 and len(iot) == 767) if set(C.IOT) <= set(a.datasets) and len(a.seeds) == 5 else None,
        share_exceeding_bound_by_dataset=(100 * R.assign(v=R.E_j >= R["T"]).groupby("dataset").v.mean()).round(2).to_dict(),
        median_E_by_dataset=med.round(1).to_dict(),
        pruned_gain_share_by_dataset=(100 * Cc.groupby("dataset").pruned_gain_share.mean()).round(3).to_dict())
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "stamp"}, indent=1, default=str))


if __name__ == "__main__":
    main()
