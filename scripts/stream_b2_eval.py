#!/usr/bin/env python3
"""B2, stage 3 (CPU): decisions and metrics for one seed (results/preregistration/2026-11-17_neuro_stream.md).

Models: Tree-to-Spike-Stream (stage 2 running logits), the per-packet teacher (stage 1 margins),
and the compact GBDT within 128 KiB (reference; the step-3 grid and selection on validation
macro-F1 at flow end, fitted here on the prefix rows with the teacher's weights).
Each: per-packet temperatures and gate on validation (src/stream_decision), then the benign
threshold tuned on validation to FAR targets 0.1, 0.25, 0.5, 1 (the matched point) and 2%.
Costs for the gate: executed SOPs (SNN), packets evaluated (trees).
Reported on test: macro-F1, FAR, DR at decision; packets-to-detection (median, 90th percentile)
on attack flows detected by both the SNN and the per-packet tree, and per model on its own
detected set; operations per packet (SOPs; node comparisons = leaf depth summed over trees);
share of executed packets with no input spikes.

    python scripts/stream_b2_eval.py <seed> [--no-compact]
Output: results/neuro/b2/seed_<S>/eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from src.data.edgeiiot_stream import load_stream  # noqa: E402
from src import stream_decision as SD  # noqa: E402

OUT = REPO / "results/neuro/b2"
BIG = REPO / "dataset/edge-iiotset/_stream/b2"
TARGETS = [0.001, 0.0025, 0.005, 0.01, 0.02]
P = 32


def node_depths(booster):
    js = json.loads(booster.save_raw("json").decode())
    out = []
    for t in js["learner"]["gradient_booster"]["model"]["trees"]:
        lc, rc = np.asarray(t["left_children"]), np.asarray(t["right_children"])
        d = np.zeros(len(lc), np.int64)
        for n in range(len(lc)):            # parents precede children in XGBoost's node order
            if lc[n] != -1:
                d[lc[n]] = d[n] + 1; d[rc[n]] = d[n] + 1
        out.append(d)
    return out


def comparisons_per_eval(booster, rows):
    import xgboost as xgb
    leaves = booster.predict(xgb.DMatrix(pd.DataFrame(rows, columns=[f"f{i}" for i in range(rows.shape[1])])),
                             pred_leaf=True).astype(np.int64)
    dep = node_depths(booster)
    return float(np.mean(sum(dep[t][leaves[:, t]] for t in range(leaves.shape[1]))))


def margins(booster, X):
    import xgboost as xgb
    n = len(X)
    m = booster.predict(xgb.DMatrix(pd.DataFrame(X.reshape(n * P, -1), columns=[f"f{i}" for i in range(X.shape[2])])),
                        output_margin=True)
    return m.reshape(n, P, -1)


def run_model(name, Zv, Zt, s, cost_v, cost_t):
    yv, yt, Lv, Lt = s.y["val"], s.y["test"], s.length["val"], s.length["test"]
    taus = SD.fit_taus(Zv, yv, Lv)
    pv, pt = SD.calibrated(Zv, taus), SD.calibrated(Zt, taus)
    theta, table = SD.gate(pv, yv, Lv, cost_v)
    fv, Pv = SD.decide(pv, Lv, theta)
    ft, Pt = SD.decide(pt, Lt, theta)
    res = dict(model=name, theta=theta, taus=taus.tolist(), gate=table,
               argmax_test=SD.scores(yt, Pt.argmax(1)), mean_packets=float((ft + 1).mean()),
               mean_cost_test=float(cost_t[np.arange(len(ft)), ft].mean()), at={})
    preds = {}
    for f in TARGETS:
        t = SD.threshold_for(Pv[yv == 0, 0], f)
        p = SD.thresholded(Pt, t)
        res["at"][f"{100 * f:g}"] = dict(threshold=t, val=SD.scores(yv, SD.thresholded(Pv, t)), test=SD.scores(yt, p))
        preds[f] = p
    return res, ft, preds


def p2d(first, detected):
    k = first[detected] + 1
    return dict(n=int(detected.sum()), median=float(np.median(k)) if len(k) else None,
                p90=float(np.percentile(k, 90)) if len(k) else None)


def compact_gbdt(s, seed, w_fn):
    import r2_gbdt_valsel as G
    import src.serialize as SER
    from stream_b2_teacher import prefix_rows
    from sklearn.metrics import f1_score
    R, f, _ = prefix_rows(s.X["train"], s.length["train"])
    y = s.y["train"][f]
    w = w_fn(f, y)
    K = len(s.class_names)
    Lv = s.length["val"]
    Xv_end = s.X["val"][np.arange(len(Lv)), Lv - 1]
    best = None
    for depth in G.GRID_DEPTH:
        m = G.fit(R, y, w, max(G.GRID_EST), depth, seed)
        for rounds in G.GRID_EST:
            for lb in G.LEAF_BITS:
                b = int(SER.pack_booster(m, rounds, leaf_bits=lb, n_classes=K)["deploy_bytes"])
                if b > G.BUDGET:
                    continue
                qb = G.quantized_booster(m, rounds, K, lb)
                f1 = 100 * f1_score(s.y["val"], G.predict(qb, Xv_end, K), average="macro")
                c = dict(depth=depth, rounds=rounds, leaf_bits=lb, bytes=b, val_macro_f1_flow_end=f1)
                if best is None or (f1, -b) > (best[0]["val_macro_f1_flow_end"], -best[0]["bytes"]):
                    best = (c, qb)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--no-compact", action="store_true")
    a = ap.parse_args()
    import xgboost as xgb
    seed = a.seed
    s = load_stream(seed)
    stu = np.load(BIG / f"seed_{seed}_student.npz")
    tz = np.load(BIG / f"seed_{seed}_teacher.npz")
    teacher = xgb.Booster(); teacher.load_model(str(OUT / f"seed_{seed}/teacher.json"))
    pk_v = np.cumsum(np.ones((len(s.y["val"]), P)), 1)
    pk_t = np.cumsum(np.ones((len(s.y["test"]), P)), 1)
    res = {}
    r_snn, f_snn, pr_snn = run_model("t2s_stream", stu["val_logits"], stu["test_logits"], s,
                                     stu["val_cum_sops"], stu["test_cum_sops"])
    r_tree, f_tree, pr_tree = run_model("tree_per_packet", tz["val_margin"], tz["test_margin"], s, pk_v, pk_t)
    yt, Lt = s.y["test"], s.length["test"]
    att = yt != 0
    for f in TARGETS:
        both = att & (pr_snn[f] != 0) & (pr_tree[f] != 0)
        key = f"{100 * f:g}"
        r_snn["at"][key]["p2d_both"] = p2d(f_snn, both); r_tree["at"][key]["p2d_both"] = p2d(f_tree, both)
        r_snn["at"][key]["p2d_own"] = p2d(f_snn, att & (pr_snn[f] != 0))
        r_tree["at"][key]["p2d_own"] = p2d(f_tree, att & (pr_tree[f] != 0))
    # operations per executed packet
    n = len(yt)
    steps = f_snn + 1
    sops_exec = stu["test_cum_sops"][np.arange(n), f_snn]
    r_snn["sops_per_packet"] = float(sops_exec.sum() / steps.sum())
    exec_mask = np.arange(P)[None, :] <= f_snn[:, None]
    r_snn["no_spike_packet_share"] = float((stu["test_in_spikes"][exec_mask] == 0).mean())
    rng = np.random.default_rng(0)
    X_eval = s.X["test"][rng.choice(n, min(n, 20000), replace=False), 0]
    r_tree["comparisons_per_packet"] = comparisons_per_eval(teacher, X_eval)
    res.update(t2s_stream=r_snn, tree_per_packet=r_tree)
    if not a.no_compact:
        counts = np.bincount(s.y["train"], minlength=len(s.class_names)).astype(np.float64)
        cls_w = counts.sum() / (counts + 1e-6)
        best, qb = compact_gbdt(s, seed, lambda f, y: cls_w[y] / s.length["train"][f])
        r_c, f_c, pr_c = run_model("compact_gbdt", margins(qb, s.X["val"]), margins(qb, s.X["test"]), s, pk_v, pk_t)
        r_c["config"] = best
        r_c["comparisons_per_packet"] = comparisons_per_eval(qb, X_eval)
        res["compact_gbdt"] = r_c
    res["feature_update_note"] = ("19 running statistics per packet, paid by every model: about 12 additions, "
                                  "4 comparisons (min, max), 3 divisions and 2 square roots")
    (OUT / f"seed_{seed}/eval.json").write_text(json.dumps(res, indent=1))
    m1 = r_snn["at"]["1"]["test"]; m2 = r_tree["at"]["1"]["test"]
    print(f"[done] seed {seed}: at val FAR<=1%: SNN F1 {m1['macro_f1']:.2f} FAR {m1['far']:.2f} | tree F1 "
          f"{m2['macro_f1']:.2f} FAR {m2['far']:.2f} | p2d median SNN {r_snn['at']['1']['p2d_both']['median']} "
          f"tree {r_tree['at']['1']['p2d_both']['median']} | SOPs/pkt {r_snn['sops_per_packet']:.0f} vs "
          f"comparisons/pkt {r_tree['comparisons_per_packet']:.0f}", flush=True)


if __name__ == "__main__":
    main()
