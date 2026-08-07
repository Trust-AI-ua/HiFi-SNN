#!/usr/bin/env python3
"""Validation-selected early exit, replayed offline from cached prefix artifacts.

No inference happens here. Every threshold is replayed from `prefix_logits`, so
the eight-threshold sweep, the class-conditional exit analysis, the risk-coverage
curve and the calibration numbers all come from one cached full-window pass.

Protocol, in order:
  A  full-window validation reference (early exit disabled)
  B  replay all eight thresholds on VALIDATION only
  C  eligibility: F1_exit,val >= F1_full,val - 1.0 ABSOLUTE macro-F1 points
  D  among eligible, pick the lowest mean executed SOPs; deterministic tie-break
     (fewer ops -> higher val macro-F1 -> lower val FAR -> higher threshold).
     If none qualifies, early exit is DISABLED for that cell -- a result, not a
     failure.
  E  apply the selected threshold to test exactly once

Test is never consulted during selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PREFIX = REPO / "results" / "prefix_outputs"
OUT = REPO / "results" / "early_exit"
THETAS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
MARGIN = 1.0        # absolute macro-F1 percentage points


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def load(ds, seed, split):
    f = PREFIX / ds / "treespike" / f"seed_{seed}" / f"{split}_prefix.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {k: d[k] for k in d.files}


def replay(pref, theta):
    """Exit at the first t whose max posterior >= theta, else run to T."""
    q = softmax(pref["prefix_logits"].astype(np.float32), axis=2)   # [N,T,K]
    conf = q.max(axis=2)                                            # [N,T]
    hit = conf >= theta
    ever = hit.any(axis=1)
    first = np.where(ever, hit.argmax(axis=1), q.shape[1] - 1)      # 0-based
    pred = q[np.arange(len(first)), first].argmax(axis=1)
    sops = pref["cum_sops"][np.arange(len(first)), first]
    spikes = (pref["cum_spk1"] + pref["cum_spk2"])[np.arange(len(first)), first]
    return pred, first + 1, ever, sops, spikes


def full_window(pref):
    T = pref["prefix_logits"].shape[1]
    pred = pref["prefix_logits"][:, -1, :].argmax(axis=1)
    return (pred, np.full(len(pred), T), pref["cum_sops"][:, -1],
            (pref["cum_spk1"] + pref["cum_spk2"])[:, -1])


def score(y, p):
    from sklearn.metrics import (f1_score, matthews_corrcoef, confusion_matrix,
                                 balanced_accuracy_score)
    tb, pb = (y != 0).astype(int), (p != 0).astype(int)
    cm = confusion_matrix(tb, pb, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return dict(
        macro_f1=100 * f1_score(y, p, average="macro", zero_division=0),
        mcc=100 * matthews_corrcoef(y, p) if len(np.unique(y)) > 1 else np.nan,
        bacc=100 * balanced_accuracy_score(y, p),
        DR=100 * tp / max(tp + fn, 1), FAR=100 * fp / max(fp + tn, 1))


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(conf)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return 100 * e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cells = sorted({(p.parts[-4], int(p.parts[-2].split("_")[1]))
                    for p in PREFIX.glob("*/treespike/seed_*/val_prefix.npz")})
    if args.datasets:
        cells = [c for c in cells if c[0] in args.datasets]
    if args.seeds:
        cells = [c for c in cells if c[1] in args.seeds]
    print(f"{len(cells)} cells with prefix artifacts\n")

    sweep_rows, final_rows, class_rows, cal_rows = [], [], [], []
    for ds, seed in cells:
        pv, pt = load(ds, seed, "val"), load(ds, seed, "test")
        if pv is None or pt is None:
            print(f"[{ds}/{seed}] missing artifact, skipped"); continue
        yv, yt = pv["labels"], pt["labels"]
        names = [str(c) for c in pv["class_names"]]
        T = pv["prefix_logits"].shape[1]

        # ---- A. full-window validation reference
        pred_f, _, sops_f, spk_f = full_window(pv)
        ref = score(yv, pred_f)
        f_full_val, ops_full_val = ref["macro_f1"], sops_f.mean()

        # ---- B. replay every threshold on validation only
        cand = []
        for th in THETAS:
            p, ex, ever, sops, spk = replay(pv, th)
            s = score(yv, p)
            row = dict(dataset=ds, seed=seed, theta=th, split="val", **s,
                       mean_exit=ex.mean(), median_exit=float(np.median(ex)),
                       pct_reaching_T=100 * float((~ever).mean()),
                       mean_sops=sops.mean(),
                       sop_reduction=100 * (1 - sops.mean() / ops_full_val),
                       spike_reduction=100 * (1 - spk.mean() / spk_f.mean()),
                       eligible=bool(s["macro_f1"] >= f_full_val - MARGIN))
            sweep_rows.append(row); cand.append(row)

        # ---- C/D. eligibility + deterministic selection
        elig = [c for c in cand if c["eligible"]]
        if elig:
            sel = sorted(elig, key=lambda c: (c["mean_sops"], -c["macro_f1"],
                                              c["FAR"], -c["theta"]))[0]
            theta, enabled = sel["theta"], True
        else:
            sel, theta, enabled = None, None, False

        # ---- E. apply to test exactly once
        pred_tf, _, sops_tf, spk_tf = full_window(pt)
        full_t = score(yt, pred_tf)
        if enabled:
            p_t, ex_t, ever_t, sops_t, spk_t = replay(pt, theta)
        else:
            p_t, ex_t, ever_t, sops_t, spk_t = (pred_tf, np.full(len(yt), T),
                                                np.zeros(len(yt), bool), sops_tf, spk_tf)
        exit_t = score(yt, p_t)
        final_rows.append(dict(
            dataset=ds, seed=seed, T=T, selected_threshold=theta,
            early_exit_disabled=not enabled,
            n_eligible=len(elig),
            full_macro_f1=full_t["macro_f1"], exit_macro_f1=exit_t["macro_f1"],
            d_macro_f1=exit_t["macro_f1"] - full_t["macro_f1"],
            full_mcc=full_t["mcc"], exit_mcc=exit_t["mcc"],
            d_mcc=exit_t["mcc"] - full_t["mcc"],
            full_FAR=full_t["FAR"], exit_FAR=exit_t["FAR"],
            d_FAR=exit_t["FAR"] - full_t["FAR"],
            full_DR=full_t["DR"], exit_DR=exit_t["DR"],
            mean_exit=float(ex_t.mean()), median_exit=float(np.median(ex_t)),
            frac_reaching_T=float((~ever_t).mean()) if enabled else 1.0,
            mean_sop_reduction=100 * (1 - sops_t.mean() / sops_tf.mean()),
            mean_spike_reduction=100 * (1 - spk_t.mean() / spk_tf.mean()),
            val_full_macro_f1=f_full_val,
            val_exit_macro_f1=sel["macro_f1"] if sel else f_full_val))

        # ---- class-conditional exit behaviour (test)
        early = ever_t if enabled else np.zeros(len(yt), bool)
        for c, nm in enumerate(names):
            m = yt == c
            if not m.any():
                continue
            e_m = m & early
            l_m = m & ~early
            class_rows.append(dict(
                dataset=ds, seed=seed, cls=nm, n=int(m.sum()),
                mean_exit=float(ex_t[m].mean()),
                frac_exit_before_T=100 * float(early[m].mean()),
                recall_full=100 * float((pred_tf[m] == c).mean()),
                recall_exit=100 * float((p_t[m] == c).mean()),
                err_early=100 * float((p_t[e_m] != c).mean()) if e_m.any() else np.nan,
                err_fullwindow=100 * float((p_t[l_m] != c).mean()) if l_m.any() else np.nan))

        # ---- calibration of the prefix posterior (diagnostic only, not selection)
        qv = softmax(pt["prefix_logits"].astype(np.float32), axis=2)
        for t in sorted({0, T // 2 - 1, T - 1}):
            conf = qv[:, t].max(axis=1)
            corr = (qv[:, t].argmax(axis=1) == yt).astype(float)
            cal_rows.append(dict(dataset=ds, seed=seed, t=t + 1, ece=ece(conf, corr),
                                 mean_conf=100 * float(conf.mean()),
                                 accuracy=100 * float(corr.mean())))

        tag = f"theta={theta}" if enabled else "DISABLED (no threshold qualified)"
        print(f"[{ds}/seed{seed}] T={T} {tag} | test F1 {full_t['macro_f1']:.2f} -> "
              f"{exit_t['macro_f1']:.2f} ({exit_t['macro_f1']-full_t['macro_f1']:+.2f} pp) "
              f"| SOPs -{100*(1-sops_t.mean()/sops_tf.mean()):.1f}% "
              f"| mean exit {ex_t.mean():.2f}/{T}", flush=True)

    for name, rows in (("sweep_validation", sweep_rows), ("selected_test", final_rows),
                       ("class_conditional", class_rows), ("calibration", cal_rows)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)

    f = pd.DataFrame(final_rows)
    if len(f):
        print("\n" + "=" * 78)
        print("SELECTED OPERATING POINTS (test, applied once)")
        print("=" * 78)
        cols = ["dataset", "seed", "selected_threshold", "early_exit_disabled",
                "full_macro_f1", "exit_macro_f1", "d_macro_f1", "d_FAR",
                "mean_exit", "T", "mean_sop_reduction"]
        print(f[cols].round(3).to_string(index=False))
        print("\n-- per dataset, mean over seeds")
        print(f.groupby("dataset")[["d_macro_f1", "d_mcc", "d_FAR", "mean_exit",
                                    "mean_sop_reduction"]].mean().round(3).to_string())
        n_dis = int(f.early_exit_disabled.sum())
        print(f"\nearly exit DISABLED in {n_dis} of {len(f)} dataset/seed cells")
        print(f"cells losing >1.0 pp macro-F1 on test: "
              f"{int((f.d_macro_f1 < -1.0).sum())} of {len(f)}")
    print(f"\nwrote {OUT}/*.csv")


if __name__ == "__main__":
    main()
