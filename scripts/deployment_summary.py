#!/usr/bin/env python3
"""Final deployment summary table + Pareto figure.

Joins predictive metrics (mean over 5 seeds) with measured systems cost
(serialized bytes, batch-1 single-thread CPU latency, executed operations,
measured on seed 42). Operations are never forced onto one scale: neural models
report MAC/SOP, tree models report comparisons.
"""
from __future__ import annotations

import json
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEP = REPO / "results" / "deployment"

# Fixed categorical order, validated (light surface, 7 slots: all checks pass;
# contrast WARN discharged by direct-labelling every point and shipping the table).
COLORS = {
    "full_gbdt": "#2a78d6", "compact_gbdt": "#eb6834", "raw_mlp_fp": "#1baf7a",
    "raw_mlp_int8": "#eda100", "dnn_tree": "#e87ba4", "t2s_full": "#008300",
    "t2s_exit": "#4a3aa7",
}
LABEL = {
    "full_gbdt": "Full GBDT", "compact_gbdt": "Compact GBDT",
    "raw_mlp_fp": "Raw MLP (FP32)", "raw_mlp_int8": "Raw MLP (int8-grid)",
    "dnn_tree": "DNN-Tree (int8-grid)", "t2s_full": "Tree-to-Spike full T",
    "t2s_exit": "Tree-to-Spike selected exit",
}
ORDER = list(COLORS)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def predictive() -> pd.DataFrame:
    rows = []
    cg = pd.read_csv(DEP / "compact_gbdt.csv")
    for ds, g in cg.groupby("dataset"):
        rows.append(dict(system="full_gbdt", dataset=ds, macro_f1=g.test_full_macro_f1.mean(),
                         mcc=g.test_full_mcc.mean(), FAR=g.test_full_FAR.mean()))
        rows.append(dict(system="compact_gbdt", dataset=ds, macro_f1=g.test_sel_macro_f1.mean(),
                         mcc=g.test_sel_mcc.mean(), FAR=g.test_sel_FAR.mean()))
    q = pd.read_csv(DEP / "int8_grid.csv")
    for ds, g in q[q.model == "rawmlp"].groupby("dataset"):
        rows.append(dict(system="raw_mlp_fp", dataset=ds, macro_f1=g.fp_macro_f1.mean(),
                         mcc=g.fp_mcc.mean(), FAR=g.fp_FAR.mean()))
        rows.append(dict(system="raw_mlp_int8", dataset=ds, macro_f1=g.int8_macro_f1.mean(),
                         mcc=g.int8_mcc.mean(), FAR=g.int8_FAR.mean()))
    for p in glob.glob(str(REPO / "results/DNN-Tree/*/seed_*/results.json")):
        r = json.load(open(p)); m = r["test_plus"]["metrics"]
        b = r["test_plus"].get("binary_ids_metrics") or {}
        rows.append(dict(system="dnn_tree", dataset=p.split("/")[-3],
                         macro_f1=100 * m["f1_macro"], mcc=100 * m["matthews_corrcoef"],
                         FAR=100 * b.get("false_alarm_rate", np.nan)))
    ee = pd.read_csv(REPO / "results/early_exit/selected_test.csv")
    for ds, g in ee.groupby("dataset"):
        rows.append(dict(system="t2s_full", dataset=ds, macro_f1=g.full_macro_f1.mean(),
                         mcc=g.full_mcc.mean(), FAR=g.full_FAR.mean()))
        rows.append(dict(system="t2s_exit", dataset=ds, macro_f1=g.exit_macro_f1.mean(),
                         mcc=g.exit_mcc.mean(), FAR=g.exit_FAR.mean()))
    return pd.DataFrame(rows).groupby(["system", "dataset"], as_index=False).mean()


def main():
    pred = predictive()
    dep = pd.read_csv(DEP / "deployment_measurements.csv")
    cost = dep.groupby("system").agg(
        total_bytes=("total_bytes", "mean"), model_bytes=("model_bytes", "mean"),
        encoder_bytes=("encoder_bytes", "mean"),
        p50_ms=("p50_ms", "mean"), p95_ms=("p95_ms", "mean"), p99_ms=("p99_ms", "mean"),
        macs=("macs", "mean"), sops=("sops", "mean"),
        tree_comparisons=("tree_comparisons", "mean")).reset_index()

    agg = pred.groupby("system").agg(macro_f1=("macro_f1", "mean"), mcc=("mcc", "mean"),
                                     FAR=("FAR", "mean")).reset_index()
    t = agg.merge(cost, on="system").set_index("system").reindex(ORDER).reset_index()

    def opcol(r):
        if r.system in ("full_gbdt", "compact_gbdt"):
            return f"{r.tree_comparisons:,.0f} cmp"
        if r.system.startswith("t2s"):
            return f"{r.sops:,.0f} SOP"
        return f"{r.macs:,.0f} MAC"
    t["ops"] = t.apply(opcol, axis=1)
    t["label"] = t.system.map(LABEL)

    print("=" * 108)
    print("DEPLOYMENT SUMMARY — predictive metrics mean over 11 datasets x 5 seeds; "
          "cost measured on seed 42")
    print("=" * 108)
    show = t[["label", "macro_f1", "mcc", "FAR", "total_bytes", "ops",
              "p50_ms", "p95_ms"]].copy()
    show.columns = ["Model", "Macro-F1", "MCC", "FAR", "Artifact B", "Ops/cmp",
                    "CPU p50 ms", "CPU p95 ms"]
    print(show.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    t.to_csv(DEP / "deployment_summary.csv", index=False)

    per = pred.merge(t[["system", "label"]], on="system")
    per.to_csv(DEP / "deployment_per_dataset.csv", index=False)

    # ---------------- Pareto: quality vs measured CPU latency ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    smin, smax = t.total_bytes.min(), t.total_bytes.max()
    for _, r in t.iterrows():
        frac = (np.log10(r.total_bytes) - np.log10(smin)) / max(
            np.log10(smax) - np.log10(smin), 1e-9)
        size = 120 + frac * 900
        ax.scatter(r.p50_ms, r.macro_f1, s=size, color=COLORS[r.system],
                   edgecolor=SURFACE, linewidth=2, zorder=3, alpha=0.95)
    # direct-label every point: discharges the palette contrast WARN and removes
    # any reliance on colour alone for identity
    off = {"full_gbdt": (12, 6), "compact_gbdt": (14, -14), "raw_mlp_fp": (10, 10),
           "raw_mlp_int8": (12, -6), "dnn_tree": (-12, -22), "t2s_full": (-14, 14),
           "t2s_exit": (12, -16)}
    for _, r in t.iterrows():
        dx, dy = off[r.system]
        ax.annotate(f"{LABEL[r.system]}\n{r.macro_f1:.1f}  ·  {r.p50_ms:.2f} ms",
                    (r.p50_ms, r.macro_f1), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8.5, color=INK,
                    ha="left" if dx > 0 else "right", linespacing=1.35)

    ax.set_xscale("log")
    ax.set_xlabel("batch-1 single-thread CPU latency, p50 (ms, log scale)  —  lower is better",
                  fontsize=9.5, color=MUTED)
    ax.set_ylabel("macro-F1 (mean over 11 datasets)  —  higher is better",
                  fontsize=9.5, color=MUTED)
    ax.set_title("Deployment frontier: predictive quality vs measured wall-clock cost",
                 fontsize=12, color=INK, pad=14, loc="left")
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.margins(x=0.22, y=0.20)
    fig.text(0.012, 0.015,
             "Bubble area ∝ total serialized artifact bytes (log-scaled). Tree models are "
             "priced in comparisons, neural models in MAC/SOP — never interconverted.\n"
             "DNN-Tree and Tree-to-Spike are evaluated from checkpoints saved after "
             "int8-grid weight projection with floating-point execution; no independent "
             "FP checkpoint was retained.",
             fontsize=7.4, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = DEP / "deployment_pareto.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"\nwrote {out}")
    print(f"wrote {DEP}/deployment_summary.csv and deployment_per_dataset.csv")


if __name__ == "__main__":
    main()
