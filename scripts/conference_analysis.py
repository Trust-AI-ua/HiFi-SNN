#!/usr/bin/env python3
"""Cross-dataset top-5 aggregation for the conference paper.

Aggregates all completed (neuron × encoding × seed) runs across the 4
target datasets (NSL-KDD, KDDCup99, CIC-IDS2017, CTU-13), ranks every
(neuron, encoding) combo by *mean F1-macro across the 4 datasets and 5
seeds* (20 measurements per combo), and produces:

  * results/conference/top5_cross_dataset.csv
  * results/conference/top5_per_dataset.csv
  * results/conference/heatmap_top5.{pdf,png}    — top-5 × 4 datasets
  * results/conference/bars_top5_per_dataset.{pdf,png}
  * results/conference/radar_top5.{pdf,png}
  * results/conference/top5_table.tex            — paper-ready LaTeX
  * results/conference/per_dataset_table.tex     — paper-ready LaTeX

Run from repo root:
    python scripts/conference_analysis.py
"""
from __future__ import annotations

import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.plotting import (apply_ieee_style, WONG_PALETTE, save_figure,
                            IEEE_RC)

DATASETS = ["nslkdd", "kddcup99", "cicids2017", "ctu13"]
DATASET_PRETTY = {
    "nslkdd": "NSL-KDD",
    "kddcup99": "KDDCup99",
    "cicids2017": "CIC-IDS2017",
    "ctu13": "CTU-13",
}

# Headline metrics to aggregate (column names in summary.csv / results.json)
METRICS = [
    "accuracy", "balanced_accuracy",
    "f1_macro", "f1_weighted",
    "precision_macro", "recall_macro",
    "roc_auc_weighted", "matthews_corrcoef", "cohen_kappa",
    "detection_rate", "false_alarm_rate",
]


def load_all_results() -> pd.DataFrame:
    """Walk results/<dataset>/SNN_*/<enc>/seed_*/results.json and return a
    DataFrame with one row per run + the headline metrics extracted."""
    rows = []
    for ds in DATASETS:
        for rj in sorted(glob.glob(str(REPO / "results" / ds / "SNN_*" / "*" /
                                        "seed_*" / "results.json"))):
            r = json.load(open(rj))
            tp = r["test_plus"]
            m = tp["metrics"]
            b = tp["binary_ids_metrics"]
            sp = tp["spike_stats"]
            row = {
                "dataset": r["dataset"],
                "neuron": r["neuron"],
                "encoding": r["encoding"],
                "seed": r["config"]["seed"],
                **{k: m.get(k) for k in ["accuracy", "balanced_accuracy",
                                          "f1_macro", "f1_weighted",
                                          "precision_macro", "recall_macro",
                                          "roc_auc_weighted",
                                          "matthews_corrcoef", "cohen_kappa"]},
                "detection_rate": b.get("detection_rate"),
                "false_alarm_rate": b.get("false_alarm_rate"),
                "spike_sparsity": sp.get("spike_sparsity"),
                "energy_spikes_per_sample": sp.get("energy_spikes_per_sample"),
                "inference_ms_per_sample": tp.get("inference_ms_per_sample"),
                "train_time_s": r["training"]["train_time_s"],
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} runs across {df['dataset'].nunique()} datasets, "
          f"{df['neuron'].nunique()} neurons × {df['encoding'].nunique()} encodings, "
          f"{df['seed'].nunique()} seeds")
    return df


# ---------------------------------------------------------------- aggregations

def cross_dataset_ranking(df: pd.DataFrame, k: int = 5):
    """Mean (and std) of each metric across 4 datasets × 5 seeds = 20 runs
    per (neuron, encoding) combo. Ranks by mean F1-macro."""
    agg = df.groupby(["neuron", "encoding"]).agg(
        acc_mean=("accuracy", "mean"),       acc_std=("accuracy", "std"),
        bal_mean=("balanced_accuracy", "mean"), bal_std=("balanced_accuracy", "std"),
        f1m_mean=("f1_macro", "mean"),       f1m_std=("f1_macro", "std"),
        f1w_mean=("f1_weighted", "mean"),    f1w_std=("f1_weighted", "std"),
        prec_mean=("precision_macro", "mean"), prec_std=("precision_macro", "std"),
        rec_mean=("recall_macro", "mean"),   rec_std=("recall_macro", "std"),
        auc_mean=("roc_auc_weighted", "mean"), auc_std=("roc_auc_weighted", "std"),
        mcc_mean=("matthews_corrcoef", "mean"), mcc_std=("matthews_corrcoef", "std"),
        kappa_mean=("cohen_kappa", "mean"),  kappa_std=("cohen_kappa", "std"),
        dr_mean=("detection_rate", "mean"),  dr_std=("detection_rate", "std"),
        far_mean=("false_alarm_rate", "mean"), far_std=("false_alarm_rate", "std"),
        sparsity_mean=("spike_sparsity", "mean"),
        energy_mean=("energy_spikes_per_sample", "mean"),
        train_time_mean=("train_time_s", "mean"),
        n=("seed", "count"),
    ).reset_index().sort_values("f1m_mean", ascending=False)
    return agg.head(k).reset_index(drop=True), agg


def per_dataset_for_top5(df: pd.DataFrame, top5: pd.DataFrame):
    """For each top-5 combo, get mean ± std on each dataset (across 5 seeds)
    for every reported metric — robustness across seeds is a paper-quality
    claim that needs explicit reporting."""
    top5_combos = list(zip(top5["neuron"], top5["encoding"]))
    sub = df[df.set_index(["neuron", "encoding"]).index.isin(top5_combos)].copy()

    pd_agg = sub.groupby(["neuron", "encoding", "dataset"]).agg(
        acc=("accuracy", "mean"),         acc_std=("accuracy", "std"),
        f1m=("f1_macro", "mean"),         f1m_std=("f1_macro", "std"),
        mcc=("matthews_corrcoef", "mean"), mcc_std=("matthews_corrcoef", "std"),
        auc=("roc_auc_weighted", "mean"), auc_std=("roc_auc_weighted", "std"),
        dr=("detection_rate", "mean"),    dr_std=("detection_rate", "std"),
        far=("false_alarm_rate", "mean"), far_std=("false_alarm_rate", "std"),
    ).reset_index()
    return pd_agg


# ---------------------------------------------------------------- plotting

def plot_heatmap_top5(top5: pd.DataFrame, per_ds: pd.DataFrame,
                      metric: str, out: Path, title: str, fmt: str = "{:.3f}"):
    """Wide heatmap: rows = top-5 combos, cols = 4 datasets, color = metric.
    Cell labels show "mean ± std" across the 5 seeds for that (variant, dataset)."""
    combos = [f"{r['neuron']}/{r['encoding']}" for _, r in top5.iterrows()]
    M = np.zeros((len(combos), len(DATASETS)))
    S = np.zeros_like(M)
    std_col = f"{metric}_std"
    for i, (_, r) in enumerate(top5.iterrows()):
        for j, ds in enumerate(DATASETS):
            sub = per_ds[(per_ds["neuron"] == r["neuron"]) &
                         (per_ds["encoding"] == r["encoding"]) &
                         (per_ds["dataset"] == ds)]
            M[i, j] = float(sub[metric].values[0]) if len(sub) else np.nan
            S[i, j] = float(sub[std_col].values[0]) if (len(sub) and std_col in sub) else 0

    # Horizontal-friendly: wide and short
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis",
                   vmin=max(0.4, np.nanmin(M) - 0.05), vmax=1.0)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_PRETTY[d] for d in DATASETS], fontsize=8)
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels(combos, fontsize=7.5)
    for i in range(len(combos)):
        for j in range(len(DATASETS)):
            v = M[i, j]; s = S[i, j]
            if not np.isnan(v):
                # Two-line annotation: mean (top), ± std (bottom)
                ax.text(j, i, f"{fmt.format(v)}\n$\\pm${fmt.format(s)}",
                        ha="center", va="center",
                        color="white" if v < 0.85 else "black", fontsize=6.5,
                        linespacing=0.9)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, out)


def plot_bars_per_dataset(top5: pd.DataFrame, per_ds: pd.DataFrame, out: Path):
    """Grouped bar chart: x = dataset, hue = top-5 combo, y = F1 macro.
    Legend placed *above* the chart so it never overlaps the (often near-1.0)
    bars on the easier datasets. Value labels are drawn above each bar with
    a thin white shadow so they read against any bar colour."""
    import matplotlib.patheffects as pe

    combos = [(r["neuron"], r["encoding"]) for _, r in top5.iterrows()]
    combo_labels = [f"{n}/{e}" for n, e in combos]
    n_combos = len(combos)
    x = np.arange(len(DATASETS))
    width = 0.16

    # Slightly taller to fit headroom + above-chart legend
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    label_effect = [pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()]
    for k, ((n, e), lbl) in enumerate(zip(combos, combo_labels)):
        vals = []
        errs = []
        for ds in DATASETS:
            sub = per_ds[(per_ds["neuron"] == n) &
                         (per_ds["encoding"] == e) &
                         (per_ds["dataset"] == ds)]
            vals.append(float(sub["f1m"].values[0]) if len(sub) else 0)
            errs.append(float(sub["f1m_std"].values[0]) if len(sub) else 0)
        offset = (k - (n_combos - 1) / 2) * width
        positions = x + offset
        ax.bar(positions, vals, width, yerr=errs, capsize=2,
               label=lbl, color=WONG_PALETTE[(k + 1) % len(WONG_PALETTE)],
               edgecolor="black", linewidth=0.4)
        # Value labels above each bar (rotated for readability with narrow bars)
        for xi, v, e_ in zip(positions, vals, errs):
            ax.text(xi, v + e_ + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5,
                    rotation=90, color="black",
                    path_effects=label_effect)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_PRETTY[d] for d in DATASETS], fontsize=9)
    ax.set_ylim(0, 1.18)  # extra headroom for rotated labels
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("F1 macro (mean $\\pm$ std over 5 seeds)", fontsize=8.5)
    ax.set_title("Top-5 SNN variants — per-dataset F1 macro", fontsize=10,
                 pad=24)
    # Legend ABOVE the bars, single horizontal row
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10),
              ncol=5, fontsize=7.5, frameon=False, handlelength=1.6,
              handletextpad=0.5, columnspacing=1.0)
    fig.tight_layout()
    save_figure(fig, out)


def plot_full_ranking(full_rank: pd.DataFrame, out: Path):
    """Wide vertical-bar chart: all 27 combos ranked by F1m, top-5 highlighted.
    Horizontal layout to save vertical space on the paper page."""
    df = full_rank.reset_index(drop=True).copy()
    df["combo"] = df["neuron"] + "/" + df["encoding"]
    df = df.sort_values("f1m_mean", ascending=False).reset_index(drop=True)

    enc_color = {"rate": WONG_PALETTE[1], "latency": WONG_PALETTE[2],
                 "delta": WONG_PALETTE[3]}
    colors = [enc_color[e] for e in df["encoding"]]
    n_top5 = 5
    edgewidths = [1.5 if i < n_top5 else 0.4 for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bars = ax.bar(range(len(df)), df["f1m_mean"],
                  yerr=df["f1m_std"], capsize=1.5,
                  color=colors,
                  edgecolor="black", linewidth=edgewidths)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["combo"], rotation=70, ha="right",
                        fontsize=6.5)
    ax.set_ylabel("F1 macro (mean $\\pm$ std)", fontsize=8)
    ax.set_ylim(0.4, 1.0)
    # Horizontal dashed line at the top-5 boundary
    cutoff = df["f1m_mean"].iloc[n_top5 - 1]
    ax.axhline(cutoff, color="black", linestyle="--", linewidth=0.6,
               alpha=0.5)
    ax.text(len(df) - 0.5, cutoff + 0.005, f"top-5 cutoff = {cutoff:.3f}",
            ha="right", va="bottom", fontsize=6.5, color="black", alpha=0.7)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=enc_color[e], edgecolor="black", label=e)
              for e in ["rate", "latency", "delta"]]
    legend.append(Patch(facecolor="none", edgecolor="black", linewidth=1.5,
                        label="top-5"))
    ax.legend(handles=legend, loc="upper right", ncol=4, fontsize=7,
              frameon=False)
    ax.set_title("All 27 SNN variants ranked by cross-dataset F1 macro "
                  "(4 datasets $\\times$ 5 seeds = 20 measurements per variant)",
                  fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, out)


def plot_encoding_effect(df: pd.DataFrame, out: Path):
    """Per-dataset encoding effect — avoids the huge cross-dataset variance
    that swamped the previous merged-into-one chart.

    Layout: 2 rows × 4 cols.
      - Row 1: F1 macro per dataset, grouped by encoding.
      - Row 2: MCC per dataset, grouped by encoding.
    Each bar's std is computed over the 9 neurons × 5 seeds = 45 measurements
    *for that (dataset, encoding) cell*, so the error bars are meaningful
    within-dataset spread rather than cross-dataset difficulty noise.
    """
    import matplotlib.patheffects as pe
    enc_color = {"rate": WONG_PALETTE[1], "latency": WONG_PALETTE[2],
                 "delta": WONG_PALETTE[3]}
    encs = ["rate", "latency", "delta"]
    metrics = [("f1_macro", "F1 macro"),
               ("matthews_corrcoef", "MCC")]
    label_effect = [pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()]

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(7.2, 4.0),
                              sharey="row")
    for col, ds in enumerate(DATASETS):
        sub_ds = df[df["dataset"] == ds]
        for row, (m, lbl) in enumerate(metrics):
            ax = axes[row, col]
            means = []; stds = []
            for enc in encs:
                vals = sub_ds[sub_ds["encoding"] == enc][m]
                means.append(float(vals.mean()))
                stds.append(float(vals.std()))
            bars = ax.bar(encs, means, yerr=stds, capsize=4,
                          color=[enc_color[e] for e in encs],
                          edgecolor="black", linewidth=0.5)
            for b, v, s in zip(bars, means, stds):
                ax.text(b.get_x() + b.get_width() / 2, v + s + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.8,
                        path_effects=label_effect)
            if row == 0:
                ax.set_title(DATASET_PRETTY[ds], fontsize=9)
            if col == 0:
                ax.set_ylabel(lbl, fontsize=8.5)
            # Set y-limits per row so headroom matches the data scale
            if row == 0:  # F1 macro
                ax.set_ylim(0, 1.15)
            else:  # MCC
                ax.set_ylim(0, 1.15)
            ax.tick_params(axis="x", labelsize=7.5)
            ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        "Encoding effect per dataset — std over 9 neurons $\\times$ 5 seeds "
        "(45 measurements per bar)",
        fontsize=9, y=1.01)
    fig.tight_layout()
    save_figure(fig, out)


def plot_efficiency_heatmap(df: pd.DataFrame, metric: str, out: Path,
                              title: str, unit: str = ""):
    """27 (neuron, encoding) combinations × 4 datasets heatmap.

    Rows are sorted by cross-dataset mean (ascending — most efficient first).
    Color uses a log normalisation because energy / inference vary across
    orders of magnitude (Lapicque/rate on NSL-KDD ≈ 5 spikes; SConv2dLSTM/rate
    on CIC-IDS2017 ≈ 2 000 spikes). Annotations report mean values per cell.
    """
    from matplotlib.colors import LogNorm
    # Per-combo, per-dataset mean
    agg = (df.groupby(["neuron", "encoding", "dataset"])[metric]
              .mean().reset_index())
    pivot = agg.pivot_table(index=["neuron", "encoding"], columns="dataset",
                             values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=DATASETS)
    # Sort by cross-dataset mean (ascending — most efficient first)
    pivot["__rank"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__rank").drop(columns="__rank")
    M = pivot.values
    combos = [f"{n}/{e}" for n, e in pivot.index]

    # 27 rows is tall by definition; balance width/height
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    # LogNorm needs strictly positive — small floor to avoid 0
    vmin = max(np.nanmin(M[M > 0]), 1e-3) if (M > 0).any() else 1e-3
    vmax = np.nanmax(M)
    im = ax.imshow(M, aspect="auto", cmap="viridis_r",
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_PRETTY[d] for d in DATASETS], fontsize=8)
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels(combos, fontsize=7)
    # Annotation: print numbers with adaptive precision so labels stay short
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            if v >= 100:    fmt = f"{v:.0f}"
            elif v >= 10:   fmt = f"{v:.1f}"
            elif v >= 1:    fmt = f"{v:.2f}"
            else:           fmt = f"{v:.3f}"
            # White text on dark cells (lower values = darker for viridis_r),
            # black on light cells. Log normalization → threshold at log midpoint.
            log_mid = np.exp((np.log(vmin) + np.log(vmax)) / 2)
            color = "white" if v < log_mid else "black"
            ax.text(j, i, fmt, ha="center", va="center",
                    color=color, fontsize=6.5)
    ax.set_title(title + (f" ({unit})" if unit else ""), fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"mean {metric} (log scale)" + (f" [{unit}]" if unit else ""),
                    fontsize=7.5)
    ax.grid(False)
    fig.tight_layout()
    save_figure(fig, out)


def plot_efficiency_ranking(df: pd.DataFrame, metric: str, out: Path,
                              title: str, ylabel: str, ascending: bool = True):
    """Wide vertical-bar chart: all 27 combos ranked by mean (metric)
    across 4 datasets × 5 seeds. Coloured by encoding."""
    agg = df.groupby(["neuron", "encoding"]).agg(
        m_mean=(metric, "mean"), m_std=(metric, "std")
    ).reset_index()
    agg["combo"] = agg["neuron"] + "/" + agg["encoding"]
    agg = agg.sort_values("m_mean", ascending=ascending).reset_index(drop=True)

    enc_color = {"rate": WONG_PALETTE[1], "latency": WONG_PALETTE[2],
                 "delta": WONG_PALETTE[3]}
    colors = [enc_color[e] for e in agg["encoding"]]
    n_best = 5  # highlight the best (= lowest if ascending) 5 combos
    edgewidths = [1.5 if i < n_best else 0.4 for i in range(len(agg))]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(range(len(agg)), agg["m_mean"], yerr=agg["m_std"],
           capsize=1.5, color=colors, edgecolor="black",
           linewidth=edgewidths)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg["combo"], rotation=70, ha="right", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=8)
    cutoff = agg["m_mean"].iloc[n_best - 1]
    ax.axhline(cutoff, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.text(len(agg) - 0.5, cutoff * 1.05,
             f"top-5 cutoff = {cutoff:.3g}",
             ha="right", va="bottom", fontsize=6.5, color="black", alpha=0.7)
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=enc_color[e], edgecolor="black", label=e)
              for e in ["rate", "latency", "delta"]]
    legend.append(Patch(facecolor="none", edgecolor="black", linewidth=1.5,
                        label="top-5"))
    ax.legend(handles=legend, loc="upper right", ncol=4, fontsize=7,
              frameon=False)
    ax.set_title(title, fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, out)


def plot_pareto(df: pd.DataFrame, out: Path):
    """Two-panel Pareto: F1 macro vs energy (left), F1 macro vs inference (right).
    Each point is a (neuron, encoding) combo averaged across datasets+seeds.
    Colour = encoding; the top-5 by F1 macro are labelled."""
    agg = df.groupby(["neuron", "encoding"]).agg(
        f1m=("f1_macro", "mean"),
        energy=("energy_spikes_per_sample", "mean"),
        infer=("inference_ms_per_sample", "mean"),
    ).reset_index()
    enc_color = {"rate": WONG_PALETTE[1], "latency": WONG_PALETTE[2],
                 "delta": WONG_PALETTE[3]}
    enc_marker = {"rate": "o", "latency": "s", "delta": "^"}

    top5 = agg.sort_values("f1m", ascending=False).head(5)["neuron"].tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    for ax, (x_col, x_label) in zip(
            [ax1, ax2],
            [("energy", "Energy (spikes/sample, log)"),
             ("infer", "Inference (ms/sample, log)")]):
        for enc in ["rate", "latency", "delta"]:
            sub = agg[agg["encoding"] == enc]
            ax.scatter(sub[x_col], sub["f1m"],
                       c=enc_color[enc], marker=enc_marker[enc],
                       edgecolor="black", linewidth=0.4, s=55,
                       label=enc, alpha=0.85)
        # Label top-5
        top = agg.sort_values("f1m", ascending=False).head(5)
        for _, r in top.iterrows():
            ax.annotate(f"{r['neuron'][:4]}", (r[x_col], r["f1m"]),
                         fontsize=6, alpha=0.8, xytext=(3, 3),
                         textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel("F1 macro (cross-dataset mean)", fontsize=8)
        ax.set_ylim(0.55, 0.85)
    ax1.set_title("Accuracy vs Energy", fontsize=9)
    ax2.set_title("Accuracy vs Inference time", fontsize=9)
    ax2.legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("Pareto: F1 macro vs efficiency — top-left is the SNN sweet spot",
                  fontsize=9, y=1.02)
    fig.tight_layout()
    save_figure(fig, out)


def write_latex_efficiency(df: pd.DataFrame, out: Path):
    """Two paper-ready blocks: top-5 most energy-efficient + top-5 fastest."""
    agg = df.groupby(["neuron", "encoding"]).agg(
        energy_mean=("energy_spikes_per_sample", "mean"),
        energy_std=("energy_spikes_per_sample", "std"),
        infer_mean=("inference_ms_per_sample", "mean"),
        infer_std=("inference_ms_per_sample", "std"),
        f1m_mean=("f1_macro", "mean"),
        acc_mean=("accuracy", "mean"),
        sparsity_mean=("spike_sparsity", "mean"),
    ).reset_index()
    top5_energy = agg.sort_values("energy_mean").head(5)
    top5_infer = agg.sort_values("infer_mean").head(5)

    def _rows(top5, value_col, value_std_col, value_fmt):
        out_rows = []
        for _, r in top5.iterrows():
            out_rows.append(
                f"{r['neuron']}/{r['encoding']} & "
                f"{value_fmt(r[value_col])} $\\pm$ {value_fmt(r[value_std_col])} & "
                f"{r['sparsity_mean']:.4f} & "
                f"{r['f1m_mean']:.4f} & {r['acc_mean']:.4f} \\\\"
            )
        return "\n".join(out_rows)

    energy_body = _rows(top5_energy, "energy_mean", "energy_std",
                         lambda v: f"{v:.2f}")
    infer_body = _rows(top5_infer, "infer_mean", "infer_std",
                        lambda v: f"{v:.4f}")

    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\renewcommand{\\arraystretch}{1.1}\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\caption{Top-5 most efficient SNN variants (cross-dataset mean over "
        "4 datasets $\\times$ 5 seeds). \\textbf{Top}: lowest energy "
        "(spikes/sample). \\textbf{Bottom}: lowest inference time "
        "(ms/sample). F1 macro and accuracy of the same variant are reported "
        "for context; quality metrics may be sub-optimal here because the "
        "ranking is by efficiency, not accuracy.}\n"
        "\\label{tab:top5_efficiency}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "\\multicolumn{5}{c}{\\textbf{(a) Top-5 by energy efficiency "
        "(lowest spikes/sample)}} \\\\\n"
        "\\midrule\n"
        "\\textbf{Variant} & \\textbf{Spikes/sample} & "
        "\\textbf{Sparsity} & \\textbf{F1 macro} & \\textbf{Accuracy} \\\\\n"
        "\\midrule\n" + energy_body + "\n"
        "\\midrule\n"
        "\\multicolumn{5}{c}{\\textbf{(b) Top-5 by inference speed "
        "(lowest ms/sample)}} \\\\\n"
        "\\midrule\n"
        "\\textbf{Variant} & \\textbf{ms/sample} & "
        "\\textbf{Sparsity} & \\textbf{F1 macro} & \\textbf{Accuracy} \\\\\n"
        "\\midrule\n" + infer_body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )
    out.write_text(tex)


def _three_top5(df: pd.DataFrame):
    """Compute the three top-5 lists and return (top5_f1, top5_energy, top5_infer)
    each as a list of dicts with rank, variant, neuron, encoding, and the
    relevant metric value."""
    agg = df.groupby(["neuron", "encoding"]).agg(
        f1m=("f1_macro", "mean"),
        energy=("energy_spikes_per_sample", "mean"),
        infer=("inference_ms_per_sample", "mean"),
    ).reset_index()
    agg["variant"] = agg["neuron"] + "/" + agg["encoding"]

    def _take(col, asc):
        sub = agg.sort_values(col, ascending=asc).head(5).reset_index(drop=True)
        return [{"rank": i + 1, "variant": r["variant"], "neuron": r["neuron"],
                  "encoding": r["encoding"], "value": float(r[col])}
                for i, r in sub.iterrows()]

    return _take("f1m", False), _take("energy", True), _take("infer", True)


def plot_top5_intersection(df: pd.DataFrame, out: Path):
    """Three-column figure showing the top-5 lists side by side. Variants
    appearing in multiple lists are highlighted with a coloured background
    so the reader can see the quality/efficiency trade-off at a glance."""
    import matplotlib.patches as mpatches

    top_f1, top_en, top_in = _three_top5(df)
    lists = [
        ("Top-5 by F1 macro",            top_f1, "F1m",    "{:.4f}"),
        ("Top-5 by energy efficiency",   top_en, "spikes/sample", "{:.1f}"),
        ("Top-5 by inference time",      top_in, "ms/sample",     "{:.4f}"),
    ]
    # Count each variant's appearances
    counts = {}
    for _, lst, _, _ in lists:
        for entry in lst:
            counts[entry["variant"]] = counts.get(entry["variant"], 0) + 1

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.axis("off")
    col_xs = [0.18, 0.50, 0.82]
    row_top = 0.74
    row_step = 0.14
    box_w = 0.30
    box_h = 0.12

    # Highlight palette: 3 → green (best all-around), 2 → yellow, 1 → none
    hilite = {3: (0.66, 0.86, 0.55), 2: (0.99, 0.93, 0.55)}

    for col_i, ((header, lst, unit, fmt), x) in enumerate(zip(lists, col_xs)):
        # Header
        ax.text(x, row_top + row_step * 0.62, header, ha="center",
                va="center", fontsize=10, weight="bold",
                transform=ax.transAxes)
        ax.text(x, row_top + row_step * 0.30, f"({unit})", ha="center",
                va="center", fontsize=7.5, style="italic",
                transform=ax.transAxes, color="#444")
        for k, entry in enumerate(lst):
            y = row_top - k * row_step
            cnt = counts[entry["variant"]]
            box_fc = hilite.get(cnt, (0.97, 0.97, 0.97))
            box = mpatches.FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w, box_h,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                linewidth=(1.2 if cnt >= 2 else 0.6),
                edgecolor=("black" if cnt >= 2 else "#888"),
                facecolor=box_fc, transform=ax.transAxes)
            ax.add_patch(box)
            # Rank number — small, top-left of box
            ax.text(x - box_w / 2 + 0.014, y + box_h / 2 - 0.018,
                     f"{entry['rank']}",
                     ha="left", va="top", fontsize=7.5, weight="bold",
                     transform=ax.transAxes, color="#666")
            # "in 2/3" badge — top-right corner
            if cnt >= 2:
                ax.text(x + box_w / 2 - 0.014, y + box_h / 2 - 0.018,
                         f"★ {cnt}/3",
                         ha="right", va="top", fontsize=7, weight="bold",
                         transform=ax.transAxes,
                         color="#1a6b1a" if cnt == 3 else "#8a6500")
            # Variant name — centered
            ax.text(x, y + 0.008, entry["variant"],
                    ha="center", va="center", fontsize=9,
                    transform=ax.transAxes,
                    weight=("bold" if cnt >= 2 else "normal"))
            # Metric value — below variant name
            ax.text(x, y - 0.026, fmt.format(entry["value"]),
                    ha="center", va="center", fontsize=8,
                    transform=ax.transAxes, color="#222")

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=hilite[3], edgecolor="black",
                        label="appears in all 3 top-5 lists (no variant)"),
        mpatches.Patch(facecolor=hilite[2], edgecolor="black",
                        label="appears in 2 of 3 top-5 lists"),
        mpatches.Patch(facecolor=(0.97, 0.97, 0.97), edgecolor="#888",
                        label="appears in 1 of 3 top-5 lists"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
                bbox_to_anchor=(0.5, -0.02), ncol=3,
                fontsize=7.5, frameon=False)
    fig.suptitle("Top-5 SNN variants across three ranking criteria "
                  "(highlighted = appears in more than one list)",
                  fontsize=10, y=0.98)
    save_figure(fig, out)


def write_latex_intersection(df: pd.DataFrame, out: Path):
    """Single side-by-side LaTeX table with the three top-5 lists,
    using \\cellcolor highlighting to show variants appearing in 2 or 3
    of the lists."""
    top_f1, top_en, top_in = _three_top5(df)
    # Count appearances per variant
    counts = {}
    for lst in (top_f1, top_en, top_in):
        for entry in lst:
            counts[entry["variant"]] = counts.get(entry["variant"], 0) + 1

    def _cell(entry):
        cnt = counts[entry["variant"]]
        bold = "\\textbf{" if cnt >= 2 else ""
        end_bold = "}" if cnt >= 2 else ""
        if cnt == 3:
            color = "\\cellcolor{paretogreen}"
        elif cnt == 2:
            color = "\\cellcolor{paretoyellow}"
        else:
            color = ""
        return f"{color}{bold}{entry['variant']}{end_bold}"

    def _val(entry, unit_fmt):
        return unit_fmt.format(entry["value"])

    rows = []
    for k in range(5):
        rows.append(
            f"{k + 1} & {_cell(top_f1[k])} & {_val(top_f1[k], '{:.4f}')} & "
            f"{_cell(top_en[k])} & {_val(top_en[k], '{:.1f}')} & "
            f"{_cell(top_in[k])} & {_val(top_in[k], '{:.4f}')} \\\\"
        )
    body = "\n".join(rows)

    # Notes block summarising which variants appear in 2 lists
    in2 = sorted(v for v, c in counts.items() if c == 2)
    in3 = sorted(v for v, c in counts.items() if c == 3)
    note_lines = []
    if in3:
        note_lines.append(
            f"Variants appearing in \\emph{{all three}} top-5 lists "
            f"(highlighted \\colorbox{{paretogreen}}{{green}}): "
            f"\\texttt{{{', '.join(in3)}}}."
        )
    else:
        note_lines.append(
            "No single variant appears in all three top-5 lists, reflecting "
            "the quality-vs-efficiency trade-off."
        )
    if in2:
        note_lines.append(
            f"Variants appearing in \\emph{{two}} of the three lists "
            f"(highlighted \\colorbox{{paretoyellow}}{{yellow}}): "
            + ", ".join(f"\\texttt{{{v}}}" for v in in2) + "."
        )

    notes = " ".join(note_lines)

    tex = (
        "% Requires: \\usepackage[table]{xcolor}\n"
        "\\definecolor{paretogreen}{RGB}{168,219,140}\n"
        "\\definecolor{paretoyellow}{RGB}{252,237,140}\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{c|ll|ll|ll}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c|}{\\textbf{Top-5 by F1 macro}} "
        "& \\multicolumn{2}{c|}{\\textbf{Top-5 by energy "
        "(spikes/sample)}} & \\multicolumn{2}{c}{"
        "\\textbf{Top-5 by inference (ms/sample)}} \\\\\n"
        "\\textbf{Rank} & \\textbf{Variant} & \\textbf{F1 macro} "
        "& \\textbf{Variant} & \\textbf{Energy} "
        "& \\textbf{Variant} & \\textbf{Inference} \\\\\n"
        "\\midrule\n" + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\caption{Top-5 SNN variants under three different ranking criteria "
        "(quality, energy, inference speed) — each computed as the "
        "cross-dataset mean over 4 datasets $\\times$ 5 seeds = "
        "20 measurements per variant. " + notes + "}\n"
        "\\label{tab:top5_intersection}\n"
        "\\end{table*}\n"
    )
    out.write_text(tex)


def write_latex_consistency(df: pd.DataFrame, out: Path):
    """Build the wide consistency table.

    Rows = the 12 unique variants that appear in any of the three top-5 lists.
    Columns = 3 metrics × 4 datasets = 12 numeric columns, plus a final
    Score column = number of (dataset, metric) cells in which the variant
    ranks in the top-3 within the 12-variant universe.

      * Accuracy ↑     — higher is better
      * Spikes/sample ↓ — lower is better (proxy for spike sparsity request)
      * Inference (ms) ↓ — lower is better

    Highlighting:
      * The single best variant in each column gets a green cell.
      * Variants ranked 2nd or 3rd in each column get yellow cells.
      * The Score column is bold-faced; the top-3 variants by overall score
        get a green-tinted row (their variant name is in green).
    """
    # Build the universe = union of top-5 by F1m, energy, inference
    top_f1, top_en, top_in = _three_top5(df)
    universe = set()
    for lst in (top_f1, top_en, top_in):
        for entry in lst:
            universe.add(entry["variant"])
    universe = sorted(universe)

    # Per-(variant, dataset) means
    agg = df.groupby(["neuron", "encoding", "dataset"]).agg(
        accuracy=("accuracy", "mean"),
        energy=("energy_spikes_per_sample", "mean"),
        infer=("inference_ms_per_sample", "mean"),
    ).reset_index()
    agg["variant"] = agg["neuron"] + "/" + agg["encoding"]
    agg = agg[agg["variant"].isin(universe)].copy()

    METRICS = [
        ("accuracy", "Accuracy $\\uparrow$",       True,  "{:.4f}"),
        ("energy",   "Spikes/sample $\\downarrow$", False, "{:.1f}"),
        ("infer",    "Inference (ms) $\\downarrow$", False, "{:.4f}"),
    ]

    # Compute ranks for each (dataset, metric_col) within the universe
    rank = {}  # rank[(variant, ds, col)] = 1..12 (1 = best)
    for ds in DATASETS:
        ds_sub = agg[agg["dataset"] == ds]
        for col, _, higher_better, _ in METRICS:
            sorted_var = ds_sub.sort_values(col, ascending=not higher_better)
            for r, v in enumerate(sorted_var["variant"].tolist()):
                rank[(v, ds, col)] = r + 1

    # Consistency score per variant: # of cells where it ranks in top-3
    score = {}
    for v in universe:
        score[v] = sum(1 for ds in DATASETS for col, _, _, _ in METRICS
                       if rank.get((v, ds, col), 99) <= 3)
    # Sort universe by score (desc), then by variant name (asc) as tiebreak
    universe = sorted(universe, key=lambda v: (-score[v], v))
    max_score = 4 * 3  # 4 datasets × 3 metrics = 12 cells

    # Identify which variants get the green "consistent top" row treatment.
    # With 12 cells × top-3 / 12 variants in universe, the expected hits per
    # variant is 3 (purely random). We highlight variants well above this:
    #   green  : ≥ 6/12 (twice random expectation)
    #   yellow : ≥ 4/12 (above random expectation)
    green_threshold = 6
    yellow_threshold = 4

    # ---- Build cell strings ----
    def _fmt_cell(v, ds, col, fmt):
        val = agg[(agg["variant"] == v) & (agg["dataset"] == ds)][col]
        if len(val) == 0:
            return "---"
        x = float(val.values[0])
        r = rank[(v, ds, col)]
        s = fmt.format(x)
        if r == 1:
            return f"\\cellcolor{{paretogreen}}\\textbf{{{s}}}"
        if r in (2, 3):
            return f"\\cellcolor{{paretoyellow}}\\textbf{{{s}}}"
        return s

    body_rows = []
    for v in universe:
        cells = []
        for col, _, _, fmt in METRICS:
            for ds in DATASETS:
                cells.append(_fmt_cell(v, ds, col, fmt))
        sc = score[v]
        sc_cell = f"\\textbf{{{sc}/{max_score}}}"
        if sc >= green_threshold:
            row_color = "\\rowcolor{paretogreen!40}"
            variant_label = f"\\textbf{{{v}}}"
        elif sc >= yellow_threshold:
            row_color = "\\rowcolor{paretoyellow!40}"
            variant_label = v
        else:
            row_color = ""
            variant_label = v
        body_rows.append(f"{row_color}{variant_label} & "
                          + " & ".join(cells) + f" & {sc_cell} \\\\")
    body = "\n".join(body_rows)

    # ---- Header rows ----
    metric_headers = " & ".join(
        f"\\multicolumn{{4}}{{c|}}{{\\textbf{{{label}}}}}"
        for _, label, _, _ in METRICS[:-1]
    ) + f" & \\multicolumn{{4}}{{c|}}{{\\textbf{{{METRICS[-1][1]}}}}}"
    dataset_headers = " & ".join(
        DATASET_PRETTY[d] for _ in METRICS for d in DATASETS
    )

    tex = (
        "% Requires: \\usepackage[table]{xcolor}\n"
        "\\definecolor{paretogreen}{RGB}{168,219,140}\n"
        "\\definecolor{paretoyellow}{RGB}{252,237,140}\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l|cccc|cccc|cccc|c}\n"
        "\\toprule\n"
        "\\textbf{Variant} & " + metric_headers + " & "
        "\\multirow{2}{*}{\\textbf{Score}} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n"
        " & " + dataset_headers + " & \\\\\n"
        "\\midrule\n" + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\caption{Per-dataset consistency of the "
        f"{len(universe)} SNN variants drawn from the union of the three "
        "top-5 lists (F1 macro, energy, inference time). Each numeric cell is "
        "the cross-seed mean of that variant on that dataset (5 seeds). "
        "Within each column, the single best variant is highlighted "
        "\\colorbox{paretogreen}{green}; 2nd and 3rd are \\colorbox{paretoyellow}{yellow}. "
        "The \\textbf{Score} column counts the number of "
        f"(dataset, metric) cells (out of {max_score}) in which the "
        "variant ranks in the top-3 within this universe. Rows with "
        f"Score $\\geq {green_threshold}$ are shaded "
        "\\colorbox{paretogreen!40}{green} (consistently strong across "
        "criteria); rows with Score "
        f"$\\geq {yellow_threshold}$ are shaded "
        "\\colorbox{paretoyellow!40}{yellow}.}\n"
        "\\label{tab:consistency}\n"
        "\\end{table*}\n"
    )
    out.write_text(tex)


def plot_radar_top5(top5: pd.DataFrame, out: Path):
    """Radar / spider chart: top-5 combos across 6 metrics."""
    # 6-axis radar — choose the most informative IEEE-friendly set
    axes_metrics = [
        ("acc_mean",  "Accuracy"),
        ("f1m_mean",  "F1 macro"),
        ("prec_mean", "Precision"),
        ("rec_mean",  "Recall"),
        ("auc_mean",  "ROC-AUC"),
        ("mcc_mean",  "MCC"),
    ]
    labels = [a[1] for a in axes_metrics]
    n_axes = len(axes_metrics)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={"projection": "polar"})
    for i, (_, r) in enumerate(top5.iterrows()):
        vals = [r[m] for m, _ in axes_metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=WONG_PALETTE[(i + 1) % len(WONG_PALETTE)],
                linewidth=1.5, label=f"{r['neuron']}/{r['encoding']}")
        ax.fill(angles, vals, color=WONG_PALETTE[(i + 1) % len(WONG_PALETTE)],
                alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=7)
    ax.set_title("Top-5 SNN variants — metric profile (cross-dataset mean)",
                 pad=18, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.05), fontsize=7,
              frameon=False)
    save_figure(fig, out)


# ---------------------------------------------------------------- LaTeX tables

def write_latex_top5(top5: pd.DataFrame, out: Path):
    """Cross-dataset top-5 with mean ± std for every metric."""
    rows = []
    for _, r in top5.iterrows():
        rows.append(
            f"{r['neuron']}/{r['encoding']} & "
            f"{r['acc_mean']:.4f} $\\pm$ {r['acc_std']:.4f} & "
            f"{r['bal_mean']:.4f} $\\pm$ {r['bal_std']:.4f} & "
            f"{r['f1m_mean']:.4f} $\\pm$ {r['f1m_std']:.4f} & "
            f"{r['prec_mean']:.4f} $\\pm$ {r['prec_std']:.4f} & "
            f"{r['rec_mean']:.4f} $\\pm$ {r['rec_std']:.4f} & "
            f"{r['auc_mean']:.4f} $\\pm$ {r['auc_std']:.4f} & "
            f"{r['mcc_mean']:.4f} $\\pm$ {r['mcc_std']:.4f} & "
            f"{r['kappa_mean']:.4f} $\\pm$ {r['kappa_std']:.4f} & "
            f"{r['dr_mean']:.4f} $\\pm$ {r['dr_std']:.4f} & "
            f"{r['far_mean']:.4f} $\\pm$ {r['far_std']:.4f} \\\\"
        )
    body = "\n".join(rows)
    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l|cccccccccc}\n"
        "\\toprule\n"
        "\\textbf{Variant} & \\textbf{Accuracy} & \\textbf{Bal. Acc.} & "
        "\\textbf{F1 (macro)} & \\textbf{Precision} & \\textbf{Recall} & "
        "\\textbf{ROC-AUC} & \\textbf{MCC} & \\textbf{Cohen's $\\kappa$} & "
        "\\textbf{DR} & \\textbf{FAR} \\\\\n"
        "\\midrule\n" + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\caption{Top-5 SNN (neuron, encoding) variants ranked by mean "
        "F1 macro across the four datasets (NSL-KDD, KDDCup99, CIC-IDS2017, "
        "CTU-13) and five random seeds (20 measurements per variant). "
        "All values are mean $\\pm$ standard deviation. "
        "DR = Detection Rate, FAR = False Alarm Rate.}\n"
        "\\label{tab:top5_cross_dataset}\n"
        "\\end{table*}\n"
    )
    out.write_text(tex)


def write_latex_per_dataset(top5: pd.DataFrame, per_ds: pd.DataFrame, out: Path):
    """For each of the top-5 variants, show every metric per dataset as
    mean ± std (across the 5 seeds), so robustness across seeds is explicit."""
    body_rows = []
    for _, t in top5.iterrows():
        first = True
        for ds in DATASETS:
            sub = per_ds[(per_ds["neuron"] == t["neuron"]) &
                          (per_ds["encoding"] == t["encoding"]) &
                          (per_ds["dataset"] == ds)]
            if len(sub) == 0:
                continue
            s = sub.iloc[0]
            variant = (f"\\multirow{{4}}{{*}}{{{t['neuron']}/{t['encoding']}}}"
                       if first else "")
            first = False
            body_rows.append(
                f"{variant} & {DATASET_PRETTY[ds]} & "
                f"{s['acc']:.4f} $\\pm$ {s['acc_std']:.4f} & "
                f"{s['f1m']:.4f} $\\pm$ {s['f1m_std']:.4f} & "
                f"{s['mcc']:.4f} $\\pm$ {s['mcc_std']:.4f} & "
                f"{s['auc']:.4f} $\\pm$ {s['auc_std']:.4f} & "
                f"{s['dr']:.4f} $\\pm$ {s['dr_std']:.4f} & "
                f"{s['far']:.4f} $\\pm$ {s['far_std']:.4f} \\\\"
            )
        body_rows.append("\\midrule")
    # Remove trailing midrule
    while body_rows and body_rows[-1] == "\\midrule":
        body_rows.pop()
    body = "\n".join(body_rows)
    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\renewcommand{\\arraystretch}{1.1}\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{llcccccc}\n"
        "\\toprule\n"
        "\\textbf{Variant} & \\textbf{Dataset} & \\textbf{Accuracy} & "
        "\\textbf{F1 macro} & \\textbf{MCC} & \\textbf{ROC-AUC} & "
        "\\textbf{DR} & \\textbf{FAR} \\\\\n"
        "\\midrule\n" + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\caption{Per-dataset performance of the top-5 cross-dataset variants. "
        "All values are mean $\\pm$ standard deviation across the 5 random seeds. "
        "Standard deviations of 0.0000 indicate exact reproducibility across all "
        "5 seeds (typical for CTU-13 with our host-aggregated features). "
        "DR = Detection Rate, FAR = False Alarm Rate.}\n"
        "\\label{tab:top5_per_dataset}\n"
        "\\end{table*}\n"
    )
    out.write_text(tex)


# ---------------------------------------------------------------- main

def main():
    apply_ieee_style()
    out_dir = REPO / "results" / "conference"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results()
    df.to_csv(out_dir / "_all_runs_raw.csv", index=False)

    top5, full_rank = cross_dataset_ranking(df, k=5)
    full_rank.to_csv(out_dir / "_full_27combo_ranking.csv", index=False)
    top5.to_csv(out_dir / "top5_cross_dataset.csv", index=False)

    per_ds = per_dataset_for_top5(df, top5)
    per_ds.to_csv(out_dir / "top5_per_dataset.csv", index=False)

    print("\n=== TOP 5 (cross-dataset, ranked by mean F1 macro) ===")
    for i, r in top5.iterrows():
        print(f"  {i+1}. {r['neuron']}/{r['encoding']:8s}  "
              f"F1m={r['f1m_mean']:.4f}±{r['f1m_std']:.4f}  "
              f"acc={r['acc_mean']:.4f}  MCC={r['mcc_mean']:.4f}  "
              f"AUC={r['auc_mean']:.4f}  DR={r['dr_mean']:.4f}  "
              f"FAR={r['far_mean']:.4f}")

    plot_heatmap_top5(top5, per_ds, "f1m", out_dir / "heatmap_top5_f1m",
                       "Top-5 SNN variants — F1 macro per dataset")
    plot_heatmap_top5(top5, per_ds, "mcc", out_dir / "heatmap_top5_mcc",
                       "Top-5 SNN variants — MCC per dataset")
    plot_heatmap_top5(top5, per_ds, "auc", out_dir / "heatmap_top5_auc",
                       "Top-5 SNN variants — ROC-AUC per dataset")
    plot_bars_per_dataset(top5, per_ds, out_dir / "bars_top5_per_dataset")
    plot_radar_top5(top5, out_dir / "radar_top5")
    plot_full_ranking(full_rank, out_dir / "full_ranking_27combos")
    plot_encoding_effect(df, out_dir / "encoding_effect")

    # Efficiency analyses (energy + inference time)
    plot_efficiency_heatmap(df, "energy_spikes_per_sample",
                             out_dir / "heatmap_energy_27x4",
                             "Energy proxy — mean spikes per sample",
                             unit="spikes")
    plot_efficiency_heatmap(df, "inference_ms_per_sample",
                             out_dir / "heatmap_inference_27x4",
                             "Inference time — mean ms per sample",
                             unit="ms")
    plot_efficiency_ranking(df, "energy_spikes_per_sample",
                             out_dir / "ranking_energy_27combos",
                             "Energy efficiency — all 27 SNN variants "
                             "ranked by mean spikes per sample (lower is better)",
                             ylabel="Energy (spikes/sample, mean $\\pm$ std)",
                             ascending=True)
    plot_efficiency_ranking(df, "inference_ms_per_sample",
                             out_dir / "ranking_inference_27combos",
                             "Inference time — all 27 SNN variants "
                             "ranked by mean ms per sample (lower is better)",
                             ylabel="Inference (ms/sample, mean $\\pm$ std)",
                             ascending=True)
    # Pareto plot — accuracy vs energy and accuracy vs inference time
    plot_pareto(df, out_dir / "pareto_accuracy_vs_efficiency")
    # LaTeX table — top-5 most efficient + top-5 fastest
    write_latex_efficiency(df, out_dir / "efficiency_table.tex")
    # Top-5 across (accuracy, energy, inference) — intersection analysis
    plot_top5_intersection(df, out_dir / "top5_intersection")
    write_latex_intersection(df, out_dir / "intersection_table.tex")
    # Per-dataset consistency table (the request that motivated this section)
    write_latex_consistency(df, out_dir / "consistency_table.tex")

    write_latex_top5(top5, out_dir / "top5_table.tex")
    write_latex_per_dataset(top5, per_ds, out_dir / "per_dataset_table.tex")

    print(f"\nwrote → {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
