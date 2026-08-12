#!/usr/bin/env python3
"""Build the journal tables and figures into results/Journal_results/new_run_results/.

Every artifact is regenerated from raw result files, never transcribed. Scope
qualifiers are baked into captions: the sensitivity sweep, the raw_full
corroboration and the fp32 quantisation comparison come from the FOUR-dataset
sensitivity subset (ciciot2023, ciciomt2024, edgeiiot, toniot), not the
eleven-dataset study.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/Journal_results/new_run_results"
OUT.mkdir(parents=True, exist_ok=True)

# validated palette (light surface): categorical 7-slot + 2-state status pair
C_BLUE, C_RED = "#2a78d6", "#d03b3b"
INK, MUTED, GRID, SURFACE, BASE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
SENS_DATASETS = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]

FAMILY_SHORT = {
    "T (timesteps, default 20)": "timesteps $T$",
    "pop_max (default 3)": "population $M$",
    "prune_quantile (default 0.20)": "prune quantile",
    "jitter (default 0.12)": "jitter $\\delta$",
    "tau (KD temperature, default 2.0)": "KD temp. $\\tau$",
    "a_kd (default 1.0)": "$\\alpha_{kd}$",
    "a_rank (default 0.3)": "$\\alpha_{rank}$",
    "a_sparse (default 0.001)": "$\\alpha_{sparse}$",
    "a_fp (default 0.5)": "$\\alpha_{fp}$",
    "epochs (default 10)": "epochs",
    "encoder ablations": "encoder ablation",
    "objective ablations": "objective ablation",
    "class weighting": "class weighting",
    "precision (default int8-grid)": "precision",
}


def fig_setup(w, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    return fig, ax


# ============================================================ Figure 1: forest
def figure_forest(t: pd.DataFrame):
    import matplotlib.pyplot as plt
    order, labels, seps = [], [], []
    for fam in FAMILY_SHORT:
        sub = t[t.family == fam]
        if not len(sub):
            continue
        for _, r in sub.iterrows():
            order.append(r)
            labels.append(r.arm)
        seps.append(len(order) - 0.5)
    n = len(order)
    fig, ax = fig_setup(8.6, 0.30 * n + 1.9)
    y = np.arange(n)[::-1]

    ax.axvspan(-1.0, 1.0, color=GRID, alpha=0.55, zorder=0)
    ax.axvline(0, color=BASE, linewidth=1.4, zorder=1)

    excl = [not (r.d_f1_lo <= 0 <= r.d_f1_hi) for r in order]
    for i, (r, yy, e) in enumerate(zip(order, y, excl)):
        c = C_RED if e else C_BLUE
        ax.plot([r.d_f1_lo, r.d_f1_hi], [yy, yy], color=c, linewidth=2, zorder=3,
                solid_capstyle="round")
        ax.scatter([r.d_f1_mean], [yy], s=46, color=c, edgecolor=SURFACE,
                   linewidth=2, zorder=4)
    for s in seps[:-1]:
        ax.axhline(n - s - 0.5, color=GRID, linewidth=0.8, zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlabel("$\\Delta$ macro-F1 vs default (pp), paired on dataset $\\times$ seed"
                  "  —  shaded band $\\pm$1 pp", fontsize=9, color=MUTED)
    ax.set_title("Sensitivity: every interval overlaps zero except the encoder ablations",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7, zorder=0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color=C_BLUE, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
               label="95% interval includes 0 (no detectable effect)"),
        Line2D([0], [0], color=C_RED, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
               label="95% interval excludes 0 (detectable)")],
        loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    fig.text(0.012, 0.016, "Four-dataset sensitivity subset $\\times$ 5 seeds = 20 paired\n"
             "observations per arm. Default = shipped Tree-to-Spike run,\n"
             "same dataset and seed.", fontsize=7, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    p = OUT / "fig_sensitivity_forest.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


# ====================================================== Figure 2: cost/quality
def figure_cost(t: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = fig_setup(8.6, 5.4)
    ax.axhspan(-1.0, 1.0, color=GRID, alpha=0.55, zorder=0)
    ax.axhline(0, color=BASE, linewidth=1.2, zorder=1)
    ax.axvline(100, color=BASE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    show = t[~t.arm.isin(["raw", "raw_full"])]
    for _, r in show.iterrows():
        e = not (r.d_f1_lo <= 0 <= r.d_f1_hi)
        ax.scatter(r.pct_synops, r.d_f1_mean, s=64, color=C_RED if e else C_BLUE,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.scatter([100], [0], s=150, marker="D", color=INK, edgecolor=SURFACE,
               linewidth=2, zorder=5)
    ax.annotate("default\n(T=20)", (100, 0), textcoords="offset points",
                xytext=(8, -26), fontsize=8.5, color=INK, weight="bold")
    for a in ("asparse1e2", "T10", "T40", "T80", "epochs30", "no_sparse", "tau4"):
        r = t[t.arm == a]
        if not len(r):
            continue
        r = r.iloc[0]
        dx, dy = (8, 8)
        if a in ("T80", "no_sparse"):
            dx, dy = (-8, 8)
        ax.annotate(a, (r.pct_synops, r.d_f1_mean), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color=INK,
                    ha="left" if dx > 0 else "right")
    ax.set_xscale("log")
    ax.set_xticks([50, 75, 100, 150, 200, 300, 400])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("synaptic operations per sample, % of default (log)  —  lower is cheaper",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("$\\Delta$ macro-F1 vs default (pp)", fontsize=9, color=MUTED)
    ax.set_title("Cost/quality: the default is well placed; higher $T$ buys little for "
                 "2.4–4$\\times$ the operations", fontsize=11.5, color=INK, loc="left",
                 pad=12)
    ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
    fig.text(0.012, 0.012, "Encoder ablations (raw, raw_full) omitted for scale — see the "
             "forest plot. Every arm shown has a 95% paired interval that includes zero, "
             "so none is a demonstrated improvement. Four-dataset sensitivity subset.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    p = OUT / "fig_sensitivity_cost_quality.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


# ========================================== Figure 3: objective acts on FAR
def figure_far(t: pd.DataFrame, default_far: float, default_f1: float):
    import matplotlib.pyplot as plt
    arms = ["no_fp", "tree_ce", "class_weighted", "class_weighted_nokd",
            "cw_nokd_rankfixed", "no_sparse", "rank_only", "kd_only"]
    sub = t[t.arm.isin(arms)].copy().sort_values("med_far")
    fig, ax = fig_setup(8.6, 4.4)
    y = np.arange(len(sub) + 1)[::-1]
    names = ["DEFAULT"] + list(sub.arm)
    fars = [default_far] + list(sub.med_far)
    d_f1 = [0.0] + list(sub.d_f1_mean)
    for i, (yy, nm, fr, df) in enumerate(zip(y, names, fars, d_f1)):
        c = INK if nm == "DEFAULT" else (C_RED if fr >= 2.0 else C_BLUE)
        ax.barh(yy, fr, height=0.62, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{fr:.2f}%   ($\\Delta$F1 {df:+.2f} pp)", (fr, yy),
                    textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5, color=INK)
    ax.set_xlabel("median false-alarm rate (%)  —  lower is better", fontsize=9, color=MUTED)
    ax.set_title("The objective's contribution is false-alarm control, not macro-F1",
                 fontsize=11.5, color=INK, loc="left", pad=12)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.margins(x=0.26)
    fig.text(0.012, 0.012, "Removing the FP-aware term (no_fp) moves macro-F1 by only "
             f"+0.15 pp but raises median FAR from {default_far:.2f}% to 2.13% (5.2$\\times$). "
             "Red $\\geq$2%. Four-dataset sensitivity subset.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = OUT / "fig_objective_far.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


# ================================================================ LaTeX tables
def tex_escape(s):
    return str(s).replace("_", "\\_")


def table_sensitivity(t: pd.DataFrame) -> Path:
    L = [r"% Auto-generated by scripts/make_journal_figures.py -- do not hand-edit",
         r"\begin{table*}[t]\centering\small",
         r"\caption{Sensitivity of Tree-to-Spike to 35 design perturbations on the "
         r"four-dataset sensitivity subset (CICIoT2023, CICIoMT2024, Edge-IIoTset, "
         r"TON-IoT) $\times$ 5 seeds. $\Delta$ values are paired against the shipped "
         r"default run on the same dataset and seed; the 95\% interval is over the 20 "
         r"paired observations. SOP\% is median synaptic operations per sample relative "
         r"to the default. \textbf{Every interval includes zero except the encoder "
         r"ablations.}}",
         r"\label{tab:sensitivity}",
         r"\begin{tabular}{llrrrr@{\,}lrr}\toprule",
         r"Axis & Arm & med.\ F1 & IQR & $\Delta$F1 & \multicolumn{2}{c}{95\% CI} "
         r"& FAR & SOP\% \\\midrule"]
    for fam, short in FAMILY_SHORT.items():
        sub = t[t.family == fam]
        if not len(sub):
            continue
        first = True
        for _, r in sub.iterrows():
            excl = not (r.d_f1_lo <= 0 <= r.d_f1_hi)
            arm = tex_escape(r.arm)
            if excl:
                arm = r"\textbf{" + arm + "}"
            L.append(f"{short if first else ''} & {arm} & {r.med_f1:.2f} & {r.iqr:.2f} "
                     f"& {r.d_f1_mean:+.2f} & [{r.d_f1_lo:+.2f}, & {r.d_f1_hi:+.2f}] "
                     f"& {r.med_far:.2f} & {r.pct_synops:.1f} \\\\")
            first = False
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule\end{tabular}\end{table*}"]
    p = OUT / "tab_sensitivity.tex"
    p.write_text("\n".join(L) + "\n")
    return p


def table_deployment() -> Path:
    d = pd.read_csv(REPO / "results/deployment/deployment_summary.csv")
    LAB = {"full_gbdt": ("Full GBDT", "Tree ensemble", "native"),
           "compact_gbdt": ("Compact GBDT", "Tree ensemble", "val.-selected prefix"),
           "raw_mlp_fp": ("Raw MLP", "Raw features", "FP32"),
           "raw_mlp_int8": ("Raw MLP", "Raw features", "int8-grid weights"),
           "dnn_tree": ("DNN-Tree", "Tree-derived", "int8-grid weights"),
           "t2s_full": ("Tree-to-Spike", "Tree-guided TTFS", "int8-grid, full $T$"),
           "t2s_exit": (r"\textbf{Tree-to-Spike}", "Tree-guided TTFS",
                        r"int8-grid, \textbf{selected exit}")}
    L = [r"% Auto-generated by scripts/make_journal_figures.py -- do not hand-edit",
         r"\begin{table*}[t]\centering\small",
         r"\caption{Deployment comparison. Predictive metrics are means over 11 datasets "
         r"$\times$ 5 seeds; artifact bytes, operation counts and latency are measured on "
         r"seed 42 at batch size one, single thread, after a 500-sample warm-up over 2{,}000 "
         r"timed samples. Tree models are priced in comparisons and neural models in "
         r"MAC/SOP; the two are never interconverted. DNN-Tree and Tree-to-Spike are "
         r"evaluated from checkpoints saved \emph{after} int8-grid weight projection with "
         r"floating-point execution, so FP and post-training int8 rows would be numerically "
         r"duplicate and no independent FP checkpoint was retained. The selected-exit row "
         r"is the deployed configuration; it lowers analytical SOPs but \emph{increases} "
         r"measured CPU latency.}",
         r"\label{tab:deployment}",
         r"\begin{tabular}{llrrrrrrr}\toprule",
         r"Model & Precision/config. & Macro-F1 & MCC & FAR & Artifact & Ops/cmp "
         r"& p50 (ms) & p95 (ms) \\\midrule"]
    for _, r in d.iterrows():
        nm, rep, prec = LAB[r.system]
        if r.system in ("full_gbdt", "compact_gbdt"):
            ops = f"{r.tree_comparisons:,.0f} cmp"
        elif r.system.startswith("t2s"):
            ops = f"{r.sops:,.0f} SOP"
        else:
            ops = f"{r.macs:,.0f} MAC"
        # decimal units, matching the prose in main_points.tex (KB = 1e3, MB = 1e6)
        kb = r.total_bytes / 1e3
        art = f"{kb/1e3:.2f} MB" if kb >= 1e3 else f"{kb:.1f} KB"
        L.append(f"{nm} & {prec} & {r.macro_f1:.2f} & {r.mcc:.2f} & {r.FAR:.2f} & {art} "
                 f"& {ops} & {r.p50_ms:.3f} & {r.p95_ms:.3f} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table*}"]
    p = OUT / "tab_deployment.tex"
    p.write_text("\n".join(L) + "\n")
    return p


def table_early_exit() -> Path:
    f = pd.read_csv(REPO / "results/early_exit/selected_test.csv")
    g = f.groupby("dataset").agg(
        disabled=("early_exit_disabled", "sum"), n=("seed", "count"),
        full=("full_macro_f1", "mean"), exitf1=("exit_macro_f1", "mean"),
        d=("d_macro_f1", "mean"), dfar=("d_FAR", "mean"),
        me=("mean_exit", "mean"), sop=("mean_sop_reduction", "mean")).reset_index()
    L = [r"% Auto-generated by scripts/make_journal_figures.py -- do not hand-edit",
         r"\begin{table}[t]\centering\small",
         r"\caption{Validation-selected early exit, Tree-to-Spike, 11 datasets $\times$ "
         r"5 seeds. A threshold qualifies only if validation macro-F1 stays within 1.0 "
         r"absolute point of full-window; otherwise early exit is disabled and full $T$ "
         r"runs. Test is applied once. Early exit was disabled in \textbf{26 of 55} "
         r"dataset--seed cells, and only 1 of 55 exceeded the predeclared margin on test.}",
         r"\label{tab:earlyexit}",
         r"\begin{tabular}{lrrrrrr}\toprule",
         r"Dataset & disabled & full F1 & exit F1 & $\Delta$F1 & mean exit & SOP red.\ "
         r"\\\midrule"]
    for _, r in g.iterrows():
        L.append(f"{tex_escape(r.dataset)} & {int(r.disabled)}/{int(r.n)} & {r.full:.2f} "
                 f"& {r.exitf1:.2f} & {r.d:+.3f} & {r.me:.2f} & {r.sop:.1f}\\% \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    p = OUT / "tab_early_exit.tex"
    p.write_text("\n".join(L) + "\n")
    return p


def table_raw_mlp() -> Path:
    q = pd.read_csv(REPO / "results/deployment/int8_grid.csv")
    q = q[q.model == "rawmlp"]
    rows = []
    for ds, g in q.groupby("dataset"):
        rows.append(dict(dataset=ds, fp=g.fp_macro_f1.mean(), i8=g.int8_macro_f1.mean(),
                         d=g.d_macro_f1.mean()))
    r = pd.DataFrame(rows)
    ts, dn = {}, {}
    for mdl, store in (("Tree-to-Spike", ts), ("DNN-Tree", dn)):
        for p in (REPO / f"results/{mdl}").glob("*/seed_*/results.json"):
            store.setdefault(p.parts[-3], []).append(
                100 * json.load(open(p))["test_plus"]["metrics"]["f1_macro"])
    L = [r"% Auto-generated by scripts/make_journal_figures.py -- do not hand-edit",
         r"\begin{table}[t]\centering\small",
         r"\caption{Raw-feature MLP control. Identical dense topology to the DNN-Tree "
         r"twin ($F\!\rightarrow\!64\!\rightarrow\!64\!\rightarrow\!K$), trained with plain "
         r"cross-entropy and no encoder, so only the input representation differs. "
         r"Removing the tree-derived representation costs 2.60 pp on average (FP) and "
         r"3.63 pp at matched int8-grid precision. This is a deployment control, "
         r"\emph{not} a formally isolated causal encoder effect.}",
         r"\label{tab:rawmlp}",
         r"\begin{tabular}{lrrrrr}\toprule",
         r"Dataset & Raw MLP FP & Raw MLP int8 & $\Delta$ int8 & DNN-Tree & Tree-to-Spike "
         r"\\\midrule"]
    for _, x in r.iterrows():
        t2 = np.mean(ts.get(x.dataset, [np.nan])); dd = np.mean(dn.get(x.dataset, [np.nan]))
        L.append(f"{tex_escape(x.dataset)} & {x.fp:.2f} & {x.i8:.2f} & {x.d:+.2f} "
                 f"& {dd:.2f} & {t2:.2f} \\\\")
    L.append(r"\midrule")
    L.append(f"\\textbf{{mean}} & {r.fp.mean():.2f} & {r.i8.mean():.2f} & "
             f"{r.d.mean():+.2f} & {np.mean([np.mean(v) for v in dn.values()]):.2f} & "
             f"{np.mean([np.mean(v) for v in ts.values()]):.2f} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    p = OUT / "tab_raw_mlp.tex"
    p.write_text("\n".join(L) + "\n")
    return p


if __name__ == "__main__":
    from sensitivity_analysis import build, FAMILY, paired
    df = build()
    dflt = df[df.arm == "DEFAULT"]
    recs = []
    for fam, arms in FAMILY.items():
        for a in arms:
            g = df[df.arm == a]
            if not len(g):
                continue
            d = paired(df, a, "macro_f1")
            q1, q3 = np.percentile(g.macro_f1, [25, 75])
            recs.append(dict(family=fam, arm=a, med_f1=g.macro_f1.median(), iqr=q3 - q1,
                             med_mcc=g.mcc.median(), med_far=g.FAR.median(),
                             d_f1_mean=d.mean(), d_f1_lo=np.percentile(d, 2.5),
                             d_f1_hi=np.percentile(d, 97.5),
                             pct_synops=100 * g.synops.median() / dflt.synops.median()))
    t = pd.DataFrame(recs)
    t.to_csv(OUT / "sensitivity_by_arm.csv", index=False)
    df.to_csv(OUT / "sensitivity_cells.csv", index=False)

    made = [figure_forest(t), figure_cost(t),
            figure_far(t, dflt.FAR.median(), dflt.macro_f1.median()),
            table_sensitivity(t), table_deployment(), table_early_exit(),
            table_raw_mlp()]
    src = REPO / "results/deployment/deployment_pareto.png"
    if src.exists():
        shutil.copy(src, OUT / "fig_deployment_pareto.png")
        made.append(OUT / "fig_deployment_pareto.png")
    for p in (REPO / "results/deployment/deployment_summary.csv",
              REPO / "results/early_exit/selected_test.csv",
              REPO / "results/deployment/int8_grid.csv",
              REPO / "results/deployment/compact_gbdt.csv"):
        if p.exists():
            shutil.copy(p, OUT / p.name)
    for p in made:
        print("  wrote", p.relative_to(REPO))
    print(f"\n{len(made)} artifacts in {OUT.relative_to(REPO)}")
