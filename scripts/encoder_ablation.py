#!/usr/bin/env python3
"""Encoder ablation: what does the GBDT teacher actually contribute?

Decomposes the teacher's role into separable questions, each with its own arm
(all at the production config/seeds; `tree` reuses the production results):

  quantile  does the teacher's split PLACEMENT (its "ruler") matter?
            equal-frequency edges, channels identical to tree (matched budget).
  nokd      does the teacher's POSTERIOR matter? tree encoder, a_kd = a_rank = 0.
  uniform   no pruning/importance, 1 channel per feature-with-edges. NOTE: this
            arm changes pruning AND allocation together (confounded) - kept for
            completeness, not for causal claims.
  all       dense upper bound (max channels per feature) - the cost of skipping
            the channel budget entirely.
  GBDT      the teacher's own test metrics (teacher_upper_bound; free, no runs).

Writes SOPs/accuracy/TTFS/early-exit comparisons (figure + table) to
results/Journal_results/. Conclusions are NOT hard-coded: read them off the
regenerated table.

Run:  python scripts/encoder_ablation.py            # runs variants (skips done) + aggregates
      python scripts/encoder_ablation.py --plot     # aggregate/plot only (no runs)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

A_GENERAL = ["kddcup99", "nslkdd", "ctu13", "iscxids2012", "cicids2017",
             "cse-cic-ids2018", "unswnb15"]
A_IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
A_DATASETS = A_GENERAL + A_IOT
# IoT is the primary table for the paper; general-purpose goes to the appendix
A_GROUPS = [(A_IOT, "iot", "IoT, IIoT, and IoMT"),
            (A_GENERAL, "general", "general-purpose network"),
            (A_DATASETS, "full", "all")]
DS_NAME = {"kddcup99": "KDDCup99", "nslkdd": "NSL-KDD", "ctu13": "CTU-13",
           "iscxids2012": "ISCX-IDS2012", "cicids2017": "CIC-IDS2017",
           "cse-cic-ids2018": "CSE-CIC-IDS2018", "unswnb15": "UNSW-NB15",
           "ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024",
           "edgeiiot": "Edge-IIoTset", "toniot": "TON-IoT"}
# Arms (tree = production results, not re-run):
#   tree     XGBoost split edges, importance-pruned + importance-budgeted channels (ours)
#   quantile equal-frequency edges, channels IDENTICAL to tree  -> does the teacher's ruler matter?
#   nokd     tree encoder, but a_kd = a_rank = 0                -> does the teacher's posterior matter?
#   uniform  tree edges, every feature-with-edges, 1 channel    -> no pruning/importance (confounded:
#            changes pruning AND allocation together; kept for completeness)
#   all      tree edges, every feature-with-edges, pop_max ch.  -> dense upper bound
# GBDT-alone (teacher's own test metrics) is reported from teacher_upper_bound, no runs needed.
MODES = ["tree", "quantile", "nokd", "uniform", "all"]
RUN_MODES = ["quantile", "nokd", "uniform", "all"]
SEEDS = [42, 43, 44, 45, 46]
# per-arm config overrides applied on top of the production config
ARM_CFG = {
    "quantile": {"encoder_mode": "quantile"},
    "nokd":     {"encoder_mode": "tree", "a_kd": 0.0, "a_rank": 0.0},
    "uniform":  {"encoder_mode": "uniform"},
    "all":      {"encoder_mode": "all"},
    # E2 (pre-registered 2026-09-24): class-supervised non-tree edges, tree channels and
    # per-feature edge count held fixed. Run cells with scripts/run_encoder_variant.py.
    "entropy":  {"encoder_mode": "entropy"},
}
# Paul Tol bright / Okabe-Ito (matches benchmark_4way).
# display names must match the table arm names exactly
ARM_LABEL = {"tree": "tree", "quantile": "quantile", "nokd": "no-KD",
             "uniform": "uniform", "all": "all"}
COLOR = {"tree": "#228833", "quantile": "#E69F00", "nokd": "#AA3377",
         "uniform": "#4477AA", "all": "#EE6677"}
OUT = REPO / "results" / "Journal_results"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
E_AC = 0.9   # pJ per spiking op (Horowitz 45nm, 32-bit)


def run_variants():
    from src.training_treespike import run_experiment
    for ds in A_DATASETS:
        base = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
        for mode in RUN_MODES:
            for seed in SEEDS:
                rd = REPO / f"results/encoder_ablation/{ds}/{mode}"
                if (rd / f"seed_{seed}" / "results.json").exists():
                    continue
                cfg = dict(base)
                cfg.update(ARM_CFG[mode])
                cfg.update(seed=seed, results_dir=str(rd),
                           summary_csv=str(REPO / "results/encoder_ablation/summary.csv"))
                print(f"[ablation] {ds}/{mode}/seed_{seed}", flush=True)
                run_experiment(cfg)


def _stat(patt, getter):
    vals = []
    for f in glob.glob(str(REPO / patt)):
        try:
            vals.append(getter(json.load(open(f))))
        except Exception:
            pass
    return (float(np.mean(vals)), float(np.std(vals, ddof=1))) if vals else (None, None)   # sample SD


PATHS = {
    "tree":     "results/Tree-to-Spike/{ds}/seed_*/results.json",
    "quantile": "results/encoder_ablation/{ds}/quantile/seed_*/results.json",
    "nokd":     "results/encoder_ablation/{ds}/nokd/seed_*/results.json",
    "uniform":  "results/encoder_ablation/{ds}/uniform/seed_*/results.json",
    "all":      "results/encoder_ablation/{ds}/all/seed_*/results.json",
}
def _SYN(r):
    """Total SOPs/sample INCLUDING the input->layer-1 term (channels x hidden[0]),
    which the stored synops_per_sample omits - that term is exactly what the encoder
    controls, so it must be counted for the ablation to be meaningful."""
    syn = r["test_plus"]["spike_stats"]["synops_per_sample"]
    ch = r["data"]["encoder_channels"]
    h0 = r["config"]["hidden"][0] if isinstance(r["config"].get("hidden"), (list, tuple)) else 64
    return syn + ch * h0
_ACC = lambda r: r["test_plus"]["metrics"]["accuracy"]
_F1 = lambda r: r["test_plus"]["metrics"]["f1_macro"]
_CH = lambda r: r["data"]["encoder_channels"]
_TTFS = lambda r: r["test_plus"]["spike_stats"]["mean_spike_time"]     # backfill_encoder_ttfs.py
_EXIT = lambda r: r["test_plus"]["early_exit"]["mean_exit_step"]
_GACC = lambda r: r["test_plus"]["teacher_upper_bound"]["accuracy"]    # GBDT-alone reference
_GF1 = lambda r: r["test_plus"]["teacher_upper_bound"]["f1_macro"]


def aggregate():
    data = {}
    for ds in A_DATASETS:
        data[ds] = {}
        for m in MODES:
            p = PATHS[m].format(ds=ds)
            data[ds][m] = {"syn": _stat(p, _SYN), "acc": _stat(p, _ACC),
                           "f1": _stat(p, _F1), "ch": _stat(p, _CH),
                           "ttfs": _stat(p, _TTFS), "exit": _stat(p, _EXIT)}
        # GBDT-alone reference (the teacher's own test metrics, stored with the tree runs)
        pt = PATHS["tree"].format(ds=ds)
        data[ds]["gbdt"] = {"acc": _stat(pt, _GACC), "f1": _stat(pt, _GF1),
                            "syn": (None, None), "ch": (None, None),
                            "ttfs": (None, None), "exit": (None, None)}
    return data


def write_table(data, DSL=None, suffix='full', group='all'):
    DSL = DSL or A_DATASETS
    md = ["### Encoder ablation - what does the teacher actually contribute?\n",
          "Mean over 5 seeds. **SOPs include the input->layer-1 term** (channels x hidden[0]); "
          "**Energy = SOPs × 0.9 pJ.** Arms: **tree** = XGBoost split edges + importance-pruned/"
          "budgeted channels (ours). **quantile** = equal-frequency edges with channels IDENTICAL "
          "to tree (same pruning/budget/jitter; per-feature edge count matched) - isolates whether "
          "the teacher's split placement (its 'ruler') matters. **nokd** = tree encoder trained "
          "WITHOUT the teacher-posterior loss terms (a_kd = a_rank = 0) - isolates the teacher's "
          "soft labels. **uniform** = tree edges, 1 channel per feature-with-edges (NOTE: changes "
          "pruning AND allocation together, so it is confounded; kept for completeness). **all** = "
          "tree edges, max channels per feature (dense upper bound). **GBDT (teacher)** = the "
          "XGBoost teacher's own test accuracy/F1, the reference ceiling (not a spiking model; no "
          "energy columns). **TTFS** = mean encoder time-to-first-spike (/20, lower = earlier). "
          "**Exit step** = mean early-exit timestep of the trained student (/20, lower = decides "
          "sooner).\n",
          "| Dataset | Encoder | Channels | SOPs/sample | Energy (pJ) | Accuracy | Macro-F1 | TTFS (/20) | Exit step (/20) |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for ds in DSL:
        for m in MODES + ["gbdt"]:
            c = data[ds][m]
            if c["acc"][0] is None:
                continue
            if m == "gbdt":
                md.append(f"| {DS_NAME[ds]} | GBDT (teacher) | -- | -- | -- | "
                          f"{c['acc'][0]*100:.1f} | {c['f1'][0]*100:.1f} | -- | -- |")
                continue
            if c["syn"][0] is None:
                continue
            ch = f"{c['ch'][0]:.0f}" if c["ch"][0] is not None else "--"
            syn = c["syn"][0]; e = syn * E_AC
            ttfs = f"{c['ttfs'][0]:.1f}" if c["ttfs"][0] is not None else "--"
            ext = f"{c['exit'][0]:.1f}" if c["exit"][0] is not None else "--"
            tag = "**tree**" if m == "tree" else m
            md.append(f"| {DS_NAME[ds]} | {tag} | {ch} | {syn:,.0f} | {e:,.0f} | "
                      f"{c['acc'][0]*100:.1f} | {c['f1'][0]*100:.1f} | {ttfs} | {ext} |")
    (OUT / "tables" / f"encoder_ablation_{suffix}.md").write_text("\n".join(md) + "\n")

    # ---------- LaTeX twin (same data, booktabs + multirow) ----------
    ARM_TEX = {"tree": r"\textbf{tree} (ours)", "quantile": "quantile", "nokd": "no-KD",
               "uniform": "uniform", "all": "all", "gbdt": "GBDT (teacher)"}
    n_arms = len(MODES) + 1
    L = [
        r"% Requires: \usepackage{multirow,booktabs}",
        r"\begin{table*}[!ht]", r"\centering", r"\renewcommand{\arraystretch}{1.05}",
        rf"\caption{{Encoder ablation over the {group} intrusion detection datasets: what "
        r"does the GBDT teacher contribute? Mean over 5 "
        r"seeds. SOPs include the input-to-first-layer term (channels $\times$ hidden$_0$); "
        r"Energy = SOPs $\times$ 0.9\,pJ. Arms: \textbf{tree} = XGBoost split edges with "
        r"importance-pruned and importance-budgeted channels (ours); \textbf{quantile} = "
        r"equal-frequency edges with channels identical to tree (same pruning, budget, "
        r"jitter, and per-feature edge count), isolating whether the teacher's split "
        r"placement matters; \textbf{no-KD} = tree encoder trained without the "
        r"teacher-posterior loss terms ($\alpha_{\mathrm{kd}} = \alpha_{\mathrm{rank}} = 0$), "
        r"isolating the teacher's soft labels; \textbf{uniform} = tree edges, one channel "
        r"per feature with edges (changes pruning and allocation together, reported for "
        r"completeness); \textbf{all} = tree edges, maximum channels per feature (dense "
        r"upper bound). GBDT (teacher) is the XGBoost teacher's own test performance (not a "
        r"spiking model). The teacher is trained cost-sensitively (inverse-frequency sample "
        r"weights), so it operates at a balanced-performance point: on strongly imbalanced "
        r"splits its raw accuracy can fall below the student's (CSE-CIC-IDS2018, UNSW-NB15) "
        r"even where its macro-F1 leads, and it is a reference, not an accuracy ceiling. TTFS = mean encoder time-to-first-spike "
        r"over all emitted spikes; Exit = mean confidence-triggered early-exit timestep of "
        r"the trained student (both /$T{=}20$, lower = earlier).}",
        rf"\label{{tab:encoder_ablation_{suffix}}}", r"{\scriptsize",
        r"\begin{tabular}{l l r r r r r r r}", r"\toprule",
        r"\textbf{Dataset} & \textbf{Encoder} & \textbf{Channels} & \textbf{SOPs} & "
        r"\textbf{Energy (pJ)} & \textbf{ACC (\%)} & \textbf{F1-M (\%)} & "
        r"\textbf{TTFS} & \textbf{Exit} \\", r"\midrule",
    ]
    cur = None
    for ds in DSL:
        for m in MODES + ["gbdt"]:
            c = data[ds][m]
            if c["acc"][0] is None or (m != "gbdt" and c["syn"][0] is None):
                continue
            if m == "gbdt":
                cells = (f"--- & --- & --- & {c['acc'][0]*100:.1f} & "
                         f"{c['f1'][0]*100:.1f} & --- & ---")
            else:
                ch = f"{c['ch'][0]:.0f}" if c["ch"][0] is not None else "---"
                syn = c["syn"][0]; e = syn * E_AC
                ttfs = f"{c['ttfs'][0]:.1f}" if c["ttfs"][0] is not None else "---"
                ext = f"{c['exit'][0]:.1f}" if c["exit"][0] is not None else "---"
                cells = (f"{ch} & {syn:,.0f} & {e:,.0f} & {c['acc'][0]*100:.1f} & "
                         f"{c['f1'][0]*100:.1f} & {ttfs} & {ext}")
            if ds != cur:
                if cur is not None:
                    L.append(r"\midrule")
                L.append(rf"\multirow{{{n_arms}}}{{*}}{{{DS_NAME[ds]}}}")
                cur = ds
            L.append(rf"& {ARM_TEX[m]} & {cells} \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    (OUT / "tables" / f"encoder_ablation_{suffix}.tex").write_text("\n".join(L))
    print(f"[wrote] {OUT/'tables'}/encoder_ablation_{suffix}.md + .tex")


def plot(data):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150,
                         "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})
    # (1) grouped bar: SOPs/sample per mode per dataset (log)
    x = np.arange(len(A_DATASETS)); w = 0.8 / len(MODES)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for i, m in enumerate(MODES):
        ys = [(data[ds][m]["syn"][0] or np.nan) for ds in A_DATASETS]
        ax.bar(x + (i - (len(MODES) - 1) / 2) * w, ys, w, label=ARM_LABEL[m], color=COLOR[m],
               edgecolor="black", linewidth=0.4)
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([DS_NAME[d] for d in A_DATASETS], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("SOPs / sample (log)")
    # the data span (~2k to ~55k) covers barely one decade, so the default log
    # locator prints a single 10^4 tick: use explicit human-readable decade+half
    # ticks and suppress the unlabelled minor ticks.
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    lo = np.nanmin([data[ds][m]["syn"][0] for ds in A_DATASETS for m in MODES
                    if data[ds][m]["syn"][0]])
    hi = np.nanmax([data[ds][m]["syn"][0] for ds in A_DATASETS for m in MODES
                    if data[ds][m]["syn"][0]])
    cand = [1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000]
    ticks = [t for t in cand if lo / 1.6 <= t <= hi * 1.6]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(
        [f"{t/1000:g}k" if t >= 1000 else f"{t:g}" for t in ticks]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(lo / 1.5, hi * 2.6)          # headroom for the legend row
    # legend on one row above the axes so it never covers the leftmost bars;
    # the axes title is dropped because the LaTeX caption carries it.
    ax.legend(frameon=False, ncol=len(MODES), loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.4,
              fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"encoder_ablation_sops.{ext}")
    plt.close(fig)

    # (2) scatter: SOPs vs Macro-F1 (efficiency frontier) - tree should sit lower-left-ish
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    marker = {ds: markers[i % len(markers)] for i, ds in enumerate(A_DATASETS)}
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for ds in A_DATASETS:
        for m in MODES:
            c = data[ds][m]
            if c["syn"][0] is None:
                continue
            ax.scatter(c["syn"][0], c["f1"][0] * 100, s=70, marker=marker[ds],
                       color=COLOR[m], edgecolor="black", zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("SOPs / sample (log) - fewer is better")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Encoder ablation: accuracy vs spike cost")
    from matplotlib.lines import Line2D
    leg1 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR[m],
                   markeredgecolor="black", markersize=8, label=ARM_LABEL[m]) for m in MODES]
    leg2 = [Line2D([0], [0], marker=marker[d], color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=8, label=DS_NAME[d]) for d in A_DATASETS]
    # legends OUTSIDE the axes (right) so they never overlap the markers
    ax.add_artist(ax.legend(handles=leg1, title="encoder", loc="upper left",
                            bbox_to_anchor=(1.02, 1.0), frameon=False))
    ax.legend(handles=leg2, title="dataset", loc="lower left",
              bbox_to_anchor=(1.02, 0.0), frameon=False, fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"encoder_ablation_frontier.{ext}")
    plt.close(fig)
    print(f"[wrote] {OUT/'figures'}/encoder_ablation_sops.* + encoder_ablation_frontier.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="aggregate/plot only (no runs)")
    args = ap.parse_args()
    if not args.plot:
        run_variants()
    data = aggregate()
    for _ds, _sfx, _grp in A_GROUPS:
        write_table(data, _ds, _sfx, _grp)
    plot(data)
    print("\nEncoder ablation summary (mean SOPs/sample | Macro-F1):")
    for ds in A_DATASETS:
        row = "  ".join(f"{m}={data[ds][m]['syn'][0]:,.0f}/{data[ds][m]['f1'][0]*100:.1f}"
                        for m in MODES if data[ds][m]['syn'][0] is not None)
        print(f"  {DS_NAME[ds]:14} {row}")


if __name__ == "__main__":
    main()
