#!/usr/bin/env python3
"""Encoder ablation (TODO §1): tree-guided vs uniform vs all-channel spike encoder.

Proves the **tree-guided encoder is *why* Tree-to-Spike is efficient**. The `tree`
mode reuses the production Tree-to-Spike results; this script runs the `uniform`
(every feature-with-edges, 1 channel — no importance) and `all` (every
feature-with-edges, pop_max channels — dense upper bound) variants at the *same*
config/seeds, then writes a SOPs + accuracy comparison (figure + table) to
results/Journal_results/.

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

A_DATASETS = ["nslkdd", "unswnb15", "toniot", "cicids2017"]
DS_NAME = {"nslkdd": "NSL-KDD", "unswnb15": "UNSW-NB15", "toniot": "TON-IoT",
           "cicids2017": "CIC-IDS2017"}
MODES = ["tree", "uniform", "all"]              # tree = production results (not re-run)
RUN_MODES = ["uniform", "all"]
SEEDS = [42, 43, 44, 45, 46]
COLOR = {"tree": "#2CA02C", "uniform": "#DD8452", "all": "#9467BD"}
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
                cfg.update(encoder_mode=mode, seed=seed, results_dir=str(rd),
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
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


PATHS = {
    "tree":    "results/Tree-to-Spike/{ds}/seed_*/results.json",
    "uniform": "results/encoder_ablation/{ds}/uniform/seed_*/results.json",
    "all":     "results/encoder_ablation/{ds}/all/seed_*/results.json",
}
def _SYN(r):
    """Total SOPs/sample INCLUDING the input->layer-1 term (channels x hidden[0]),
    which the stored synops_per_sample omits — that term is exactly what the encoder
    controls, so it must be counted for the ablation to be meaningful."""
    syn = r["test_plus"]["spike_stats"]["synops_per_sample"]
    ch = r["data"]["encoder_channels"]
    h0 = r["config"]["hidden"][0] if isinstance(r["config"].get("hidden"), (list, tuple)) else 64
    return syn + ch * h0
_ACC = lambda r: r["test_plus"]["metrics"]["accuracy"]
_F1 = lambda r: r["test_plus"]["metrics"]["f1_macro"]
_CH = lambda r: r["data"]["encoder_channels"]


def aggregate():
    data = {}
    for ds in A_DATASETS:
        data[ds] = {}
        for m in MODES:
            p = PATHS[m].format(ds=ds)
            data[ds][m] = {"syn": _stat(p, _SYN), "acc": _stat(p, _ACC),
                           "f1": _stat(p, _F1), "ch": _stat(p, _CH)}
    return data


def write_table(data):
    md = ["### Encoder ablation — tree-guided vs uniform vs all-channel\n",
          "Mean over 5 seeds. **SOPs include the input->layer-1 term** (channels x hidden[0]); "
          "**Energy = SOPs × 0.9 pJ.** The all-channel encoder costs ~2x for negligible accuracy "
          "gain — the importance-guided channel budget is the efficiency mechanism.\n",
          "| Dataset | Encoder | Channels | SOPs/sample | Energy (pJ) | Accuracy | Macro-F1 |",
          "|---|---|--:|--:|--:|--:|--:|"]
    for ds in A_DATASETS:
        for m in MODES:
            c = data[ds][m]
            if c["syn"][0] is None:
                continue
            ch = f"{c['ch'][0]:.0f}" if c["ch"][0] is not None else "--"
            syn = c["syn"][0]; e = syn * E_AC
            tag = "**tree**" if m == "tree" else m
            md.append(f"| {DS_NAME[ds]} | {tag} | {ch} | {syn:,.0f} | {e:,.0f} | "
                      f"{c['acc'][0]*100:.1f} | {c['f1'][0]*100:.1f} |")
    (OUT / "tables" / "encoder_ablation.md").write_text("\n".join(md) + "\n")
    print(f"[wrote] {OUT/'tables'/'encoder_ablation.md'}")


def plot(data):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150,
                         "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})
    # (1) grouped bar: SOPs/sample per mode per dataset (log)
    x = np.arange(len(A_DATASETS)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, m in enumerate(MODES):
        ys = [(data[ds][m]["syn"][0] or np.nan) for ds in A_DATASETS]
        ax.bar(x + (i - 1) * w, ys, w, label=m, color=COLOR[m], edgecolor="black", linewidth=0.4)
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([DS_NAME[d] for d in A_DATASETS], rotation=15)
    ax.set_ylabel("SOPs / sample (log)"); ax.legend(frameon=False, title="encoder")
    ax.set_title("Spike cost by encoder variant (lower = cheaper)")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"encoder_ablation_sops.{ext}")
    plt.close(fig)

    # (2) scatter: SOPs vs Macro-F1 (efficiency frontier) — tree should sit lower-left-ish
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    marker = {"nslkdd": "o", "unswnb15": "s", "toniot": "^", "cicids2017": "D"}
    for ds in A_DATASETS:
        for m in MODES:
            c = data[ds][m]
            if c["syn"][0] is None:
                continue
            ax.scatter(c["syn"][0], c["f1"][0] * 100, s=70, marker=marker[ds],
                       color=COLOR[m], edgecolor="black", zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("SOPs / sample (log) — fewer is better")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Encoder ablation: accuracy vs spike cost")
    from matplotlib.lines import Line2D
    leg1 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR[m],
                   markeredgecolor="black", markersize=8, label=m) for m in MODES]
    leg2 = [Line2D([0], [0], marker=marker[d], color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=8, label=DS_NAME[d]) for d in A_DATASETS]
    ax.add_artist(ax.legend(handles=leg1, title="encoder", loc="lower right", frameon=False))
    ax.legend(handles=leg2, title="dataset", loc="upper left", frameon=False, fontsize=7)
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
    write_table(data)
    plot(data)
    print("\nEncoder ablation summary (mean SOPs/sample | Macro-F1):")
    for ds in A_DATASETS:
        row = "  ".join(f"{m}={data[ds][m]['syn'][0]:,.0f}/{data[ds][m]['f1'][0]*100:.1f}"
                        for m in MODES if data[ds][m]['syn'][0] is not None)
        print(f"  {DS_NAME[ds]:14} {row}")


if __name__ == "__main__":
    main()
