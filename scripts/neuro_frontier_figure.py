#!/usr/bin/env python3
"""Energy-accuracy frontier figure (amendments 5, 5b and 6). Runs in .venv. Reads results only.

Data:
  results/neuro/frontier/frontier.csv   seed-42 frontier: 7 variants per dataset (28), production
                                        (T = 20), C1-gate early exit and the fixed-point Vanilla SNN
  results/neuro/confirm/confirm.csv     five-seed confirmation (Edge-IIoTset T = 10, CICIoMT2024 early
                                        exit, the paired fixed-point Vanilla)
Training records give the epochs, pop_max and prune quantile of every Tree-to-Spike row.

Figure: four panels (Edge-IIoTset, TON-IoT, CIC-IoT2023, CICIoMT2024); x = SANA-FE energy per inference
(nJ, log scale), y = test macro-F1 of the simulated network (bit-exact emulation of Lava's arithmetic,
whole test split). Shaded: less energy than the seed-42 Vanilla and higher macro-F1. Line: Pareto front of
the seed-42 rows. Five-seed means with SD bars (across seeds) on the two confirmation panels.

    .venv/bin/python scripts/neuro_frontier_figure.py
Output: results/neuro/frontier/frontier_seed42.csv, results/neuro/confirm/confirm_five_seed.csv,
        results/neuro/frontier/frontier.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FR = REPO / "results/neuro/frontier"
CF = REPO / "results/neuro/confirm"
ORDER = ["edgeiiot", "toniot", "ciciot2023", "ciciomt2024"]
TITLE = {"edgeiiot": "Edge-IIoTset", "toniot": "TON-IoT", "ciciot2023": "CIC-IoT2023", "ciciomt2024": "CICIoMT2024"}
RUN = {"t2s_full": "results/Tree-to-Spike/{ds}", "t2s_exit": "results/Tree-to-Spike/{ds}",
       "uniform": "results/encoder_ablation/{ds}/uniform", "vanilla_fixed": "results/{ds}/SNN_LeakyParallel/latency"}
TYPE = {"T10": "window", "T40": "window", "T80": "window", "q40": "pruning", "M2": "channel budget",
        "prune_only": "channel budget", "uniform": "uniform", "t2s_full": "production", "t2s_exit": "early exit",
        "vanilla_fixed": "vanilla"}
NAME = {"T10": "T2S, T = 10", "T40": "T2S, T = 40", "T80": "T2S, T = 80", "q40": "T2S, prune q = 0.40",
        "M2": "T2S, pop_max 2", "prune_only": "T2S, pop_max 1", "uniform": "T2S, uniform allocation",
        "t2s_full": "T2S, production (T = 20)", "t2s_exit": "T2S, early exit (C1 gate)",
        "vanilla_fixed": "Vanilla SNN, fixed point"}

# palette: validated (validate_palette.js, light, all pairs): blue, orange, aqua; variants neutral gray, Vanilla ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, GRAY, MUTED, SHADE = "#0b0b0b", "#8a8984", "#52514e", "#e3eefb"


def config(ds, row):
    p = REPO / RUN.get(row, f"results/ladder/{{ds}}/{row}").format(ds=ds) / "seed_42/results.json"
    return json.loads(p.read_text())["config"]


def seed42_table():
    fr = pd.read_csv(FR / "frontier.csv")
    thr = json.loads((REPO / "results/journal_r2/exit_calibration/thresholds.json").read_text())
    rows = []
    for _, r in fr.iterrows():
        c = config(r.dataset, r.row)
        van = r.row == "vanilla_fixed"
        if van:
            ch = int(np.load(REPO / f"results/neuro/inputs/{r.dataset}_vanilla_seed42.npz")["spike_times"].shape[1])
        else:
            ch = int(r.channels)
        gate = thr[f"{r.dataset}/42"]["gate_c1"]
        ee = ("C1 gate" if gate is not None else "C1 gate not admitted (full window)") if r.row == "t2s_exit" else "no"
        rows.append(dict(dataset=r.dataset, network=NAME[r.row], variant_type=TYPE[r.row],
                         T=int(c.get("T") or 20) if not van else 25, channels=ch,
                         pruning="" if van else c["prune_quantile"], early_exit=ee, seed=42,
                         energy_nJ=r.energy_nJ_mean, energy_sd=r.energy_nJ_sd, latency_us=r.latency_us_mean,
                         macro_f1=r.simulated_macro_f1, epochs=int(c["epochs"]),
                         pop_max="" if van else int(c["pop_max"]), mean_steps=r.mean_steps,
                         row_id=r.row, pareto=bool(r.pareto), dominates_vanilla=bool(r.dominates_vanilla)))
    return pd.DataFrame(rows)


def five_seed_table():
    c = pd.read_csv(CF / "confirm.csv")
    keep = {"edgeiiot": ["T10", "vanilla_fixed"], "ciciomt2024": ["t2s_exit", "vanilla_fixed"]}
    rows = []
    for ds, rws in keep.items():
        for rw in rws:
            g = c[(c.dataset == ds) & (c.row == rw)].sort_values("seed")
            if sorted(g.seed) != list(range(42, 47)):
                raise SystemExit(f"{ds}/{rw}: five seeds expected, got {sorted(g.seed)}")
            for _, r in g.iterrows():
                rows.append(dict(dataset=ds, network=NAME[rw], row_id=rw, seed=str(int(r.seed)), energy_nJ=r.energy_nJ_mean,
                                 energy_sd_flows=r.energy_nJ_sd, latency_us=r.latency_us_mean,
                                 macro_f1=r.simulated_macro_f1_full_test,
                                 exit_gate="" if rw != "t2s_exit" else ("disabled (full window)" if pd.isna(r.exit_gate) else r.exit_gate),
                                 mean_steps=r.mean_steps))
            for stat, f in (("mean", np.mean), ("sd", lambda x: np.std(x, ddof=1))):
                rows.append(dict(dataset=ds, network=NAME[rw], row_id=rw, seed=stat, energy_nJ=f(g.energy_nJ_mean),
                                 latency_us=f(g.latency_us_mean), macro_f1=f(g.simulated_macro_f1_full_test),
                                 mean_steps=f(g.mean_steps)))
    return pd.DataFrame(rows)


def pareto(d):
    """Non-dominated (min energy, max macro-F1) points, in energy order."""
    pts = d.sort_values(["energy_nJ", "macro_f1"], ascending=[True, False])
    out, best = [], -np.inf
    for _, r in pts.iterrows():
        if r.macro_f1 > best:
            out.append(r); best = r.macro_f1
    return pd.DataFrame(out)


MK = {"window": "o", "pruning": "v", "channel budget": "s", "uniform": "D"}


def main():
    s42 = seed42_table()
    s5 = five_seed_table()
    out42 = s42.drop(columns=["row_id"])
    out42.to_csv(FR / "frontier_seed42.csv", index=False)
    s5.drop(columns=["row_id"]).to_csv(CF / "confirm_five_seed.csv", index=False)
    # the Pareto set drawn must be the one the frontier summary reported
    for ds in ORDER:
        d = s42[s42.dataset == ds]
        key = lambda x: set(zip(x.energy_nJ.round(6), x.macro_f1.round(6)))   # identical points count once
        if key(pareto(d)) != key(d[d.pareto]):
            raise SystemExit(f"{ds}: Pareto set {sorted(pareto(d).row_id)} differs from the frontier summary's "
                             f"{sorted(d[d.pareto].row_id)}")

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "text.color": INK})
    W = 15 / 2.54
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.86))
    for ax, ds in zip(axes.flat, ORDER):
        d = s42[s42.dataset == ds]
        v = d[d.row_id == "vanilla_fixed"].iloc[0]
        has5 = ds in s5.dataset.values
        e_all = list(d.energy_nJ) + (list(s5[s5.dataset == ds].query("seed not in ['mean', 'sd']").energy_nJ) if has5 else [])
        f_all = list(d.macro_f1) + (list(s5[s5.dataset == ds].query("seed not in ['mean', 'sd']").macro_f1) if has5 else [])
        x0, x1 = min(e_all) / 1.25, max(e_all) * 1.25
        yr = max(f_all) - min(f_all)
        y0, y1 = min(f_all) - 0.10 * yr, max(f_all) + 0.12 * yr
        ax.set_xscale("log"); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        # dominance region: less energy than the seed-42 Vanilla, higher macro-F1
        ax.add_patch(plt.Rectangle((x0, v.macro_f1), v.energy_nJ - x0, y1 - v.macro_f1, facecolor=SHADE,
                                   edgecolor="none", zorder=0))
        # Pareto front
        pf = pareto(d)
        ax.plot(pf.energy_nJ, pf.macro_f1, color=MUTED, lw=0.7, zorder=2)
        # variants
        for _, r in d[d.variant_type.isin(MK)].iterrows():
            ax.scatter(r.energy_nJ, r.macro_f1, marker=MK[r.variant_type], s=16, facecolor=GRAY, edgecolor="white",
                       linewidth=0.5, zorder=3)
            if r.variant_type == "window":
                left = ds == "ciciomt2024" and r["T"] == 10        # clear of the T20 label
                ax.annotate(f"T{r['T']}", (r.energy_nJ, r.macro_f1), xytext=(-3 if left else 3, 3), textcoords="offset points",
                            ha="right" if left else "left", fontsize=8, color=MUTED, zorder=6)
        p = d[d.row_id == "t2s_full"].iloc[0]
        ex = d[d.row_id == "t2s_exit"].iloc[0]
        ax.scatter(p.energy_nJ, p.macro_f1, marker="o", s=34, facecolor=BLUE, edgecolor="white", linewidth=0.7, zorder=4)
        dx, dy, ha = (0, 5, "center") if ds == "ciciomt2024" else (3, 3, "left")   # clear of the uniform marker
        ax.annotate("T20", (p.energy_nJ, p.macro_f1), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    ha=ha, color=MUTED, zorder=6)
        if ex.early_exit == "C1 gate":
            ax.scatter(ex.energy_nJ, ex.macro_f1, marker="p", s=44, facecolor=ORANGE, edgecolor="white", linewidth=0.7, zorder=4)
        else:
            ax.text(0.98, 0.03, "exit gate not\nadmitted at seed 42:\nearly exit = T20", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.scatter(v.energy_nJ, v.macro_f1, marker="X", s=40, facecolor=INK, edgecolor="white", linewidth=0.5, zorder=4)
        # five-seed means, SD across seeds in both directions
        if has5:
            for rid, col, mk in (("T10", AQUA, "o"), ("t2s_exit", AQUA, "p"), ("vanilla_fixed", INK, "X")):
                g = s5[(s5.dataset == ds) & (s5.row_id == rid)]
                if not len(g):
                    continue
                m, sd = g[g.seed == "mean"].iloc[0], g[g.seed == "sd"].iloc[0]
                ax.errorbar(m.energy_nJ, m.macro_f1, xerr=sd.energy_nJ, yerr=sd.macro_f1, fmt="none", ecolor=col,
                            elinewidth=0.9, capsize=2, capthick=0.9, zorder=5)
                ax.scatter(m.energy_nJ, m.macro_f1, marker=mk, s=40, facecolor="none", edgecolor=col, linewidth=1.1, zorder=5)
        if ds == "ciciot2023":
            ax.text(0.98, 0.77, "variants: 10-epoch runs\n(uniform: 20); T20, early\nexit, Vanilla: 20 epochs", transform=ax.transAxes,
                    ha="right", va="center", fontsize=8, color=MUTED)
        ax.set_title(TITLE[ds], loc="left", fontweight="bold")
        ticks = [t for t in (200, 300, 500, 1000, 2000, 3000) if x0 <= t <= x1]
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:,.0f}"))
        ax.grid(axis="y", color="#e6e5e1", lw=0.5, zorder=-1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Energy per inference (nJ, SANA-FE)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1, simulated (%)")
    h = [Line2D([], [], ls="none", marker="o", ms=4, mfc=GRAY, mec="white", label="Window (T labelled)"),
         Line2D([], [], ls="none", marker="v", ms=4, mfc=GRAY, mec="white", label="Pruning (q = 0.40)"),
         Line2D([], [], ls="none", marker="s", ms=4, mfc=GRAY, mec="white", label="Channel budget"),
         Line2D([], [], ls="none", marker="D", ms=3.5, mfc=GRAY, mec="white", label="Uniform allocation"),
         Line2D([], [], ls="none", marker="o", ms=5.5, mfc=BLUE, mec="white", label="Production (T20)"),
         Line2D([], [], ls="none", marker="p", ms=6, mfc=ORANGE, mec="white", label="Early exit (C1 gate)"),
         Line2D([], [], ls="none", marker="X", ms=6, mfc=INK, mec="white", label="Vanilla SNN"),
         Line2D([], [], color=AQUA, marker="o", ms=5, mfc="none", mec=AQUA, lw=0.9, label="Five-seed mean ± SD, T2S"),
         Line2D([], [], color=INK, marker="X", ms=5, mfc="none", mec=INK, lw=0.9, label="Five-seed mean ± SD, Vanilla"),
         Line2D([], [], color=MUTED, lw=0.7, label="Pareto front"),
         Patch(facecolor=SHADE, edgecolor="none", label="Beats Vanilla (seed 42)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0.17, 1, 1), h_pad=0.8, w_pad=0.8)
    fig.savefig(FR / "frontier.pdf")
    fig.savefig(REPO / "results/neuro/frontier/frontier_preview.png", dpi=200)
    print(out42.drop(columns=["pareto", "dominates_vanilla"]).round(2).to_string(index=False))
    print(s5.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
