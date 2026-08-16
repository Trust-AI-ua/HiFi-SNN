#!/usr/bin/env python3
"""Final paper figures, rendered at exact IEEE print scale.

Rendered at the width they will occupy on the page, so LaTeX never rescales
them and the font sizes here are the font sizes a reader sees. That is why
``savefig.bbox`` is **not** "tight": tight cropping changes the output width,
so a figure declared at ``\\columnwidth`` would silently be scaled and every
label would shrink. Constrained layout is used instead.

Four figures, one per manuscript claim, each drawn from a single inferential
population:

  fig_screening   27 configurations x 20 blocks   (column width, tall)
  fig_axes        strict encoding + neuron axes   (text width, two panels)
  fig_readout     temporal convergence            (text width, two panels)
  fig_pareto      macro-F1 against SOPs           (column width)

Colour is never the only channel. Encodings carry a distinct marker as well as
a hue from the Wong colourblind-safe palette, and the selected configuration is
labelled in text rather than identified by colour alone.

Captions are emitted to `paper/fig_captions.tex` so their statistical scope --
which population, how many blocks -- stays tied to the figure that shows it.

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
A = ROOT / "results/analysis"
FIG = ROOT / "paper/figures"

# IEEEtran, US letter, two columns.
COL_W = 3.487           # \columnwidth in inches
TEXT_W = 7.16           # \textwidth in inches

# Sized for reading at 100%, not while zoomed in. 7pt is the practical floor
# for a serif tick label in print; nothing here goes below it.
RC = {
    "figure.dpi": 200, "savefig.dpi": 600,
    "savefig.bbox": None,          # exact width; see module docstring
    "savefig.pad_inches": 0.0,
    "font.family": "serif", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.1,
}

ENC = {
    "latency": {"c": "#0072B2", "m": "o", "ls": "-"},
    "rate":    {"c": "#D55E00", "m": "s", "ls": "--"},
    "delta":   {"c": "#009E73", "m": "^", "ls": ":"},
}
SEL = "LeakyParallel/latency"
PROTO_LABEL = {"nslkdd_v2": "NSL-KDD", "kddcup99_v2": "KDDCup99",
               "cicids2017_v2": "CIC-IDS2017", "ctu13_v2_f0": "CTU-13 f0",
               "ctu13_v2_f1": "CTU-13 f1"}
ORDER = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2", "ctu13_v2_f0",
         "ctu13_v2_f1"]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  {name:16s} {w:.3f} x {h:.3f} in")


# ---------------------------------------------------------------- 1. screening
def fig_screening() -> str:
    mr = pd.read_csv(A / "v1/mean_rank.csv", index_col=0)
    fr = json.load(open(A / "v1/friedman.json"))
    cd = fr["nemenyi_critical_difference"]
    best = mr.mean_rank.iloc[0]

    fig, ax = plt.subplots(figsize=(COL_W, 4.15), layout="constrained")
    y = np.arange(len(mr))
    within = mr.mean_rank <= best + cd
    colors = ["#0072B2" if w else "#BBBBBB" for w in within]
    ax.barh(y, mr.mean_rank, color=colors, height=0.68, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(mr.index, fontsize=6.2)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.axvspan(best, best + cd, color="#CC79A7", alpha=0.13, zorder=1)
    ax.axvline(best + cd, color="#CC79A7", lw=0.9, ls="--", zorder=2)
    # Top of the plot, not the bottom: with the y-axis inverted, index 0 is
    # the top row, so the label sat on the worst bar and overlapped it.
    ax.text(best + cd + 0.3, -0.4, f"CD = {cd:.2f}", color="#8B2E63",
            fontsize=6.5, va="center")
    ax.set_xlabel("mean rank over 20 blocks")   # shorter: full text overran
    ax.set_xlim(0, mr.mean_rank.max() * 1.12)
    ax.set_ylim(len(mr) - 0.4, -1.1)            # headroom for the CD label
    ax.grid(axis="y", visible=False)
    save(fig, "fig_screening")
    return (f"Mean macro-F1 rank of all {fr['k_variants']} configurations in the "
            f"\\textbf{{screening}} study, over {fr['n_blocks']} paired "
            f"(dataset $\\times$ seed) blocks. Friedman "
            f"$\\chi^2={fr['friedman_chi2']:.1f}$, $p={fr['friedman_p']:.1e}$. "
            f"The shaded band spans one Nemenyi critical difference "
            f"($\\mathrm{{CD}}={cd:.2f}$) from the leader; the "
            f"{len(fr['statistically_indistinguishable_from_best'])} "
            f"configurations inside it (blue) are not separable from "
            f"\\texttt{{{SEL}}}, which is therefore reported as top-ranked "
            f"rather than uniquely best.")


# ---------------------------------------------------------------- 2. axes
def fig_axes() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.35), layout="constrained")
    caps = []
    for ax, axis, title in ((axes[0], "encoding", "(a) encoding axis"),
                            (axes[1], "neuron", "(b) neuron axis")):
        mr = pd.read_csv(A / f"v2_{axis}_axis/mean_rank.csv", index_col=0)
        d = json.load(open(A / f"v2_{axis}_axis/summary.json"))
        caps.append(d)
        y = np.arange(len(mr))
        cols = ["#0072B2" if i == 0 else "#9DBBD0" for i in range(len(mr))]
        ax.barh(y, mr.mean_rank, color=cols, height=0.62, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(mr.index, fontsize=7)
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.invert_yaxis()
        for i, (v, w) in enumerate(zip(mr.mean_rank, mr.best_block_count)):
            ax.text(v + 0.06, i,
                    f"{v:.2f} ({int(w)} win{'' if int(w) == 1 else 's'})",
                    va="center", fontsize=6.2)
        ax.set_xlabel(f"mean rank over {d['n_blocks']} blocks")
        ax.set_xlim(0, mr.mean_rank.max() * 1.42)
        ax.set_title(title, fontsize=8, loc="left")
        ax.grid(axis="y", visible=False)
    save(fig, "fig_axes")
    e, n = caps
    return (f"Strict confirmation along the two design axes. "
            f"\\textbf{{(a)}} encoding axis: three encodings with the neuron "
            f"held at \\texttt{{LeakyParallel}}, over {e['n_blocks']} paired "
            f"(protocol $\\times$ seed) blocks (Friedman $p={e['friedman_p']:.1e}$). "
            f"\\textbf{{(b)}} neuron axis: nine families with the encoding held "
            f"at latency, over {n['n_blocks']} blocks ($p={n['friedman_p']:.1e}$). "
            f"Bars show mean rank, annotated with blocks won. Each panel holds "
            f"one factor fixed, so neither --- nor the two together --- "
            f"reconstructs a strict ranking of all 27 configurations. Note that "
            f"in (b) the leading family does not win the most individual "
            f"blocks.")


# ---------------------------------------------------------------- 3. readout
def fig_readout() -> str:
    curves = pd.read_csv(A / "early_readout/curves.csv")
    summ = pd.read_csv(A / "early_readout/summary.csv")
    curves = curves[curves.neuron == "LeakyParallel"]
    summ = summ[summ.neuron == "LeakyParallel"]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.6), layout="constrained")

    # (a) per-seed traces on the strongest case
    sub = curves[curves.protocol == "nslkdd_v2"]
    for enc in ("latency", "rate", "delta"):
        st = ENC[enc]
        g = sub[sub.encoding == enc]
        for _, sg in g.groupby("seed"):
            sg = sg.sort_values("t")
            axes[0].plot(sg.t, sg.f1_macro, color=st["c"], lw=0.45, alpha=0.40)
        m = g.groupby("t").f1_macro.median()
        axes[0].plot(m.index, m.values, color=st["c"], lw=1.5, ls=st["ls"],
                     label=enc, zorder=4)
    axes[0].set_xlabel("timestep $t$")
    axes[0].set_ylabel("macro-F1 at $t$")
    axes[0].set_title("(a) NSL-KDD, every seed shown", fontsize=8, loc="left")
    axes[0].legend(loc="lower right", ncols=3, handlelength=1.6,
                   columnspacing=1.0)

    # (b) cross-protocol T99, latency vs rate
    ax = axes[1]
    x = np.arange(len(ORDER))
    lat = [summ[(summ.protocol == p) & (summ.encoding == "latency")]
           .T99_median.iloc[0] for p in ORDER]
    rat = [summ[(summ.protocol == p) & (summ.encoding == "rate")]
           .T99_median.iloc[0] for p in ORDER]
    ax.plot(x - 0.11, lat, ENC["latency"]["m"], color=ENC["latency"]["c"],
            ms=5, ls="none", label="latency")
    ax.plot(x + 0.11, rat, ENC["rate"]["m"], color=ENC["rate"]["c"],
            ms=4.5, ls="none", label="rate")
    for i, (a, b) in enumerate(zip(lat, rat)):
        ax.plot([i - 0.11, i + 0.11], [a, b], color="#999999", lw=0.6, zorder=0)
        # Annotate the earlier point, beneath it, so the word names the marker
        # it sits next to. Placing it above the pair left it ambiguous which
        # encoding the label referred to.
        faster = "latency" if a < b else "rate"
        xf = i - 0.11 if a < b else i + 0.11
        ax.annotate(faster, (xf, min(a, b)), xytext=(0, -9),
                    textcoords="offset points", ha="center", fontsize=5.8,
                    color=ENC[faster]["c"])
    ax.set_xticks(x)
    ax.set_xticklabels([PROTO_LABEL[p] for p in ORDER], fontsize=6.4,
                       rotation=18, ha="right")
    # Padding at both ends: the leftmost annotation sits at x=-0.11 and was
    # clipped by the default limits.
    ax.set_xlim(-0.55, len(ORDER) - 0.45)
    ax.set_ylabel("median $T_{99}$ (timesteps)")
    ax.set_ylim(0, 27)
    ax.set_title("(b) convergence step, all strict protocols", fontsize=8,
                 loc="left")
    ax.legend(loc="upper left", ncols=2, handlelength=1.0, columnspacing=1.0)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_readout")

    nsl = summ[summ.protocol == "nslkdd_v2"].set_index("encoding")
    return (f"Temporal evidence accumulation, \\texttt{{LeakyParallel}}, five "
            f"seeds. \\textbf{{(a)}} macro-F1 as a function of readout step on "
            f"NSL-KDD; thin lines are individual seeds and thick lines the "
            f"median, plotted this way because $T_{{99}}$ is discrete and "
            f"skewed and a symmetric band would misrepresent it. "
            f"\\textbf{{(b)}} median $T_{{99}}$ for latency and rate on all "
            f"five strict protocols. Latency resolves earlier on two and later "
            f"on three, so the NSL-KDD advantage "
            f"({int(nsl.loc['latency','T99_median'])} against "
            f"{int(nsl.loc['rate','T99_median'])} steps) is the strongest case "
            f"rather than the typical one.")


# ---------------------------------------------------------------- 4. pareto
def fig_pareto() -> str:
    eff = pd.read_csv(A / "efficiency/summary.csv")
    eff = eff[eff.neuron == "LeakyParallel"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.75), layout="constrained")
    # Every point of one marker shape is the same configuration evaluated on a
    # different protocol, so labelling a single point with the configuration
    # name would be wrong. Points carry their protocol instead, which also
    # makes the figure readable without relying on colour.
    short = {"nslkdd_v2": "N", "kddcup99_v2": "K", "cicids2017_v2": "C",
             "ctu13_v2_f0": "T0", "ctu13_v2_f1": "T1"}
    for enc in ("latency", "rate", "delta"):
        g = eff[eff.encoding == enc]
        st = ENC[enc]
        ax.scatter(g.sops_mean, g.f1_mean, s=26, marker=st["m"], color=st["c"],
                   label=enc, zorder=3, edgecolor="white", linewidth=0.4)
        for _, r in g.iterrows():
            ax.annotate(short[r.protocol], (r.sops_mean, r.f1_mean),
                        xytext=(4.5, 2.5), textcoords="offset points",
                        fontsize=5.6, color="#444444")
    # The CIC-IDS2017 points cannot exceed 0.80 under the fixed label universe
    # because U2R has no test support. Without this line the "C" points read as
    # a harder dataset rather than a metric with a structural ceiling, and a
    # caption alone does not reach a reader skimming the figure.
    ax.axhline(0.80, color="#8B2E63", lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.text(ax.get_xlim()[1] if False else 1.5e3, 0.815,
            "CIC-IDS2017 ceiling (no U2R in test)", fontsize=5.6,
            color="#8B2E63", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("synaptic operations per sample (log)")
    ax.set_ylabel("macro-F1 (fixed universe)")   # shorter: full text overran
    ax.set_xlim(1.4e3, 8e4)
    ax.legend(loc="lower left", ncols=3, handlelength=1.0, columnspacing=0.8,
              handletextpad=0.3)
    ax.set_ylim(0.4, 1.06)
    save(fig, "fig_pareto")

    piv = eff.pivot(index="protocol", columns="encoding", values="sops_mean")
    r = piv["rate"] / piv["latency"]
    return (f"Detection quality against synaptic-operation demand, "
            f"\\texttt{{LeakyParallel}}, one point per strict protocol and "
            f"encoding (seed means). Rate encoding requires "
            f"${r.min():.1f}\\times$ to ${r.max():.1f}\\times$ more operations "
            f"per sample than latency on every protocol, while emitting only "
            f"$1.7\\times$ to $2.5\\times$ more spikes in total: the saving "
            f"comes from where the spikes occur, ahead of the high-fanout "
            f"input projection, not merely from how many there are. Macro-F1 "
            f"is the fixed-universe metric, capped at 0.80 on CIC-IDS2017 "
            f"where U2R has no test support. Point labels give the protocol: "
            f"N NSL-KDD, K KDDCup99, C CIC-IDS2017, T0/T1 CTU-13 folds.")


# ------------------------------------------------------------------ 5. timing
def fig_timing() -> str:
    """The value of spike timing alone, at matched input activity.

    The paper's strongest single result and, until now, the only major one
    without a figure. Horizontal bars because the protocol names are long and
    the quantity is signed: a diverging layout about zero shows at a glance that
    this is not an effect with a direction.
    """
    d = pd.read_csv(A / "e1_timing_isolation.csv", index_col=0)
    d = d.reindex([p for p in ORDER if p in d.index])
    d = d.sort_values("timing_premium_pp")            # ascending: best on top

    fig, ax = plt.subplots(figsize=(COL_W, 2.5), layout="constrained")
    y = np.arange(len(d))
    pos = d.timing_premium_pp > 0
    ax.barh(y, d.timing_premium_pp,
            color=[ENC["latency"]["c"] if v else ENC["delta"]["c"]
                   for v in pos],
            height=0.62, zorder=3, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#444444", lw=0.8, zorder=4)

    for i, (proto, r) in enumerate(d.iterrows()):
        v = r.timing_premium_pp
        # Annotate with the two macro-F1 values the difference is taken between,
        # so the bar cannot be read without its operands.
        txt = f"{r.f1_timing:.3f} / {r.f1_notiming:.3f}"
        off, ha = (1.6, "left") if v > 0 else (-1.6, "right")
        ax.text(v + off, i, txt, va="center", ha=ha, fontsize=6.0,
                color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([PROTO_LABEL[p] for p in d.index], fontsize=7)
    ax.set_xlabel("value of spike timing (pp macro-F1)")
    # Asymmetric padding: the annotations run outward from the bar end, so the
    # negative side needs room for a label AND must not reach the tick labels.
    lo, hi = d.timing_premium_pp.min(), d.timing_premium_pp.max()
    rng = hi - lo
    ax.set_xlim(lo - 0.60 * rng, hi + 0.30 * rng)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_timing")
    return (f"\\textbf{{What spike timing is worth, in isolation.}} Each bar is "
            f"latency coding minus delta coding with both gates set to "
            f"$\\vartheta=0.01$, a construction under which the two arms emit "
            f"\\emph{{identical input spike sets}} (maximum difference "
            f"$\\FNTimingMaxDiff$ spikes per sample) and differ only in whether "
            f"the spike carries timing information. Annotations give the two "
            f"macro-F1 values the difference is taken between, with timing "
            f"first. LeakyParallel, three seeds, five confirmation protocols. "
            f"The effect spans $\\FNTimingSpread$ percentage points and changes "
            f"sign, so temporal coding is not a property that helps or hurts in "
            f"general.")


# --------------------------------------------------------- 6. sparsity cascade
def fig_sparsity_cascade() -> str:
    """Contraction at the input, dilution in the total, recovery in operations.

    The three ratios are the whole argument of the efficiency subsection and
    they are hard to compare in a table because they live on different scales.
    A log axis with a rule at unity makes the contract-then-re-expand shape
    legible in one look.
    """
    e = pd.read_csv(A / "efficiency/summary.csv")
    e = e[e.neuron == SEL.split("/")[0]]
    piv = lambda c: e.pivot(index="protocol", columns="encoding", values=c)
    series = {
        "input spikes": piv("input_spikes")["rate"] / piv("input_spikes")["latency"],
        "total spikes": piv("total_spikes")["rate"] / piv("total_spikes")["latency"],
        "synaptic ops": piv("sops_mean")["rate"] / piv("sops_mean")["latency"],
    }
    hue = {"input spikes": "#0072B2", "total spikes": "#999999",
           "synaptic ops": "#CC79A7"}

    fig, ax = plt.subplots(figsize=(COL_W, 2.45), layout="constrained")
    protos = [p for p in ORDER if p in series["input spikes"].index]
    x = np.arange(len(protos))
    w = 0.26
    for k, (name, ser) in enumerate(series.items()):
        ax.bar(x + (k - 1) * w, [ser[p] for p in protos], width=w,
               label=name, color=hue[name], zorder=3,
               edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="#444444", lw=0.8, ls="--", zorder=4)
    ax.text(len(protos) - 0.42, 1.04, "parity", fontsize=6, color="#444444",
            ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylabel("rate / latency")
    ax.set_xticks(x)
    ax.set_xticklabels([PROTO_LABEL[p] for p in protos], fontsize=6.4,
                       rotation=12, ha="right")
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(["1x", "2x", "5x", "10x", "20x"])
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              handlelength=1.1, columnspacing=1.0)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_sparsity_cascade")
    return (f"\\textbf{{Where latency's efficiency advantage is created and where "
            f"it is lost.}} Ratios of rate to latency on a logarithmic axis, "
            f"LeakyParallel, five confirmation protocols, five seeds. Input "
            f"activity separates by $\\FNInpRatioMin\\times$ to "
            f"$\\FNInpRatioMax\\times$, a separation with a closed form that "
            f"depends on neither dimensionality nor sparsity. Most of it is then "
            f"diluted: whole-network spike counts differ by only "
            f"$\\FNTotRatioMin\\times$ to $\\FNTotRatioMax\\times$. It "
            f"reappears in synaptic operations "
            f"($\\FNSopRatioMin\\times$ to $\\FNSopRatioMax\\times$) because "
            f"latency's reduction falls ahead of the high-fanout input "
            f"projection. Where spikes occur matters more than how many there "
            f"are.")


def main() -> int:
    matplotlib.rcParams.update(RC)
    print("rendering at IEEE print scale "
          f"(column {COL_W}in, text {TEXT_W}in):")
    caps = {
        "fig:screening": fig_screening(),
        "fig:axes": fig_axes(),
        "fig:readout": fig_readout(),
        "fig:pareto": fig_pareto(),
        "fig:timing": fig_timing(),
        "fig:sparsitycascade": fig_sparsity_cascade(),
    }
    widths = {"fig:screening": "\\columnwidth", "fig:axes": "\\textwidth",
              "fig:readout": "\\textwidth", "fig:pareto": "\\columnwidth",
              "fig:timing": "\\columnwidth",
              "fig:sparsitycascade": "\\columnwidth"}
    stars = {"fig:axes": "*", "fig:readout": "*"}
    out = ["% GENERATED by scripts/make_figures.py — do not edit by hand.",
           "% Each caption states the population and block count it draws on,",
           "% so a reader need not infer the statistical scope from Methods.",
           "% Figures are rendered at the exact width declared here; do NOT add",
           "% a scale factor, or the font sizes will no longer be as designed.",
           ""]
    for label, text in caps.items():
        name = label.split(":")[1]
        s = stars.get(label, "")
        out += [f"\\begin{{figure{s}}}[t]", "  \\centering",
                f"  \\includegraphics[width={widths[label]}]{{figures/fig_{name}}}",
                f"  \\caption{{{text}}}", f"  \\label{{{label}}}",
                f"\\end{{figure{s}}}", ""]
    (ROOT / "paper/fig_captions.tex").write_text("\n".join(out))
    print(f"\ncaptions -> {ROOT / 'paper/fig_captions.tex'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
