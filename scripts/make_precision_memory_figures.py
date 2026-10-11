#!/usr/bin/env python3
"""The precision-profile and memory-band figures as standalone vector PDFs.

Matplotlib versions of the manuscript's two pgfplots figures (fig:precision_profile and
fig:memory_band), for journals that want figure files rather than TikZ. Every plotted value
is read from results/tables/numbers.tex, the same macros the TikZ code types:

  Figures/precision_profile.pdf        IoT-mean macro-F1 by weight precision
                                       (JNPc{Tts,Dnn}{Fp,PerTensor,PerChannel}IotMean, JNPle*IotMean)
  Figures/memory_band_sensitivity.pdf  DNN-Tree / Tree-to-Spike total energy vs SRAM price
                                       (JNBand{Dep,Full}{Cold,Reuse}{Five,...,Hundred})

Sized like the TikZ: \\columnwidth (3.5 in) wide, a 4.3 cm axis block plus the legend above it,
Times-like serif at the TikZ font sizes (scriptsize ticks and legend, footnotesize labels), and
fonts embedded as TrueType. Include with \\includegraphics[width=\\columnwidth]{...}.

    python scripts/make_precision_memory_figures.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Figures"

COL_W = 3.5                 # IEEE \columnwidth, inches
AXIS_H = 4.3 / 2.54         # pgfplots height=4.3cm (axis plus its labels)
LEGEND_ROW = 0.15           # one legend row above the axis, inches
SCRIPT, FOOT = 7, 8         # \scriptsize, \footnotesize at 10 pt
GRID = (0.91, 0.91, 0.91)   # gray!18
BLUE70, BLUE75 = (0, 0, 0.70), (0, 0, 0.75)          # blue!70!black, blue!75!black
ORANGE80, TEAL80 = (0.80, 0.40, 0), (0, 0.40, 0.40)  # orange!80!black, teal!80!black
THICK, THIN, MARK = 0.8, 0.4, 4.0                    # TikZ thick / default line (pt), mark diameter (pt)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": THIN,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": THIN, "ytick.major.width": THIN, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.labelsize": SCRIPT, "ytick.labelsize": SCRIPT, "axes.labelsize": FOOT, "legend.fontsize": SCRIPT,
    "savefig.dpi": 600})


def macros() -> dict[str, float]:
    out = {}
    for line in (REPO / "results/tables/numbers.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(?:\\ensuremath\{)?([-+]?[\d.]+)\}?\}%", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def figure(legend_rows=1, bottom=0.36):
    LEGEND_H = 0.09 + LEGEND_ROW * legend_rows
    fig = plt.figure(figsize=(COL_W, AXIS_H + LEGEND_H))
    # axis box inside the 3.5 in x 4.3 cm block, leaving room for tick and axis labels
    left, right = 0.42, 0.06          # inches; bottom leaves room for the x labels
    ax = fig.add_axes([left / COL_W, bottom / (AXIS_H + LEGEND_H),
                       1 - (left + right) / COL_W, (AXIS_H - bottom - 0.06) / (AXIS_H + LEGEND_H)])
    ax.grid(True, which="major", color=GRID, linewidth=THIN)
    ax.set_axisbelow(True)
    return fig, ax


def legend(ax, ncol, order=None):
    h, l = ax.get_legend_handles_labels()
    if order:
        h, l = [h[i] for i in order], [l[i] for i in order]
    ax.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol, frameon=False,
              handlelength=2.2, columnspacing=1.2, handletextpad=0.5, borderaxespad=0)


def precision_profile(M):
    keys = ("Fp", "PerTensor", "PerChannel")
    series = [("Tree-to-Spike", [M[f"JNPcTts{k}IotMean"] for k in keys], BLUE70, "o"),
              ("DNN-Tree", [M[f"JNPcDnn{k}IotMean"] for k in keys], ORANGE80, "s"),
              ("PLE-MLP", [M[f"JNPle{k}IotMean"] for k in keys], TEAL80, "^")]
    fig, ax = figure(bottom=0.24)   # no x-axis title
    x = [0, 1, 2]
    for name, y, color, marker in series:
        ax.plot(x, y, color=color, linewidth=THICK, marker=marker, markersize=MARK, markerfacecolor=color,
                markeredgecolor=color, label=name, clip_on=False, zorder=3)
    ax.set_xlim(-0.35, 2.35)   # room for the outer category labels, as pgfplots enlarges its limits
    ax.set_xticks(x, ["fp32", "Per-tensor int8", "Per-channel int8"])
    ax.set_ylim(82.5, 93)
    ax.set_yticks([84, 86, 88, 90, 92])
    ax.set_ylabel("Mean macro-F1 (%)")
    legend(ax, 3)
    fig.savefig(OUT / "precision_profile.pdf")
    plt.close(fig)


def memory_band(M):
    words = ("Five", "Ten", "Twenty", "Fifty", "Hundred")
    sram = [5, 10, 20, 50, 100]
    series = [("Cold/deployed", "DepCold", BLUE75, "o", True, "-"),
              ("Reuse/deployed", "DepReuse", ORANGE80, "s", True, "-"),
              ("Cold/full", "FullCold", BLUE75, "o", False, (0, (3, 3))),
              ("Reuse/full", "FullReuse", ORANGE80, "s", False, (0, (3, 3)))]
    fig, ax = figure(legend_rows=2)
    for name, key, color, marker, filled, ls in series:
        y = [M[f"JNBand{key}{w}"] for w in words]
        ax.plot(sram, y, color=color, linewidth=THICK if filled else THIN * 1.5, linestyle=ls, marker=marker,
                markersize=MARK, markerfacecolor=color if filled else "white", markeredgecolor=color,
                markeredgewidth=THIN * 1.5, label=name, zorder=3)
    ax.plot([4, 115], [1, 1], color="black", linewidth=THIN, linestyle=(0, (1, 1)), zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(4, 115)
    ax.xaxis.set_major_locator(FixedLocator(sram))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(s) for s in sram])
    ax.set_ylim(0.35, 1.5)
    ax.set_yticks([0.5, 0.75, 1, 1.25, 1.5], ["0.5", "0.75", "1", "1.25", "1.5"])
    ax.set_xlabel("SRAM cost (pJ/access)", labelpad=1)
    ax.set_ylabel("DNN-Tree / Tree-to-Spike")
    legend(ax, 2, order=[0, 2, 1, 3])   # matplotlib fills columns first; this gives pgfplots' row order
    fig.savefig(OUT / "memory_band_sensitivity.pdf")
    plt.close(fig)


def main():
    M = macros()
    OUT.mkdir(exist_ok=True)
    precision_profile(M)
    memory_band(M)
    print("[wrote] Figures/precision_profile.pdf, Figures/memory_band_sensitivity.pdf")


if __name__ == "__main__":
    main()
