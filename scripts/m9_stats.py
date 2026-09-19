#!/usr/bin/env python3
"""M9: Friedman / Iman-Davenport / Nemenyi over the four models, the
Tree-to-Spike vs DNN-Tree equivalence analysis, and the CD diagram.

Input : results/diagnostics/per_seed_all_cells.csv (dataset, model, seed, f1),
        test macro-F1 as reported (present classes only), 11 datasets x 4 models
        x 5 seeds. Blocks are datasets; each block value is the 5-seed mean.
Output: results/Journal_results/figures/cd_diagram.{pdf,png}
        results/diagnostics/m9_stats.txt (everything printed below)

Re-run after any change to per-dataset means (in particular after the M3
quantization analysis).

Changes from the draft in the manuscript comment block:
  * x-axis orientation: the draft's set_xlim(hi, lo) put rank k on the left while
    labelling the best models "left"; here rank 1 is on the left.
  * cliques: the draft drew a bar for every start index, re-drawing sub-cliques
    already covered by a longer bar; only maximal cliques are drawn.
  * label rows: inner labels sit lower than outer ones on both sides, so no
    horizontal leader crosses another model's vertical line.

    python scripts/m9_stats.py
"""
from __future__ import annotations

import csv
import io
import statistics as st
import subprocess
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results/diagnostics/per_seed_all_cells.csv"
FIG = REPO / "results/Journal_results/figures/cd_diagram"
TXT = REPO / "results/diagnostics/m9_stats.txt"

MODELS = ["dnn", "treespike", "vanilla", "tasnn"]
NAMES = {"vanilla": "Vanilla SNN", "treespike": "Tree-to-Spike",
         "dnn": "DNN-Tree", "tasnn": "TASNN"}
ALPHA_Q = 2.569            # Nemenyi q_0.05 for k=4 (Demsar 2006, Table 5a)
MARGIN = 1.0               # equivalence margin, macro-F1 points


def load():
    cells = {}
    for r in csv.DictReader(open(SRC)):
        cells.setdefault((r["dataset"], r["model"]), []).append(float(r["f1"]))
    datasets = sorted({d for d, _ in cells})
    for d in datasets:
        for m in MODELS:
            n = len(cells.get((d, m), []))
            assert n == 5, f"{d}/{m}: {n} seeds"
    M = np.array([[st.mean(cells[(d, m)]) for m in MODELS] for d in datasets])
    return datasets, M


def analyse(M):
    R = np.array([stats.rankdata(-row) for row in M])      # 1 = best; ties averaged
    N, k = M.shape
    Rj = R.mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(Rj ** 2) - k * (k + 1) ** 2 / 4)
    p_chi = stats.chi2.sf(chi2, k - 1)
    Ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_F = stats.f.sf(Ff, k - 1, (k - 1) * (N - 1))
    CD = ALPHA_Q * np.sqrt(k * (k + 1) / (6 * N))

    print(f"N={N} datasets, k={k} models")
    print("mean ranks: " + "  ".join(f"{NAMES[m]}={Rj[j]:.3f}" for j, m in enumerate(MODELS)))
    print(f"Friedman chi2_F={chi2:.3f} df={k-1} p={p_chi:.3e}")
    print(f"Iman-Davenport F_F={Ff:.3f} df=({k-1},{(k-1)*(N-1)}) p={p_F:.3e}")
    print(f"Nemenyi CD={CD:.3f}")
    for a in range(k):
        for b in range(a + 1, k):
            d = abs(Rj[a] - Rj[b])
            print(f"  {NAMES[MODELS[a]]} vs {NAMES[MODELS[b]]}: {d:.3f} "
                  f"{'SIG' if d > CD else 'ns'}")

    # --- equivalence: Tree-to-Spike vs DNN-Tree ---
    it, idn = MODELS.index("treespike"), MODELS.index("dnn")
    diff = M[:, it] - M[:, idn]
    print(f"\nTree-to-Spike - DNN-Tree: mean diff {diff.mean():.3f}  "
          f"median {np.median(diff):.3f}  sd {diff.std(ddof=1):.3f}")
    print("Wilcoxon p=%.4f" % stats.wilcoxon(M[:, it], M[:, idn]).pvalue)
    t_lo = stats.ttest_1samp(diff, -MARGIN)
    t_hi = stats.ttest_1samp(diff, MARGIN)
    p_lo = t_lo.pvalue / 2 if t_lo.statistic > 0 else 1 - t_lo.pvalue / 2
    p_hi = t_hi.pvalue / 2 if t_hi.statistic < 0 else 1 - t_hi.pvalue / 2
    print(f"TOST (margin +/-{MARGIN}): p={max(p_lo, p_hi):.4f}")
    rng = np.random.default_rng(42)
    bs = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"bootstrap 95% CI [{lo:.3f}, {hi:.3f}]  "
          f"contained in margin: {lo > -MARGIN and hi < MARGIN}")
    return Rj, CD


def maximal_cliques(srt, CD):
    """Maximal runs [i, j] of the sorted ranks with srt[j] - srt[i] <= CD."""
    k = len(srt)
    runs = []
    for i in range(k):
        j = i
        while j + 1 < k and srt[j + 1] - srt[i] <= CD:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in runs):
            runs.append((i, j))
    return runs


def draw(Rj, CD):
    k = len(Rj)
    order = np.argsort(Rj)
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 1.25))
    lo_r, hi_r = 1, k
    ax.set_xlim(lo_r - 0.25, hi_r + 0.25)          # rank 1 (best) on the left
    ax.set_ylim(0.3, 1.3)
    ax.axis("off")
    y_axis = 0.85
    ax.plot([lo_r, hi_r], [y_axis, y_axis], "k-", lw=1)
    for t in range(lo_r, hi_r + 1):
        ax.plot([t, t], [y_axis, y_axis + 0.05], "k-", lw=1)
        ax.text(t, y_axis + 0.08, str(t), ha="center", va="bottom", fontsize=8)
    for t in np.arange(lo_r + 0.5, hi_r, 1.0):
        ax.plot([t, t], [y_axis, y_axis + 0.025], "k-", lw=0.6)

    half = k // 2
    for pos, j in enumerate(order):
        left = pos < half
        depth = pos if left else (k - 1 - pos)      # 0 = outermost label
        y = 0.55 - 0.16 * depth
        x_end = lo_r - 0.2 if left else hi_r + 0.2
        ax.plot([Rj[j], Rj[j]], [y_axis, y], "k-", lw=0.8)
        ax.plot([Rj[j], x_end], [y, y], "k-", lw=0.8)
        ax.text(x_end, y, f"{NAMES[MODELS[j]]} ({Rj[j]:.2f}) ", fontsize=7,
                ha="right" if left else "left", va="center")

    srt = Rj[order]
    y_bar = 0.74
    for i, j in maximal_cliques(srt, CD):
        ax.plot([srt[i] - 0.03, srt[j] + 0.03], [y_bar, y_bar], "k-", lw=2.5,
                solid_capstyle="butt")
        y_bar -= 0.06

    # CD bracket above the axis, anchored at rank 1
    yc = 1.17
    ax.plot([lo_r, lo_r + CD], [yc, yc], "k-", lw=0.8)
    for x in (lo_r, lo_r + CD):
        ax.plot([x, x], [yc - 0.025, yc + 0.025], "k-", lw=0.8)
    ax.text(lo_r + CD / 2, yc + 0.03, f"CD = {CD:.2f}", ha="center", va="bottom", fontsize=7)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "scripts/m9_stats.py"}
    fig.savefig(FIG.with_suffix(".pdf"), bbox_inches="tight", metadata=meta)
    fig.savefig(FIG.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    buf = io.StringIO()
    with redirect_stdout(buf):
        datasets, M = load()
        print("datasets: " + ", ".join(datasets))
        Rj, CD = analyse(M)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=REPO, text=True).strip()
    except Exception:
        sha = "unknown"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = (f"# generated {ts} by scripts/m9_stats.py at git {sha}\n"
            f"# source: {SRC.relative_to(REPO)}\n")
    TXT.write_text(head + buf.getvalue())
    print(head + buf.getvalue(), end="")
    draw(Rj, CD)
    print(f"[wrote] {FIG.with_suffix('.pdf').relative_to(REPO)} (+ .png), "
          f"{TXT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
