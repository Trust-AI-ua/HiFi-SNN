#!/usr/bin/env python3
"""Statistics on the IoT subset (the paper's primary scope): descriptive results
and the Friedman omnibus, for the main text. No post-hoc test and no CD diagram:
at N = 4 blocks the Nemenyi critical difference (2.345 for k = 4) exceeds any
separation four datasets can support. The pooled eleven-dataset analysis stays in
scripts/m9_stats.py, for the appendix.

The Friedman chi-square is referred to its asymptotic distribution only for
comparison. At N = 4 that approximation is poor, so the reported p-value is EXACT:
under H0 each block's ranking is equally likely to be any permutation of its
observed rank vector (ties kept), and all combinations are enumerated.

Pairwise comparisons are reported as sign counts over the four datasets, not
tests: the smallest attainable two-sided sign-test p-value at n = 4 is 0.125.

Input : results/diagnostics/per_seed_all_cells.csv (test macro-F1, 5 seeds)
Output: results/diagnostics/m9_iot_stats.txt, results/tables/tab_iot_stats.md

    python scripts/m9_iot.py
"""
from __future__ import annotations

import io
import itertools
import subprocess
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results/diagnostics/per_seed_all_cells.csv"
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
DSN = {"ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024",
       "edgeiiot": "Edge-IIoTset", "toniot": "TON-IoT"}
MODELS = ["treespike", "dnn", "vanilla", "tasnn"]
NAMES = {"treespike": "Tree-to-Spike", "dnn": "DNN-Tree",
         "vanilla": "Vanilla SNN", "tasnn": "TASNN"}


def friedman_chi2(R):
    N, k = R.shape
    Rj = R.mean(0)
    return 12 * N / (k * (k + 1)) * (np.sum(Rj ** 2) - k * (k + 1) ** 2 / 4)


def exact_p(R):
    perms = [sorted(set(itertools.permutations(row))) for row in R]
    obs = friedman_chi2(R)
    hit = tot = 0
    for combo in itertools.product(*perms):
        tot += 1
        hit += friedman_chi2(np.array(combo)) >= obs - 1e-12
    return hit / tot, tot


def main():
    c = pd.read_csv(SRC)
    c = c[c.dataset.isin(IOT)]
    mean = c.pivot_table(index="dataset", columns="model", values="f1", aggfunc="mean")
    sd = c.pivot_table(index="dataset", columns="model", values="f1", aggfunc="std")
    n = c.groupby(["dataset", "model"]).seed.nunique()
    assert (n == 5).all(), n[n != 5]
    M = mean.loc[IOT, MODELS].to_numpy()
    R = np.array([stats.rankdata(-row) for row in M])
    N, k = M.shape
    chi2 = friedman_chi2(R)
    p_asym = stats.chi2.sf(chi2, k - 1)
    Ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_F = stats.f.sf(Ff, k - 1, (k - 1) * (N - 1))
    p_exact, n_perm = exact_p(R)

    print(f"IoT subset: N={N} datasets, k={k} models, 5 seeds per cell, test macro-F1")
    print("\nper-dataset mean +/- SD (rank)")
    for i, d in enumerate(IOT):
        print(f"  {DSN[d]:13s} " + "  ".join(
            f"{NAMES[m]}={mean.loc[d, m]:.2f}+/-{sd.loc[d, m]:.2f} ({R[i, j]:.1f})"
            for j, m in enumerate(MODELS)))
    print("\nmean rank: " + "  ".join(f"{NAMES[m]}={R[:, j].mean():.2f}" for j, m in enumerate(MODELS)))
    print("mean macro-F1: " + "  ".join(f"{NAMES[m]}={M[:, j].mean():.2f}" for j, m in enumerate(MODELS)))
    print(f"\nFriedman chi2_F={chi2:.3f} df={k-1}")
    print(f"  exact p = {p_exact:.4f}  ({n_perm:,} rank configurations enumerated)")
    print(f"  asymptotic p = {p_asym:.4f}  (for comparison only; unreliable at N=4)")
    print(f"Iman-Davenport F_F={Ff:.3f} df=({k-1},{(k-1)*(N-1)}) p={p_F:.4f}")
    print("\nTree-to-Spike against each model, per dataset (sign counts; "
          "smallest attainable two-sided sign-test p at n=4 is 0.125)")
    it = MODELS.index("treespike")
    rows = []
    for j, m in enumerate(MODELS):
        if m == "treespike":
            continue
        diff = M[:, it] - M[:, j]
        wins = int((diff > 0).sum())
        print(f"  vs {NAMES[m]:12s} wins {wins}/4  mean diff {diff.mean():+.2f}  "
              f"range [{diff.min():+.2f}, {diff.max():+.2f}]  "
              f"per dataset " + " ".join(f"{x:+.2f}" for x in diff))
        rows.append(dict(comparison=f"Tree-to-Spike vs {NAMES[m]}", wins=f"{wins}/4",
                         mean_diff=round(float(diff.mean()), 2),
                         **{DSN[d]: round(float(x), 2) for d, x in zip(IOT, diff)}))
    return mean, sd, R, rows, dict(chi2=chi2, p_exact=p_exact, p_asym=p_asym, Ff=Ff, p_F=p_F)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        mean, sd, R, rows, fr = main()
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = f"# generated {ts} by scripts/m9_iot.py at git {sha}\n# source: {SRC.relative_to(REPO)}\n"
    (REPO / "results/diagnostics/m9_iot_stats.txt").write_text(head + buf.getvalue())
    print(head + buf.getvalue(), end="")
    md = ["| Dataset | " + " | ".join(NAMES[m] for m in MODELS) + " |",
          "|---|" + "---|" * len(MODELS)]
    for i, d in enumerate(IOT):
        md.append(f"| {DSN[d]} | " + " | ".join(
            f"{mean.loc[d, m]:.2f} ± {sd.loc[d, m]:.2f} ({R[i, j]:.0f})" for j, m in enumerate(MODELS)) + " |")
    md.append("| **Mean rank** | " + " | ".join(f"{R[:, j].mean():.2f}" for j in range(len(MODELS))) + " |")
    md += ["", f"Friedman chi2_F = {fr['chi2']:.2f} (df 3), exact p = {fr['p_exact']:.4f} "
           f"(asymptotic {fr['p_asym']:.4f}); Iman-Davenport F_F = {fr['Ff']:.2f}, p = {fr['p_F']:.4f}. "
           "Macro-F1, five-seed mean ± SD (rank). No post-hoc test at N = 4.", "",
           "| Comparison | Wins | Mean diff | " + " | ".join(DSN[d] for d in IOT) + " |",
           "|---|---|---|" + "---|" * len(IOT)]
    for r in rows:
        md.append(f"| {r['comparison']} | {r['wins']} | {r['mean_diff']:+.2f} | "
                  + " | ".join(f"{r[DSN[d]]:+.2f}" for d in IOT) + " |")
    (REPO / "results/tables/tab_iot_stats.md").write_text("\n".join(md) + "\n")
    print("[wrote] results/diagnostics/m9_iot_stats.txt, results/tables/tab_iot_stats.md")
