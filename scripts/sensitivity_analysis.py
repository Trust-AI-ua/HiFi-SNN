#!/usr/bin/env python3
"""Sensitivity sweep analysis: is Tree-to-Spike's default stable in its neighbourhood?

This is a SENSITIVITY sweep, not 700 independent experiments, so the statistics are
handled accordingly:

  * dataset is the generalisation unit (4 datasets);
  * seeds are within-configuration optimisation repetitions (5 per cell);
  * every arm-vs-default comparison is PAIRED on (dataset, seed) -- the same
    partition and the same optimisation seed on both sides;
  * no significance test treats the 700 cells as independent samples.

The default configuration is NOT part of the ladder: it is the shipped Tree-to-Spike
run for the same dataset and seed (`results/Tree-to-Spike/<ds>/seed_<s>/`), so the
paired difference isolates exactly the parameter the arm overrides. The ladder arms are
10-epoch runs, so CICIoT2023's default is its 10-epoch run
(`results/fixed_budget_ciciot2023/`), not the 20-epoch block of amendment A1.

The question is robustness -- broad plateaus, catastrophic regions, cost trade-offs,
and whether the default sits inside a stable region rather than on an isolated
optimum. It is NOT "which arm scores highest".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from run_ladder import ARMS, DATASETS, SEEDS          # noqa: E402

OUT = REPO / "results" / "ladder"

# Arms grouped by the design axis they perturb, with the default value shown.
FAMILY = {
    "T (timesteps, default 20)": ["T10", "T40", "T80"],
    "pop_max (default 3)": ["M2", "M5"],
    "prune_quantile (default 0.20)": ["q10", "q40"],
    "jitter (default 0.12)": ["d000", "d006", "d024"],
    "tau (KD temperature, default 2.0)": ["tau1", "tau4"],
    "a_kd (default 1.0)": ["akd05", "akd20", "rank_only"],
    "a_rank (default 0.3)": ["arank01", "arank06", "kd_only"],
    "a_sparse (default 0.001)": ["asparse1e4", "asparse1e2", "no_sparse"],
    "a_fp (default 0.5)": ["afp025", "afp100", "no_fp"],
    "epochs (default 10)": ["epochs30"],
    "encoder ablations": ["raw", "raw_full", "tree_only", "prune_only", "alloc_only"],
    "objective ablations": ["tree_ce"],
    "class weighting": ["class_weighted", "class_weighted_nokd", "cw_nokd_rankfixed"],
    "precision (default int8-grid)": ["fp32"],
}


def load_cell(path: Path) -> dict | None:
    if not path.exists():
        return None
    r = json.load(open(path))
    tp = r["test_plus"]
    m, b = tp["metrics"], (tp.get("binary_ids_metrics") or {})
    sp = tp.get("spike_stats") or {}
    return {
        "macro_f1": 100 * m["f1_macro"], "mcc": 100 * m["matthews_corrcoef"],
        "FAR": 100 * b.get("false_alarm_rate", np.nan),
        "DR": 100 * b.get("detection_rate", np.nan),
        "synops": sp.get("synops_per_sample", np.nan),
        "spike_density": 100 * sp.get("spike_density", np.nan),
        "spikes_per_sample": sp.get("energy_spikes_per_sample", np.nan),
        "channels": r["data"].get("encoder_channels", np.nan),
        "params": r["model_size"].get("trainable_parameters", np.nan),
    }


def build() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for s in SEEDS:
            root = "results/fixed_budget_ciciot2023" if ds == "ciciot2023" else "results"
            d = load_cell(REPO / f"{root}/Tree-to-Spike/{ds}/seed_{s}/results.json")
            if d:
                rows.append({"arm": "DEFAULT", "dataset": ds, "seed": s, **d})
            for a in ARMS:
                c = load_cell(REPO / f"results/ladder/{ds}/{a}/seed_{s}/results.json")
                if c:
                    rows.append({"arm": a, "dataset": ds, "seed": s, **c})
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, arm: str, col: str) -> np.ndarray:
    """arm - DEFAULT, paired on (dataset, seed)."""
    k = ["dataset", "seed"]
    a = df[df.arm == arm].set_index(k)[col]
    b = df[df.arm == "DEFAULT"].set_index(k)[col]
    idx = a.index.intersection(b.index)
    return (a.reindex(idx) - b.reindex(idx)).dropna().to_numpy()


if __name__ == "__main__":
    df = build()
    n_def = int((df.arm == "DEFAULT").sum())
    print(f"loaded {len(df)} cells: {n_def} default + {len(df)-n_def} ladder "
          f"({df.dataset.nunique()} datasets x {df.seed.nunique()} seeds)\n")
    df.to_csv(OUT / "sensitivity_cells.csv", index=False)

    dflt = df[df.arm == "DEFAULT"]
    print("=" * 100)
    print("DEFAULT CONFIGURATION reference (median over 4 datasets x 5 seeds)")
    print("=" * 100)
    print(f"  macro-F1 {dflt.macro_f1.median():6.2f}   MCC {dflt.mcc.median():6.2f}   "
          f"FAR {dflt.FAR.median():5.2f}   synops/sample {dflt.synops.median():9.0f}   "
          f"spike density {dflt.spike_density.median():5.2f}%")
    print("  per dataset macro-F1: " + "  ".join(
        f"{d}={g.macro_f1.median():.2f}" for d, g in dflt.groupby("dataset")))

    # ---------------------------------------------------------------- per arm
    recs = []
    for fam, arms in FAMILY.items():
        for a in arms:
            g = df[df.arm == a]
            if not len(g):
                continue
            d_f1 = paired(df, a, "macro_f1")
            d_syn = paired(df, a, "synops")
            q1, q3 = np.percentile(g.macro_f1, [25, 75])
            # dataset is the generalisation unit: mean the paired delta within each
            # dataset first, then summarise across the 4 datasets
            per_ds = [paired(df[df.dataset == ds], a, "macro_f1").mean()
                      for ds in DATASETS if len(paired(df[df.dataset == ds], a, "macro_f1"))]
            recs.append(dict(
                family=fam, arm=a, n=len(g),
                med_f1=g.macro_f1.median(), iqr=q3 - q1,
                med_mcc=g.mcc.median(), med_far=g.FAR.median(),
                med_synops=g.synops.median(), med_density=g.spike_density.median(),
                med_channels=g.channels.median(),
                d_f1_mean=d_f1.mean() if len(d_f1) else np.nan,
                d_f1_lo=np.percentile(d_f1, 2.5) if len(d_f1) else np.nan,
                d_f1_hi=np.percentile(d_f1, 97.5) if len(d_f1) else np.nan,
                pct_synops=100 * g.synops.median() / dflt.synops.median(),
                n_ds_worse=int(sum(1 for x in per_ds if x < 0)),
                n_ds=len(per_ds),
                ds_delta_min=min(per_ds) if per_ds else np.nan,
                ds_delta_max=max(per_ds) if per_ds else np.nan))
    t = pd.DataFrame(recs)
    t.to_csv(OUT / "sensitivity_by_arm.csv", index=False)

    print("\n" + "=" * 100)
    print("PAIRED DELTA vs DEFAULT (macro-F1 pp) — paired on dataset+seed; "
          "95% interval over 20 pairs")
    print("=" * 100)
    for fam in FAMILY:
        sub = t[t.family == fam]
        if not len(sub):
            continue
        print(f"\n{fam}")
        print(f"  {'arm':20s} {'medF1':>7} {'IQR':>6} {'ΔF1':>7} {'95% CI':>16} "
              f"{'ΔMCC-med':>9} {'FAR':>6} {'synops%':>8} {'ds worse':>9}")
        for _, r in sub.iterrows():
            print(f"  {r.arm:20s} {r.med_f1:7.2f} {r.iqr:6.2f} {r.d_f1_mean:+7.2f} "
                  f"[{r.d_f1_lo:+6.2f},{r.d_f1_hi:+6.2f}] {r.med_mcc:9.2f} "
                  f"{r.med_far:6.2f} {r.pct_synops:8.1f} {r.n_ds_worse:>4}/{r.n_ds}")

    # ------------------------------------------------------- robustness verdict
    print("\n" + "=" * 100)
    print("ROBUSTNESS: is the default inside a stable region?")
    print("=" * 100)
    tuning = t[~t.family.str.contains("ablation|class weighting|precision")]
    stable = tuning[tuning.d_f1_mean.abs() <= 1.0]
    print(f"  continuous-hyperparameter arms: {len(tuning)}")
    print(f"  within +/-1.0 pp of default:    {len(stable)}  "
          f"({100*len(stable)/max(len(tuning),1):.0f}%)")
    print(f"  arms BETTER than default:       {int((tuning.d_f1_mean > 0).sum())}")
    print(f"  arms worse by >1 pp:            {int((tuning.d_f1_mean < -1).sum())}")
    print(f"  arms worse by >5 pp:            {int((tuning.d_f1_mean < -5).sum())}")
    print("\n  best tuning arms (may be noise, not a recommendation):")
    for _, r in tuning.nlargest(3, "d_f1_mean").iterrows():
        print(f"    {r.arm:20s} {r.d_f1_mean:+.2f} pp  CI[{r.d_f1_lo:+.2f},{r.d_f1_hi:+.2f}]"
              f"  synops {r.pct_synops:.0f}% of default")
    print("\n  catastrophic regions (< -5 pp):")
    cat = t[t.d_f1_mean < -5]
    if len(cat):
        for _, r in cat.iterrows():
            print(f"    {r.arm:20s} {r.d_f1_mean:+.2f} pp  ({r.family})")
    else:
        print("    none among tuning arms")

    print("\n  cost/quality trade-off — arms cheaper than default with <=1 pp loss:")
    cheap = t[(t.pct_synops < 95) & (t.d_f1_mean >= -1.0)].sort_values("pct_synops")
    for _, r in cheap.iterrows():
        print(f"    {r.arm:20s} synops {r.pct_synops:5.1f}% of default, "
              f"ΔF1 {r.d_f1_mean:+.2f} pp")

    print(f"\nwrote {OUT}/sensitivity_cells.csv and sensitivity_by_arm.csv")
