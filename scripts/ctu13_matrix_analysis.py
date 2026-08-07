#!/usr/bin/env python3
"""Analyse the 420-run CTU-13 matrix: three effects, per fold and per scenario.

Folds are the statistical unit -- seeds measure optimisation variability within a
fold. Effects are therefore computed per (fold, seed) as paired differences, then
averaged over seeds within a fold, and only then summarised across folds. A single
grand mean is never the headline: fold 1 averages a 56%-FAR scenario against a
0.12%-DR scenario, so unqualified means hide opposite failure modes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "results/ctu13/matrix"
OUT = REPO / "results/ctu13"
MODELS = ["gbdt", "vanilla", "treespike", "dnn", "tasnn"]
NICE = {"gbdt": "GBDT teacher", "vanilla": "Vanilla SNN", "treespike": "Tree-to-Spike",
        "dnn": "DNN-Tree", "tasnn": "TASNN"}
CONDS = ["legacy_random", "causal_random", "causal_scenario"]


def load() -> tuple:
    rows, groups = [], []
    for p in sorted(MATRIX.rglob("results.json")):
        rel = p.relative_to(MATRIX).parts       # cond/fs/model/foldN/seed_S
        cond, fs, model, fold, seed = rel[0], rel[1], rel[2], rel[3], rel[4]
        r = json.load(open(p))
        tp = r["test_plus"]; m = tp["metrics"]; b = tp.get("binary_ids_metrics") or {}
        base = dict(condition=cond, features=fs, model=model,
                    fold=int(fold.replace("fold", "")),
                    seed=int(seed.replace("seed_", "")))
        rows.append({**base,
                     "macro_f1": 100 * m["f1_macro"],
                     "mcc": 100 * m["matthews_corrcoef"],
                     "DR": 100 * b.get("detection_rate", np.nan),
                     "FAR": 100 * b.get("false_alarm_rate", np.nan),
                     "dupfree_macro_f1": 100 * m["duplicate_free"]["metrics"]["macro_f1"]
                     if "duplicate_free" in m else np.nan,
                     "dupfree_delta_pp": m["duplicate_free"]["delta_macro_f1_pp"]
                     if "duplicate_free" in m else np.nan,
                     "dupfree_excluded_pct": m["duplicate_free"]["pct_excluded"]
                     if "duplicate_free" in m else np.nan})
        for g, blk in m.get("per_group", {}).get("groups", {}).items():
            groups.append({**base, "scenario": int(g),
                           "macro_f1": 100 * blk["macro_f1"],
                           "DR": 100 * blk["DR"], "FAR": 100 * blk["FAR"],
                           "n": blk["n"], "support_normal": blk["support_normal"],
                           "support_attack": blk["support_attack"],
                           "pred_attack_frac": blk["pred_attack_frac"]})
    return pd.DataFrame(rows), pd.DataFrame(groups)


def paired(df: pd.DataFrame, fs: str, a: str, b: str, metric="macro_f1") -> pd.DataFrame:
    """b - a, paired on (model, fold, seed). Both conditions ran identical
    partitions per (condition, fold), so the pairing is exact."""
    k = ["model", "fold", "seed"]
    left = df[(df.features == fs) & (df.condition == a)].set_index(k)[metric]
    right = df[(df.features == fs) & (df.condition == b)].set_index(k)[metric]
    return (right - left).rename("delta").reset_index()


def fmt(x, nd=2):
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


if __name__ == "__main__":
    df, gdf = load()
    print(f"loaded {len(df)} runs, {len(gdf)} scenario-level records\n")
    assert len(df) == 420, f"expected 420 runs, found {len(df)}"

    # ---------------------------------------------------------------- headline
    print("=" * 78)
    print("1. MATRIX A -- combined features, macro-F1 (mean over 5 seeds, by fold)")
    print("=" * 78)
    a = df[df.features == "combined"]
    for cond in CONDS:
        sub = a[a.condition == cond]
        piv = sub.pivot_table(index="model", columns="fold", values="macro_f1")
        piv = piv.reindex(MODELS)
        piv["mean"] = piv.mean(axis=1)
        piv["min_fold"] = piv[[0, 1, 2, 3]].min(axis=1)
        print(f"\n-- {cond}")
        print(piv.round(2).to_string())

    # ------------------------------------------------------------ three effects
    print("\n" + "=" * 78)
    print("2. THE THREE EFFECTS (macro-F1 pp, paired on fold+seed)")
    print("=" * 78)

    agg = paired(df, "combined", "legacy_random", "causal_random")
    scn = paired(df, "combined", "causal_random", "causal_scenario")
    print("\n-- delta_aggregation = causal_random - legacy_random  (combined)")
    t = agg.groupby("model")["delta"].agg(["mean", "std", "min", "max"]).reindex(MODELS)
    print(t.round(2).to_string())
    print("\n   by fold:")
    print(agg.pivot_table(index="model", columns="fold", values="delta")
             .reindex(MODELS).round(2).to_string())

    print("\n-- delta_scenario = causal_scenario - causal_random  (combined)")
    t = scn.groupby("model")["delta"].agg(["mean", "std", "min", "max"]).reindex(MODELS)
    print(t.round(2).to_string())
    print("\n   by fold:")
    print(scn.pivot_table(index="model", columns="fold", values="delta")
             .reindex(MODELS).round(2).to_string())

    # host interaction, Matrix B models only
    print("\n-- delta_host = (comb_scen - base_scen) - (comb_rand - base_rand)")
    k = ["model", "fold", "seed"]
    pieces = {}
    for fs in ("base", "combined"):
        for cond in ("causal_random", "causal_scenario"):
            s = df[(df.features == fs) & (df.condition == cond)]
            pieces[(fs, cond)] = s.set_index(k)["macro_f1"]
        # align
    idx = pieces[("base", "causal_random")].index
    hi = ((pieces[("combined", "causal_scenario")].reindex(idx)
           - pieces[("base", "causal_scenario")].reindex(idx))
          - (pieces[("combined", "causal_random")].reindex(idx)
             - pieces[("base", "causal_random")].reindex(idx))).rename("delta_host")
    hi = hi.dropna().reset_index()
    t = hi.groupby("model")["delta_host"].agg(["mean", "std", "min", "max"])
    print(t.round(2).to_string())
    print("\n   by fold:")
    print(hi.pivot_table(index="model", columns="fold", values="delta_host")
            .round(2).to_string())

    # component parts, so the interaction is readable
    print("\n   components (macro-F1, mean over folds+seeds):")
    comp = df[df.model.isin(hi.model.unique())].pivot_table(
        index="model", columns=["condition", "features"], values="macro_f1")
    print(comp.round(2).to_string())

    # -------------------------------------------------------- per scenario
    print("\n" + "=" * 78)
    print("3. PER-SCENARIO, causal_scenario / combined (mean over seeds)")
    print("=" * 78)
    g = gdf[(gdf.condition == "causal_scenario") & (gdf.features == "combined")]
    piv = g.pivot_table(index="scenario", columns="model", values="macro_f1")
    piv = piv.reindex(columns=MODELS)
    piv["spread"] = piv.max(axis=1) - piv.min(axis=1)
    sup = g.groupby("scenario")[["support_normal", "support_attack"]].first()
    piv.insert(0, "botnet%", (100 * sup.support_attack
                              / (sup.support_normal + sup.support_attack)).round(1))
    print(piv.round(2).to_string())

    print("\n-- FAR by scenario (causal_scenario / combined)")
    print(g.pivot_table(index="scenario", columns="model", values="FAR")
           .reindex(columns=MODELS).round(2).to_string())

    # base vs combined per scenario, Matrix B models
    print("\n-- host-context effect per scenario: combined - base (causal_scenario)")
    gb = gdf[gdf.condition == "causal_scenario"]
    bb = gb.pivot_table(index=["scenario", "model"], columns="features",
                        values="macro_f1").dropna()
    bb["delta"] = bb["combined"] - bb["base"]
    print(bb.reset_index().pivot_table(index="scenario", columns="model",
                                       values="delta").round(2).to_string())

    # ------------------------------------------------------- duplicate-free
    print("\n" + "=" * 78)
    print("4. DUPLICATE-FREE SENSITIVITY")
    print("=" * 78)
    d = df.groupby("condition")[["dupfree_excluded_pct", "dupfree_delta_pp"]].agg(
        ["mean", "max"])
    print(d.round(4).to_string())
    print(f"\nlargest |delta| anywhere: {df.dupfree_delta_pp.abs().max():.4f} pp")

    df.to_csv(OUT / "matrix_runs.csv", index=False)
    gdf.to_csv(OUT / "matrix_scenarios.csv", index=False)
    agg.assign(effect="aggregation").to_csv(OUT / "matrix_delta_aggregation.csv", index=False)
    scn.assign(effect="scenario").to_csv(OUT / "matrix_delta_scenario.csv", index=False)
    hi.to_csv(OUT / "matrix_delta_host.csv", index=False)
    print(f"\nwrote {OUT}/matrix_runs.csv, matrix_scenarios.csv, matrix_delta_*.csv")
