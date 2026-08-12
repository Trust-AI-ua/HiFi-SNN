#!/usr/bin/env python3
"""Reproduce the crossed 2x2 encoder x objective attribution from committed results.

Cell mapping (all four cells are committed runs; nothing is retrained):

                    | CE objective            | full objective
  Raw TTFS encoder  | ladder arm `raw`        | ladder arm `raw_full`
  Tree TTFS encoder | ladder arm `tree_ce`    | shipped Tree-to-Spike default

  `raw`      = encoder_mode raw + a_kd/a_rank/a_fp/a_sparse all 0  -> Raw + CE
  `raw_full` = encoder_mode raw, objective untouched               -> Raw + full
  `tree_ce`  = tree encoder + all four loss weights 0              -> Tree + CE
  default    = tree encoder + full objective                       -> Tree + full

Emits both main effects, the interaction, the MCC encoder effect, and the two
conditional encoder contrasts (under CE and under the full objective) whose
closeness is the additivity evidence.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from statistics_utils import (BOOTSTRAP_DESIGN, hierarchical_bootstrap,  # noqa: E402
                             paired_by_dataset)

PRIMARY = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
CELLS = {("raw", "ce"): ("ladder", "raw"), ("raw", "full"): ("ladder", "raw_full"),
         ("tree", "ce"): ("ladder", "tree_ce"), ("tree", "full"): ("default", None)}


def collect(kind, arm, metric="f1_macro", scale=100.0):
    out = defaultdict(dict)
    pat = (str(REPO / f"results/Tree-to-Spike/*/seed_*/results.json") if kind == "default"
           else str(REPO / f"results/ladder/*/{arm}/seed_*/results.json"))
    for p in glob.glob(pat):
        parts = p.split("/")
        ds = parts[-3] if kind == "default" else parts[-4]
        if ds not in PRIMARY:
            continue
        seed = int(parts[-2].split("_")[1])
        out[ds][seed] = scale * json.load(open(p))["test_plus"]["metrics"][metric]
    return out


def combine(a, b, sign_b=-1.0):
    """Per-(ds,seed) linear combination a + sign_b*b, kept as {ds:{seed:val}}."""
    out = defaultdict(dict)
    for ds in set(a) & set(b):
        for s in set(a[ds]) & set(b[ds]):
            out[ds][s] = a[ds][s] + sign_b * b[ds][s]
    return out


def avg(a, b):
    out = defaultdict(dict)
    for ds in set(a) & set(b):
        for s in set(a[ds]) & set(b[ds]):
            out[ds][s] = 0.5 * (a[ds][s] + b[ds][s])
    return out


def cellmeans(metric, scale=100.0):
    return {k: collect(*v, metric=metric, scale=scale) for k, v in CELLS.items()}


def effects(metric="f1_macro", scale=100.0):
    C = cellmeans(metric, scale)
    rc, rf = C[("raw", "ce")], C[("raw", "full")]
    tc, tf = C[("tree", "ce")], C[("tree", "full")]
    res = {}
    # encoder main effect: mean(tree) - mean(raw), paired per (ds, seed)
    res["encoder_main"] = hierarchical_bootstrap(
        paired_by_dataset(avg(tc, tf), avg(rc, rf)))
    res["objective_main"] = hierarchical_bootstrap(
        paired_by_dataset(avg(rf, tf), avg(rc, tc)))
    res["interaction"] = hierarchical_bootstrap(
        paired_by_dataset(combine(tf, tc), combine(rf, rc)))
    res["encoder_under_ce"] = hierarchical_bootstrap(paired_by_dataset(tc, rc))
    res["encoder_under_full"] = hierarchical_bootstrap(paired_by_dataset(tf, rf))
    cm = {f"{e}_{o}": round(float(np.mean(
        [np.mean(list(C[(e, o)][ds].values())) for ds in sorted(C[(e, o)])])), 4)
        for e, o in CELLS}
    return res, cm


def threshold_placement():
    """Tree split thresholds vs matched equal-frequency (quantile) thresholds.

    Identical channel budget and identical objective -- only edge PLACEMENT differs.
    Distinct from both the 2x2 encoder effect and the raw_full arm.
    """
    tree = collect("default", None)
    quant = defaultdict(dict)
    for p in glob.glob(str(REPO / "results/encoder_ablation/*/quantile/seed_*/results.json")):
        parts = p.split("/")
        ds = parts[-4]
        if ds not in PRIMARY:
            continue
        quant[ds][int(parts[-2].split("_")[1])] = \
            100 * json.load(open(p))["test_plus"]["metrics"]["f1_macro"]
    return hierarchical_bootstrap(paired_by_dataset(tree, quant))


if __name__ == "__main__":
    out = {"scope": "four primary datasets x 5 seeds",
           "cell_mapping": {f"{e}+{o}": (v[1] or "shipped Tree-to-Spike default")
                            for (e, o), v in CELLS.items()},
           "bootstrap_design": BOOTSTRAP_DESIGN}
    print("=== crossed 2x2: encoder x objective ===\n")
    for metric, scale, lab in (("f1_macro", 100.0, "macro-F1"),
                               ("matthews_corrcoef", 100.0, "MCC")):
        res, cm = effects(metric, scale)
        out[lab] = {"cell_means": cm, "effects": {k: v for k, v in res.items()}}
        print(f"-- {lab} cell means")
        print(f"   Raw  + CE   {cm['raw_ce']:6.2f}      Raw  + full {cm['raw_full']:6.2f}")
        print(f"   Tree + CE   {cm['tree_ce']:6.2f}      Tree + full {cm['tree_full']:6.2f}")
        for k in ("encoder_main", "objective_main", "interaction",
                  "encoder_under_ce", "encoder_under_full"):
            r = res[k]
            print(f"   {k:20s} {r['estimate_pp']:+7.2f}  "
                  f"CI [{r['ci95_pp'][0]:+7.2f}, {r['ci95_pp'][1]:+7.2f}]")
        if lab == "macro-F1":
            d = abs(res["encoder_under_ce"]["estimate_pp"]
                    - res["encoder_under_full"]["estimate_pp"])
            print(f"   additivity: conditional encoder effects differ by {d:.2f} pp")
        print()
    tp = threshold_placement()
    out["threshold_placement"] = {
        "comparison": "tree split thresholds minus matched quantile thresholds, macro-F1",
        "note": "identical channel budget and identical objective; only edge placement differs",
        **tp}
    print(f"-- threshold placement (tree - quantile, matched budget)")
    print(f"   estimate {tp['estimate_pp']:+.2f} pp   "
          f"95% CI [{tp['ci95_pp'][0]:+.2f}, {tp['ci95_pp'][1]:+.2f}]")
    print(f"   per dataset: " + "  ".join(f"{k}={v:+.2f}" for k,v in tp["per_dataset_mean_pp"].items()) + "\n")

    p = REPO / "results/Journal_results/new_run_results/repro_crossed_2x2.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"wrote {p.relative_to(REPO)}")
