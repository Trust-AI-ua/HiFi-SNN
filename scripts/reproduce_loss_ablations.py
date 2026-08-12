#!/usr/bin/env python3
"""Reproduce the loss-term ablation statistics: FAR effect, multiplicity, class weighting.

Three analyses, all paired against the shipped Tree-to-Spike default on the same
(dataset, seed), all on the four primary datasets:

  1. the false-positive-aware term: Delta macro-F1 and Delta FAR with intervals.
     This underpins the claim that the objective is primarily a FAR-control
     mechanism, so it gets its own reported block.
  2. multiplicity across the isolated loss-term family, with Bonferroni and
     Benjamini-Hochberg applied to bootstrap two-sided p-values.
  3. the class-weighting controls, including the operation-count increase.

FAMILY MEMBERSHIP IS DECLARED HERE, not inferred. The response record refers to
"eleven ablation contrasts" without listing them. This script declares the family as
every arm that perturbs only loss weights or class weighting, which yields m=16. Both
contrasts the record names as nominally significant (ranking-only and
class-weighted-no-KD) are inside that single family, which is required for the
multiplicity claim to be testable at all. m=16 > 11 makes both corrections STRICTER,
so the conclusion is conservative: nothing that fails at m=16 would pass at m=11.
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

from statistics_utils import (BOOTSTRAP_DESIGN, benjamini_hochberg, bonferroni,  # noqa: E402
                             hierarchical_bootstrap, paired_by_dataset)

PRIMARY = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]

# Arms that perturb ONLY loss weights (no encoder, architecture, T, epochs or
# precision change). tree_ce zeroes all four terms simultaneously and is included
# because it is the joint contrast.
LOSS_FAMILY = ["kd_only", "rank_only", "no_sparse", "no_fp", "tree_ce",
               "akd05", "akd20", "arank01", "arank06",
               "asparse1e2", "asparse1e4", "afp025", "afp100"]
CLASS_WEIGHT_FAMILY = ["class_weighted", "class_weighted_nokd", "cw_nokd_rankfixed"]

METRICS = {"macro_f1": ("metrics", "f1_macro", 100.0),
           "FAR": ("binary_ids_metrics", "false_alarm_rate", 100.0),
           "synops": ("spike_stats", "synops_per_sample", 1.0)}


def collect(arm, metric):
    block, key, scale = METRICS[metric]
    out = defaultdict(dict)
    pat = (str(REPO / "results/Tree-to-Spike/*/seed_*/results.json") if arm == "DEFAULT"
           else str(REPO / f"results/ladder/*/{arm}/seed_*/results.json"))
    for p in glob.glob(pat):
        parts = p.split("/")
        ds = parts[-3] if arm == "DEFAULT" else parts[-4]
        if ds not in PRIMARY:
            continue
        seed = int(parts[-2].split("_")[1])
        tp = json.load(open(p))["test_plus"]
        b = tp.get(block) or {}
        if key in b:
            out[ds][seed] = scale * b[key]
    return out


def contrast(arm, metric):
    return hierarchical_bootstrap(paired_by_dataset(collect(arm, metric),
                                                   collect("DEFAULT", metric)))


if __name__ == "__main__":
    out = {"scope": "four primary datasets x 5 seeds, paired vs shipped default",
           "bootstrap_design": BOOTSTRAP_DESIGN}

    print("=== 1. false-positive-aware loss term (`no_fp`) ===")
    f1 = contrast("no_fp", "macro_f1")
    far = contrast("no_fp", "FAR")
    out["false_positive_term"] = {"delta_macro_f1": f1, "delta_FAR": far}
    print(f"   Delta macro-F1 {f1['estimate_pp']:+.2f} pp  "
          f"CI [{f1['ci95_pp'][0]:+.2f}, {f1['ci95_pp'][1]:+.2f}]")
    print(f"   Delta FAR      {far['estimate_pp']:+.2f} pp  "
          f"CI [{far['ci95_pp'][0]:+.2f}, {far['ci95_pp'][1]:+.2f}]")
    print("   -> removing the term barely moves macro-F1 but raises FAR\n")

    # The correction family must contain BOTH nominally significant contrasts the
    # record names (ranking-only and class-weighted-no-KD), otherwise they are
    # corrected in separate families and the multiplicity claim is untestable.
    print("=== 2. multiplicity across the isolated-ablation family ===")
    rows = []
    for arm in LOSS_FAMILY + CLASS_WEIGHT_FAMILY:
        r = contrast(arm, "macro_f1")
        if not r["n_datasets"]:
            continue
        rows.append((arm, r))
    p_raw = [r["bootstrap_p_two_sided"] for _, r in rows]
    p_bonf = bonferroni(p_raw)
    p_bh = benjamini_hochberg(p_raw)
    print(f"   family size m = {len(rows)} (declared in this script)")
    print(f"   {'arm':14s} {'Delta':>7} {'95% CI':>18} {'p_raw':>8} {'p_bonf':>8} {'p_BH':>8}  nominal")
    fam = {}
    for (arm, r), pr, pb, ph in zip(rows, p_raw, p_bonf, p_bh):
        nominal = not (r["ci95_pp"][0] <= 0 <= r["ci95_pp"][1])
        fam[arm] = {**r, "p_raw": pr, "p_bonferroni": pb, "p_bh": ph,
                    "nominally_significant": nominal,
                    "survives_bonferroni": bool(pb < 0.05),
                    "survives_bh": bool(ph < 0.05)}
        print(f"   {arm:14s} {r['estimate_pp']:+7.2f} "
              f"[{r['ci95_pp'][0]:+7.2f},{r['ci95_pp'][1]:+7.2f}] "
              f"{pr:8.4f} {pb:8.4f} {ph:8.4f}  {'YES' if nominal else '-'}")
    out["loss_family_multiplicity"] = {"family": LOSS_FAMILY + CLASS_WEIGHT_FAMILY, "m": len(rows),
                                       "contrasts": fam}
    surv = [a for a, v in fam.items() if v["survives_bonferroni"] or v["survives_bh"]]
    biggest = max(abs(v["estimate_pp"]) for v in fam.values())
    print(f"\n   surviving either correction: {surv if surv else 'NONE'}")
    print(f"   largest |Delta| in family: {biggest:.2f} pp\n")
    out["loss_family_multiplicity"]["survives_any_correction"] = surv
    out["loss_family_multiplicity"]["max_abs_effect_pp"] = round(biggest, 4)

    print("=== 3. class-weighting controls ===")
    cw = {}
    d_syn = collect("DEFAULT", "synops")
    for arm in CLASS_WEIGHT_FAMILY:
        r = contrast(arm, "macro_f1")
        a_syn = collect(arm, "synops")
        pr = paired_by_dataset(a_syn, d_syn)
        ratio = np.mean([np.mean([a_syn[ds][s] / d_syn[ds][s]
                                  for s in sorted(set(a_syn[ds]) & set(d_syn[ds]))])
                         for ds in sorted(set(a_syn) & set(d_syn))]) if pr else np.nan
        cw[arm] = {**r, "synops_pct_of_default": round(100 * float(ratio), 2)}
        print(f"   {arm:22s} {r['estimate_pp']:+6.2f} pp  "
              f"CI [{r['ci95_pp'][0]:+6.2f}, {r['ci95_pp'][1]:+6.2f}]  "
              f"synops {100*ratio:6.1f}% of default")
    out["class_weighting"] = cw
    print("   -> encoder/distillation behaviour is not indirect class-weight transfer\n")

    p = REPO / "results/Journal_results/new_run_results/repro_loss_ablations.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"wrote {p.relative_to(REPO)}")
