#!/usr/bin/env python3
"""Reproduce the dense-twin one-point non-inferiority test from committed results.

The predeclared margin is encoded here, not inferred at reading time: Tree-to-Spike
is non-inferior to its dense twin only if the LOWER bound of the 95% interval on
(T2S - DNN-Tree) macro-F1 lies at or above -1.0 pp.

Emits a machine-readable verdict so no downstream text has to restate it.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from statistics_utils import (BOOTSTRAP_DESIGN, hierarchical_bootstrap,  # noqa: E402
                             paired_by_dataset)

TARGET_DOMAIN = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
MARGIN_PP = -1.0
ALL_ELEVEN = ["nslkdd", "kddcup99", "unswnb15", "cicids2017", "cse-cic-ids2018",
              "iscxids2012", "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]


def collect(model_dir: str, datasets, metric="f1_macro", scale=100.0):
    out = defaultdict(dict)
    for p in glob.glob(str(REPO / f"results/{model_dir}/*/seed_*/results.json")):
        ds = p.split("/")[-3]
        if ds not in datasets:
            continue
        seed = int(p.split("/")[-2].split("_")[1])
        out[ds][seed] = scale * json.load(open(p))["test_plus"]["metrics"][metric]
    return out


def run(datasets, label):
    t2s = collect("Tree-to-Spike", datasets)
    dnn = collect("DNN-Tree", datasets)
    pairs = paired_by_dataset(t2s, dnn)
    r = hierarchical_bootstrap(pairs)
    established = bool(r["ci95_pp"][0] >= MARGIN_PP)
    r.update(scope=label, margin_pp=MARGIN_PP, noninferiority_established=established,
             comparison="Tree-to-Spike minus DNN-Tree, macro-F1, percentage points",
             criterion="CI lower bound >= margin",
             bootstrap_design=BOOTSTRAP_DESIGN)
    return r


if __name__ == "__main__":
    print("=== dense-twin one-point non-inferiority ===")
    print(f"predeclared margin: {MARGIN_PP:+.1f} pp on (Tree-to-Spike - DNN-Tree)\n")
    res = {}
    for datasets, label in ((TARGET_DOMAIN, "four target-domain datasets (PRIMARY)"),
                            (ALL_ELEVEN, "all eleven datasets (descriptive)")):
        r = run(datasets, label)
        res[label] = r
        print(f"-- {label}")
        print(f"   per dataset: " + "  ".join(
            f"{k}={v:+.2f}" for k, v in r["per_dataset_mean_pp"].items()))
        print(f"   estimate {r['estimate_pp']:+.2f} pp   "
              f"95% CI [{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]   "
              f"n_ds={r['n_datasets']} n_pairs={r['n_pairs_total']}")
        print(f"   noninferiority_established: {r['noninferiority_established']}\n")
    out = REPO / "results/Journal_results/new_run_results/repro_noninferiority.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out.relative_to(REPO)}")
