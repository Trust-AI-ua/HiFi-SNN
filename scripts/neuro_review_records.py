#!/usr/bin/env python3
"""Amendment 8, part 5: existing selection records flattened to CSV (results/preregistration/2026-11-17_neuro_stream.md).
Runs in .venv; no new numbers, only reshaping of saved records.

  gbdt_budget_candidates.csv  revision-2 validation-selected GBDT (results/journal_r2/gbdt_valsel): every candidate per
                              dataset and seed, its bytes, whether it fits the 128 KiB budget, validation macro-F1, and
                              the selected one with its test metrics (candidates not selected were never scored on test)
  far_operating_points.csv    revision-2 matched FAR (results/journal_r2/matched_far): per dataset, seed and operating point
                              (Tree-to-Spike, GBDT argmax, A matched to the student's validation FAR, B FAR <= 1%), the
                              validation and the test macro-F1, FAR and DR separately, with the threshold chosen on validation

    .venv/bin/python scripts/neuro_review_records.py
Output: results/neuro/review/p5_selection/
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/neuro/review/p5_selection"
IOT = ("edgeiiot", "ciciomt2024", "toniot", "ciciot2023")
SEEDS = (42, 43, 44, 45, 46)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g, f = [], []
    for ds in IOT:
        for s in SEEDS:
            d = json.loads((REPO / f"results/journal_r2/gbdt_valsel/{ds}/seed_{s}.json").read_text())
            key = lambda c: (c["n_estimators"], c["max_depth"], c["gain_prune"], c["leaf_bits"], c["rounds"])
            sk = key(d["selected"])
            for c in d["configs"]:
                row = dict(dataset=ds, seed=s, budget_bytes=d["budget_bytes"], **c, selected=key(c) == sk)
                if row["selected"]:
                    row.update({k: v for k, v in d["selected"].items() if k.startswith("test_")})
                g.append(row)
            m = json.loads((REPO / f"results/journal_r2/matched_far/{ds}/seed_{s}.json").read_text())
            for op, rec in (("tree_to_spike", m["t2s"]), ("gbdt_argmax", {"test": m["gbdt_argmax_test"]}),
                            ("gbdt_A_matched_far", m["A_matched"]), ("gbdt_B_far_le_1pct", m["B_far1"])):
                for split in ("val", "test"):
                    if split in rec:
                        r = rec[split]
                        f.append(dict(dataset=ds, seed=s, operating_point=op, split=split, macro_f1=r.get("macro_f1"),
                                      far=r.get("far", r.get("test_far")), dr=r.get("dr", r.get("test_dr")),
                                      threshold=rec.get("threshold"), target_far=rec.get("target_far"),
                                      gbdt_config=json.dumps({k: m["gbdt_config"][k] for k in ("n_estimators", "max_depth", "leaf_bits", "rounds", "bytes")})
                                      if op.startswith("gbdt") else None))
    pd.DataFrame(g).to_csv(OUT / "gbdt_budget_candidates.csv", index=False)
    pd.DataFrame(f).to_csv(OUT / "far_operating_points.csv", index=False)
    print(len(g), "GBDT candidates;", len(f), "FAR rows")


if __name__ == "__main__":
    main()
