#!/usr/bin/env python3
"""A2: which classes the Lava vs integer-NumPy disagreements fall in (2,000-flow sample, seed 42).

Reads results/neuro/inputs/<ds>_seed42.npz and results/neuro/a2_lava/<ds>_<variant>_perflow.npz.
Per dataset, variant and true class: sample size, disagreeing flows, and how each disagreement
changes correctness (integer right / Lava wrong, integer wrong / Lava right, both wrong).

    .venv/bin/python scripts/neuro_a2_disagreements.py
Output: results/neuro/a2_lava/disagreements_by_class.csv, disagreements_by_class.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
INP, A2 = REPO / "results/neuro/inputs", REPO / "results/neuro/a2_lava"
DS = ["toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]


def main():
    rows = []
    for ds in DS:
        z = np.load(INP / f"{ds}_seed42.npz", allow_pickle=True)
        cn = [str(c) for c in z["class_names"]]
        for variant in ("subtract", "zero"):
            p = np.load(A2 / f"{ds}_{variant}_perflow.npz")
            s = p["sample"]
            if not np.array_equal(s, z["sample"]):
                raise SystemExit(f"{ds}/{variant}: per-flow file is for another sample")
            y, ip, lp = z["labels"][s], z["int_pred"][s], p["pred"]
            d = ip != lp
            for c in np.unique(y):
                m = y == c
                rows.append(dict(dataset=ds, variant=variant, cls=cn[c], n=int(m.sum()), disagree=int((d & m).sum()),
                                 int_right_lava_wrong=int((d & m & (ip == y)).sum()),
                                 int_wrong_lava_right=int((d & m & (lp == y)).sum()),
                                 both_wrong=int((d & m & (ip != y) & (lp != y)).sum()),
                                 lava_pred_classes="; ".join(f"{cn[a]}:{b}" for a, b in zip(*np.unique(lp[d & m], return_counts=True)))))
    df = pd.DataFrame(rows)
    tot = df.groupby(["dataset", "variant"]).disagree.transform("sum")
    df["share_of_class_pct"] = 100 * df.disagree / df.n
    df["share_of_disagreements_pct"] = np.where(tot > 0, 100 * df.disagree / tot.where(tot > 0, 1), 0.0)
    df.to_csv(A2 / "disagreements_by_class.csv", index=False)
    out = ["# A2: Lava vs integer NumPy disagreements by true class (2,000-flow sample, seed 42)", "",
           "Subtractive reset (the P5 variant): every disagreeing flow.", "",
           "| Dataset | Disagreeing flows | True class (sample size) | Integer right, Lava wrong | Integer wrong, Lava right | Both wrong | Lava predicted |",
           "|---|--:|---|--:|--:|--:|---|"]
    sub = df[(df.variant == "subtract") & (df.disagree > 0)]
    for ds in DS:
        r = sub[sub.dataset == ds]
        for _, x in r.iterrows():
            out.append(f"| {ds} | {x.disagree} | {x.cls} ({x.n}) | {x.int_right_lava_wrong} | {x.int_wrong_lava_right} | "
                       f"{x.both_wrong} | {x.lava_pred_classes} |")
    out += ["", "Zero reset: share of each dataset's disagreements by true class (top 3), and the share of that class that disagrees.", "",
            "| Dataset | Disagreeing flows | Top classes: share of disagreements (share of class) |", "|---|--:|---|"]
    zz = df[df.variant == "zero"]
    for ds in DS:
        r = zz[zz.dataset == ds].sort_values("disagree", ascending=False)
        out.append(f"| {ds} | {r.disagree.sum()} | " + "; ".join(
            f"{x.cls} {x.share_of_disagreements_pct:.0f}% ({x.share_of_class_pct:.0f}%)" for _, x in r.head(3).iterrows()) + " |")
    (A2 / "disagreements_by_class.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
