#!/usr/bin/env python3
"""B1 checks before any B2 model (results/preregistration/2026-11-17_neuro_stream.md).

On the seed-42 training split of the stream dataset: the Layer B shortcut screen of
scripts/leakage_feature_audit.py (indicator a_j = [x_j > min_j]; NMI I(a_j; y) / H(y); classes
with >= 100 training flows that the indicator rules out or in while 0.05 <= P(a_j) <= 0.95) on
the features after packet 1, after packet 4, and after the flow's last packet. Also, descriptive:
the share of test flows whose final state and length equal some training flow's.

    python scripts/stream_checks.py
Output: results/neuro/b1_stream/{shortcut_screen.csv, checks.md}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.edgeiiot_stream import load_stream, state_after  # noqa: E402

REP = REPO / "results/neuro/b1_stream"


def screen(X, y, names, feats, where):
    classes, counts = np.unique(y, return_counts=True)
    hy = mutual_info_score(y, y)
    rows = []
    for j, f in enumerate(feats):
        a = X[:, j] > X[:, j].min()
        p = a.mean()
        per = {int(c): a[y == c].mean() for c in classes}
        big = [int(c) for c, n in zip(classes, counts) if n >= 100]
        mid = 0.05 <= p <= 0.95
        rows.append(dict(state=where, feature=f, frac_active=p,
                         nmi=(mutual_info_score(a, y) / hy) if p not in (0.0, 1.0) else 0.0,
                         constant=bool(X[:, j].min() == X[:, j].max()),
                         excludes=";".join(names[c] for c in big if mid and per[c] == 0.0),
                         includes=";".join(names[c] for c in big if mid and per[c] == 1.0),
                         per_class_active=";".join(f"{names[c]}={per[c]:.3f}" for c in classes)))
    return rows


def main():
    s = load_stream(42)
    names, feats = s.class_names, s.features
    rows = []
    for k in (1, 4, "last"):
        rows += screen(state_after(s.X["train"], s.length["train"], k), s.y["train"], names, feats, f"after_{k}")
    df = pd.DataFrame(rows)
    df.to_csv(REP / "shortcut_screen.csv", index=False)
    # exact duplicates of (final state, length) between test and train
    key = lambda X, L: pd.util.hash_pandas_object(pd.DataFrame(np.c_[state_after(X, L, "last"), L]), index=False).to_numpy()
    ktr, kte = key(s.X["train"], s.length["train"]), key(s.X["test"], s.length["test"])
    dup = np.isin(kte, ktr)
    dshare = pd.Series(dup).groupby(s.y["test"]).mean()
    out = ["# B1 checks (seed-42 split)", "",
           f"Flows: train {len(s.y['train']):,}, validation {len(s.y['val']):,}, test {len(s.y['test']):,}.", "",
           "## Shortcut screen: flagged indicators", "",
           "| State | Feature | P(active) | NMI | Rules out | Rules in |", "|---|---|--:|--:|---|---|"]
    flag = df[(df.excludes != "") | (df.includes != "")]
    for r in flag.itertuples():
        out.append(f"| {r.state} | {r.feature} | {r.frac_active:.3f} | {r.nmi:.3f} | {r.excludes} | {r.includes} |")
    out += ["", "## Highest indicator NMI per state", ""]
    for st, g in df.groupby("state", sort=False):
        out.append(f"- {st}: " + ", ".join(f"{r.feature} {r.nmi:.2f}" for r in g.nlargest(5, "nmi").itertuples()))
    out += ["", "## Test flows whose final state and length equal a training flow's", "",
            f"All classes: {100 * dup.mean():.1f}%. Per class: " +
            ", ".join(f"{names[c]} {100 * v:.1f}%" for c, v in dshare.items()), ""]
    (REP / "checks.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
