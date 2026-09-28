#!/usr/bin/env python3
"""B2: is the entropy control already a per-feature supervised tree? (pre-registered, Part B)

For each IoT dataset at seed 42 (split and teacher guards), for every kept feature j:
- the entropy cutter's cuts, exactly as the E2 encoder places them (teacher-weighted,
  count |E_j|). Only information-gain cuts are compared (D3): the cutter adds median
  fillers only after every positive-gain split is used, so its first |E_j| - fillers cuts
  are the information-gain cuts, recovered by calling it with that count;
- sklearn DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=|E_j|+1), no depth
  limit, the teacher's class weights as sample weights, on the same training column.

Tree cuts are matched one-to-one to the nearest entropy cut. A pair coincides when it
splits the feature's distinct training values the same way (tree: x > t; cutter: x >= c),
or when the cuts differ by less than the smallest gap between adjacent entropy cuts of the
feature (single-cut features: partition criterion only).

Rule: >= 95% of cuts coincide (pooled over the IoT four) -> same construction, nothing
further runs; otherwise a fourth arm, 20 runs under the E2 protocol.

    python scripts/partB_b2_entropy_tree.py
Output: results/journal_partB/B2/<run_id>/{per_feature.csv,summary.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

RULE = 0.95


def match(tree_cuts, ent_cuts, u):
    """Greedy one-to-one nearest matching; returns (n_exact, n_coincide, offsets)."""
    if len(tree_cuts) == 0 or len(ent_cuts) == 0:
        return 0, 0, []
    D = np.abs(tree_cuts[:, None] - ent_cuts[None, :])
    pairs, used_t, used_e = [], set(), set()
    for flat in np.argsort(D, axis=None, kind="stable"):
        i, j = divmod(int(flat), D.shape[1])
        if i in used_t or j in used_e:
            continue
        pairs.append((i, j))
        used_t.add(i); used_e.add(j)
        if len(used_t) == len(tree_cuts) or len(used_e) == len(ent_cuts):
            break
    gap = np.diff(np.sort(ent_cuts)).min() if len(ent_cuts) > 1 else None
    exact = coincide = 0
    offsets = []
    for i, j in pairs:
        t, c = float(tree_cuts[i]), float(ent_cuts[j])
        same_partition = np.array_equal(u > t, u >= c)
        close = gap is not None and abs(t - c) < gap
        exact += int(np.float32(t) == np.float32(c))
        coincide += int(same_partition or close)
        offsets.append(abs(t - c))
    return exact, coincide, offsets


def main():
    from sklearn.tree import DecisionTreeClassifier
    from src.models.tree_to_spike import TreeGuidedSpikeEncoder, entropy_cuts, teacher_class_weights
    C.no_thread_caps()
    st = C.stamp("B2")
    out = C.run_dir(st)
    rows = []
    for ds in C.IOT:
        seed = 42
        cfg = C.t2s_config(ds, seed)
        b, arrays = C.load_bundle(ds, seed)
        Xtr, ytr = arrays[0], arrays[1]
        teacher, _ = C.fit_teacher_guarded(ds, seed, b, arrays)
        power = float(cfg.get("teacher_cost_power", 1.0))
        enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                     jitter=cfg["jitter"], encoder_mode="tree").fit(teacher, Xtr.shape[1], X=Xtr)
        w = teacher_class_weights(ytr, power)
        sw = w[ytr].astype(np.float64)
        for j in sorted({j for j, _ in enc.channels_}):
            k = len(enc.edges_[j])
            x = Xtr[:, j].astype(np.float32)
            full, n_fill = entropy_cuts(x, ytr, k, class_weight=w)
            ig = entropy_cuts(x, ytr, k - n_fill, class_weight=w)[0] if k - n_fill > 0 else np.zeros(0, np.float32)
            if not np.isin(ig, full).all():      # guard: fillers only follow the information-gain cuts
                sys.exit(f"{ds} feature {j}: information-gain cuts are not a subset of the full cut set")
            clf = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=k + 1, random_state=0)
            clf.fit(x.reshape(-1, 1), ytr, sample_weight=sw)
            tt = clf.tree_
            tree_cuts = np.sort(tt.threshold[tt.children_left != -1]).astype(np.float64)
            u = np.unique(x).astype(np.float64)
            ne, nc, offs = match(tree_cuts, ig.astype(np.float64), u)
            rows.append(dict(dataset=ds, seed=seed, feature=int(j), E_j=k, fillers=int(n_fill), ig_cuts=len(ig),
                             tree_cuts=len(tree_cuts), matched=len(offs), exact=ne, coincide=nc,
                             median_offset=float(np.median(offs)) if offs else np.nan,
                             max_offset=float(np.max(offs)) if offs else np.nan))
        d = pd.DataFrame([r for r in rows if r["dataset"] == ds])
        print(f"{ds:12s} features {len(d)} | IG cuts {d.ig_cuts.sum()} fillers {d.fillers.sum()} | tree cuts "
              f"{d.tree_cuts.sum()} | coincide {d.coincide.sum()}/{d.ig_cuts.sum()} "
              f"({100*d.coincide.sum()/max(d.ig_cuts.sum(),1):.2f}%) exact {d.exact.sum()}", flush=True)
        pd.DataFrame(rows).to_csv(out / "per_feature.csv", index=False)
    D = pd.DataFrame(rows)
    share = D.coincide.sum() / D.ig_cuts.sum()
    per = D.groupby("dataset").apply(lambda g: dict(coincide_pct=100 * g.coincide.sum() / g.ig_cuts.sum(),
                                                    exact_pct=100 * g.exact.sum() / g.ig_cuts.sum(),
                                                    ig_cuts=int(g.ig_cuts.sum()), tree_cuts=int(g.tree_cuts.sum()),
                                                    fillers=int(g.fillers.sum()),
                                                    median_offset=float(np.nanmedian(g.median_offset)))).to_dict()
    summary = dict(stamp=st, pooled_coincide_pct=100 * share, pooled_exact_pct=100 * D.exact.sum() / D.ig_cuts.sum(),
                   rule_pct=100 * RULE, outcome=("same construction: nothing further runs" if share >= RULE
                                                 else "below 95%: the fourth arm runs (20 runs, E2 protocol)"),
                   per_dataset=per)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "stamp"}, indent=1, default=str))


if __name__ == "__main__":
    main()
