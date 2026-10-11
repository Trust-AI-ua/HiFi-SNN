#!/usr/bin/env python3
"""B5: train-only vocabulary and cross-partition duplicates (pre-registered, Part B).

Per dataset and seed, on the arrays the stored Tree-to-Spike run trained and tested on
(split guard in partB_common.load_bundle):

Vocabulary. Every categorical loader one-hot encodes over the union of categories before
the split, so the one-hot block is the tail of the feature vector. It is located as the
longest tail of columns whose values are exactly 0 or 1 in every partition, shrunk from
the left until every row has the same number of ones in the block (one per categorical
field; numeric columns pass through log1p, so they cannot be exactly {0, 1} in general).
A one-hot column is unseen in training when it is all-zero in the training partition
(MinMax keeps such a column at its raw value, so a test row carrying the category still
shows a 1). Reported: one-hot columns, columns unseen in training, those present in
test, and the share of test rows carrying an unseen category.
Trigger (pre-registered): > 0.1% of test rows on any seed of a primary (IoT) dataset.

Duplicates. Exact feature-vector duplicates between the training and test partitions,
per test class, plus label conflicts (the same vector with another label in training).
Rows are hashed (pandas, 64-bit); every hash match is confirmed byte-for-byte against a\nrepresentative training row, and rejected matches are counted.
A class is dominated when more than 50% of its test rows are duplicated in training.

    python scripts/partB_b5_data_checks.py [--datasets ...] [--seeds ...]
Output: results/journal_partB/B5/<run_id>/{vocabulary,duplicates}.csv, summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

TRIGGER_PCT = 0.1
DOMINATED = 0.5


def onehot_block(parts):
    X = np.concatenate(parts, axis=0)
    binary = ((X == 0) | (X == 1)).all(axis=0)
    start = X.shape[1]
    while start > 0 and binary[start - 1]:
        start -= 1
    if start == X.shape[1]:
        return None, "no {0,1} tail"
    for s in range(start, X.shape[1]):
        rs = X[:, s:].sum(axis=1)
        if rs.min() == rs.max() and rs.min() >= 1:
            return list(range(s, X.shape[1])), f"constant row sum {int(rs.min())}"
    return None, "tail row sums never constant"


def vocabulary(ds, seed, Xtr, Xval, Xte):
    cols, how = onehot_block([Xtr, Xval, Xte])
    if cols is None:
        return dict(dataset=ds, seed=seed, n_onehot=0, n_unseen_train=0, n_unseen_in_test=0,
                    test_rows_affected=0, n_test=len(Xte), affected_pct=0.0, block=how)
    B_tr, B_te = Xtr[:, cols], Xte[:, cols]
    unseen = B_tr.max(axis=0) == 0
    in_test = unseen & (B_te.max(axis=0) > 0)
    rows = (B_te[:, unseen] > 0).any(axis=1) if unseen.any() else np.zeros(len(Xte), bool)
    return dict(dataset=ds, seed=seed, n_onehot=len(cols), n_unseen_train=int(unseen.sum()),
                n_unseen_in_test=int(in_test.sum()), test_rows_affected=int(rows.sum()), n_test=len(Xte),
                affected_pct=100 * rows.mean(), block=how)


def duplicates(ds, seed, Xtr, ytr, Xte, yte, names):
    htr = pd.util.hash_pandas_object(pd.DataFrame(Xtr), index=False).to_numpy()
    hte = pd.util.hash_pandas_object(pd.DataFrame(Xte), index=False).to_numpy()
    g = pd.DataFrame({"h": htr, "y": ytr, "i": np.arange(len(htr))}).groupby("h")
    rep = g.i.first()                      # one representative training row per hash
    ymin, ymax = g.y.min(), g.y.max()      # the training labels carried by that vector
    hit = np.isin(hte, rep.index.to_numpy())
    idx = np.where(hit)[0]
    r = rep.reindex(hte[idx]).to_numpy()
    same = (Xtr[r] == Xte[idx]).all(axis=1)          # byte-for-byte confirmation
    confirmed = np.zeros(len(Xte), bool)
    confirmed[idx[same]] = True
    conflict = np.zeros(len(Xte), bool)
    lo, hi = ymin.reindex(hte[idx]).to_numpy(), ymax.reindex(hte[idx]).to_numpy()
    conflict[idx[same]] = ((lo != yte[idx]) | (hi != yte[idx]))[same]
    rows = []
    for c in np.unique(yte):
        m = yte == c
        rows.append(dict(dataset=ds, seed=seed, cls=names[int(c)], n_test=int(m.sum()),
                         n_dup=int(confirmed[m].sum()), dup_pct=100 * confirmed[m].mean(),
                         n_label_conflict=int(conflict[m].sum()),
                         dominated=bool(confirmed[m].mean() > DOMINATED)))
    return rows, int((~same).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=C.ALL)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(C.SEEDS))
    a = ap.parse_args()
    C.no_thread_caps()
    st = C.stamp("B5")
    out = C.run_dir(st)
    voc, dup, coll = [], [], 0
    for ds in a.datasets:
        for s in a.seeds:
            b, (Xtr, ytr, Xval, yval, Xte, yte) = C.load_bundle(ds, s)
            v = vocabulary(ds, s, Xtr, Xval, Xte)
            voc.append(v)
            d, c = duplicates(ds, s, Xtr, ytr, Xte, yte, b.class_names)
            dup += d
            coll += c
            print(f"{ds:16s} {s} one-hot {v['n_onehot']:4d} unseen {v['n_unseen_train']:3d} "
                  f"(in test {v['n_unseen_in_test']:3d}) affected {v['affected_pct']:.4f}% | "
                  f"dup test rows {100*sum(x['n_dup'] for x in d)/len(yte):.2f}%", flush=True)
            pd.DataFrame(voc).to_csv(out / "vocabulary.csv", index=False)
            pd.DataFrame(dup).to_csv(out / "duplicates.csv", index=False)
    V, D = pd.DataFrame(voc), pd.DataFrame(dup)
    prim = V[V.dataset.isin(C.IOT)]
    fired = sorted(prim[prim.affected_pct > TRIGGER_PCT].dataset.unique().tolist())
    dom = D[D.dominated][["dataset", "seed", "cls", "dup_pct"]].to_dict("records")
    summary = dict(stamp=st, trigger_pct=TRIGGER_PCT, vocabulary_trigger_fired_for=fired,
                   max_affected_pct_by_dataset=V.groupby("dataset").affected_pct.max().round(4).to_dict(),
                   dominated_classes=dom, hash_collisions_rejected=coll,
                   dup_pct_by_dataset=D.groupby("dataset").apply(
                       lambda g: round(100 * g.n_dup.sum() / g.n_test.sum(), 3)).to_dict())
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "stamp"}, indent=1, default=str))


if __name__ == "__main__":
    main()
