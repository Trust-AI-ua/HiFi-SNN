#!/usr/bin/env python3
"""CICIoMT2024 duplicate-free test mask + a model-free bound on its impact.

The provider train/test partition is capture-file disjoint and is NOT changed.
This only flags test rows whose canonical input hash also occurs in training.

Before spending any inference, it computes a rigorous worst-case bound: removing
r rows from a class of support s can move that class's recall by at most r/s, so
the macro-F1 change is bounded regardless of what any model predicts. If the
bound is negligible the re-evaluation is a formality; if it is not, the exact
per-model numbers matter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "ciciomt2024"


def main():
    from src.data.ciciomt2024 import _build_full_arrays, _category, BENIGN_FINE
    from src.data._label_modes import apply_label_mode

    Xtr, ytr_fine, Xte, yte_fine, fine_names = _build_full_arrays(
        REPO / "dataset/cic-iomt-2024")
    ytr, class_names = apply_label_mode("category", ytr_fine, fine_names,
                                        _category, BENIGN_FINE)
    yte, _ = apply_label_mode("category", yte_fine, fine_names, _category, BENIGN_FINE)
    print(f"train {Xtr.shape}  test {Xte.shape}  classes {class_names}")

    # canonical input hash over the raw feature matrix, before any scaling
    tr_df = pd.DataFrame(np.asarray(Xtr))
    te_df = pd.DataFrame(np.asarray(Xte))
    htr = pd.util.hash_pandas_object(tr_df, index=False).to_numpy()
    hte = pd.util.hash_pandas_object(te_df, index=False).to_numpy()
    dup = np.asarray(pd.Index(hte).isin(pd.Index(htr)), dtype=bool)

    n_te = len(yte)
    print(f"\ntest rows            {n_te}")
    print(f"duplicated in train  {int(dup.sum())} ({100*dup.mean():.4f}%)")
    print(f"duplicate-free test  {int((~dup).sum())}")

    # label agreement on the overlap
    agree = None
    if dup.any():
        pos = {}
        for i, h in enumerate(htr):
            pos.setdefault(h, i)
        idx = np.where(dup)[0]
        same = [ytr[pos[hte[i]]] == yte[i] for i in idx]
        agree = 100 * float(np.mean(same))
        print(f"label agreement on overlap: {agree:.2f}% ({int(np.sum(same))}/{len(same)})")

    print("\nclass distribution of the removed rows, and the bound they imply:")
    rows = []
    for c, name in enumerate(class_names):
        s = int((yte == c).sum())
        r = int(((yte == c) & dup).sum())
        # recall for class c can move by at most r/s; precision by at most r/(preds_c)
        # which is unobservable model-free, so the recall bound is the reportable one.
        rows.append({"class": name, "support": s, "removed": r,
                     "pct_of_class": round(100 * r / max(s, 1), 4),
                     "max_recall_shift_pp": round(100 * r / max(s, 1), 4)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))

    worst = t.max_recall_shift_pp.max()
    macro_bound = t.max_recall_shift_pp.sum() / len(class_names)
    print(f"\nworst single-class recall shift  <= {worst:.4f} pp")
    print(f"macro-averaged bound             <= {macro_bound:.4f} pp")

    np.save(OUT / "dupfree_test_mask.npy", ~dup)
    json.dump({"n_test": n_te, "n_duplicated": int(dup.sum()),
               "pct_duplicated": float(100 * dup.mean()),
               "label_agreement_pct": agree,
               "class_table": rows,
               "worst_class_recall_bound_pp": float(worst),
               "macro_bound_pp": float(macro_bound),
               "partition": "provider capture-file split, UNCHANGED"},
              open(OUT / "dupfree_summary.json", "w"), indent=2)
    print(f"\nwrote {OUT}/dupfree_test_mask.npy and dupfree_summary.json")


if __name__ == "__main__":
    main()
