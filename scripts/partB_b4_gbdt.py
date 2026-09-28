#!/usr/bin/env python3
"""B4 (amendment A3): validation-selected GBDT on CICIoMT2024's provider split.

The provider's pcap-file-level partition (test flows come from capture files never used in
training) is the paper's non-random evaluation. Tree-to-Spike, DNN-Tree and the entropy arm
already exist on it; this adds a validation-selected GBDT. Per seed (42-46), on the stored
runs' exact split (split guard): cost-weighted XGBoost through the production teacher
function (_fit_teacher), rounds {50, 100, 200} x depth {3, 4, 6}, everything else as the
teacher; the configuration with the highest validation macro-F1 is selected (ties: fewer
rounds, then shallower) and scored on test once.

    python scripts/partB_b4_gbdt.py [--seeds ...]
Output: results/journal_partB/B4_gbdt/<run_id>/{grid.csv,selected.csv,per_class.csv}
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

DS = "ciciomt2024"
GRID = list(itertools.product((50, 100, 200), (3, 4, 6)))


def main():
    from sklearn.metrics import f1_score
    from src.training_treespike import _fit_teacher
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=list(C.SEEDS))
    a = ap.parse_args()
    C.no_thread_caps()
    st = C.stamp("B4_gbdt")
    out = C.run_dir(st)
    grid, sel, per = [], [], []
    for seed in a.seeds:
        b, (Xtr, ytr, Xva, yva, Xte, yte) = C.load_bundle(DS, seed)
        cols = [f"f{i}" for i in range(Xtr.shape[1])]
        best = None
        for n, d in GRID:
            cfg = C.t2s_config(DS, seed)
            cfg.update(teacher_n_estimators=n, teacher_max_depth=d)
            m, _, present = _fit_teacher(Xtr, ytr, b.num_classes, cfg)
            pv = present[m.predict(pd.DataFrame(Xva, columns=cols))]
            vf1 = 100 * f1_score(yva, pv, average="macro", zero_division=0)
            grid.append(dict(seed=seed, rounds=n, depth=d, val_macro_f1=vf1))
            key = (vf1, -n, -d)
            if best is None or key > best[0]:
                best = (key, n, d, m, present)
            print(f"seed {seed} rounds {n} depth {d}: val macro-F1 {vf1:.2f}", flush=True)
            pd.DataFrame(grid).to_csv(out / "grid.csv", index=False)
        _, n, d, m, present = best
        pt = present[m.predict(pd.DataFrame(Xte, columns=cols))]
        mm = compute_metrics(yte, pt, y_score=None, class_names=b.class_names)
        bb = compute_binary_ids_metrics(yte, pt, normal_class_index=0)
        sel.append(dict(seed=seed, rounds=n, depth=d, val_macro_f1=best[0][0], test_macro_f1=100 * mm["f1_macro"],
                        test_accuracy=100 * mm["accuracy"], FAR=100 * bb["false_alarm_rate"], DR=100 * bb["detection_rate"]))
        for c, name in enumerate(b.class_names):
            msk = yte == c
            per.append(dict(seed=seed, cls=name, support_test=int(msk.sum()),
                            support_train=int(b.class_counts_train.get(name, 0)),
                            recall=100 * float((pt[msk] == c).mean()) if msk.any() else np.nan))
        print(f"[selected] seed {seed}: rounds {n} depth {d} | test macro-F1 {sel[-1]['test_macro_f1']:.2f} "
              f"FAR {sel[-1]['FAR']:.2f}", flush=True)
        pd.DataFrame(sel).to_csv(out / "selected.csv", index=False)
        pd.DataFrame(per).to_csv(out / "per_class.csv", index=False)


if __name__ == "__main__":
    main()
