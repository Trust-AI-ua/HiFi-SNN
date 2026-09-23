#!/usr/bin/env python3
"""Teacher (full GBDT) versus Tree-to-Spike student, paired by seed, IoT subset.

Each seed fits its own teacher and trains its own student on the same split, so
the natural pairing is by seed. Teacher values are the persisted
``teacher_upper_bound`` of each Tree-to-Spike run, which the five-seed teacher
refit (scripts/teacher_per_dataset.py --seeds 42..46) reproduces exactly.

Per dataset: paired mean difference (student - teacher) and its SD over five
seeds, a t-based 95% CI (df = 4), and how many seeds the student wins. Across the
four datasets, the sign count is reported rather than a test.

    python scripts/teacher_vs_student.py
Output: results/tables/tab_teacher_vs_student.{csv,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
DSN = {"ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024",
       "edgeiiot": "Edge-IIoTset", "toniot": "TON-IoT"}
SEEDS = [42, 43, 44, 45, 46]


def main():
    rows = []
    for ds in IOT:
        for s in SEEDS:
            r = json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{s}/results.json").read_text())
            tp = r["test_plus"]
            rows.append(dict(dataset=ds, seed=s,
                             t_f1=100 * tp["teacher_upper_bound"]["f1_macro"],
                             s_f1=100 * tp["metrics"]["f1_macro"],
                             t_mcc=100 * tp["teacher_upper_bound"]["matthews_corrcoef"],
                             s_mcc=100 * tp["metrics"]["matthews_corrcoef"]))
    d = pd.DataFrame(rows)
    out = []
    for ds, g in d.groupby("dataset", sort=False):
        rec = {"dataset": DSN[ds]}
        for met in ("f1", "mcc"):
            diff = (g[f"s_{met}"] - g[f"t_{met}"]).to_numpy()
            m, sdv = diff.mean(), diff.std(ddof=1)
            half = stats.t.ppf(0.975, len(diff) - 1) * sdv / np.sqrt(len(diff))
            rec.update({f"teacher_{met}": g[f"t_{met}"].mean(), f"student_{met}": g[f"s_{met}"].mean(),
                        f"diff_{met}": m, f"sd_{met}": sdv,
                        f"ci_lo_{met}": m - half, f"ci_hi_{met}": m + half,
                        f"seed_wins_{met}": f"{int((diff > 0).sum())}/{len(diff)}"})
        out.append(rec)
    t = pd.DataFrame(out)
    t.to_csv(REPO / "results/tables/tab_teacher_vs_student.csv", index=False)
    print("student - teacher, paired by seed (5 seeds), IoT subset\n")
    for met, lab in (("f1", "macro-F1"), ("mcc", "MCC")):
        print(f"{lab}:")
        for _, r in t.iterrows():
            print(f"  {r.dataset:13s} teacher={r[f'teacher_{met}']:6.2f} student={r[f'student_{met}']:6.2f} "
                  f"diff={r[f'diff_{met}']:+6.2f} +/- {r[f'sd_{met}']:.2f}  "
                  f"95% CI [{r[f'ci_lo_{met}']:+.2f}, {r[f'ci_hi_{met}']:+.2f}]  "
                  f"student wins {r[f'seed_wins_{met}']} seeds")
        pos = int((t[f"diff_{met}"] > 0).sum())
        print(f"  student ahead on {pos}/4 datasets; mean diff {t[f'diff_{met}'].mean():+.2f}\n")
    md = ["| Dataset | Teacher F1 | Student F1 | Diff (± SD) | 95% CI | Seed wins | Diff MCC | Seed wins MCC |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        md.append(f"| {r.dataset} | {r.teacher_f1:.2f} | {r.student_f1:.2f} | {r.diff_f1:+.2f} ± {r.sd_f1:.2f} | "
                  f"[{r.ci_lo_f1:+.2f}, {r.ci_hi_f1:+.2f}] | {r.seed_wins_f1} | {r.diff_mcc:+.2f} | {r.seed_wins_mcc} |")
    (REPO / "results/tables/tab_teacher_vs_student.md").write_text("\n".join(md) + "\n")
    print("[wrote] results/tables/tab_teacher_vs_student.{csv,md}")


if __name__ == "__main__":
    main()
