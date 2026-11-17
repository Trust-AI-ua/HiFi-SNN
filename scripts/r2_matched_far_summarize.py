#!/usr/bin/env python3
"""Summarize the matched false-alarm check (results/preregistration/2026-11-17_r2_matched_far.md).

Reads results/journal_r2/matched_far_sym/<ds>/seed_<S>.json (which carry (A) to (C) as well
as amendment 1's (D) and (E)) and writes results/journal_r2/matched_far_sym/SUMMARY.md and
summary.csv. Five-seed means, test split, percent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results/journal_r2/matched_far_sym"
DS = [("ciciomt2024", "CICIoMT2024"), ("ciciot2023", "CICIoT2023"), ("edgeiiot", "Edge-IIoTset"), ("toniot", "TON-IoT")]
SEEDS = range(42, 47)
M = ("macro_f1", "far", "dr")


def main():
    rows, curve = [], []
    for ds, name in DS:
        for s in SEEDS:
            r = json.loads((D / ds / f"seed_{s}.json").read_text())
            pts = {"T2S argmax": r["t2s"]["test"], "GBDT argmax": r["gbdt_argmax_test"],
                   "GBDT (A) matched": r["A_matched"]["test"], "GBDT (B) val FAR<=1%": r["B_far1"]["test"],
                   "T2S (D) val FAR<=1%": r["D_t2s_far1"]["test"], "GBDT (C) at T2S test FAR": r["C_test_roc"]["gbdt_at_t2s_test_far"]}
            for p, v in pts.items():
                rows.append(dict(dataset=name, seed=s, point=p, **{k: v[k] for k in M}))
            rows.append(dict(dataset=name, seed=s, point="AUC GBDT", macro_f1=100 * r["C_test_roc"]["auc_gbdt"]))
            rows.append(dict(dataset=name, seed=s, point="AUC T2S", macro_f1=100 * r["C_test_roc"]["auc_t2s"]))
            for e in r["E_curve"]:
                for mdl in ("gbdt", "t2s"):
                    curve.append(dict(dataset=name, seed=s, target=e["target_far"], model=mdl, **{k: e[mdl][k] for k in M}))
    df, cv = pd.DataFrame(rows), pd.DataFrame(curve)
    mean = df.groupby(["dataset", "point"], sort=False)[list(M)].mean()
    cmean = cv.groupby(["dataset", "target", "model"])[list(M)].mean()
    iot = mean.groupby(level="point", sort=False).mean()
    ciot = cmean.groupby(level=["target", "model"]).mean()
    out = []
    w = out.append
    w("# Matched false-alarm check (revision 2, step 3b): results\n")
    w("Pre-registration: `results/preregistration/2026-11-17_r2_matched_far.md` (rule R5; amendment 1 adds (D), (E)).")
    w("Five-seed means (42 to 46), test split, percent. Thresholds are set on validation; test is scored once per threshold.\n")
    w("## (A) to (D), per dataset\n")
    pts = ["T2S argmax", "GBDT argmax", "GBDT (A) matched", "GBDT (B) val FAR<=1%", "T2S (D) val FAR<=1%", "GBDT (C) at T2S test FAR"]
    w("| Dataset | " + " | ".join(pts) + " |")
    w("|---|" + "--:|" * len(pts))
    for _, name in DS + [(None, "IoT mean")]:
        src = iot if name == "IoT mean" else mean.loc[name]
        w(f"| {name} | " + " | ".join(f"{src.loc[p, 'macro_f1']:.2f} / {src.loc[p, 'far']:.2f} / {src.loc[p, 'dr']:.2f}" for p in pts) + " |")
    w("\nCells are macro-F1 / FAR / DR. (C) is a readout on the test ROC curve, not a deployable operating point.\n")
    w("Benign-versus-attack test ROC AUC (score 1 - p_benign), percent:\n")
    w("| Dataset | GBDT | Tree-to-Spike |\n|---|--:|--:|")
    for _, name in DS + [(None, "IoT mean")]:
        src = iot if name == "IoT mean" else mean.loc[name]
        w(f"| {name} | {src.loc['AUC GBDT', 'macro_f1']:.2f} | {src.loc['AUC T2S', 'macro_f1']:.2f} |")
    w("\n## Rule R5\n")
    held = []
    for _, name in DS:
        g, t = mean.loc[(name, "GBDT (A) matched"), "macro_f1"], mean.loc[(name, "T2S argmax"), "macro_f1"]
        ok = g < t
        held.append(ok)
        nseed = int((df[(df.dataset == name) & (df.point == "GBDT (A) matched")].set_index("seed").macro_f1
                     < df[(df.dataset == name) & (df.point == "T2S argmax")].set_index("seed").macro_f1).sum())
        w(f"- {name}: matched GBDT {g:.2f} vs Tree-to-Spike {t:.2f} ({g - t:+.2f}); advantage "
          f"{'holds' if ok else 'does not hold'}; Tree-to-Spike ahead on {nseed}/5 seeds.")
    gi, ti = iot.loc["GBDT (A) matched", "macro_f1"], iot.loc["T2S argmax", "macro_f1"]
    w(f"- IoT mean: matched GBDT {gi:.2f} vs Tree-to-Spike {ti:.2f} ({gi - ti:+.2f}).")
    sup = sum(held) >= 3 and gi < ti
    w(f"\n**Verdict:** the false-alarm advantage holds on {sum(held)}/4 datasets "
      f"{'and' if gi < ti else 'and not'} on the IoT mean. The claim \"with far fewer false alarms\" is "
      f"{'supported' if sup else 'not supported'}; per R5 the paper "
      f"{'keeps it' if sup else 'reports the matched-FAR results and drops it as a general claim'}.\n")
    w("## (E) Macro-F1 against FAR (amendment 1, descriptive)\n")
    w("Both models under the same benign-threshold rule, tuned on validation to each FAR target. "
      "Cells are test macro-F1 / test FAR; the last column counts seeds on which Tree-to-Spike has the higher macro-F1.\n")
    for _, name in DS + [(None, "IoT mean")]:
        w(f"### {name}\n\n| Target FAR | GBDT | Tree-to-Spike | Difference (T2S - GBDT) | T2S ahead |\n|--:|--:|--:|--:|--:|")
        for tg in sorted(cv.target.unique()):
            src = ciot if name == "IoT mean" else cmean.loc[name]
            g, t = src.loc[(tg, "gbdt")], src.loc[(tg, "t2s")]
            sub = cv[cv.target == tg] if name == "IoT mean" else cv[(cv.target == tg) & (cv.dataset == name)]
            p = sub.pivot_table(index=["dataset", "seed"], columns="model", values="macro_f1")
            w(f"| {tg:g}% | {g.macro_f1:.2f} / {g.far:.2f} | {t.macro_f1:.2f} / {t.far:.2f} | "
              f"{t.macro_f1 - g.macro_f1:+.2f} | {int((p.t2s > p.gbdt).sum())}/{len(p)} |")
        w("")
    (D / "SUMMARY.md").write_text("\n".join(out))
    pd.concat([mean.reset_index(), iot.reset_index().assign(dataset="IoT mean")]).to_csv(D / "summary.csv", index=False)
    pd.concat([cmean.reset_index(), ciot.reset_index().assign(dataset="IoT mean")]).to_csv(D / "curve_summary.csv", index=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
