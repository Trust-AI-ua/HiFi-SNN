#!/usr/bin/env python3
"""Summarize revision-2 steps 2 and 3 and apply the pre-registered decision rules R2 and R3
(results/preregistration/2026-11-16_r2_integer_gbdt.md).

    python scripts/r2_summarize.py
Output: results/journal_r2/{integer_summary.csv, integer_per_seed.csv, gbdt_valsel_summary.csv,
        gbdt_valsel_per_seed.csv, gbdt_guard.csv, SUMMARY.md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R2 = REPO / "results/journal_r2"
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
NAME = {"ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024", "edgeiiot": "Edge-IIoTset", "toniot": "TON-IoT"}
SEEDS = list(range(42, 47))
sd = lambda v: float(np.std(v, ddof=1))


def integer():
    rows = []
    for ds in IOT:
        for s in SEEDS:
            r = json.loads((R2 / f"integer/{ds}/seed_{s}.json").read_text())
            I, F = r["integer"], r["float"]
            rows.append(dict(dataset=ds, seed=s, tts_float=F["tts"]["macro_f1"], tts_int=I["tts_q8_leak8"]["macro_f1"],
                             tts_int_leak16=I["tts_q8_leak16"]["macro_f1"], tts_int_q0=I["tts_q0_leak8"]["macro_f1"],
                             tts_agree=100 * I["tts_q8_leak8"]["agreement_with_float"], tts_far_float=F["tts"]["far"],
                             tts_far_int=I["tts_q8_leak8"]["far"], tts_max_int=I["tts_q8_leak8"]["max_abs_int"],
                             dnn_float=F["dnn"]["macro_f1"], dnn_int=I["dnn_u8"]["macro_f1"],
                             dnn_agree=100 * I["dnn_u8"]["agreement_with_float"], dnn_far_float=F["dnn"]["far"],
                             dnn_far_int=I["dnn_u8"]["far"], dnn_max_int=I["dnn_u8"]["max_abs_int"]))
    p = pd.DataFrame(rows)
    for m in ("tts", "dnn"):
        p[f"{m}_delta"] = p[f"{m}_int"] - p[f"{m}_float"]
    p["tts_delta_leak16"] = p.tts_int_leak16 - p.tts_float
    p["tts_delta_q0"] = p.tts_int_q0 - p.tts_float
    p.to_csv(R2 / "integer_per_seed.csv", index=False)
    g = p.groupby("dataset")
    s = pd.DataFrame({c: g[c].mean() for c in p.columns if c not in ("dataset", "seed")}).loc[IOT]
    s["tts_delta_min_seed"] = g.tts_delta.min()
    s["tts_int_sd"] = g.tts_int.agg(sd)
    s["dnn_int_sd"] = g.dnn_int.agg(sd)
    s.loc["IoT mean"] = s.loc[IOT].mean()
    s.to_csv(R2 / "integer_summary.csv")
    return p, s


def gbdt():
    rows, guard = [], []
    t2s = {(ds, x): 100 * json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{x}/results.json").read_text())
           ["test_plus"]["metrics"]["f1_macro"] for ds in IOT for x in SEEDS}
    t2s_far = {(ds, x): 100 * json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{x}/results.json").read_text())
               ["test_plus"]["binary_ids_metrics"]["false_alarm_rate"] for ds in IOT for x in SEEDS}
    for ds in IOT:
        for x in SEEDS:
            r = json.loads((R2 / f"gbdt_valsel/{ds}/seed_{x}.json").read_text())
            rec = json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{x}/results.json").read_text())
            if r["split"]["n_test"] != rec["test_plus"]["n_samples"]:
                raise SystemExit(f"{ds}/{x}: GBDT split differs from Tree-to-Spike")
            sel = r["selected"]
            rows.append(dict(dataset=ds, seed=x, config=f"{sel['n_estimators']}x{sel['max_depth']}"
                             + (" more-rounds" if sel["gain_prune"] == "0.9" else "") + f" {sel['leaf_bits']}b",
                             rounds=sel["rounds"], kib=sel["bytes"] / 1024, within=r["n_within_budget"],
                             val_f1=sel["val_macro_f1"], gbdt_f1=sel["test_macro_f1"], gbdt_far=sel["test_far"],
                             gbdt_dr=sel["test_dr"], tts_f1=t2s[(ds, x)], tts_far=t2s_far[(ds, x)]))
            for gr in r.get("guard", []):
                guard.append(dict(dataset=ds, **gr))
    p = pd.DataFrame(rows)
    p["gbdt_minus_tts"] = p.gbdt_f1 - p.tts_f1
    p.to_csv(R2 / "gbdt_valsel_per_seed.csv", index=False)
    g = p.groupby("dataset")
    s = pd.DataFrame(dict(gbdt_f1=g.gbdt_f1.mean(), gbdt_f1_sd=g.gbdt_f1.agg(sd), tts_f1=g.tts_f1.mean(),
                          tts_f1_sd=g.tts_f1.agg(sd), gbdt_minus_tts=g.gbdt_minus_tts.mean(),
                          seeds_gbdt_ahead=g.gbdt_minus_tts.apply(lambda v: int((v > 0).sum())),
                          gbdt_far=g.gbdt_far.mean(), tts_far=g.tts_far.mean(), kib=g.kib.mean(),
                          configs=g.config.apply(lambda v: "; ".join(sorted(set(v)))))).loc[IOT]
    s.loc["IoT mean", ["gbdt_f1", "tts_f1", "gbdt_minus_tts", "gbdt_far", "tts_far", "kib"]] = \
        s.loc[IOT, ["gbdt_f1", "tts_f1", "gbdt_minus_tts", "gbdt_far", "tts_far", "kib"]].mean()
    s.to_csv(R2 / "gbdt_valsel_summary.csv")
    gd = pd.DataFrame(guard)
    if len(gd):
        gd["d_f1"] = gd.test_sklearn_f1 - gd.published_f1
        gd.to_csv(R2 / "gbdt_guard.csv", index=False)
    return p, s, gd


def main():
    pi, si = integer()
    pg, sg, gd = gbdt()
    iot_d = si.loc["IoT mean", "tts_delta"]
    worst = si.loc[IOT, "tts_delta"].min()
    hurts = (iot_d <= -0.5) or (worst <= -1.0)
    wins = int((sg.loc[IOT, "gbdt_f1"] > sg.loc[IOT, "tts_f1"]).sum())
    mean_win = sg.loc["IoT mean", "gbdt_f1"] > sg.loc["IoT mean", "tts_f1"]
    r3 = wins >= 2 or mean_win
    diff_int = si.loc["IoT mean", "tts_int"] - si.loc["IoT mean", "dnn_int"]
    diff_float = si.loc["IoT mean", "tts_float"] - si.loc["IoT mean", "dnn_float"]
    f = lambda v, d=2: f"{v:+.{d}f}"
    L = ["# Revision 2, steps 1-3: results and pre-registered decisions", "",
         "Pre-registration: `results/preregistration/2026-11-16_r2_integer_gbdt.md`. IoT four, seeds 42-46.", "",
         "## Step 2. Integer-only execution", "",
         "Macro-F1 (%), five-seed means. Float = the reported deployed model (int8-grid weights, float arithmetic), "
         "reproduced exactly from each checkpoint. Integer = int8 weights, int32 accumulators, Q = 8, leak (v*230)>>8.", "",
         "| Dataset | T2S float | T2S integer | Change | Worst seed | Agreement | T2S FAR int | DNN float | DNN integer | Change | Agreement |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for ds in IOT + ["IoT mean"]:
        r = si.loc[ds]; n = NAME.get(ds, ds)
        L.append(f"| {n} | {r.tts_float:.2f} | {r.tts_int:.2f} | {f(r.tts_delta)} | {f(r.tts_delta_min_seed)} | "
                 f"{r.tts_agree:.2f}% | {r.tts_far_int:.2f} | {r.dnn_float:.2f} | {r.dnn_int:.2f} | {f(r.dnn_delta)} | {r.dnn_agree:.2f}% |")
    L += ["", f"Variants (Tree-to-Spike, IoT-mean change): 16-bit leak {f(si.loc['IoT mean', 'tts_delta_leak16'])}, "
          f"Q = 0 {f(si.loc['IoT mean', 'tts_delta_q0'])}. Largest integer magnitude: Tree-to-Spike "
          f"{int(pi.tts_max_int.max()):,}, DNN-Tree {int(pi.dnn_max_int.max()):,} (int32 limit 2,147,483,647).", "",
          f"Tree-to-Spike minus DNN-Tree, IoT mean: {f(diff_float)} with the reported models, {f(diff_int)} under integer execution.", "",
          f"**R2:** IoT-mean change {f(iot_d)}, worst dataset {f(worst)}. Rule: hurts if <= -0.5 or any dataset <= -1.0. "
          f"**{'Integer execution hurts the spiking model.' if hurts else 'Integer execution does not hurt the spiking model.'}**", "",
          "## Step 3. Validation-selected compact GBDT within 128 KiB", "",
          "Selected on validation macro-F1 among the 48 configurations within 128 KiB (packed format); test scored once. "
          "\"more-rounds\" is the sweep's mislabelled gain-prune arm.", "",
          "| Dataset | GBDT (val-selected, <=128 KiB) | Tree-to-Spike | GBDT minus T2S | Seeds GBDT ahead | GBDT FAR | T2S FAR | Mean KiB | Configurations |",
          "|---|--:|--:|--:|--:|--:|--:|--:|---|"]
    for ds in IOT:
        r = sg.loc[ds]
        L.append(f"| {NAME[ds]} | {r.gbdt_f1:.2f} ± {r.gbdt_f1_sd:.2f} | {r.tts_f1:.2f} ± {r.tts_f1_sd:.2f} | {f(r.gbdt_minus_tts)} | "
                 f"{int(r.seeds_gbdt_ahead)}/5 | {r.gbdt_far:.2f} | {r.tts_far:.2f} | {r.kib:.1f} | {r.configs} |")
    r = sg.loc["IoT mean"]
    L.append(f"| IoT mean | {r.gbdt_f1:.2f} | {r.tts_f1:.2f} | {f(r.gbdt_minus_tts)} | | {r.gbdt_far:.2f} | {r.tts_far:.2f} | {r.kib:.1f} | |")
    if len(gd):
        L += ["", f"Guard (seed 42, all four datasets, {len(gd)} configurations): round counts equal "
              f"{int((gd.rounds == gd.published_rounds).sum())}/{len(gd)}, packed bytes equal "
              f"{int((gd.bytes == gd.published_bytes).sum())}/{len(gd)}, largest test macro-F1 difference "
              f"{gd.d_f1.abs().max():.4f} (the published values have three decimals)."]
    L += ["", f"**R3:** the GBDT is ahead on {wins} of 4 datasets; IoT mean {'higher' if mean_win else 'lower'}. "
          f"Rule: wins on more benchmarks if >= 2 of 4 or a higher IoT mean. "
          f"**{'The tree wins on more benchmarks; the framing needs reassessment.' if r3 else 'The tree does not win on more benchmarks.'}**", ""]
    L += ["## Deviation from the pre-registered implementation", "",
          "The per-depth packed-prediction check first compared labels of the in-model quantized booster and "
          "`src/serialize.predict_packed`. On Edge-IIoTset seed 45, depth 6, one of 300 rows differed: the packed "
          "sums of classes 7 and 10 tie exactly at 8-bit leaves, and XGBoost's per-class intercept (which "
          "`predict_packed` omits) breaks the tie the other way. The check was replaced by a stricter one "
          "(identical leaf indices, and margins equal up to a constant per class) and that cell was rerun; "
          "the other 19 cells had passed the label check. Selection and scoring are unchanged. The intercepts "
          "(4 B per class, at most 56 B) are not in the packed price; every selected model keeps at least 378 B "
          "of headroom under 128 KiB.", ""]
    (R2 / "SUMMARY.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
