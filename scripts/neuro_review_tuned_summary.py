#!/usr/bin/env python3
"""Amendment 8, part 2 summary: original and validation-tuned fixed-point Vanilla SNN, five seeds
(results/preregistration/2026-11-17_neuro_stream.md). Runs in .venv after scripts/neuro_a8_runner.sh.

Per dataset and seed: selected p; full-test macro-F1 of the float record, the original and the tuned fixed-point
Vanilla (Lava emulation); per-class F1; prediction agreement of the tuned network with the float record (GPU replay)
and with the original fixed-point network; energy and latency (SANA-FE); footprint. Recovery = (tuned - original) /
(float - original) on five-seed means. The amendment-5 rule re-applied on five-seed means to the two Tree-to-Spike
rows with a five-seed energy claim, against the tuned Vanilla SNN.

    .venv/bin/python scripts/neuro_review_tuned_summary.py
Output: results/neuro/review/a8_tuned_vanilla/{tuned_vanilla.csv, tuned_per_class_f1.csv, TUNED.md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
INP = REPO / "results/neuro/inputs"
CONF = REPO / "results/neuro/confirm"
OUT = REPO / "results/neuro/review/a8_tuned_vanilla"
SEEDS = (42, 43, 44, 45, 46)
CLAIM = {"edgeiiot": "T10", "ciciomt2024": "t2s_exit"}


def main():
    from sklearn.metrics import f1_score
    from neuro_stagewise import vanilla_split, lava_vanilla, batched
    from neuro_a2_vanilla import vanilla_lava_params
    sel = json.loads((OUT / "selection.json").read_text())
    rows, pc = [], []
    for ds in CLAIM:
        for s in SEEDS:
            t = json.loads((OUT / f"{ds}_seed{s}_vanilla_tuned.json").read_text())
            o = json.loads((CONF / f"{ds}_seed{s}_vanilla_fixed.json").read_text())
            c = json.loads((CONF / f"{ds}_seed{s}_{CLAIM[ds]}.json").read_text())
            tp = np.load(OUT / f"{ds}_seed{s}_vanilla_tuned_pred.npz")
            y = tp["labels"]
            v = dict(np.load(INP / f"{ds}_vanilla_seed{s}.npz")); vi = np.load(INP / f"{ds}_vanilla_int_seed{s}.npz")
            vl, _, dv = vanilla_lava_params(v, vi)
            T = int(v["T"])
            orig = batched(lambda x: lava_vanilla(vl, x, T, dv=dv)[0], vi["spike_times_full"].astype(np.int64)).argmax(1)
            if abs(100 * f1_score(y, orig, average="macro") - o["simulated_macro_f1_full_test"]) > 1e-9:
                raise SystemExit(f"{ds}/{s}: original Vanilla emulation is not the confirmation's")
            sw = REPO / f"results/neuro/review/p1_stagewise/{ds}_seed{s}_test.npz"
            if sw.exists():
                z = np.load(sw); fl = z["pred_vanilla_float_gpu"].astype(np.int64)
                if not np.array_equal(z["labels"], y):
                    raise SystemExit("stagewise test split differs")
            else:
                _, yy, oc, _, _ = vanilla_split(ds, s, "test")
                if not np.array_equal(yy, y):
                    raise SystemExit("GPU replay split differs")
                fl = oc.argmax(1)
            rec = json.loads((REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_{s}/results.json").read_text())
            f_float = 100 * rec["test_plus"]["metrics"]["f1_macro"]
            if abs(100 * f1_score(y, fl, average="macro") - f_float) > 1e-9:
                raise SystemExit(f"{ds}/{s}: float predictions are not the record")
            pr = tp["pred"].astype(np.int64)
            names = [str(n) for n in np.load(INP / f"{ds}_seed{s}.npz")["class_names"]]
            rows.append(dict(dataset=ds, seed=s, selected_p=t["selected_p"], val_f1_original=sel[f"{ds}/{s}"]["candidates"][0]["val_macro_f1"],
                             val_f1_selected=max(cc["val_macro_f1"] for cc in sel[f"{ds}/{s}"]["candidates"] if cc["p"] == t["selected_p"]),
                             f1_float=f_float, f1_original=o["simulated_macro_f1_full_test"], f1_tuned=t["simulated_macro_f1_full_test"],
                             agree_tuned_float=float((pr == fl).mean()), agree_original_float=float((orig == fl).mean()),
                             agree_tuned_original=float((pr == orig).mean()),
                             energy_original_nJ=o["energy_nJ_mean"], energy_tuned_nJ=t["energy_nJ_mean"],
                             latency_original_us=o["latency_us_mean"], latency_tuned_us=t["latency_us_mean"],
                             weight_bytes_int8=t["weight_bytes_int8"], nonzero_synapses_tuned=t["nonzero_synapses"],
                             nonzero_synapses_original=int((vi["P_Wh"] != 0).sum() + (vi["P_Wo"] != 0).sum()),
                             claim_row=CLAIM[ds], claim_f1=c["simulated_macro_f1_full_test"], claim_energy_nJ=c["energy_nJ_mean"]))
            for k, (a, b, f) in enumerate(zip(*[100 * f1_score(y, p, average=None, labels=range(len(names))) for p in (orig, pr, fl)])):
                pc.append(dict(dataset=ds, seed=s, class_name=names[k], support=int((y == k).sum()), f1_float=f, f1_original=a, f1_tuned=b))
    R = pd.DataFrame(rows); R.to_csv(OUT / "tuned_vanilla.csv", index=False)
    P = pd.DataFrame(pc); P.to_csv(OUT / "tuned_per_class_f1.csv", index=False)
    ms = lambda x: f"{x.mean():.2f} ({x.std(ddof=1):.2f})"
    L = ["# Validation-tuned fixed-point Vanilla SNN (amendment 8, part 2)", "",
         "Five candidates per seed (per-tensor |W| percentile mapped to 127: 100 = the amendment-4 rule, 99.99, 99.9, 99.5, 99), "
         "selected on the whole validation split by the Lava emulation's macro-F1, frozen (`selection.json`, committed) before "
         "any test evaluation. Selected networks through the amendment-6 chain under the guard (Lava = emulation bit for bit, "
         "SANA-FE gate 100% on every seed). Full-test macro-F1; five-seed mean (SD).", ""]
    for ds, g in R.groupby("dataset", sort=False):
        rec = (g.f1_tuned.mean() - g.f1_original.mean()) / (g.f1_float.mean() - g.f1_original.mean())
        L += [f"## {ds}", "", "| Seed | p | Float record | Original fixed point | Tuned fixed point | Change | Agreement tuned/float | Energy original / tuned (nJ) |",
              "|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in g.itertuples():
            L.append(f"| {r.seed} | {r.selected_p:g} | {r.f1_float:.2f} | {r.f1_original:.2f} | {r.f1_tuned:.2f} | {r.f1_tuned - r.f1_original:+.2f} | "
                     f"{100 * r.agree_tuned_float:.1f}% | {r.energy_original_nJ:.1f} / {r.energy_tuned_nJ:.1f} |")
        L.append(f"| mean | | {ms(g.f1_float)} | {ms(g.f1_original)} | {ms(g.f1_tuned)} | {(g.f1_tuned - g.f1_original).mean():+.2f} | "
                 f"{100 * g.agree_tuned_float.mean():.1f}% | {g.energy_original_nJ.mean():.1f} / {g.energy_tuned_nJ.mean():.1f} |")
        dom = g.claim_f1.mean() > g.f1_tuned.mean() and g.claim_energy_nJ.mean() <= g.energy_tuned_nJ.mean()
        wins = int(((g.claim_f1 > g.f1_tuned) & (g.claim_energy_nJ <= g.energy_tuned_nJ)).sum())
        gap = g.f1_float.mean() - g.f1_original.mean()
        rtxt = (f"{rec:.2f}" + (" (most of the loss)" if rec >= 0.5 else " (not most of the loss)")) if gap >= 1.0 else \
            "not defined: the original fixed-point network was already within 1 point of the float record"
        L += ["", f"- Recovery (tuned - original) / (float - original): {rtxt}; float minus original {gap:+.2f} points; "
              f"agreement with the float record {100 * g.agree_original_float.mean():.1f}% (original), {100 * g.agree_tuned_float.mean():.1f}% (tuned).",
              f"- Footprint: {int(g.weight_bytes_int8.iloc[0])} int8 weight bytes, unchanged; nonzero synapses {g.nonzero_synapses_original.mean():.0f} "
              f"(original) and {g.nonzero_synapses_tuned.mean():.0f} (tuned); SANA-FE maps every dense connection, zeros included.",
              f"- Amendment-5 rule against the tuned Vanilla SNN: {CLAIM[ds]} at {g.claim_energy_nJ.mean():.1f} nJ and {g.claim_f1.mean():.2f} macro-F1 "
              f"against {g.energy_tuned_nJ.mean():.1f} nJ and {g.f1_tuned.mean():.2f}: " + ("dominates" if dom else "does not dominate")
              + f" on five-seed means; per-seed wins {wins}/5.", ""]
        q = P[P.dataset == ds].groupby("class_name")[["support", "f1_float", "f1_original", "f1_tuned"]].mean()
        q["recovered"] = q.f1_tuned - q.f1_original
        L += ["| Class | Support (mean) | Float | Original | Tuned | Change |", "|---|--:|--:|--:|--:|--:|"]
        for k, r in q.sort_values("f1_original").iterrows():
            L.append(f"| {k} | {r.support:.0f} | {r.f1_float:.2f} | {r.f1_original:.2f} | {r.f1_tuned:.2f} | {r.recovered:+.2f} |")
        L.append("")
    (OUT / "TUNED.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
