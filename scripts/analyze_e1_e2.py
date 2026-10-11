#!/usr/bin/env python3
"""Apply the pre-registered rules to E1 and E2 (results/preregistration/
2026-09-24_matched_info_and_entropy_edges.md, with its amendment). Refuses to report
while any cell is missing or any new run's split differs from the stored Tree-to-Spike
run.

E1 is reported per dataset at three precisions (fp32, per-tensor int8, per-channel
int8); the registered outcome is decided on fp32 only. A dataset is flagged when it
accounts for more than half of a change (level input -> spike-time input, at any
precision). E2 reports each dataset's fallback share (median filler cuts / entropy
cuts, pooled over seeds); above one half, that dataset's comparison is labelled as
largely against quantile placement, which Table V already covers.

    python scripts/analyze_e1_e2.py
Output: results/tables/tab_e1_matched_info.{csv,md}, results/tables/tab_e2_entropy.{csv,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
SEEDS = range(42, 47)


def rj(p):
    return json.loads(Path(p).read_text())


def split_sig(r):
    # split (test size, per-class training counts) and the teacher: its recorded test
    # macro-F1 must equal the stored run's bitwise, so both arms share one teacher
    return (r["test_plus"].get("n_samples"), json.dumps(r["data"].get("class_counts_train"), sort_keys=True),
            r["test_plus"]["teacher_upper_bound"]["f1_macro"])


def check_splits(pattern):
    bad = []
    for ds in IOT:
        for s in SEEDS:
            ref = rj(REPO / f"results/Tree-to-Spike/{ds}/seed_{s}/results.json")
            new = rj(REPO / pattern.format(d=ds, s=s))
            if split_sig(new) != split_sig(ref):
                bad.append(f"{ds}/{s}")
    return bad


def md(df):
    cols = [df.index.name or ""] + list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        out.append(f"| {i} | " + " | ".join(f"{v:+.2f}" if isinstance(v, float) and ("gap" in c or c.startswith(("d_", "chg_"))) else
                                           (f"{v:.2f}" if isinstance(v, float) else str(v)) for c, v in r.items()) + " |")
    return "\n".join(out) + "\n"


def e1():
    E = "results/DNN-TQ/{d}/seed_{s}/results.json"
    miss = [f"{d}/{s}" for d in IOT for s in SEEDS if not (REPO / E.format(d=d, s=s)).exists()]
    if miss:
        print(f"E1: {len(miss)} cells missing; not reported"); return
    bad = check_splits(E)
    if bad:
        sys.exit(f"E1: split or teacher differs from Tree-to-Spike on {bad}; refused")
    pc = pd.read_csv(REPO / "results/tables/tab_perchannel.csv")
    P = {"fp32": "fp32_f1", "pt": "per_tensor_f1", "pc": "per_channel_f1"}
    ref = pc.groupby(["model", "dataset"])[list(P.values())].mean()
    # The CICIoT2023 block records (amendment A1) store the three precisions under
    # test_plus.precisions; the 10-epoch records under fp32_before_projection.
    def prec(r, k):
        t_ = r["test_plus"]
        if "precisions" in t_:
            return 100 * t_["precisions"][k]["metrics"]["f1_macro"]
        if k == "per_tensor_int8":
            return 100 * t_["metrics"]["f1_macro"]
        f = t_["fp32_before_projection"]
        return 100 * (f if k == "fp32" else f["per_channel_int8"])["metrics"]["f1_macro"]
    rows = []
    for d in IOT:
        rs = [rj(REPO / E.format(d=d, s=s)) for s in SEEDS]
        if any("precisions" not in r["test_plus"] and "per_channel_int8" not in r["test_plus"]["fp32_before_projection"]
               for r in rs):
            sys.exit(f"E1: {d} has cells without the per-channel score (pre-amendment runs); refused")
        new = {"fp32": np.mean([prec(r, "fp32") for r in rs]), "pt": np.mean([prec(r, "per_tensor_int8") for r in rs]),
               "pc": np.mean([prec(r, "per_channel_int8") for r in rs])}
        row = {"dataset": d}
        for k, col in P.items():
            t2s, lvl = ref.loc[("Tree-to-Spike", d), col], ref.loc[("DNN-Tree", d), col]
            row.update({f"dnn_level_{k}": lvl, f"dnn_tq_{k}": new[k], f"t2s_{k}": t2s,
                        f"gap_level_{k}": lvl - t2s, f"gap_tq_{k}": new[k] - t2s, f"chg_{k}": new[k] - lvl})
        rows.append(row)
    t = pd.DataFrame(rows).set_index("dataset")
    g_tq, g_lv = t.gap_tq_fp32.mean(), t.gap_level_fp32.mean()
    outcome = ("A: matched information closes the gap" if g_tq <= 0.35 else
               "B: about one point is the cost of spiking computation" if g_tq >= 0.70 else
               f"C: input resolution explains {100 * (1 - g_tq / g_lv):.0f}% of the gap")
    # dominance: dataset d's share of the IoT-mean change is chg_d / (4 * mean chg)
    flags = []
    for k, name in (("fp32", "fp32"), ("pt", "per-tensor int8"), ("pc", "per-channel int8")):
        c = t[f"chg_{k}"]
        tot = c.sum()
        if (c > 0).any() and (c < 0).any():
            flags.append(f"the {name} change has mixed signs across datasets "
                         f"({', '.join(f'{d} {v:+.2f}' for d, v in c.items())}); its IoT mean is a net")
        for d, v in c.items():
            share = v / tot if abs(tot) > 1e-12 else np.inf
            if share > 0.5:
                flags.append(f"{d} accounts for {100 * share:.0f}% of the {name} change "
                             f"({v:+.2f} of the four-dataset sum {tot:+.2f}; IoT mean {tot / 4:+.2f})")
    t.loc["IoT mean"] = t.mean()
    t.to_csv(REPO / "results/tables/tab_e1_matched_info.csv")
    note = ("\nColumns: five-seed means of macro-F1 (%); level = production DNN-Tree input, tq = spike-time "
            "input; gap = dense minus Tree-to-Spike; chg = tq minus level. Outcome decided on fp32.\n")
    dom = ("\nSingle-dataset dominance (> half of a change) and sign notes:\n" + "\n".join(f"- {f}" for f in flags) + "\n"
           if flags else "\nNo single dataset accounts for more than half of any change.\n")
    (REPO / "results/tables/tab_e1_matched_info.md").write_text(
        md(t.round(2)) + note + f"\nPre-registered outcome: {outcome}\n" + dom)
    print(t.round(2).T.to_string())
    print(f"\nE1 gap_TQ = {g_tq:+.2f} (level input {g_lv:+.2f}) -> {outcome}")
    print(dom.strip())


def e2():
    E = "results/encoder_ablation/{d}/entropy/seed_{s}/results.json"
    miss = [f"{d}/{s}" for d in IOT for s in SEEDS if not (REPO / E.format(d=d, s=s)).exists()]
    if miss:
        print(f"E2: {len(miss)} cells missing; not reported"); return
    bad = check_splits(E)
    if bad:
        sys.exit(f"E2: split or teacher differs from Tree-to-Spike on {bad}; refused")
    def arm(pat, d):
        rs = [rj(REPO / pat.format(d=d, s=s)) for s in SEEDS]
        f1 = [100 * r["test_plus"]["metrics"]["f1_macro"] for r in rs]
        ops = [r["test_plus"]["spike_stats"]["synops_per_sample"] + r["data"]["encoder_channels"] * r["config"]["hidden"][0] for r in rs]
        return np.mean(f1), np.std(f1, ddof=1), np.mean(ops)
    def fallback(d):
        dg = [rj(REPO / E.format(d=d, s=s))["data"].get("entropy_edges") for s in SEEDS]
        if any(g is None for g in dg):
            sys.exit(f"E2: {d} has cells without recorded edge diagnostics (pre-amendment runs); refused")
        if any(g["cost_power"] != 1.0 for g in dg):
            sys.exit(f"E2: {d} entropy cutter not weighted with the teacher's power 1.0; refused")
        fill, tot = sum(g["filler_total"] for g in dg), sum(g["edges_entropy_total"] for g in dg)
        short = sum(g["edges_tree_total"] - g["edges_entropy_total"] for g in dg)
        return fill / max(tot, 1), short / len(dg)
    rows = []
    for d in IOT:
        tf, tsd, to = arm("results/Tree-to-Spike/{d}/seed_{s}/results.json", d)
        ef, esd, eo = arm(E, d)
        fb, short = fallback(d)
        rows.append(dict(dataset=d, tree=tf, tree_sd=tsd, entropy=ef, entropy_sd=esd, d_tree_minus_entropy=tf - ef,
                         tree_ksops=to / 1000, entropy_ksops=eo / 1000, fallback_share=fb,
                         edge_shortfall_per_run=short,
                         reading=("largely against quantile placement (Table V)" if fb > 0.5
                                  else "against supervised entropy placement")))
    t = pd.DataFrame(rows).set_index("dataset")
    dm, ahead_t, ahead_e = t.d_tree_minus_entropy.mean(), int((t.d_tree_minus_entropy > 0).sum()), int((t.d_tree_minus_entropy < 0).sum())
    outcome = ("A: tree clearly wins (tree-derived placement is the mechanism)" if dm >= 1.0 and ahead_t >= 3 else
               "B: match (supervised threshold placement)" if abs(dm) < 1.0 else
               "C: entropy clearly wins" if dm <= -1.0 and ahead_e >= 3 else
               "D: mixed; no mechanism claim beyond supervised placement")
    t.loc["IoT mean"] = t.drop(columns="reading").mean()
    t.loc["IoT mean", "reading"] = ""
    t.to_csv(REPO / "results/tables/tab_e2_entropy.csv")
    scope = ("\nScope: E2 isolates edge placement only; channel selection, the per-feature edge budget and "
             "pruning stay tree-derived in both arms.\n")
    fb = [f"- {d}: fallback share {t.loc[d, 'fallback_share']:.2f} -> {t.loc[d, 'reading']}" for d in IOT]
    (REPO / "results/tables/tab_e2_entropy.md").write_text(
        md(t.round(2)) + f"\nPre-registered outcome: {outcome}\n" + scope + "\nFallback reading:\n" + "\n".join(fb) + "\n")
    print(t.round(2).to_string()); print(f"\nE2 d = {dm:+.2f}, tree ahead on {ahead_t}/4 -> {outcome}")
    print(scope.strip()); print("\n".join(fb))


if __name__ == "__main__":
    e1(); e2()
