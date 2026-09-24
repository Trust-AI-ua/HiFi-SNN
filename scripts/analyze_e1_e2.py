#!/usr/bin/env python3
"""Apply the pre-registered rules to E1 and E2 (results/preregistration/
2026-09-24_matched_info_and_entropy_edges.md). Refuses to report while any cell is
missing or any new run's split differs from the stored Tree-to-Spike run.

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
    return (r["test_plus"].get("n_samples"), json.dumps(r["data"].get("class_counts_train"), sort_keys=True))


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
        out.append(f"| {i} | " + " | ".join(f"{v:+.2f}" if isinstance(v, float) and ("gap" in c or c.startswith("d")) else
                                           (f"{v:.2f}" if isinstance(v, float) else str(v)) for c, v in r.items()) + " |")
    return "\n".join(out) + "\n"


def e1():
    E = "results/DNN-TQ/{d}/seed_{s}/results.json"
    miss = [f"{d}/{s}" for d in IOT for s in SEEDS if not (REPO / E.format(d=d, s=s)).exists()]
    if miss:
        print(f"E1: {len(miss)} cells missing; not reported"); return
    bad = check_splits(E)
    if bad:
        sys.exit(f"E1: splits differ from Tree-to-Spike on {bad}; refused")
    pc = pd.read_csv(REPO / "results/tables/tab_perchannel.csv")
    ref = pc.groupby(["model", "dataset"])[["fp32_f1", "per_tensor_f1"]].mean()
    rows = []
    for d in IOT:
        rs = [rj(REPO / E.format(d=d, s=s)) for s in SEEDS]
        fp = np.mean([100 * r["test_plus"]["fp32_before_projection"]["metrics"]["f1_macro"] for r in rs])
        q8 = np.mean([100 * r["test_plus"]["metrics"]["f1_macro"] for r in rs])
        t_fp, t_q = ref.loc[("Tree-to-Spike", d)]
        l_fp, l_q = ref.loc[("DNN-Tree", d)]
        rows.append(dict(dataset=d, dnn_level_fp32=l_fp, dnn_tq_fp32=fp, t2s_fp32=t_fp, gap_level_fp32=l_fp - t_fp,
                         gap_tq_fp32=fp - t_fp, dnn_level_int8=l_q, dnn_tq_int8=q8, t2s_int8=t_q, gap_tq_int8=q8 - t_q))
    t = pd.DataFrame(rows).set_index("dataset")
    g_tq, g_lv = t.gap_tq_fp32.mean(), t.gap_level_fp32.mean()
    outcome = ("A: matched information closes the gap" if g_tq <= 0.35 else
               "B: about one point is the cost of spiking computation" if g_tq >= 0.70 else
               f"C: input resolution explains {100 * (1 - g_tq / g_lv):.0f}% of the gap")
    t.loc["IoT mean"] = t.mean()
    t.to_csv(REPO / "results/tables/tab_e1_matched_info.csv")
    (REPO / "results/tables/tab_e1_matched_info.md").write_text(md(t.round(2)) + f"\nPre-registered outcome: {outcome}\n")
    print(t.round(2).to_string()); print(f"\nE1 gap_TQ = {g_tq:+.2f} (level input {g_lv:+.2f}) -> {outcome}")


def e2():
    E = "results/encoder_ablation/{d}/entropy/seed_{s}/results.json"
    miss = [f"{d}/{s}" for d in IOT for s in SEEDS if not (REPO / E.format(d=d, s=s)).exists()]
    if miss:
        print(f"E2: {len(miss)} cells missing; not reported"); return
    bad = check_splits(E)
    if bad:
        sys.exit(f"E2: splits differ from Tree-to-Spike on {bad}; refused")
    def arm(pat, d):
        rs = [rj(REPO / pat.format(d=d, s=s)) for s in SEEDS]
        f1 = [100 * r["test_plus"]["metrics"]["f1_macro"] for r in rs]
        ops = [r["test_plus"]["spike_stats"]["synops_per_sample"] + r["data"]["encoder_channels"] * r["config"]["hidden"][0] for r in rs]
        return np.mean(f1), np.std(f1, ddof=1), np.mean(ops)
    rows = []
    for d in IOT:
        tf, tsd, to = arm("results/Tree-to-Spike/{d}/seed_{s}/results.json", d)
        ef, esd, eo = arm(E, d)
        rows.append(dict(dataset=d, tree=tf, tree_sd=tsd, entropy=ef, entropy_sd=esd, d_tree_minus_entropy=tf - ef,
                         tree_ksops=to / 1000, entropy_ksops=eo / 1000))
    t = pd.DataFrame(rows).set_index("dataset")
    dm, ahead_t, ahead_e = t.d_tree_minus_entropy.mean(), int((t.d_tree_minus_entropy > 0).sum()), int((t.d_tree_minus_entropy < 0).sum())
    outcome = ("A: tree clearly wins (tree-derived placement is the mechanism)" if dm >= 1.0 and ahead_t >= 3 else
               "B: match (supervised threshold placement)" if abs(dm) < 1.0 else
               "C: entropy clearly wins" if dm <= -1.0 and ahead_e >= 3 else
               "D: mixed; no mechanism claim beyond supervised placement")
    t.loc["IoT mean"] = t.mean()
    t.to_csv(REPO / "results/tables/tab_e2_entropy.csv")
    (REPO / "results/tables/tab_e2_entropy.md").write_text(md(t.round(2)) + f"\nPre-registered outcome: {outcome}\n")
    print(t.round(2).to_string()); print(f"\nE2 d = {dm:+.2f}, tree ahead on {ahead_t}/4 -> {outcome}")


if __name__ == "__main__":
    e1(); e2()
