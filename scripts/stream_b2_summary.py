#!/usr/bin/env python3
"""B3: go/no-go rule R9 over seeds 42 to 46 (results/preregistration/2026-11-17_neuro_stream.md).

At the matched point (validation FAR <= 1% for both), on five-seed means:
  Go if Tree-to-Spike-Stream's test macro-F1 is at most 2.0 points below the per-packet tree's,
  and (a) its median packets-to-detection (attack flows detected by both) is lower than the
  tree's, or (b) the tree's node comparisons per packet are at least 5 times its SOPs per packet.

    python scripts/stream_b2_summary.py
Output: results/neuro/b2/SUMMARY.md, summary.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
B2 = REPO / "results/neuro/b2"
SEEDS = range(42, 47)
MODELS = ("t2s_stream", "tree_per_packet", "compact_gbdt")


def main():
    rows = []
    for s in SEEDS:
        e = json.loads((B2 / f"seed_{s}/eval.json").read_text())
        for m in MODELS:
            if m not in e:
                continue
            r = e[m]
            for tgt, v in r["at"].items():
                rows.append(dict(seed=s, model=m, target_far=float(tgt), **v["test"],
                                 p2d_both_median=v["p2d_both"]["median"] if "p2d_both" in v else None,
                                 p2d_both_p90=v["p2d_both"]["p90"] if "p2d_both" in v else None,
                                 p2d_own_median=v["p2d_own"]["median"] if "p2d_own" in v else None,
                                 p2d_own_p90=v["p2d_own"]["p90"] if "p2d_own" in v else None,
                                 ops_per_packet=r.get("sops_per_packet", r.get("comparisons_per_packet")),
                                 no_spike_share=r.get("no_spike_packet_share"), theta=r["theta"],
                                 mean_packets=r["mean_packets"]))
    df = pd.DataFrame(rows)
    df.to_csv(B2 / "summary.csv", index=False)
    mean = df.groupby(["model", "target_far"]).mean(numeric_only=True)
    s_, t_ = mean.loc[("t2s_stream", 1.0)], mean.loc[("tree_per_packet", 1.0)]
    d_f1 = s_["macro_f1"] - t_["macro_f1"]
    a_ = s_["p2d_both_median"] < t_["p2d_both_median"]
    ratio = t_["ops_per_packet"] / s_["ops_per_packet"]
    b_ = ratio >= 5
    go = (d_f1 >= -2.0) and (a_ or b_)
    out = ["# B2 stream pilot: results and rule R9", "",
           "Five-seed means (42 to 46), test split, percent. Matched point: benign threshold tuned on validation "
           "to FAR <= 1% for each model, after its gate.", "",
           "| Model | Target FAR | Macro-F1 | FAR | DR | P2D median (both) | P2D p90 (both) | Ops per packet |",
           "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for (m, tgt), r in mean.iterrows():
        out.append(f"| {m} | {tgt:g}% | {r.macro_f1:.2f} | {r.far:.2f} | {r.dr:.2f} | "
                   f"{r.p2d_both_median if pd.notna(r.p2d_both_median) else float('nan'):.1f} | "
                   f"{r.p2d_both_p90 if pd.notna(r.p2d_both_p90) else float('nan'):.1f} | {r.ops_per_packet:.0f} |")
    out += ["", "## Rule R9", "",
            f"- Macro-F1, Tree-to-Spike-Stream minus per-packet tree: {d_f1:+.2f} (limit -2.00): "
            f"{'within' if d_f1 >= -2.0 else 'outside'}.",
            f"- (a) median packets-to-detection: {s_['p2d_both_median']:.2f} vs {t_['p2d_both_median']:.2f}: "
            f"{'lower' if a_ else 'not lower'}.",
            f"- (b) tree comparisons per packet / SNN SOPs per packet: {ratio:.2f} (needs >= 5): "
            f"{'met' if b_ else 'not met'}. Synaptic operations and node comparisons are different operations.",
            f"- Share of executed packets with no input spikes (SNN): {100 * s_['no_spike_share']:.1f}%.", "",
            f"**Decision: {'GO' if go else 'NO-GO'}.** "
            + ("Continue with B4." if go else "The paper ships with scope A only."), ""]
    (B2 / "SUMMARY.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
