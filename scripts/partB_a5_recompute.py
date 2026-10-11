#!/usr/bin/env python3
"""A5 consequence: Table VIII and the E1 references with the four Edge-IIoTset M3 cells replaced.

The M3 reruns for Edge-IIoTset seeds 45 and 46 (Tree-to-Spike and DNN-Tree) were made in a
thread-capped pool and did not reproduce the stored runs; the A5 reruns in the default
environment do (results/journal_partB/A5_m3_edge). Their per-channel scores
(results/journal_partB/A5_m3_edge/perchannel, from scripts/perchannel_check.py with the
same guards) replace the four rows of results/tables/tab_perchannel.csv in a new table; the
original is left as it is. Recomputed: Table VIII values (fp32, per-tensor, per-channel;
per dataset and IoT mean), and the E1 quantities that use these references (gap with level
input, gap with spike-time input, percent closed, Edge-IIoTset's share), with the registered
E1 outcome re-derived (A <= +0.35, B >= +0.70, else C).

    python scripts/partB_a5_recompute.py
Output: results/journal_partB/A5_m3_edge/{tab_perchannel_A5.csv,e1_recomputed.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

A5 = C.REPO / "results/journal_partB/A5_m3_edge"
COLS = ["fp32_f1", "per_tensor_f1", "per_channel_f1"]


def main():
    pc = pd.read_csv(C.REPO / "results/tables/tab_perchannel.csv")
    new = pc.copy()
    for m in ("DNN-Tree", "Tree-to-Spike"):
        for s in (45, 46):
            r = json.loads((A5 / f"perchannel/{m}_edgeiiot_seed{s}.json").read_text())
            i = new.index[(new.model == m) & (new.dataset == "edgeiiot") & (new.seed == s)]
            if len(i) != 1:
                sys.exit(f"{m} edgeiiot {s}: row not found")
            for c in COLS:
                new.loc[i, c] = r[c]
            new.loc[i, "device"] = r.get("device")
    new.to_csv(A5 / "tab_perchannel_A5.csv", index=False)
    g_old = pc.groupby(["model", "dataset"])[COLS].mean()
    g_new = new.groupby(["model", "dataset"])[COLS].mean()
    tq = {}
    for d in C.IOT:
        rs = [json.loads((C.REPO / f"results/DNN-TQ/{d}/seed_{s}/results.json").read_text()) for s in C.SEEDS]
        tq[d] = {"fp32_f1": np.mean([100 * r["test_plus"]["fp32_before_projection"]["metrics"]["f1_macro"] for r in rs]),
                 "per_tensor_f1": np.mean([100 * r["test_plus"]["metrics"]["f1_macro"] for r in rs]),
                 "per_channel_f1": np.mean([100 * r["test_plus"]["fp32_before_projection"]["per_channel_int8"]["metrics"]["f1_macro"] for r in rs])}

    def e1(g):
        out = {}
        for c in COLS:
            lvl = {d: g.loc[("DNN-Tree", d), c] - g.loc[("Tree-to-Spike", d), c] for d in C.IOT}
            gtq = {d: tq[d][c] - g.loc[("Tree-to-Spike", d), c] for d in C.IOT}
            chg = {d: tq[d][c] - g.loc[("DNN-Tree", d), c] for d in C.IOT}
            out[c] = dict(gap_level=lvl, gap_tq=gtq, gap_level_iot=float(np.mean(list(lvl.values()))),
                          gap_tq_iot=float(np.mean(list(gtq.values()))),
                          edge_share_pct=float(100 * chg["edgeiiot"] / sum(chg.values())))
        f = out["fp32_f1"]
        f["closed_pct"] = float(100 * (1 - f["gap_tq_iot"] / f["gap_level_iot"]))
        f["outcome"] = "A" if f["gap_tq_iot"] <= 0.35 else "B" if f["gap_tq_iot"] >= 0.70 else "C"
        return out

    res = dict(stamp=C.stamp("A5_recompute"), old=e1(g_old), new=e1(g_new),
               table8_old=g_old.round(4).reset_index().to_dict("records"),
               table8_new=g_new.round(4).reset_index().to_dict("records"),
               table8_iot_mean_new={m: g_new.loc[m].mean().round(4).to_dict() for m in ("DNN-Tree", "Tree-to-Spike")},
               table8_iot_mean_old={m: g_old.loc[m].mean().round(4).to_dict() for m in ("DNN-Tree", "Tree-to-Spike")})
    (A5 / "e1_recomputed.json").write_text(json.dumps(res, indent=1, default=float))
    for k in ("old", "new"):
        f = res[k]["fp32_f1"]
        print(f"{k}: gap level {f['gap_level_iot']:+.4f} | gap TQ {f['gap_tq_iot']:+.4f} | closed {f['closed_pct']:.1f}% | "
              f"Edge share {f['edge_share_pct']:.1f}% | outcome {f['outcome']} | Edge gaps level {f['gap_level']['edgeiiot']:+.2f} tq {f['gap_tq']['edgeiiot']:+.2f}")
    print("Table VIII IoT means old:", res["table8_iot_mean_old"])
    print("Table VIII IoT means new:", res["table8_iot_mean_new"])
    for m in ("DNN-Tree", "Tree-to-Spike"):
        print(m, "Edge-IIoTset old", g_old.loc[(m, "edgeiiot")].round(2).to_dict(), "new", g_new.loc[(m, "edgeiiot")].round(2).to_dict())


if __name__ == "__main__":
    main()
