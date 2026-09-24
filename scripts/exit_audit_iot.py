#!/usr/bin/env python3
"""Early-exit audit statistics on the IoT four, all five seeds, from prefix artifacts.

Same definitions as scripts/evidence_arrival.py, which computes them at seed 42 only
by replaying the checkpoint:
  t'          first step whose argmax equals the true label, among samples correct at
              the full window (1-indexed); reported as median and p90 (skewed)
  inclusion   share of test samples on which t' is defined = correct at the full window
  wrong-exit  share of ALL test samples that cross a FIXED diagnostic theta = 0.90 with
              a wrong label (not the validation-selected threshold, so it is defined
              where the gate rejects exit)
The prefix artifacts hold the same running confidence (softmax of the cumulative
output membrane / t) that replay() recomputes, so seed 42 must reproduce
results/evidence_arrival.json before the other seeds are reported.

    python scripts/exit_audit_iot.py
Output: results/tables/tab_exit_audit_iot.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
TH = 0.90


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def stats(ds, seed):
    z = np.load(REPO / f"results/prefix_outputs/{ds}/treespike/seed_{seed}/test_prefix.npz", allow_pickle=True)
    q = softmax(z["prefix_logits"].astype(np.float32))
    conf, pred, y = q.max(-1), q.argmax(-1), z["labels"]
    T = pred.shape[1]
    ok = pred[:, -1] == y
    hit = (pred == y[:, None]) & ok[:, None]
    tp = hit.argmax(1)[ok] + 1
    fired = conf >= TH
    ex = fired.any(1)
    t_star = np.where(ex, fired.argmax(1), T - 1)
    wrong = ex & (pred[np.arange(len(y)), t_star] != y)
    return dict(dataset=ds, seed=seed, n=len(y), inclusion=100 * ok.mean(),
                t_median=float(np.median(tp)), t_p90=float(np.percentile(tp, 90)),
                wrong_exit=100 * wrong.mean(), exit_share=100 * ex.mean()), tp


def main():
    ref = json.loads((REPO / "results/evidence_arrival.json").read_text())
    rows, pooled = [], {}
    for ds in IOT:
        for s in range(42, 47):
            r, tp = stats(ds, s)
            rows.append(r)
            pooled.setdefault(ds, []).append(tp)
        # guard: seed 42 from the prefixes must equal the checkpoint replay
        r42 = [x for x in rows if x["dataset"] == ds and x["seed"] == 42][0]
        e = ref[ds]
        inc = 100 * sum(v["frac_defined"] * v["n"] for v in e["defined"].values()) / sum(v["n"] for v in e["defined"].values())
        diffs = dict(median=r42["t_median"] - e["median"], p90=r42["t_p90"] - e["p90"],
                     wrong=r42["wrong_exit"] - 100 * e["wrong_exit_frac"], inclusion=r42["inclusion"] - inc)
        bad = {k: v for k, v in diffs.items() if abs(v) > 0.01}
        print(f"guard {ds}: seed-42 prefix vs replay differences {({k: round(v, 4) for k, v in diffs.items()})}"
              f" -> {'OK' if not bad else 'FAIL'}")
        if bad:
            raise SystemExit(f"{ds}: prefix statistics do not reproduce evidence_arrival.json; refused")
    d = pd.DataFrame(rows)
    d.to_csv(REPO / "results/tables/tab_exit_audit_iot.csv", index=False)
    print("\nper seed")
    print(d.round(2).to_string(index=False))
    print("\nfive-seed summary (median/p90 pooled over all included samples; shares = mean over seeds)")
    for ds in IOT:
        tp = np.concatenate(pooled[ds]); g = d[d.dataset == ds]
        print(f"  {ds:12s} t' median {np.median(tp):.0f} p90 {np.percentile(tp, 90):.0f} | "
              f"inclusion {g.inclusion.mean():.2f}% (range {g.inclusion.min():.2f}-{g.inclusion.max():.2f}) | "
              f"wrong-exit {g.wrong_exit.mean():.2f}% (range {g.wrong_exit.min():.2f}-{g.wrong_exit.max():.2f})")


if __name__ == "__main__":
    main()
