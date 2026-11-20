#!/usr/bin/env python3
"""Amendment 8, part 5: the C1-gate validation search written out candidate by candidate
(results/preregistration/2026-11-17_neuro_stream.md). Runs in .venv; recomputes, from the cached prefix
outputs, what scripts/r2_exit_calibration.py's gate() evaluated, and checks that the selected threshold
equals results/journal_r2/exit_calibration/thresholds.json.

Rule (scripts/early_exit_sweep.py, scripts/r2_exit_calibration.py): per-step temperatures tau_t fitted on
validation (NLL); for each theta in THETAS, exit at the first step whose calibrated maximum class probability
reaches theta (else the full window); a theta is eligible if its validation macro-F1 is at least the full
window's minus MARGIN (1.0 point); among eligible thetas the lowest mean SOPs wins (then higher macro-F1,
lower FAR, higher theta); none eligible = exit disabled (gate_c1 = None).

    .venv/bin/python scripts/neuro_review_gate_search.py
Output: results/neuro/review/p5_selection/gate_search_c1.csv, selected_policy.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/neuro/review/p5_selection"


def main():
    import early_exit_sweep as EE
    from r2_exit_calibration import fit_tau, IOT, SEEDS
    thr = json.loads((REPO / "results/journal_r2/exit_calibration/thresholds.json").read_text())
    ps = pd.read_csv(REPO / "results/journal_r2/exit_calibration/per_seed.csv")
    rows, sel = [], []
    for ds in IOT:
        for seed in SEEDS:
            pv = EE.load(ds, seed, "val")
            yv, zv = pv["labels"], pv["prefix_logits"].astype(np.float32)
            T = zv.shape[1]
            tau = np.array([fit_tau(zv[:, t], yv) for t in range(T)])
            if not np.allclose(tau.round(4), thr[f"{ds}/{seed}"]["tau"]):
                raise SystemExit(f"{ds}/{seed}: temperatures differ from thresholds.json")
            pv2 = dict(pv, prefix_logits=zv / tau[None, :, None])
            pred_f, _, sops_f, _ = EE.full_window(pv2)
            sf = EE.score(yv, pred_f)
            cand = []
            for th in EE.THETAS:
                p, ex, ever, sops, _ = EE.replay(pv2, th)
                s = EE.score(yv, p)
                cand.append(dict(dataset=ds, seed=seed, theta=th, val_macro_f1=s["macro_f1"], val_FAR=s["FAR"],
                                 val_mean_sops=float(sops.mean()), val_full_window_macro_f1=sf["macro_f1"],
                                 val_full_window_FAR=sf["FAR"], val_full_window_mean_sops=float(sops_f.mean()),
                                 val_mean_exit_step=float(np.mean(ex)),
                                 eligible=bool(s["macro_f1"] >= sf["macro_f1"] - EE.MARGIN)))
            elig = [c for c in cand if c["eligible"]]
            chosen = sorted(elig, key=lambda c: (c["val_mean_sops"], -c["val_macro_f1"], c["val_FAR"], -c["theta"]))[0]["theta"] if elig else None
            if chosen != thr[f"{ds}/{seed}"]["gate_c1"]:
                raise SystemExit(f"{ds}/{seed}: recomputed gate {chosen} is not thresholds.json's {thr[f'{ds}/{seed}']['gate_c1']}")
            for c in cand:
                c["selected"] = c["theta"] == chosen
            rows += cand
            t = ps[(ps.dataset == ds) & (ps.seed == seed) & (ps.op == "C1-gate")].iloc[0]
            sel.append(dict(dataset=ds, seed=seed, policy="C1-gate", gate_c1=chosen, exit_disabled=chosen is None,
                            n_eligible=len(elig), val_macro_f1=next((c["val_macro_f1"] for c in cand if c["selected"]), sf["macro_f1"]),
                            val_FAR=next((c["val_FAR"] for c in cand if c["selected"]), sf["FAR"]),
                            test_macro_f1=t.macro_f1, test_FAR=t.far, test_full_window_macro_f1=t.full_macro_f1,
                            test_full_window_FAR=t.full_far, test_mean_exit_step=t.mean_exit, test_wrong_exit=t.wrong_exit,
                            test_sop_reduction=t.sop_reduction))
            print(ds, seed, "gate", chosen, "eligible", len(elig), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "gate_search_c1.csv", index=False)
    pd.DataFrame(sel).to_csv(OUT / "selected_policy.csv", index=False)


if __name__ == "__main__":
    main()
