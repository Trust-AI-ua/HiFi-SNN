#!/usr/bin/env python3
"""Full-test macro-F1 of the networks Lava and SANA-FE run (descriptive; added after the A2 Vanilla
result, results/preregistration/2026-11-17_neuro_stream.md, amendment 4 note).

A2 checks each Lava run against a NumPy emulation of Lava's arithmetic, bit for bit, on the
2,000-flow sample, and A3 checks SANA-FE against Lava on the same flows. This script runs those
emulations (src/lava_sublif: emulate for Tree-to-Spike with the subtractive reset, emulate_vanilla
for the Vanilla SNN) on the whole test split, so the energy can be paired with the accuracy of the
network it was measured on. Also Tree-to-Spike with early exit: each test flow's C1-gate exit step
(the rule of scripts/neuro_a3_sanafe.py exit_steps, applied to the whole test split; no exit where the
gate disables it), and the emulated readout over that flow's window. Runs in .venv-neuro through
scripts/neuro_guard.py.

    .venv-neuro/bin/python scripts/neuro_lava_fulltest.py [datasets...]
Output: results/neuro/a2_lava/lava_fulltest.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/a2_lava"
DS = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
CH = 20000


def exit_steps_all(ds, z):
    """C1-gate exit step (1-based window length) for every test flow; None if the gate disables exit."""
    thr = json.loads((REPO / "results/journal_r2/exit_calibration/thresholds.json").read_text())[f"{ds}/42"]
    T = int(z["T"])
    if thr["gate_c1"] is None:
        return None, None
    p = np.load(REPO / f"results/prefix_outputs/{ds}/treespike/seed_42/test_prefix.npz")
    if not np.array_equal(p["labels"], z["labels"]):
        raise SystemExit(f"{ds}: prefix outputs are not in the export's test order")
    out = []
    for i in range(0, len(z["labels"]), CH):
        zl = p["prefix_logits"][i:i + CH].astype(np.float64) / np.asarray(thr["tau"])[None, :, None]
        zl -= zl.max(2, keepdims=True)
        prob = np.exp(zl); prob /= prob.sum(2, keepdims=True)
        fired = prob.max(2) >= thr["gate_c1"]
        out.append(np.where(fired.any(1), fired.argmax(1), T - 1) + 1)
    return np.concatenate(out), thr["gate_c1"]


def main():
    from neuro_a2_lava import lava_params, DU, DV
    from neuro_a2_vanilla import vanilla_lava_params
    from src.lava_sublif.sublif import emulate, emulate_vanilla
    rows = []
    for ds in (sys.argv[1:] or DS):
        z = np.load(INP / f"{ds}_seed42.npz")
        v = np.load(INP / f"{ds}_vanilla_seed42.npz")
        vi = np.load(INP / f"{ds}_vanilla_int_seed42.npz")
        lp, _ = lava_params(z)
        st, y, T = z["spike_times"].astype(np.int64), z["labels"], int(z["T"])
        pred = np.concatenate([emulate(lp, st[i:i + CH], T, du=DU, dv=DV, reset="subtract")[0].argmax(1)
                               for i in range(0, len(st), CH)])
        samp = z["sample"]
        lava = np.load(OUT / f"{ds}_subtract_perflow.npz")
        if not np.array_equal(pred[samp], lava["pred"]):
            raise SystemExit(f"{ds}: full-test emulation differs from the Lava run on the sample")
        f1 = lambda yy, p: 100 * f1_score(yy, p, average="macro")
        rows.append(dict(dataset=ds, model="t2s", n_test=int(len(y)), lava_arith_macro_f1=f1(y, pred),
                         int_numpy_macro_f1=f1(y, z["int_pred"]), agreement_with_int_numpy=100 * float(np.mean(pred == z["int_pred"]))))
        ex, gate = exit_steps_all(ds, z)
        if ex is None:
            epred, mean_steps = pred, float(T)
        else:
            epred = np.empty_like(pred)
            for k in np.unique(ex):
                idx = np.flatnonzero(ex == k)
                epred[idx] = np.concatenate([emulate(lp, st[idx[i:i + CH]], int(k), du=DU, dv=DV, reset="subtract")[0].argmax(1)
                                             for i in range(0, len(idx), CH)])
            mean_steps = float(ex.mean())
            per = pd.read_csv(REPO / f"results/neuro/a3_sanafe/per_flow_{ds}.csv")
            pe = per[(per.variant == "t2s_exit") & (per.mapping == "base")].sort_values("flow")
            if not np.array_equal(pe.pred.values, epred[samp]):
                raise SystemExit(f"{ds}: full-test exit emulation differs from SANA-FE's exit run on the sample")
        rows.append(dict(dataset=ds, model="t2s_exit", n_test=int(len(y)), lava_arith_macro_f1=f1(y, epred),
                         int_numpy_macro_f1=None, agreement_with_int_numpy=None, exit_gate=gate, mean_steps=mean_steps))
        vl, _, vdv = vanilla_lava_params(v, vi)
        vst, vy, VT = vi["spike_times_full"].astype(np.int64), vi["labels_full"], int(v["T"])
        vpred = np.concatenate([emulate_vanilla(vl, vst[i:i + CH], VT, du=DU, dv=vdv)[0].argmax(1)
                                for i in range(0, len(vst), CH)])
        lv = np.load(OUT / f"{ds}_vanilla_perflow.npz")
        if not np.array_equal(vpred[vi["sample"]], lv["pred"]):
            raise SystemExit(f"{ds}: full-test Vanilla emulation differs from the Lava run on the sample")
        rows.append(dict(dataset=ds, model="vanilla", n_test=int(len(vy)), lava_arith_macro_f1=f1(vy, vpred),
                         int_numpy_macro_f1=f1(vy, vi["int_pred_full"]),
                         agreement_with_int_numpy=100 * float(np.mean(vpred == vi["int_pred_full"]))))
        print(rows[-3:], flush=True)
    p = OUT / "lava_fulltest.csv"
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old[~old.dataset.isin(new.dataset)], new])
    new.sort_values(["model", "dataset"]).to_csv(p, index=False)


if __name__ == "__main__":
    main()
