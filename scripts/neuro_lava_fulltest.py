#!/usr/bin/env python3
"""Full-test macro-F1 of the networks Lava and SANA-FE run (descriptive; added after the A2 Vanilla
result, results/preregistration/2026-11-17_neuro_stream.md, amendment 4 note).

A2 checks each Lava run against a NumPy emulation of Lava's arithmetic, bit for bit, on the
2,000-flow sample, and A3 checks SANA-FE against Lava on the same flows. This script runs those
emulations (src/lava_sublif: emulate for Tree-to-Spike with the subtractive reset, emulate_vanilla
for the Vanilla SNN) on the whole test split, so the energy can be paired with the accuracy of the
network it was measured on. Runs in .venv-neuro through scripts/neuro_guard.py.

    .venv-neuro/bin/python scripts/neuro_lava_fulltest.py [datasets...]
Output: results/neuro/a2_lava/lava_fulltest.csv
"""
from __future__ import annotations

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
        print(rows[-2], rows[-1], flush=True)
    p = OUT / "lava_fulltest.csv"
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old[~old.dataset.isin(new.dataset)], new])
    new.sort_values(["model", "dataset"]).to_csv(p, index=False)


if __name__ == "__main__":
    main()
