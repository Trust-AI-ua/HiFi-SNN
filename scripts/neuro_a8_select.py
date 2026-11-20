#!/usr/bin/env python3
"""Amendment 8, part 2: validation selection of the fixed-point Vanilla SNN's weight scale
(results/preregistration/2026-11-17_neuro_stream.md). Runs in .venv; touches no test-split prediction.

  select   for each dataset and seed: every candidate p (per-tensor |W| percentile mapped to 127; weights
           beyond it saturate) through Lava's range check and the bit-exact Lava emulation on the whole
           validation split; the highest validation macro-F1 wins (ties: larger p). Writes
           results/neuro/review/a8_tuned_vanilla/selection.json.
  export   reads selection.json and writes the selected network's integer parameters to
           results/neuro/inputs/<ds>_vanilla_tuned_seed<S>.npz for scripts/neuro_a8_tuned.py.

    .venv/bin/python scripts/neuro_a8_select.py select <dataset> [<dataset> ...]
    .venv/bin/python scripts/neuro_a8_select.py export <dataset> [<dataset> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/review/a8_tuned_vanilla"
SEL = OUT / "selection.json"
CANDIDATES = (100.0, 99.99, 99.9, 99.5, 99.0)
SEEDS = (42, 43, 44, 45, 46)


def select(datasets):
    from sklearn.metrics import f1_score
    from neuro_a2_vanilla import vanilla_lava_params
    from neuro_stagewise import vanilla_split, vanilla_params, lava_vanilla, batched
    sel = json.loads(SEL.read_text()) if SEL.exists() else {}
    for ds in datasets:
        for seed in SEEDS:
            v = dict(np.load(INP / f"{ds}_vanilla_seed{seed}.npz"))
            st, y, oc_gpu, T, _ = vanilla_split(ds, seed, "val")
            cands = []
            for p in CANDIDATES:
                P = vanilla_params(v, (p, p))
                c = dict(p=p, s_h=P["s_h"], s_o=P["s_o"], saturated_weights_h=int((np.abs(np.round(v["W_ih"] / P["s_h"])) > 127).sum()),
                         saturated_weights_o=int((np.abs(np.round(v["W_out"] / P["s_o"])) > 127).sum()),
                         zero_weights=int((P["Wh"] == 0).sum() + (P["Wo"] == 0).sum()))
                try:
                    vl, info, dv = vanilla_lava_params(v, dict(P_Wh=P["Wh"], P_Wo=P["Wo"], s_h=P["s_h"], s_o=P["s_o"]))
                except SystemExit as e:
                    cands.append(dict(c, admissible=False, reason=str(e))); continue
                oc, _, sat = batched(lambda x: lava_vanilla(vl, x, T, dv=dv), st)
                cands.append(dict(c, admissible=True, vth=[i["vth"] for i in info], bias_exp_max=[i["bias_exp_max"] for i in info],
                                  val_macro_f1=100 * f1_score(y, oc.argmax(1), average="macro"), val_saturated_frac=float(sat.mean())))
                print(ds, seed, p, round(cands[-1]["val_macro_f1"], 3), flush=True)
            ok = [c for c in cands if c["admissible"]]
            best = max(ok, key=lambda c: (round(c["val_macro_f1"], 10), c["p"]))
            sel[f"{ds}/{seed}"] = dict(selected_p=best["p"], val_macro_f1_float_gpu=100 * f1_score(y, oc_gpu.argmax(1), average="macro"),
                                       n_val=int(len(y)), candidates=cands)
            OUT.mkdir(parents=True, exist_ok=True)
            SEL.write_text(json.dumps(sel, indent=1))


def export(datasets):
    from neuro_stagewise import vanilla_params
    sel = json.loads(SEL.read_text())
    for ds in datasets:
        for seed in SEEDS:
            p = sel[f"{ds}/{seed}"]["selected_p"]
            v = dict(np.load(INP / f"{ds}_vanilla_seed{seed}.npz"))
            vi = np.load(INP / f"{ds}_vanilla_int_seed{seed}.npz")
            P = vanilla_params(v, (p, p))
            np.savez_compressed(INP / f"{ds}_vanilla_tuned_seed{seed}.npz", P_Wh=P["Wh"], P_Wo=P["Wo"], s_h=P["s_h"], s_o=P["s_o"],
                                p=np.float64(p), sample=vi["sample"], labels_full=vi["labels_full"],
                                spike_times_full=vi["spike_times_full"])
            print(ds, seed, "selected p", p, flush=True)


if __name__ == "__main__":
    {"select": select, "export": export}[sys.argv[1]](sys.argv[2:])
