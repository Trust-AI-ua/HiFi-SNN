#!/usr/bin/env python3
"""Amendment 8, part 2: the validation-selected fixed-point Vanilla SNN through the amendment-6 chain
(results/preregistration/2026-11-17_neuro_stream.md). Runs in .venv-neuro through scripts/neuro_guard.py,
after scripts/neuro_a8_select.py select + export.

As scripts/neuro_confirm.py run_vanilla, with the selected integer weights (inputs/<ds>_vanilla_tuned_seed<S>.npz):
Lava fixed point on the seed's 2,000-flow sample, equal bit for bit to the NumPy emulation; SANA-FE with the
amendment-2 gate; energy and latency per inference; the full-test predictions of the simulated network.

    .venv-neuro/bin/python scripts/neuro_a8_tuned.py <dataset> <seed>
Output: results/neuro/review/a8_tuned_vanilla/<ds>_seed<S>_vanilla_tuned.json, per_flow_<ds>_seed<S>_vanilla_tuned.csv,
        <ds>_seed<S>_vanilla_tuned_pred.npz (full-test predictions and output spike counts)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/review/a8_tuned_vanilla"
CH = 20000


def main():
    import neuro_confirm as C
    from neuro_a2_vanilla import vanilla_lava_params, run_lava, DU
    from src.lava_sublif.sublif import emulate_vanilla
    ds, seed = sys.argv[1], int(sys.argv[2])
    sel = json.loads((OUT / "selection.json").read_text())[f"{ds}/{seed}"]
    v = np.load(INP / f"{ds}_vanilla_seed{seed}.npz")
    vt = np.load(INP / f"{ds}_vanilla_tuned_seed{seed}.npz")
    if float(vt["p"]) != sel["selected_p"]:
        raise SystemExit("tuned export is not the selected candidate")
    T = int(v["T"])
    st = v["spike_times"].astype(np.int64)
    vl, info, dv = vanilla_lava_params(v, vt)
    t0 = time.time()
    oc, cnt, sec = run_lava(vl, st, T, dv)
    e_oc, e_cnt, _ = emulate_vanilla(vl, st, T, du=DU, dv=dv)
    if not (np.array_equal(oc, e_oc) and np.array_equal(cnt, e_cnt)):
        raise SystemExit(f"{ds}/{seed}/vanilla_tuned: Lava differs from its NumPy emulation")
    vst, vy = vt["spike_times_full"].astype(np.int64), vt["labels_full"]
    full_oc = np.concatenate([emulate_vanilla(vl, vst[i:i + CH], T, du=DU, dv=dv)[0] for i in range(0, len(vst), CH)])
    full = full_oc.argmax(1)
    if not np.array_equal(full[vt["sample"]], oc.argmax(1)):
        raise SystemExit(f"{ds}/{seed}: full-test tuned emulation differs from Lava on the sample")
    hid = dict(reset_mode="none", reverse_threshold=0.0, reverse_reset=0.0, reverse_reset_mode="saturate")
    layers = [dict(W=vl[0]["W"].astype(np.float64), bias=vl[0]["bias_eff"] / 64.0, leak=(4096 - dv) / 4096,
                   thr=float(vl[0]["vth"]), soma="loihi_lif", attrs=hid),
              dict(W=vl[1]["W"].astype(np.float64), bias=vl[1]["bias_eff"] / 64.0, leak=(4096 - dv) / 4096,
                   thr=float(vl[1]["vth"]), soma="loihi_lif_delayed")]
    d, rate = C.sanafe_rows(layers, st, np.full(len(st), T), "spikes",
                            lambda i, al, vec: np.array_equal(al, cnt[i]) and np.array_equal(vec, oc[i]))
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / f"per_flow_{ds}_seed{seed}_vanilla_tuned.csv", index=False)
    np.savez_compressed(OUT / f"{ds}_seed{seed}_vanilla_tuned_pred.npz", pred=full.astype(np.int8), out_counts=full_oc.astype(np.int16),
                        labels=vy)
    nz = int((vt["P_Wh"] != 0).sum() + (vt["P_Wo"] != 0).sum())
    C.OUT = OUT
    C.row(ds, seed, "vanilla_tuned", d, rate, 100 * f1_score(vy, full, average="macro"),
          dict(T=T, selected_p=float(vt["p"]), s_h=float(vt["s_h"]), s_o=float(vt["s_o"]), vth=[i["vth"] for i in info],
               weight_bytes_int8=int(vt["P_Wh"].size + vt["P_Wo"].size), nonzero_synapses=nz, lava_seconds=round(sec, 1)))
    print(f"[done] {ds} seed {seed} vanilla_tuned in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
