#!/usr/bin/env python3
"""Amendment 4: Lava fixed-point execution of the seed-42 Vanilla SNN (LeakyParallel, latency)
(results/preregistration/2026-11-17_neuro_stream.md).

Runs in .venv-neuro, through scripts/neuro_guard.py. Inputs: results/neuro/inputs/<ds>_vanilla_seed42.npz
(scripts/neuro_export_vanilla.py) and <ds>_vanilla_int_seed42.npz (scripts/neuro_vanilla_int.py).

Network: RingBuffer (latency spike trains) -> Dense -> ReluLIFReset (hidden: no reset, floor at
zero) -> Dense -> SubLIFReset (output: snnTorch's delayed subtractive reset); src/lava_sublif,
Loihi2SimCfg(select_tag="fixed_pt"). Fixed point as in A2: the int8 weights as 2W, vth = round(2 thr / s),
bias = round(2 b / s * 64) as mantissa << exponent; du = 4095; dv = 614 (beta = 3482/4096 = 0.85010).
One 32-step slot per flow; readout: output spike count over the T aligned steps, argmax.
Check: every Lava run must equal the NumPy emulation (emulate_vanilla) bit for bit.
Reported with the P5 thresholds (agreement with integer NumPy >= 99%, |delta macro-F1| <= 0.5),
descriptively: P5 is the Tree-to-Spike rule.

    .venv-neuro/bin/python scripts/neuro_a2_vanilla.py [datasets...]
Output: results/neuro/a2_lava/<ds>_vanilla.json, <ds>_vanilla_perflow.npz, a2_vanilla.csv
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/a2_lava"
DS = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
SLOT, DU = 32, 4095


def vanilla_lava_params(v, vi):
    """Lava parameters for the hidden and output layers, and the leak dv (from beta)."""
    beta = float(np.unique(v["beta_h"])[0])
    if len(np.unique(v["beta_h"])) != 1 or abs(float(v["beta_out"]) - beta) > 1e-6:
        raise SystemExit("one scalar beta expected for both layers")
    dv = int(round(4096 * (1 - beta)))
    layers, info = [], []
    for name, W, s, b, thr in (("hidden", vi["P_Wh"], float(vi["s_h"]), v["b_h"], float(v["thr_h"])),
                               ("output", vi["P_Wo"], float(vi["s_o"]), v["b_out"], float(v["thr_out"]))):
        vth = int(round(2.0 * thr / s))
        bv = np.round(2.0 * b.astype(np.float64) / s * 64).astype(np.int64)
        exp = np.zeros_like(bv)
        while np.any(np.abs(np.round(bv / 2.0 ** exp)) > 4095):
            exp = np.where(np.abs(np.round(bv / 2.0 ** exp)) > 4095, exp + 1, exp)
        if exp.max() > 7 or vth > 131071 or vth < 1 or np.abs(W).max() > 127:
            raise SystemExit(f"{name}: parameters out of Lava's range (exp {exp.max()}, vth {vth})")
        mant = np.round(bv / 2.0 ** exp).astype(np.int64)
        layers.append(dict(W=2 * W.astype(np.int64), bias_mant=mant, bias_exp=exp, bias_eff=mant << exp, vth=vth))
        info.append(dict(layer=name, vth=vth, exact_theta=thr / s, bias_exp_max=int(exp.max())))
    return layers, info, dv


def run_lava(layers, st, T, dv):
    from lava.proc.dense.process import Dense
    from lava.proc.io.source import RingBuffer
    from lava.proc.monitor.process import Monitor
    from lava.magma.core.run_configs import Loihi2SimCfg
    from lava.magma.core.run_conditions import RunSteps
    from src.lava_sublif.sublif import ReluLIFReset, SubLIFReset
    n, C = st.shape
    steps = n * SLOT
    data = np.zeros((C, steps), np.int32)
    f, c = np.nonzero(st >= 0)
    data[c, f * SLOT + st[f, c]] = 1
    src = RingBuffer(data=data)
    prev, neurons = src.s_out, []
    for k, (L, P) in enumerate(zip(layers, (ReluLIFReset, SubLIFReset)), 1):
        d = Dense(weights=L["W"], num_message_bits=0)
        nr = P(shape=(L["W"].shape[0],), du=DU, dv=dv, bias_mant=L["bias_mant"], bias_exp=L["bias_exp"],
               vth=L["vth"], reset_interval=SLOT, reset_offset=1 + k)
        prev.connect(d.s_in); d.a_out.connect(nr.a_in)
        prev = nr.s_out; neurons.append(nr)
    ms = [Monitor() for _ in neurons]
    for m, nr in zip(ms, neurons):
        m.probe(nr.s_out, steps)
    t0 = time.time()
    neurons[-1].run(condition=RunSteps(num_steps=steps), run_cfg=Loihi2SimCfg(select_tag="fixed_pt"))
    S = [m.get_data()[nr.name]["s_out"].astype(np.int64) for m, nr in zip(ms, neurons)]
    neurons[-1].stop()
    sec = time.time() - t0
    base = np.arange(n)[:, None] * SLOT + np.arange(T)[None, :]
    oc = S[1][base + 2].sum(1)                                  # [n, K]
    cnt = np.stack([S[k][base + k + 1].sum((1, 2)) for k in range(2)], 1)
    return oc, cnt, sec


def main():
    from src.lava_sublif.sublif import emulate_vanilla
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in (sys.argv[1:] or DS):
        v = np.load(INP / f"{ds}_vanilla_seed42.npz")
        vi = np.load(INP / f"{ds}_vanilla_int_seed42.npz")
        T = int(v["T"])
        st = v["spike_times"].astype(np.int64)
        y, ip, fp = v["labels"], vi["int_pred"], v["pred"]
        layers, info, dv = vanilla_lava_params(v, vi)
        if T + 3 > SLOT:
            raise SystemExit("slot too short")
        oc, cnt, sec = run_lava(layers, st, T, dv)
        e_oc, e_cnt, vmax = emulate_vanilla(layers, st, T, du=DU, dv=dv)
        if not (np.array_equal(oc, e_oc) and np.array_equal(cnt, e_cnt)):
            raise SystemExit(f"{ds}/vanilla: Lava differs from its NumPy emulation; the timing alignment is wrong")
        lp = oc.argmax(1)
        f1 = lambda p: 100 * f1_score(y, p, average="macro")
        ic = np.stack([vi["int_hidden_counts"], vi["int_out_counts"].sum(1)], 1)
        fc = np.stack([v["hidden_counts"], v["out_counts"]], 1)
        r = dict(dataset=ds, variant="vanilla", n=int(len(y)), T=T, lava_seconds=round(sec, 1), parameters=info,
                 du=DU, dv=dv, beta_fixed_point=(4096 - dv) / 4096, slot=SLOT,
                 agreement_int_numpy=float(np.mean(lp == ip)), agreement_float=float(np.mean(lp == fp)),
                 int_numpy_vs_float=float(np.mean(ip == fp)),
                 macro_f1=dict(lava=f1(lp), int_numpy=f1(ip), float=f1(fp)),
                 spikes_per_flow=dict(lava=cnt.mean(0).tolist(), int_numpy=ic.mean(0).tolist(), float=fc.mean(0).tolist()),
                 spike_count_exact_share=[float(np.mean(cnt[:, k] == ic[:, k])) for k in range(2)],
                 max_abs_v=int(vmax), saturated=bool(vmax >= 2 ** 23 - 1), emulation_equal=True)
        r["delta_f1_vs_int_numpy"] = r["macro_f1"]["lava"] - r["macro_f1"]["int_numpy"]
        r["p5_thresholds_met"] = bool(r["agreement_int_numpy"] >= 0.99 and abs(r["delta_f1_vs_int_numpy"]) <= 0.5)
        (OUT / f"{ds}_vanilla.json").write_text(json.dumps(r, indent=1))
        np.savez_compressed(OUT / f"{ds}_vanilla_perflow.npz", out_counts=oc, counts=cnt, pred=lp)
        rows.append(dict(dataset=ds, agree_int=100 * r["agreement_int_numpy"], agree_float=100 * r["agreement_float"],
                         f1_lava=r["macro_f1"]["lava"], f1_int=r["macro_f1"]["int_numpy"], f1_float=r["macro_f1"]["float"],
                         delta_f1=r["delta_f1_vs_int_numpy"], p5_thresholds_met=r["p5_thresholds_met"],
                         spikes_lava="/".join(f"{x:.1f}" for x in r["spikes_per_flow"]["lava"]),
                         spikes_int="/".join(f"{x:.1f}" for x in r["spikes_per_flow"]["int_numpy"]),
                         spikes_float="/".join(f"{x:.1f}" for x in r["spikes_per_flow"]["float"]),
                         max_abs_v=vmax, seconds=r["lava_seconds"]))
        print(rows[-1], flush=True)
    p = OUT / "a2_vanilla.csv"
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old[~old.dataset.isin(new.dataset)], new])
    new.sort_values("dataset").to_csv(p, index=False)


if __name__ == "__main__":
    main()
