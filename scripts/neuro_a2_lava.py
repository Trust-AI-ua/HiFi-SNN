#!/usr/bin/env python3
"""A2: Lava fixed-point execution of the seed-42 Tree-to-Spike students
(results/preregistration/2026-11-17_neuro_stream.md, rules R7 and R8).

Runs in .venv-neuro. Inputs: results/neuro/inputs/<ds>_seed42.npz (scripts/neuro_export.py).

Network: RingBuffer (the encoder's spike trains) -> Dense -> LIF, three times;
Loihi2SimCfg(select_tag="fixed_pt"). Variants (rule R8):
  zero      Lava's own LIF (LIFReset: the standard LIF plus a periodic state reset), reset to
            zero in the step of the spike
  subtract  src/lava_sublif SubLIFReset: the same fixed-point arithmetic with snnTorch's delayed
            subtractive reset
Fixed point. Lava truncates the LSB of 8-bit mixed-sign weights, so the int8 weights W are
given as 2W (exact), and threshold and bias are doubled with them: vth = round(2 / s), bias =
round(2 b / s * 64) as mantissa << exponent (13-bit mantissa, smallest exponent that fits).
du = 4095 (with Lava's offset the current decays fully each step: the current is the step's
input), dv = 410 (beta = 3686/4096 = 0.89990).
Time: one 32-step slot per flow, flows back to back; the periodic reset of layer k fires at the
slot step its input arrives (each Dense delays by one step). Readout: the output layer's
voltage summed over the T aligned steps (Lava's monitor), argmax.
Check: every Lava run must equal the NumPy emulation of the same arithmetic bit for bit.

    .venv-neuro/bin/python scripts/neuro_a2_lava.py <zero|subtract> [datasets...]
Output: results/neuro/a2_lava/<ds>_<variant>.json, a2_lava.csv
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
SLOT, DU, DV = 32, 4095, 410


def lava_params(z):
    layers, info = [], []
    for i in (1, 2, 3):
        W = z[f"W{i}"].astype(np.int64)
        s, b = float(z[f"s{i}"]), z[f"b{i}"].astype(np.float64)
        vth = int(round(2.0 / s))
        bv = np.round(2.0 * b / s * 64).astype(np.int64)
        exp = np.zeros_like(bv)
        while np.any(np.abs(np.round(bv / 2.0 ** exp)) > 4095):
            exp = np.where(np.abs(np.round(bv / 2.0 ** exp)) > 4095, exp + 1, exp)
        if exp.max() > 7 or vth > 131071 or vth < 1:
            raise SystemExit(f"layer {i}: parameters out of Lava's range (exp {exp.max()}, vth {vth})")
        mant = np.round(bv / 2.0 ** exp).astype(np.int64)
        eff = mant << exp
        layers.append(dict(W=2 * W, bias_mant=mant, bias_exp=exp, bias_eff=eff, vth=vth))
        info.append(dict(layer=i, vth=vth, theta_weight_units=vth / 2.0, exact_theta=1.0 / s,
                         bias_exp_max=int(exp.max()),
                         bias_quant_err_weight_units=float(np.abs(eff - 2.0 * b / s * 64).max() / 128.0)))
    return layers, info


def run_lava(layers, st, T, variant):
    from lava.proc.dense.process import Dense
    from lava.proc.lif.process import LIFReset
    from lava.proc.io.source import RingBuffer
    from lava.proc.monitor.process import Monitor
    from lava.magma.core.run_configs import Loihi2SimCfg
    from lava.magma.core.run_conditions import RunSteps
    from src.lava_sublif.sublif import SubLIFReset
    n, C = st.shape
    steps = n * SLOT
    data = np.zeros((C, steps), np.int32)
    data[np.arange(C)[None, :].repeat(n, 0), (np.arange(n)[:, None] * SLOT + st)] = 1
    src = RingBuffer(data=data)
    prev, neurons = src.s_out, []
    P = LIFReset if variant == "zero" else SubLIFReset
    for k, L in enumerate(layers, 1):
        d = Dense(weights=L["W"], num_message_bits=0)
        nr = P(shape=(L["W"].shape[0],), du=DU, dv=DV, bias_mant=L["bias_mant"], bias_exp=L["bias_exp"],
               vth=L["vth"], reset_interval=SLOT, reset_offset=1 + k)
        prev.connect(d.s_in); d.a_out.connect(nr.a_in)
        prev = nr.s_out; neurons.append(nr)
    mv = Monitor(); mv.probe(neurons[-1].v, steps)
    ms = [Monitor() for _ in neurons]
    for m, nr in zip(ms, neurons):
        m.probe(nr.s_out, steps)
    t0 = time.time()
    neurons[-1].run(condition=RunSteps(num_steps=steps), run_cfg=Loihi2SimCfg(select_tag="fixed_pt"))
    V = mv.get_data()[neurons[-1].name]["v"].astype(np.int64)
    S = [m.get_data()[nr.name]["s_out"].astype(np.int64) for m, nr in zip(ms, neurons)]
    neurons[-1].stop()
    sec = time.time() - t0
    base = np.arange(n)[:, None] * SLOT + np.arange(T)[None, :]
    acc = V[base + 3].sum(1)
    cnt = np.stack([S[k][base + k + 1].sum((1, 2)) for k in range(3)], 1)
    return acc, cnt, sec


def main():
    variant = sys.argv[1]
    if variant not in ("zero", "subtract"):
        raise SystemExit("variant: zero or subtract")
    from src.lava_sublif.sublif import emulate
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in (sys.argv[2:] or DS):
        z = np.load(INP / f"{ds}_seed42.npz")
        T, samp = int(z["T"]), z["sample"]
        st = z["spike_times"][samp].astype(np.int64)
        y = z["labels"][samp]
        ip, fp = z["int_pred"][samp], z["float_logits"][samp].argmax(1)
        layers, info = lava_params(z)
        acc, cnt, sec = run_lava(layers, st, T, variant)
        e_acc, e_cnt, vmax = emulate(layers, st, T, du=DU, dv=DV, reset=variant)
        if not (np.array_equal(acc, e_acc) and np.array_equal(cnt, e_cnt)):
            raise SystemExit(f"{ds}/{variant}: Lava differs from its NumPy emulation; the timing alignment is wrong")
        lp = acc.argmax(1)
        f1 = lambda p: 100 * f1_score(y, p, average="macro")
        ic = z["int_counts"]
        r = dict(dataset=ds, variant=variant, n=int(len(samp)), T=T, lava_seconds=round(sec, 1),
                 parameters=info, du=DU, dv=DV, beta_fixed_point=(4096 - DV) / 4096, slot=SLOT,
                 agreement_int_numpy=float(np.mean(lp == ip)), agreement_snntorch=float(np.mean(lp == fp)),
                 int_numpy_vs_snntorch=float(np.mean(ip == fp)),
                 macro_f1=dict(lava=f1(lp), int_numpy=f1(ip), snntorch=f1(fp)),
                 spikes_per_flow=dict(lava=cnt.mean(0).tolist(), int_numpy=ic.mean(0).tolist()),
                 spike_count_exact_share=[float(np.mean(cnt[:, k] == ic[:, k])) for k in range(3)],
                 max_abs_v=int(vmax), saturated=bool(vmax >= 2 ** 23 - 1),
                 emulation_equal=True)
        r["delta_f1_vs_int_numpy"] = r["macro_f1"]["lava"] - r["macro_f1"]["int_numpy"]
        r["R7_pass"] = bool(r["agreement_int_numpy"] >= 0.99 and abs(r["delta_f1_vs_int_numpy"]) <= 0.5)
        (OUT / f"{ds}_{variant}.json").write_text(json.dumps(r, indent=1))
        rows.append(dict(dataset=ds, variant=variant, agree_int=100 * r["agreement_int_numpy"],
                         agree_snntorch=100 * r["agreement_snntorch"], f1_lava=r["macro_f1"]["lava"],
                         f1_int=r["macro_f1"]["int_numpy"], f1_snntorch=r["macro_f1"]["snntorch"],
                         delta_f1=r["delta_f1_vs_int_numpy"], R7_pass=r["R7_pass"],
                         spikes_lava="/".join(f"{v:.1f}" for v in r["spikes_per_flow"]["lava"]),
                         spikes_int="/".join(f"{v:.1f}" for v in r["spikes_per_flow"]["int_numpy"]),
                         max_abs_v=vmax, seconds=r["lava_seconds"]))
        print(rows[-1], flush=True)
    p = OUT / "a2_lava.csv"
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old[~(old.dataset.isin(new.dataset) & (old.variant == variant))], new])
    new.sort_values(["variant", "dataset"]).to_csv(p, index=False)


if __name__ == "__main__":
    main()
