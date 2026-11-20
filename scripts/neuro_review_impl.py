#!/usr/bin/env python3
"""Amendment 8, part 5 (implementation records; results/preregistration/2026-11-17_neuro_stream.md). Runs in .venv.

  deployment_params.csv   per dataset, seed, model and layer: weight scale, integer threshold and bias range,
                          leak, Lava vth / bias mantissa and exponent / dv, accumulator peak, from the exports
  trace_<ds>_seed<S>.csv  one test flow through the Lava fixed-point emulation of the production student and of
                          the Vanilla SNN, every step and layer: input spikes, the chosen neuron's current u,
                          membrane v (after leak, input, bias and reset), threshold, spike, reset applied; the
                          trace's readout is checked against the emulation's

    .venv/bin/python scripts/neuro_review_impl.py
Output: results/neuro/review/p4_implementation/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/review/p4_implementation"
DATASETS = ("edgeiiot", "ciciomt2024", "toniot", "ciciot2023")
SEEDS = (42, 43, 44, 45, 46)


def decay(x, d):
    p = int(x) * (4096 - d)
    return int(np.sign(p)) * (abs(p) >> 12)


def trace(layers, st, T, dv, kind, watch):
    """Step-by-step Lava fixed point for one flow (scalar Python ints; same arithmetic as the batched emulators).
    kind: t2s (all layers subtractive, delayed reset) or vanilla (hidden: ReLU clip at 0, no reset)."""
    M = 2 ** 23
    u = [np.zeros(L["W"].shape[0], np.int64) for L in layers]
    v = [np.zeros_like(x) for x in u]; r = [np.zeros_like(x) for x in u]
    rows, acc, oc = [], np.zeros(layers[-1]["W"].shape[0], np.int64), np.zeros(layers[-1]["W"].shape[0], np.int64)
    for t in range(T):
        s = (st == t).astype(np.int64)
        for i, L in enumerate(layers):
            n_in = int(s.sum())
            a = L["W"].astype(np.int64) @ s
            uu = np.array([decay(x, 4096) for x in u[i]], np.int64) + (a << 6)      # du = 4095: current lasts one step
            uu = np.where(uu > M, uu - 2 * M, uu); uu = np.where(uu <= -M, uu + 2 * M, uu)
            u[i] = uu
            th = int(L["vth"]) << 6
            leak = np.array([decay(x, dv) for x in v[i]], np.int64)
            vv = leak + u[i] + L["bias_eff"]
            relu = kind == "vanilla" and i == 0
            reset = np.zeros_like(vv) if relu else r[i] * th
            vv = np.clip(vv, 0, M - 1) if relu else np.clip(vv - reset, -M + 1, M - 1)
            s = (vv > th).astype(np.int64)
            j = watch[i]
            rows.append(dict(step=t, layer=i + 1, input_spikes=n_in, neuron=j, v_prev=int(v[i][j]), leak_term=int(leak[j]),
                             u=int(u[i][j]), bias_eff=int(L["bias_eff"][j]), reset_subtracted=int(reset[j]), v=int(vv[j]),
                             threshold=th, spike=int(s[j]), layer_spikes=int(s.sum())))
            v[i] = vv
            if not relu:
                r[i] = s
            if i == len(layers) - 1:
                oc += s
        acc += v[-1]
    return pd.DataFrame(rows), acc, oc


def main():
    from neuro_a2_lava import lava_params, DV
    from neuro_a2_vanilla import vanilla_lava_params
    from neuro_stagewise import lava_t2s, lava_vanilla
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            z = np.load(INP / f"{ds}_seed{seed}.npz")
            lp, info = lava_params(z)
            for k, (L, I) in enumerate(zip(lp, info), 1):
                rows.append(dict(dataset=ds, seed=seed, model="tree_to_spike", layer=k, n_in=L["W"].shape[1], n_out=L["W"].shape[0],
                                 T=int(z["T"]), weight_scale=float(z[f"s{k}"]), int8_weight_max_abs=int(np.abs(z[f"W{k}"]).max()),
                                 int_threshold_q8=int(z[f"theta{k}"]), int_bias_q8_max_abs=int(np.abs(z[f"bq{k}"]).max()),
                                 int_leak="(v * 230) >> 8 (beta 0.8984; float 0.9)", lava_weights="2 * int8",
                                 lava_vth=L["vth"], lava_threshold_effective=L["vth"] << 6, lava_bias_exp_max=int(L["bias_exp"].max()),
                                 lava_bias_mant_max_abs=int(np.abs(L["bias_mant"]).max()), lava_dv=DV, lava_du=4095,
                                 reset="delayed subtractive (r = [v_prev > theta])", readout="argmax of summed output membrane"))
            v = np.load(INP / f"{ds}_vanilla_seed{seed}.npz"); vi = np.load(INP / f"{ds}_vanilla_int_seed{seed}.npz")
            vl, vinfo, dv = vanilla_lava_params(v, vi)
            for k, (L, I, nm) in enumerate(zip(vl, vinfo, ("hidden", "output")), 1):
                s = float(vi["s_h"] if k == 1 else vi["s_o"])
                rows.append(dict(dataset=ds, seed=seed, model="vanilla_snn", layer=k, n_in=L["W"].shape[1], n_out=L["W"].shape[0],
                                 T=int(v["T"]), weight_scale=s, int8_weight_max_abs=int(np.abs(vi["P_Wh"] if k == 1 else vi["P_Wo"]).max()),
                                 int_threshold_q8=int(vi["P_th_h"] if k == 1 else vi["P_th_o"]),
                                 int_bias_q8_max_abs=int(np.abs(vi["P_bh"] if k == 1 else vi["P_bo"]).max()),
                                 int_leak=f"(v * {int(vi['P_mul'])}) >> 8 (float beta {float(v['beta_out']):.4f})", lava_weights="2 * int8",
                                 lava_vth=L["vth"], lava_threshold_effective=L["vth"] << 6, lava_bias_exp_max=int(L["bias_exp"].max()),
                                 lava_bias_mant_max_abs=int(np.abs(L["bias_mant"]).max()), lava_dv=dv, lava_du=4095,
                                 reset="none, ReLU clip at 0 (LeakyParallel)" if k == 1 else "delayed subtractive",
                                 readout="" if k == 1 else "argmax of output spike count"))
    pd.DataFrame(rows).to_csv(OUT / "deployment_params.csv", index=False)
    ds, seed = "edgeiiot", 42
    z = np.load(INP / f"{ds}_seed{seed}.npz"); lp, _ = lava_params(z)
    v = np.load(INP / f"{ds}_vanilla_seed{seed}.npz"); vi = np.load(INP / f"{ds}_vanilla_int_seed{seed}.npz")
    vl, _, dv = vanilla_lava_params(v, vi)
    i = int(z["sample"][0])
    st_t = z["spike_times"][i].astype(np.int64); st_v = vi["spike_times_full"][i].astype(np.int64)
    acc_ref, cnt_ref = lava_t2s(lp, st_t[None], int(z["T"]))
    oc_ref, vcnt_ref, _ = lava_vanilla(vl, st_v[None], int(v["T"]), dv=dv)
    # watch, per layer, the neuron that spikes most on this flow (a spiking neuron, so a reset is visible)
    tw, tv = [], []
    for L_, st_, T_, dv_, kind, n_layers in ((lp, st_t, int(z["T"]), DV, "t2s", 3), (vl, st_v, int(v["T"]), dv, "vanilla", 2)):
        watch = []
        for li in range(n_layers):
            best, bj = -1, 0
            for j in range(L_[li]["W"].shape[0]):
                d, _, _ = trace(L_, st_, T_, dv_, kind, [j if k == li else 0 for k in range(n_layers)])
                c = d[d.layer == li + 1].spike.sum()
                if c > best:
                    best, bj = c, j
                if L_[li]["W"].shape[0] > 32 and j >= 31:
                    break
            watch.append(bj)
        d, acc, oc = trace(L_, st_, T_, dv_, kind, watch)
        if kind == "t2s" and not np.array_equal(acc, acc_ref[0]):
            raise SystemExit("trace differs from the student's Lava emulation")
        if kind == "vanilla" and not np.array_equal(oc, oc_ref[0]):
            raise SystemExit("trace differs from the Vanilla Lava emulation")
        d.insert(0, "model", "tree_to_spike" if kind == "t2s" else "vanilla_snn")
        (tw if kind == "t2s" else tv).append(d)
    out = pd.concat(tw + tv)
    out.insert(0, "flow_id", i); out.insert(0, "seed", seed); out.insert(0, "dataset", ds)
    out.to_csv(OUT / f"trace_{ds}_seed{seed}.csv", index=False)
    print(f"trace flow {i} (class {int(z['labels'][i])}): student pred {int(acc_ref[0].argmax())}, Vanilla pred {int(oc_ref[0].argmax())}")


if __name__ == "__main__":
    main()
