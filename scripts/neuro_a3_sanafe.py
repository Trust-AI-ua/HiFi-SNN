#!/usr/bin/env python3
"""A3: SANA-FE energy and latency per inference on Loihi (results/preregistration/2026-11-17_neuro_stream.md,
amendment 2). Run through scripts/neuro_guard.py in .venv-neuro, after
scripts/neuro_a3_install_check.py has passed, scripts/build_sanafe_plugin.sh and
scripts/make_sanafe_arch.py have built the plugin and its architecture file, and A2 has written
the per-flow Lava outputs of the P5-passing variant.

Architecture: SANA-FE's bundled loihi.yaml plus one soma unit, loihi_lif_delayed (the plugin
src/sanafe_plugin/loihi_lif_delayed.cpp: the built-in Loihi LIF arithmetic with snnTorch's
delayed subtractive reset; energy and latency costs copied from loihi_lif).

Tree-to-Spike: the Lava network of A2 in Lava's integer units (weights 2W, threshold vth, bias =
effective bias / 64, leak 3686/4096), so SANA-FE's 1/64-grid truncation is Lava's. Soma
loihi_lif_delayed for the subtract variant, the built-in loihi_lif with hard reset for zero.
Layer k's bias is switched on at step k, when its first input arrives (SANA-FE delays one step
per layer), matching Lava's per-layer reset offsets. Consistency gate (amendment 2): per flow,
SANA-FE's spikes per layer in the aligned windows and its readout must equal Lava's; energy is
reported with the match rate and only if it is 100% (otherwise the run stops).
Vanilla SNN: hidden layer = built-in loihi_lif, no reset, clipped at 0 (reverse threshold 0,
saturate: the ReLU of LeakyParallel), per-neuron leak; output = loihi_lif_delayed; weights,
biases and thresholds in units of each layer's int8 scale (max |W| / 127, unrounded). No Lava
reference exists for it; its predictions and hidden spike counts are compared with the trained
float model and reported.

Each flow runs from a reset chip for its input steps plus one step per layer (fill and drain
included in energy and latency). Mappings: base = one layer per core; split = each hidden
layer across two cores (sensitivity). Variants: t2s_full, t2s_exit (C1-gate exit step of step 4,
full window where the gate disables exit), vanilla_full.

    .venv-neuro/bin/python scripts/neuro_a3_sanafe.py <zero|subtract> [datasets...]
Output: results/neuro/a3_sanafe/{per_flow_<ds>.csv, energy_sanafe.csv, mapping.json}
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sanafe

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/a3_sanafe"
ARCH = REPO / "src/sanafe_plugin/build/loihi_delayed.yaml"
DS = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
CAL_ERR_ENERGY = 0.10   # SANA-FE's reported energy error against first-generation Loihi (about 10%)


def build(layers, n_in, mapping):
    """layers: dicts W [out, in], bias [out], leak (scalar or [out]), thr, soma, extra attrs."""
    arch = sanafe.load_arch(str(ARCH))
    net = sanafe.Network()
    groups = [net.create_neuron_group("in", n_in, model_attributes={"spikes": [False]})]
    for i, L in enumerate(layers):
        n_out = L["W"].shape[0]
        g = net.create_neuron_group(f"l{i + 1}", n_out, model_attributes=dict(
            threshold=float(L["thr"]), force_update=True, bias=0.0, **L.get("attrs", {})))
        leak = np.broadcast_to(np.asarray(L["leak"], np.float64), (n_out,))
        for j, nr in enumerate(g):
            nr.set_attributes(soma_hw_name=L["soma"], log_spikes=True, log_potential=(i == len(layers) - 1),
                              model_attributes={"leak_decay": float(leak[j])})
        groups[-1].connect_neurons_dense(g, {"w": np.asarray(L["W"], np.float64).T.flatten()})
        groups.append(g)
    cores = arch.cores()
    for i, nr in enumerate(groups[0]):
        nr.set_attributes(soma_hw_name=f"loihi_inputs[{i}]")
        nr.map_to_core(cores[0])
    plan, c = {"in": [0]}, 1
    for k, g in enumerate(groups[1:], 1):
        parts = 2 if (mapping == "split" and k < len(groups) - 1) else 1
        per = math.ceil(len(g) / parts)
        plan[f"l{k}"] = list(range(c, c + parts))
        for j, nr in enumerate(g):
            nr.map_to_core(cores[c + j // per])
        c += parts
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    return chip, plan


def simulate(chip, layers, st, steps, readout):
    """Returns energy, latency, aligned spikes per layer, total spikes per layer, readout vector, prediction."""
    nl = len(layers)
    total = steps + nl
    chip.reset()
    for k in range(1, nl + 1):
        for nr in chip.mapped_neuron_groups[f"l{k}"]:
            nr.set_attributes(model_attributes={"bias": 0.0})
    for i, nr in enumerate(chip.mapped_neuron_groups["in"]):
        nr.set_attributes(model_attributes={"spikes": [bool(t == st[i]) for t in range(total)]})
    trace, pots, energy, lat = [], [], 0.0, 0.0
    for t in range(total):
        if 1 <= t <= nl:                                   # layer t's first input arrives now
            for j, nr in enumerate(chip.mapped_neuron_groups[f"l{t}"]):
                nr.set_attributes(model_attributes={"bias": float(layers[t - 1]["bias"][j])})
        r = chip.sim(1, spike_trace=True, potential_trace=True)
        trace += r["spike_trace"]; pots += r["potential_trace"]
        energy += float(r["energy"]["total"]); lat += float(r["sim_time"])
    per_step = np.zeros((total, nl), np.int64)
    out_spk = np.zeros(layers[-1]["W"].shape[0])
    for t, step in enumerate(trace):
        for s in step:
            g, j = str(s).split(".")
            if g != "in":
                per_step[t, int(g[1:]) - 1] += 1
                if int(g[1:]) == nl:
                    out_spk[int(j)] += 1 if (nl <= t < nl + steps) else 0
    aligned = np.array([per_step[k + 1:k + 1 + steps, k].sum() for k in range(nl)])
    pot = np.asarray(pots, np.float64)[nl:nl + steps].sum(0)
    vec = pot if readout == "potential" else out_spk
    return energy, lat, aligned, per_step.sum(0), vec, int(vec.argmax())


def exit_steps(ds, z):
    thr = json.loads((REPO / "results/journal_r2/exit_calibration/thresholds.json").read_text())[f"{ds}/42"]
    T = int(z["T"])
    if thr["gate_c1"] is None:
        return np.full(len(z["sample"]), T), None
    p = np.load(REPO / f"results/prefix_outputs/{ds}/treespike/seed_42/test_prefix.npz")
    if not np.array_equal(p["labels"], z["labels"]):
        raise SystemExit(f"{ds}: prefix outputs are not in the export's test order")
    zl = p["prefix_logits"][z["sample"]].astype(np.float64) / np.asarray(thr["tau"])[None, :, None]
    zl -= zl.max(2, keepdims=True)
    prob = np.exp(zl); prob /= prob.sum(2, keepdims=True)
    fired = prob.max(2) >= thr["gate_c1"]
    first = np.where(fired.any(1), fired.argmax(1), T - 1)
    return first + 1, thr["gate_c1"]


def main():
    variant = sys.argv[1]
    if variant not in ("zero", "subtract"):
        raise SystemExit("variant: zero or subtract (the P5-passing A2 variant)")
    if not json.loads((OUT / "install_check.json").read_text())["passed"]:
        raise SystemExit("SANA-FE installation check has not passed")
    from neuro_a2_lava import lava_params, DU, DV
    from src.lava_sublif.sublif import emulate
    rows, maps = [], {}
    for ds in (sys.argv[2:] or DS):
        z = np.load(INP / f"{ds}_seed42.npz")
        v = np.load(INP / f"{ds}_vanilla_seed42.npz")
        lava = np.load(REPO / f"results/neuro/a2_lava/{ds}_{variant}_perflow.npz")
        T, samp = int(z["T"]), z["sample"]
        if not np.array_equal(lava["sample"], samp):
            raise SystemExit(f"{ds}: A2 per-flow outputs are for a different sample")
        st = z["spike_times"][samp].astype(np.int64)
        lp, _ = lava_params(z)
        soma = "loihi_lif_delayed" if variant == "subtract" else "loihi_lif"
        attrs = {} if variant == "subtract" else {"reset_mode": "hard"}
        t2s = [dict(W=L["W"].astype(np.float64), bias=L["bias_eff"] / 64.0, leak=(4096 - DV) / 4096, thr=float(L["vth"]),
                    soma=soma, attrs=attrs) for L in lp]
        sc = [float(np.abs(v["W_ih"]).max()) / 127, float(np.abs(v["W_out"]).max()) / 127]
        van = [dict(W=v["W_ih"] / sc[0], bias=v["b_h"] / sc[0], leak=v["beta_h"], thr=float(v["thr_h"]) / sc[0],
                    soma="loihi_lif", attrs=dict(reset_mode="none", reverse_threshold=0.0, reverse_reset=0.0,
                                                 reverse_reset_mode="saturate")),
               dict(W=v["W_out"] / sc[1], bias=v["b_out"] / sc[1], leak=float(v["beta_out"]), thr=float(v["thr_out"]) / sc[1],
                    soma="loihi_lif_delayed")]
        ex, theta = exit_steps(ds, z)
        e_acc_exit = None
        if theta is not None:   # Lava reference for the truncated windows: the emulation, checked equal to Lava at T
            e_acc_exit = [emulate(lp, st[i:i + 1], int(ex[i]), du=DU, dv=DV, reset=variant) for i in range(len(st))]
        enc_ops = int(sum(math.ceil(math.log2(e + 1)) for e in z["n_edges"])) if "n_edges" in z.files else None
        runs = [("t2s_full", "base", t2s, st, np.full(len(samp), T), "potential"),
                ("t2s_full", "split", t2s, st, np.full(len(samp), T), "potential"),
                ("t2s_exit", "base", t2s, st, ex, "potential"),
                ("vanilla_full", "base", van, v["spike_times"].astype(np.int64), np.full(len(v["labels"]), int(v["T"])), "spikes")]
        per = []
        for name, mapping, layers, sts, steps, readout in runs:
            t0 = time.time()
            chip, plan = build(layers, sts.shape[1], mapping)
            maps[f"{ds}/{name}/{mapping}"] = plan
            match = []
            for i in range(len(sts)):
                e, lat, al, tot, vec, pred = simulate(chip, layers, sts[i], int(steps[i]), readout)
                if name == "t2s_full":
                    m = bool(np.array_equal(al, lava["counts"][i]) and np.array_equal(np.round(vec * 64), lava["acc"][i]))
                    ref = int(lava["pred"][i])
                elif name == "t2s_exit":
                    if e_acc_exit is None:
                        m = bool(np.array_equal(al, lava["counts"][i])); ref = int(lava["pred"][i])
                    else:
                        ea, ec, _ = e_acc_exit[i]
                        m = bool(np.array_equal(al, ec[0]) and np.array_equal(np.round(vec * 64), ea[0]))
                        ref = int(ea[0].argmax())
                else:
                    m = bool(al[0] == v["hidden_counts"][i]); ref = int(v["pred"][i])
                match.append(m)
                per.append(dict(dataset=ds, variant=name, mapping=mapping, flow=i, steps=int(steps[i]), energy_J=e,
                                latency_s=lat, pred=pred, ref_pred=ref, matches_reference=m,
                                **{f"spikes_l{k + 1}": int(c) for k, c in enumerate(al)},
                                **{f"spikes_total_l{k + 1}": int(c) for k, c in enumerate(tot)}))
            d = pd.DataFrame([p for p in per if p["variant"] == name and p["mapping"] == mapping])
            rate = float(np.mean(match))
            if name.startswith("t2s") and rate < 1.0:
                pd.DataFrame(per).to_csv(OUT / f"per_flow_{ds}.csv", index=False)
                raise SystemExit(f"{ds}/{name}/{mapping}: SANA-FE matches Lava on {100 * rate:.2f}% of flows; "
                                 "energy not reported (amendment 2)")
            rows.append(dict(dataset=ds, variant=name, mapping=mapping, lava_variant=variant if name.startswith("t2s") else None,
                             n=len(d), mean_steps=d.steps.mean(), reference_match_rate=rate,
                             energy_nJ_mean=1e9 * d.energy_J.mean(), energy_nJ_sd=1e9 * d.energy_J.std(),
                             energy_nJ_cal_low=1e9 * d.energy_J.mean() * (1 - CAL_ERR_ENERGY),
                             energy_nJ_cal_high=1e9 * d.energy_J.mean() * (1 + CAL_ERR_ENERGY),
                             latency_us_mean=1e6 * d.latency_s.mean(), latency_us_sd=1e6 * d.latency_s.std(),
                             agreement_with_reference=float((d.pred == d.ref_pred).mean()),
                             spikes_per_flow="/".join(f"{d[c].mean():.1f}" for c in d.columns if c.startswith("spikes_l")),
                             exit_theta=theta if name == "t2s_exit" else None,
                             host_encoder_comparisons=enc_ops if name.startswith("t2s") else None,
                             cores="; ".join(f"{k}:{c}" for k, c in plan.items()), seconds=round(time.time() - t0, 1)))
            print(rows[-1], flush=True)
        pd.DataFrame(per).to_csv(OUT / f"per_flow_{ds}.csv", index=False)
    p = OUT / "energy_sanafe.csv"
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old[~old.dataset.isin(new.dataset)], new])
    new.to_csv(p, index=False)
    mp = OUT / "mapping.json"
    allm = json.loads(mp.read_text()) if mp.exists() else {}
    allm.update(maps)
    mp.write_text(json.dumps(allm, indent=1))


if __name__ == "__main__":
    main()
