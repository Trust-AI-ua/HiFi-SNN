#!/usr/bin/env python3
"""A3: SANA-FE energy and latency per inference on Loihi (results/preregistration/2026-11-17_neuro_stream.md).

Runs in .venv-neuro, after scripts/neuro_a3_install_check.py has passed. Inputs:
results/neuro/inputs/<ds>_seed42.npz and <ds>_vanilla_seed42.npz, the step-4 exit thresholds
(results/journal_r2/exit_calibration/thresholds.json) and the cached prefix outputs.

Networks on SANA-FE's bundled Loihi architecture (loihi.yaml), float weights (the checkpoints'):
  Tree-to-Spike  input C -> 64 -> 64 -> K, LIF beta 0.9, threshold 1, subtractive ("soft") reset
  Vanilla SNN    input F -> 128 (leaky, no reset, clipped at 0: the ReLU of LeakyParallel) -> K
                 (LIF, subtractive reset)
Input neurons are the architecture's host-driven input units, replaying the recorded spike
trains of the 2,000-flow sample. Biased layers update every step (force_update), as the
trained models do. Mapping "base": one layer per core (input on core 0, then one core per
layer). Mapping "split": each hidden layer split across two cores (sensitivity).
Variants: t2s_full (T steps), t2s_exit (each flow stops at its C1-gate exit step of step 4;
the full window where the gate disables exit), vanilla_full. Each flow is simulated from a
reset chip for its input steps plus one step per layer, so the last input reaches the output.
Reported per flow: energy (J), simulated latency (s), spikes per layer, and the prediction
from the output (T2S: summed output potential; Vanilla: output spike count) for agreement with
the trained model. Host encoder (Tree-to-Spike): binary-search comparisons per flow,
sum over kept features of ceil(log2(edges + 1)).

    .venv-neuro/bin/python scripts/neuro_a3_sanafe.py [datasets...]
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
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/a3_sanafe"
DS = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
CAL_ERR_ENERGY = 0.10   # SANA-FE's reported energy error against first-generation Loihi (about 10%)


def build(layers, n_in, mapping):
    """layers: list of dicts(W [in,out] float, bias, beta (scalar or array), thr, reset, relu)."""
    arch = sanafe.load_loihi()
    net = sanafe.Network()
    groups = [net.create_neuron_group("in", n_in, model_attributes={"spikes": [False]})]
    for i, L in enumerate(layers):
        n_out = L["W"].shape[1]
        attrs = dict(threshold=float(L["thr"]), reset_mode=L["reset"], force_update=True)
        if L.get("relu"):
            attrs.update(reverse_threshold=0.0, reverse_reset=0.0, reverse_reset_mode="saturate")
        g = net.create_neuron_group(f"l{i + 1}", n_out, model_attributes=attrs)
        beta = np.broadcast_to(np.asarray(L["beta"], np.float64), (n_out,))
        for j, n in enumerate(g):
            n.set_attributes(model_attributes={"bias": float(L["bias"][j]), "leak_decay": float(beta[j])})
        groups[-1].connect_neurons_dense(g, {"w": np.asarray(L["W"], np.float64).flatten()})
        groups.append(g)
    cores = arch.cores()
    plan = {}
    for i, n in enumerate(groups[0]):
        n.set_attributes(soma_hw_name=f"loihi_inputs[{i}]")
        n.map_to_core(cores[0])
    plan["in"] = [0]
    c = 1
    for k, g in enumerate(groups[1:], 1):
        hidden = k < len(groups) - 1
        parts = 2 if (mapping == "split" and hidden) else 1
        n = len(g)
        per = math.ceil(n / parts)
        plan[f"l{k}"] = list(range(c, c + parts))
        for j, nr in enumerate(g):
            nr.set_attributes(log_spikes=True, log_potential=(k == len(groups) - 1))
            nr.map_to_core(cores[c + j // per])
        c += parts
    chip = sanafe.SpikingChip(arch)
    chip.load(net)
    return chip, plan


def simulate(chip, st, steps, n_layers, readout):
    """st: [C] spike times (-1 = no spike). Returns energy, latency, spikes per layer, prediction."""
    chip.reset()
    total = steps + n_layers
    for i, n in enumerate(chip.mapped_neuron_groups["in"]):
        n.set_attributes(model_attributes={"spikes": [bool(t == st[i]) for t in range(total)]})
    r = chip.sim(total, spike_trace=True, potential_trace=True)
    cnt = np.zeros(n_layers + 1, np.int64)
    out_spk = None
    for step in r["spike_trace"]:
        for s in step:
            g = str(s).split(".")[0]
            cnt[0 if g == "in" else int(g[1:])] += 1
    pot = np.asarray(r["potential_trace"], np.float64)            # [steps, K]
    if readout == "potential":
        pred = int(pot.sum(0).argmax())
    else:                                                         # output spike count
        k = pot.shape[1]
        out_spk = np.zeros(k)
        for step in r["spike_trace"]:
            for s in step:
                g, j = str(s).split(".")
                if g == f"l{n_layers}":
                    out_spk[int(j)] += 1
        pred = int(out_spk.argmax())
    return float(r["energy"]["total"]), float(r["sim_time"]), cnt, pred


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
    chk = json.loads((OUT / "install_check.json").read_text())
    if not chk["passed"]:
        raise SystemExit("SANA-FE installation check has not passed")
    rows, maps = [], {}
    for ds in (sys.argv[1:] or DS):
        z = np.load(INP / f"{ds}_seed42.npz")
        v = np.load(INP / f"{ds}_vanilla_seed42.npz")
        T, samp = int(z["T"]), z["sample"]
        st = z["spike_times"][samp].astype(np.int64)
        ref_t = z["float_logits"][samp].argmax(1)
        t2s = [dict(W=(z[f"W{i}"].astype(np.float64) * float(z[f"s{i}"])).T, bias=z[f"b{i}"], beta=float(z["beta"]),
                    thr=1.0, reset="soft") for i in (1, 2, 3)]
        van = [dict(W=v["W_ih"].T, bias=v["b_h"], beta=v["beta_h"], thr=float(v["thr_h"]), reset="none", relu=True),
               dict(W=v["W_out"].T, bias=v["b_out"], beta=float(v["beta_out"]), thr=float(v["thr_out"]), reset="soft")]
        ex, theta = exit_steps(ds, z)
        enc_ops = int(sum(math.ceil(math.log2(e + 1)) for e in z["n_edges"])) if "n_edges" in z.files else None
        runs = [("t2s_full", "base", t2s, st, np.full(len(samp), T), "potential", ref_t),
                ("t2s_full", "split", t2s, st, np.full(len(samp), T), "potential", ref_t),
                ("t2s_exit", "base", t2s, st, ex, "potential", ref_t),
                ("vanilla_full", "base", van, v["spike_times"].astype(np.int64), np.full(len(v["labels"]), int(v["T"])),
                 "spikes", v["pred"])]
        per = []
        for name, mapping, layers, sts, steps, readout, ref in runs:
            t0 = time.time()
            chip, plan = build(layers, sts.shape[1], mapping)
            maps[f"{ds}/{name}/{mapping}"] = plan
            for i in range(len(sts)):
                e, lat, cnt, pred = simulate(chip, sts[i], int(steps[i]), len(layers), readout)
                per.append(dict(dataset=ds, variant=name, mapping=mapping, flow=i, steps=int(steps[i]), energy_J=e,
                                latency_s=lat, pred=pred, ref_pred=int(ref[i]),
                                **{f"spikes_l{k}": int(c) for k, c in enumerate(cnt)}))
            d = pd.DataFrame([p for p in per if p["variant"] == name and p["mapping"] == mapping])
            rows.append(dict(dataset=ds, variant=name, mapping=mapping, n=len(d), mean_steps=d.steps.mean(),
                             energy_nJ_mean=1e9 * d.energy_J.mean(), energy_nJ_sd=1e9 * d.energy_J.std(),
                             energy_nJ_cal_low=1e9 * d.energy_J.mean() * (1 - CAL_ERR_ENERGY),
                             energy_nJ_cal_high=1e9 * d.energy_J.mean() * (1 + CAL_ERR_ENERGY),
                             latency_us_mean=1e6 * d.latency_s.mean(), latency_us_sd=1e6 * d.latency_s.std(),
                             agreement_with_model=float((d.pred == d.ref_pred).mean()),
                             spikes_per_flow="/".join(f"{d[c].mean():.1f}" for c in d.columns if c.startswith("spikes_l")),
                             exit_theta=theta if name == "t2s_exit" else None,
                             host_encoder_comparisons=enc_ops if name.startswith("t2s") else None,
                             cores="; ".join(f"{k}:{v}" for k, v in plan.items()), seconds=round(time.time() - t0, 1)))
            print(rows[-1], flush=True)
        pd.DataFrame(per).to_csv(OUT / f"per_flow_{ds}.csv", index=False)
    p = OUT / "energy_sanafe.csv"
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old[~old.dataset.isin(new.dataset)], new])
    new.to_csv(p, index=False)
    (OUT / "mapping.json").write_text(json.dumps(maps, indent=1))


if __name__ == "__main__":
    main()
