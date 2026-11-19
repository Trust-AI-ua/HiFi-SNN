#!/usr/bin/env python3
"""Amendment 6: five-seed confirmation of the two frontier dominance results
(results/preregistration/2026-11-17_neuro_stream.md). Runs in .venv-neuro through scripts/neuro_guard.py.

For one dataset, one seed and a list of models, each through the A2/A3 chain unchanged:
  t2s       the reported Tree-to-Spike model (inputs/<ds>_seed<S>.npz): full window, and early exit
            with that seed's C1 gate (results/journal_r2/exit_calibration/thresholds.json, the
            full window where the gate disables exit), exit steps from that seed's prefix outputs
  <tag>     a trained variant (inputs/<ds>_<tag>_seed<S>.npz), full window
  vanilla   the fixed-point Vanilla SNN (inputs/<ds>_vanilla_seed<S>.npz, <ds>_vanilla_int_seed<S>.npz)
Per model: Lava fixed point on the seed's stratified 2,000-flow sample, equal bit for bit to the NumPy
emulation; SANA-FE (one layer per core) with the amendment-2 gate (every flow's per-layer spikes and
readout equal Lava's, or the emulation's for exit windows; otherwise the run stops); energy and latency
per inference; the test macro-F1 of the simulated network on the whole test split (bit-exact emulation).

    .venv-neuro/bin/python scripts/neuro_confirm.py <dataset> <seed> <model> [model ...]
Output: results/neuro/confirm/<ds>_seed<S>_<row>.json (rows t2s_full, t2s_exit, <tag>, vanilla_fixed),
        per_flow_<ds>_seed<S>_<model>.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/confirm"
CH = 20000


def exit_steps(ds, seed, z, idx):
    """C1-gate exit step (window length) for test flows idx; (None, None) if the gate disables exit."""
    thr = json.loads((REPO / "results/journal_r2/exit_calibration/thresholds.json").read_text())[f"{ds}/{seed}"]
    T = int(z["T"])
    if thr["gate_c1"] is None:
        return None, None
    p = np.load(REPO / f"results/prefix_outputs/{ds}/treespike/seed_{seed}/test_prefix.npz")
    if not np.array_equal(p["labels"], z["labels"]):
        raise SystemExit(f"{ds}/{seed}: prefix outputs are not in the export's test order")
    out = []
    for i in range(0, len(idx), CH):
        zl = p["prefix_logits"][idx[i:i + CH]].astype(np.float64) / np.asarray(thr["tau"])[None, :, None]
        zl -= zl.max(2, keepdims=True)
        prob = np.exp(zl); prob /= prob.sum(2, keepdims=True)
        fired = prob.max(2) >= thr["gate_c1"]
        out.append(np.where(fired.any(1), fired.argmax(1), T - 1) + 1)
    return np.concatenate(out), thr["gate_c1"]


def sanafe_rows(layers, sts, steps, readout, check):
    """Run SANA-FE per flow; check(i, aligned, readout_vector) -> bool. Returns per-flow rows and the match rate."""
    from neuro_a3_sanafe import build, simulate
    chip, _ = build(layers, sts.shape[1], "base")
    per = []
    for i in range(len(sts)):
        e, lat, al, tot, vec, pred = simulate(chip, layers, sts[i], int(steps[i]), readout)
        per.append(dict(flow=i, steps=int(steps[i]), energy_J=e, latency_s=lat, pred=pred, matches=bool(check(i, al, vec)),
                        **{f"spikes_l{k + 1}": int(c) for k, c in enumerate(al)}))
    d = pd.DataFrame(per)
    return d, float(d.matches.mean())


def row(ds, seed, name, d, rate, f1_full, extra):
    r = dict(dataset=ds, seed=seed, row=name, n_sample=int(len(d)), sanafe_match_rate=rate, mean_steps=float(d.steps.mean()),
             energy_nJ_mean=1e9 * d.energy_J.mean(), energy_nJ_sd=1e9 * d.energy_J.std(),
             latency_us_mean=1e6 * d.latency_s.mean(), latency_us_sd=1e6 * d.latency_s.std(),
             spikes_per_flow="/".join(f"{d[c].mean():.1f}" for c in d.columns if c.startswith("spikes_l")),
             simulated_macro_f1_full_test=f1_full, **extra)
    if rate < 1.0:
        raise SystemExit(f"{ds}/{seed}/{name}: SANA-FE matches the reference on {100 * rate:.2f}% of flows; no energy reported")
    (OUT / f"{ds}_seed{seed}_{name}.json").write_text(json.dumps(r, indent=1))
    print({k: v for k, v in r.items()}, flush=True)


def run_t2s(ds, seed, model):
    import neuro_a2_lava as A2
    from neuro_a2_lava import lava_params, run_lava, DU, DV
    from src.lava_sublif.sublif import emulate
    z = np.load(INP / (f"{ds}_seed{seed}.npz" if model == "t2s" else f"{ds}_{model}_seed{seed}.npz"))
    T, samp = int(z["T"]), z["sample"]
    st_all, y_all = z["spike_times"].astype(np.int64), z["labels"]
    st = st_all[samp]
    lp, _ = lava_params(z)
    A2.SLOT = max(32, T + 8)
    acc, cnt, sec = run_lava(lp, st, T, "subtract")
    e_acc, e_cnt, _ = emulate(lp, st, T, du=DU, dv=DV, reset="subtract")
    if not (np.array_equal(acc, e_acc) and np.array_equal(cnt, e_cnt)):
        raise SystemExit(f"{ds}/{seed}/{model}: Lava differs from its NumPy emulation")
    f1 = lambda yy, p: 100 * f1_score(yy, p, average="macro")
    full = np.concatenate([emulate(lp, st_all[i:i + CH], T, du=DU, dv=DV, reset="subtract")[0].argmax(1)
                           for i in range(0, len(st_all), CH)])
    if not np.array_equal(full[samp], acc.argmax(1)):
        raise SystemExit(f"{ds}/{seed}/{model}: full-test emulation differs from Lava on the sample")
    layers = [dict(W=L["W"].astype(np.float64), bias=L["bias_eff"] / 64.0, leak=(4096 - DV) / 4096, thr=float(L["vth"]),
                   soma="loihi_lif_delayed", attrs={}) for L in lp]
    extra = dict(T=T, channels=int(st.shape[1]), lava_agreement_with_int_numpy_sample=float(np.mean(acc.argmax(1) == z["int_pred"][samp])),
                 lava_seconds=round(sec, 1))
    d, rate = sanafe_rows(layers, st, np.full(len(st), T), "potential",
                          lambda i, al, vec: np.array_equal(al, cnt[i]) and np.array_equal(np.round(vec * 64), acc[i]))
    d.to_csv(OUT / f"per_flow_{ds}_seed{seed}_{model}_full.csv", index=False)
    row(ds, seed, "t2s_full" if model == "t2s" else model, d, rate, f1(y_all, full), extra)
    if model != "t2s":
        return
    ex_s, gate = exit_steps(ds, seed, z, samp)
    if ex_s is None:                       # gate disabled: the exit row is the full window
        row(ds, seed, "t2s_exit", d, rate, f1(y_all, full), dict(extra, exit_gate=None))
        return
    ref = [emulate(lp, st[i:i + 1], int(ex_s[i]), du=DU, dv=DV, reset="subtract") for i in range(len(st))]
    d, rate = sanafe_rows(layers, st, ex_s, "potential",
                          lambda i, al, vec: np.array_equal(al, ref[i][1][0]) and np.array_equal(np.round(vec * 64), ref[i][0][0]))
    d.to_csv(OUT / f"per_flow_{ds}_seed{seed}_{model}_exit.csv", index=False)
    ex_all, _ = exit_steps(ds, seed, z, np.arange(len(y_all)))
    epred = np.empty(len(y_all), np.int64)
    for k in np.unique(ex_all):
        idx = np.flatnonzero(ex_all == k)
        epred[idx] = np.concatenate([emulate(lp, st_all[idx[i:i + CH]], int(k), du=DU, dv=DV, reset="subtract")[0].argmax(1)
                                     for i in range(0, len(idx), CH)])
    if not np.array_equal(epred[samp], d.pred.values):
        raise SystemExit(f"{ds}/{seed}: full-test exit emulation differs from SANA-FE's exit run on the sample")
    row(ds, seed, "t2s_exit", d, rate, f1(y_all, epred), dict(extra, exit_gate=gate, mean_steps_full_test=float(ex_all.mean())))


def run_vanilla(ds, seed):
    from neuro_a2_vanilla import vanilla_lava_params, run_lava, DU
    from src.lava_sublif.sublif import emulate_vanilla
    v = np.load(INP / f"{ds}_vanilla_seed{seed}.npz")
    vi = np.load(INP / f"{ds}_vanilla_int_seed{seed}.npz")
    T = int(v["T"])
    st = v["spike_times"].astype(np.int64)
    vl, _, dv = vanilla_lava_params(v, vi)
    oc, cnt, sec = run_lava(vl, st, T, dv)
    e_oc, e_cnt, _ = emulate_vanilla(vl, st, T, du=DU, dv=dv)
    if not (np.array_equal(oc, e_oc) and np.array_equal(cnt, e_cnt)):
        raise SystemExit(f"{ds}/{seed}/vanilla: Lava differs from its NumPy emulation")
    vst, vy = vi["spike_times_full"].astype(np.int64), vi["labels_full"]
    full = np.concatenate([emulate_vanilla(vl, vst[i:i + CH], T, du=DU, dv=dv)[0].argmax(1) for i in range(0, len(vst), CH)])
    if not np.array_equal(full[vi["sample"]], oc.argmax(1)):
        raise SystemExit(f"{ds}/{seed}: full-test Vanilla emulation differs from Lava on the sample")
    hid = dict(reset_mode="none", reverse_threshold=0.0, reverse_reset=0.0, reverse_reset_mode="saturate")
    layers = [dict(W=vl[0]["W"].astype(np.float64), bias=vl[0]["bias_eff"] / 64.0, leak=(4096 - dv) / 4096,
                   thr=float(vl[0]["vth"]), soma="loihi_lif", attrs=hid),
              dict(W=vl[1]["W"].astype(np.float64), bias=vl[1]["bias_eff"] / 64.0, leak=(4096 - dv) / 4096,
                   thr=float(vl[1]["vth"]), soma="loihi_lif_delayed")]
    d, rate = sanafe_rows(layers, st, np.full(len(st), T), "spikes",
                          lambda i, al, vec: np.array_equal(al, cnt[i]) and np.array_equal(vec, oc[i]))
    d.to_csv(OUT / f"per_flow_{ds}_seed{seed}_vanilla.csv", index=False)
    row(ds, seed, "vanilla_fixed", d, rate, 100 * f1_score(vy, full, average="macro"),
        dict(T=T, lava_agreement_with_int_numpy_sample=float(np.mean(oc.argmax(1) == vi["int_pred"])), lava_seconds=round(sec, 1)))


def main():
    ds, seed, models = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    OUT.mkdir(parents=True, exist_ok=True)
    for m in models:
        t0 = time.time()
        run_vanilla(ds, seed) if m == "vanilla" else run_t2s(ds, seed, m)
        print(f"[done] {ds} seed {seed} {m} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
