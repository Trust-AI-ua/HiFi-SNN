#!/usr/bin/env python3
"""Energy-accuracy frontier (exploratory; results/preregistration/2026-11-17_neuro_stream.md, amendment 5).

Runs in .venv-neuro through scripts/neuro_guard.py. For one dataset and a list of Tree-to-Spike
variants exported by scripts/neuro_export.py --run/--tag, each through the A2/A3 chain unchanged:
  1. Lava fixed point (SubLIFReset, the P5 variant) on the variant's stratified 2,000-flow sample,
     checked bit for bit against the NumPy emulation (scripts/neuro_a2_lava.py: lava_params, run_lava);
  2. SANA-FE, full window, base mapping (one layer per core), with the amendment-2 gate: every
     flow's per-layer spikes and readout must equal Lava's, or no energy is reported;
  3. the full-test macro-F1 of the simulated network (the bit-exact emulation on the whole test split).
Also reports agreement with integer NumPy on the sample, the float and integer full-test macro-F1,
latency, spikes per layer and the host encoder's comparisons.

    .venv-neuro/bin/python scripts/neuro_frontier.py <dataset> <tag> [tag ...]
Output: results/neuro/frontier/<ds>_<tag>.json, per_flow_<ds>_<tag>.csv
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/frontier"
CH = 20000


def main():
    from neuro_a2_lava import lava_params, run_lava, DU, DV
    from neuro_a3_sanafe import build, simulate
    from src.lava_sublif.sublif import emulate
    ds, tags = sys.argv[1], sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        t0 = time.time()
        z = np.load(INP / f"{ds}_{tag}_seed42.npz")
        T, samp = int(z["T"]), z["sample"]
        st_all, y_all = z["spike_times"].astype(np.int64), z["labels"]
        st, y = st_all[samp], y_all[samp]
        lp, info = lava_params(z)
        acc, cnt, sec = run_lava(lp, st, T, "subtract")
        e_acc, e_cnt, vmax = emulate(lp, st, T, du=DU, dv=DV, reset="subtract")
        if not (np.array_equal(acc, e_acc) and np.array_equal(cnt, e_cnt)):
            raise SystemExit(f"{ds}/{tag}: Lava differs from its NumPy emulation")
        lpred = acc.argmax(1)
        layers = [dict(W=L["W"].astype(np.float64), bias=L["bias_eff"] / 64.0, leak=(4096 - DV) / 4096, thr=float(L["vth"]),
                       soma="loihi_lif_delayed", attrs={}) for L in lp]
        chip, plan = build(layers, st.shape[1], "base")
        per, match = [], []
        for i in range(len(st)):
            e, lat, al, tot, vec, pred = simulate(chip, layers, st[i], T, "potential")
            m = bool(np.array_equal(al, cnt[i]) and np.array_equal(np.round(vec * 64), acc[i]))
            match.append(m)
            per.append(dict(flow=i, energy_J=e, latency_s=lat, pred=pred, lava_pred=int(lpred[i]), matches_lava=m,
                            **{f"spikes_l{k + 1}": int(c) for k, c in enumerate(al)}))
        d = pd.DataFrame(per)
        d.to_csv(OUT / f"per_flow_{ds}_{tag}.csv", index=False)
        rate = float(np.mean(match))
        if rate < 1.0:
            raise SystemExit(f"{ds}/{tag}: SANA-FE matches Lava on {100 * rate:.2f}% of flows; energy not reported")
        full = np.concatenate([emulate(lp, st_all[i:i + CH], T, du=DU, dv=DV, reset="subtract")[0].argmax(1)
                               for i in range(0, len(st_all), CH)])
        if not np.array_equal(full[samp], lpred):
            raise SystemExit(f"{ds}/{tag}: full-test emulation differs from Lava on the sample")
        f1 = lambda yy, p: 100 * f1_score(yy, p, average="macro")
        enc_ops = int(sum(math.ceil(math.log2(e + 1)) for e in z["n_edges"]))
        r = dict(dataset=ds, variant=tag, run_dir=str(z["run_dir"]), T=T, channels=int(st.shape[1]), n_sample=int(len(samp)),
                 sanafe_match_rate=rate, energy_nJ_mean=1e9 * d.energy_J.mean(), energy_nJ_sd=1e9 * d.energy_J.std(),
                 latency_us_mean=1e6 * d.latency_s.mean(), latency_us_sd=1e6 * d.latency_s.std(),
                 spikes_per_flow="/".join(f"{d[c].mean():.1f}" for c in d.columns if c.startswith("spikes_l")),
                 simulated_macro_f1_full_test=f1(y_all, full), float_macro_f1_full_test=float(z["float_f1"]),
                 int_numpy_macro_f1_full_test=float(z["int_f1"]),
                 lava_agreement_with_int_numpy_sample=float(np.mean(lpred == z["int_pred"][samp])),
                 host_encoder_comparisons=enc_ops, lava_seconds=round(sec, 1), max_abs_v=int(vmax),
                 seconds=round(time.time() - t0, 1), parameters=info)
        (OUT / f"{ds}_{tag}.json").write_text(json.dumps(r, indent=1))
        print({k: v for k, v in r.items() if k != "parameters"}, flush=True)


if __name__ == "__main__":
    main()
