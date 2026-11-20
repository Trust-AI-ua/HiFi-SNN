#!/usr/bin/env python3
"""Amendment 8, part 3 and the per-flow simulator records (results/preregistration/2026-11-17_neuro_stream.md).
Runs in .venv; reads only existing traces (results/neuro/confirm, results/neuro/review/a8_tuned_vanilla).

  per_flow_records.csv.gz  one row per benchmark, seed, configuration and sampled flow: flow_id (row index in
                           that seed's test split; the split is drawn per seed), class, energy, latency,
                           executed steps, exit decision, prediction, full-window prediction
  class_counts.csv         per benchmark and seed: class counts on the 2,000-flow sample and the full test split
  energy_prevalence.csv    per benchmark, seed and configuration: sample-mean energy, prevalence-weighted energy
                           (sum over classes of test share x class mean), the paired per-flow difference against
                           the fixed-point Vanilla on the same flows (plain and prevalence-weighted), mean steps
  energy_by_class.csv      per benchmark, seed, configuration and class: mean energy, mean steps, paired difference
  ENERGY.md                the five-seed summary for the two five-seed energy claims, and every other row

    .venv/bin/python scripts/neuro_review_energy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
INP = REPO / "results/neuro/inputs"
CONF = REPO / "results/neuro/confirm"
TUNED = REPO / "results/neuro/review/a8_tuned_vanilla"
OUT = REPO / "results/neuro/review/p2_simulator"
SEEDS = (42, 43, 44, 45, 46)
CONFIGS = {"edgeiiot": ["t2s_full", "t2s_exit", "T10", "vanilla", "vanilla_tuned"],
           "ciciomt2024": ["t2s_full", "t2s_exit", "vanilla", "vanilla_tuned"],
           "toniot": ["t2s_full", "t2s_exit", "T10", "vanilla"],
           "ciciot2023": ["t2s_full", "t2s_exit", "vanilla"]}
CLAIMS = (("edgeiiot", "T10"), ("ciciomt2024", "t2s_exit"))


def per_flow(ds, seed, cfg):
    """Per-flow frame for one configuration, or None; the exit row falls back to the full window where the
    seed's gate is disabled (no exit file), as in amendments 6 and 7."""
    f = {"t2s_full": CONF / f"per_flow_{ds}_seed{seed}_t2s_full.csv", "t2s_exit": CONF / f"per_flow_{ds}_seed{seed}_t2s_exit.csv",
         "T10": CONF / f"per_flow_{ds}_seed{seed}_T10_full.csv", "vanilla": CONF / f"per_flow_{ds}_seed{seed}_vanilla.csv",
         "vanilla_tuned": TUNED / f"per_flow_{ds}_seed{seed}_vanilla_tuned.csv"}[cfg]
    gate_disabled = cfg == "t2s_exit" and not f.exists() and (CONF / f"per_flow_{ds}_seed{seed}_t2s_full.csv").exists()
    if gate_disabled:
        f = CONF / f"per_flow_{ds}_seed{seed}_t2s_full.csv"
    if not f.exists():
        return None, None
    return pd.read_csv(f), gate_disabled


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    thr = json.loads((REPO / "results/journal_r2/exit_calibration/thresholds.json").read_text())
    recs, counts, prev, bycls = [], [], [], []
    for ds, cfgs in CONFIGS.items():
        for seed in SEEDS:
            z = np.load(INP / f"{ds}_seed{seed}.npz")
            y, samp, names, T = z["labels"], z["sample"], [str(c) for c in z["class_names"]], int(z["T"])
            vi = np.load(INP / f"{ds}_vanilla_int_seed{seed}.npz")
            if not np.array_equal(vi["sample"], samp) or not np.array_equal(vi["labels_full"], y):
                raise SystemExit(f"{ds}/{seed}: Vanilla and Tree-to-Spike samples differ; flows cannot be paired")
            ys = y[samp]
            K = len(names)
            cs, cf = np.bincount(ys, minlength=K), np.bincount(y, minlength=K)
            pi = cf / cf.sum()
            counts += [dict(dataset=ds, seed=seed, class_id=k, class_name=names[k], n_sample=int(cs[k]), n_test=int(cf[k]),
                            test_share=float(pi[k])) for k in range(K)]
            full, _ = per_flow(ds, seed, "t2s_full")
            van, _ = per_flow(ds, seed, "vanilla")
            for cfg in cfgs:
                d, gd = per_flow(ds, seed, cfg)
                if d is None:
                    continue
                if not np.array_equal(d.flow.values, np.arange(len(samp))):
                    raise SystemExit(f"{ds}/{seed}/{cfg}: per-flow rows are not the sample in order")
                Tc = 10 if cfg == "T10" else (25 if cfg.startswith("vanilla") else T)
                e = 1e9 * d.energy_J.values
                rec = pd.DataFrame(dict(dataset=ds, seed=seed, config=cfg, sample_pos=d.flow.values, flow_id=samp,
                                        class_id=ys, class_name=np.array(names)[ys], energy_nJ=e, latency_us=1e6 * d.latency_s.values,
                                        steps=d.steps.values, window=Tc,
                                        exited_early=(d.steps.values < Tc) if cfg == "t2s_exit" else False,
                                        exit_gate=(None if gd else thr[f"{ds}/{seed}"]["gate_c1"]) if cfg == "t2s_exit" else None,
                                        pred=d.pred.values,
                                        full_window_pred=full.pred.values if cfg == "t2s_exit" else d.pred.values))
                recs.append(rec)
                cm = np.array([e[ys == k].mean() if (ys == k).any() else np.nan for k in range(K)])
                sm = np.array([d.steps.values[ys == k].mean() if (ys == k).any() else np.nan for k in range(K)])
                de = e - 1e9 * van.energy_J.values
                dm = np.array([de[ys == k].mean() if (ys == k).any() else np.nan for k in range(K)])
                ok = cs > 0
                w = pi[ok] / pi[ok].sum()
                prev.append(dict(dataset=ds, seed=seed, config=cfg, gate_disabled=bool(gd), n_sample=int(len(e)),
                                 energy_nJ_sample_mean=float(e.mean()), energy_nJ_prevalence=float((w * cm[ok]).sum()),
                                 paired_diff_vs_vanilla_nJ_sample=float(de.mean()), paired_diff_vs_vanilla_nJ_prevalence=float((w * dm[ok]).sum()),
                                 paired_diff_sd_nJ=float(de.std(ddof=1)), share_flows_below_vanilla=float((de < 0).mean()),
                                 mean_steps_sample=float(d.steps.mean()), mean_steps_prevalence=float((w * sm[ok]).sum()),
                                 classes_absent_from_sample=int((~ok).sum())))
                bycls += [dict(dataset=ds, seed=seed, config=cfg, class_id=k, class_name=names[k], n_sample=int(cs[k]), test_share=float(pi[k]),
                               energy_nJ_mean=float(cm[k]), mean_steps=float(sm[k]), paired_diff_vs_vanilla_nJ=float(dm[k]))
                          for k in range(K) if cs[k] > 0]
    pd.concat(recs).to_csv(OUT / "per_flow_records.csv.gz", index=False, float_format="%.6g")
    pd.DataFrame(counts).to_csv(OUT / "class_counts.csv", index=False)
    P = pd.DataFrame(prev); P.to_csv(OUT / "energy_prevalence.csv", index=False, float_format="%.6g")
    pd.DataFrame(bycls).to_csv(OUT / "energy_by_class.csv", index=False, float_format="%.6g")
    L = ["# Simulated energy: sample mean, prevalence-weighted, paired against the fixed-point Vanilla (amendment 8, part 3)", "",
         "Five-seed means (SD over seeds). Paired difference = configuration minus the Vanilla SNN on the same sampled flows; "
         "negative means less energy than the Vanilla SNN. Prevalence = class means on the stratified 2,000-flow sample weighted "
         "by the full test split's class shares.", "",
         "| Benchmark | Configuration | Seeds | Energy, sample (nJ) | Energy, prevalence (nJ) | Paired diff, sample (nJ) | "
         "Paired diff, prevalence (nJ) | Flows below Vanilla | Mean steps, sample / prevalence |", "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    ms = lambda s: f"{s.mean():.1f} ({s.std(ddof=1):.1f})" if len(s) > 1 else f"{s.mean():.1f}"
    for (ds, cfg), g in P.groupby(["dataset", "config"], sort=False):
        L.append(f"| {ds} | {cfg} | {len(g)} | {ms(g.energy_nJ_sample_mean)} | {ms(g.energy_nJ_prevalence)} | "
                 f"{ms(g.paired_diff_vs_vanilla_nJ_sample)} | {ms(g.paired_diff_vs_vanilla_nJ_prevalence)} | "
                 f"{100 * g.share_flows_below_vanilla.mean():.1f}% | {g.mean_steps_sample.mean():.2f} / {g.mean_steps_prevalence.mean():.2f} |")
    L += ["", "## The two five-seed energy claims", ""]
    for ds, cfg in CLAIMS:
        g = P[(P.dataset == ds) & (P.config == cfg)]
        L.append(f"- **{ds} {cfg}**: paired difference {g.paired_diff_vs_vanilla_nJ_sample.mean():.1f} nJ on the sample, "
                 f"{g.paired_diff_vs_vanilla_nJ_prevalence.mean():.1f} nJ prevalence-weighted; traffic-mix share of the sample "
                 f"difference {(g.paired_diff_vs_vanilla_nJ_sample - g.paired_diff_vs_vanilla_nJ_prevalence).mean():.1f} nJ; "
                 f"per seed (sample / prevalence): " + ", ".join(f"{r.seed}: {r.paired_diff_vs_vanilla_nJ_sample:.1f} / "
                                                                f"{r.paired_diff_vs_vanilla_nJ_prevalence:.1f}" for r in g.itertuples()))
        if cfg == "t2s_exit":
            a = g[~g.gate_disabled]
            L.append(f"  - gate admitted on {len(a)} of {len(g)} seeds; admitted seeds only: {a.paired_diff_vs_vanilla_nJ_sample.mean():.1f} nJ "
                     f"(sample), {a.paired_diff_vs_vanilla_nJ_prevalence.mean():.1f} nJ (prevalence); gate-disabled seeds: "
                     + ", ".join(f"{r.seed} ({r.paired_diff_vs_vanilla_nJ_sample:.1f} nJ)" for r in g[g.gate_disabled].itertuples()))
    L += ["", "Where the gate is evaluated and what is charged: the C1 gate (per-step temperature, then the maximum class "
          "probability against `gate_c1`) is evaluated on the host from the float student's per-step prefix outputs "
          "(`results/prefix_outputs/<ds>/treespike/seed_<S>/test_prefix.npz`); the chip then runs that flow for the gate's "
          "window. SANA-FE (`src/sanafe_plugin/build/loihi_delayed.yaml`, the bundled Loihi prices) charges only the "
          "chip, for the executed steps plus one fill/drain step per layer: every mapped neuron's access (51.2 pJ) and "
          "update (21.6 pJ) each step (force_update), each output spike (69.3 pJ) and outgoing message (111 pJ), each "
          "synaptic event (35.5 pJ per dense connection of a spiking input; zero weights are mapped and charged), and "
          "network-on-chip hops (3.0 to 4.2 pJ). The gate's computation, the host encoder and input transfer are not charged."]
    (OUT / "ENERGY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
