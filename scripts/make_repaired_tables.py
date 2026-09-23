#!/usr/bin/env python3
"""Tables that the repaired re-runs make final (no latency, no energy).

Emits CSV + LaTeX for:
  tab_repaired_predictive   ACC / BACC / F1-M / MCC / DR / FAR, 4 models x the two
                            repaired datasets (rows for Tables IV / VIII)
  tab_repaired_anatomy      input width, output dim, encoder channels, parameters,
                            spike density, SOPs or MACs, mean exit step
                            (rows for Tables V / IX)
  tab_fc1_ratio             first-layer max|W|/p99|W| and the fraction of weights
                            the int8 scale rounds to zero, before and after repair
  tab_m6_per_seed           repaired CSE-CIC-IDS2018 DNN-Tree fp32 / int8 / delta
  tab_gbdt_envelope         per dataset: full ensemble bytes, whether it fits
                            256 KB, and the best compressed configuration that does

Latency, energy and artifact sizes are deliberately absent: they come from the
deployment measurement, which runs last on a quiet machine.

    python scripts/make_repaired_tables.py
Output: results/tables/<name>.{csv,tex}
"""
from __future__ import annotations

import glob
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TAB = REPO / "results/tables"
REPAIRED = ["cse-cic-ids2018", "edgeiiot"]
LABEL = {"cse-cic-ids2018": "CSE-CIC-IDS2018", "edgeiiot": "Edge-IIoTset"}
MODELS = [("Tree-to-Spike", "results/Tree-to-Spike/{d}"), ("DNN-Tree", "results/DNN-Tree/{d}"),
          ("TASNN", "results/TASNN/{d}"), ("Vanilla SNN", "results/{d}/SNN_LeakyParallel/latency")]


def prov():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                      text=True).strip()
    except Exception:
        sha = "unknown"
    return (f"generated {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} by "
            f"scripts/make_repaired_tables.py at git {sha}")


def runs(base, d):
    return [json.load(open(f)) for f in sorted(glob.glob(str(REPO / base.format(d=d) /
                                                          "seed_*/results.json")))]


def write(df, name, caption, label, floatfmt="%.2f"):
    head = prov()
    with open(TAB / f"{name}.csv", "w") as f:
        f.write(f"# {head}\n")
        df.to_csv(f, index=False)
    body = df.to_latex(index=False, float_format=floatfmt, escape=False,
                       column_format="l" * 2 + "r" * (len(df.columns) - 2))
    (TAB / f"{name}.tex").write_text(
        f"% {head}\n% source: results/tables/{name}.csv\n"
        f"\\begin{{table}}[t]\\centering\\small\n\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n{body}\\end{{table}}\n")
    print(f"[wrote] {name}.csv/.tex ({len(df)} rows)")


def predictive():
    rows = []
    for d in REPAIRED:
        for name, base in MODELS:
            rs = runs(base, d)
            if len(rs) != 5:
                continue
            m = [r["test_plus"]["metrics"] for r in rs]
            b = [r["test_plus"]["binary_ids_metrics"] for r in rs]
            g = lambda xs: (100 * np.mean(xs), 100 * np.std(xs, ddof=1))
            acc, bacc = g([x["accuracy"] for x in m]), g([x["balanced_accuracy"] for x in m])
            f1, mcc = g([x["f1_macro"] for x in m]), g([x["matthews_corrcoef"] for x in m])
            dr, far = g([x["detection_rate"] for x in b]), g([x["false_alarm_rate"] for x in b])
            rows.append(dict(Dataset=LABEL[d], Model=name,
                             ACC=acc[0], ACC_sd=acc[1], BACC=bacc[0], BACC_sd=bacc[1],
                             F1M=f1[0], F1M_sd=f1[1], MCC=mcc[0], MCC_sd=mcc[1],
                             DR=dr[0], DR_sd=dr[1], FAR=far[0], FAR_sd=far[1]))
    write(pd.DataFrame(rows), "tab_repaired_predictive",
          "Predictive performance on the two repaired corpora, mean $\\pm$ sample SD over "
          "five seeds. Tree-to-Spike and DNN-Tree are the deployed int8-grid artifacts; "
          "Vanilla SNN and TASNN are floating point.", "tab:repaired:predictive")


def anatomy():
    rows = []
    for d in REPAIRED:
        for name, base in MODELS:
            rs = runs(base, d)
            if len(rs) != 5:
                continue
            r0, tp = rs[0], [r["test_plus"] for r in rs]
            def per_sample(t):
                """SOPs where recorded, else spikes/sample for the SNNs that do not
                record synops, else MACs for the dense twin."""
                ss = t.get("spike_stats", {})
                for k in ("synops_per_sample", "energy_spikes_per_sample"):
                    if k in ss:
                        return ss[k], k
                return t["compute"]["macs_per_sample"], "macs_per_sample"

            arith = [per_sample(t)[0] for t in tp]
            unit = per_sample(tp[0])[1]
            dens = [t.get("spike_stats", {}).get("spike_density", np.nan) * 100 for t in tp]
            exit_step = [t["early_exit"]["mean_exit_step"] if "early_exit" in t else np.nan
                         for t in tp]
            rows.append(dict(Dataset=LABEL[d], Model=name,
                             InputWidth=r0["data"]["num_features"],
                             Channels=r0["data"].get("encoder_channels", np.nan),
                             OutputDim=r0["data"]["num_classes"],
                             Params=r0["model_size"]["trainable_parameters"],
                             SpikeDensity=np.mean(dens),
                             ArithPerSample=np.mean(arith), ArithUnit=unit,
                             MeanExitStep=np.mean(exit_step)))
    write(pd.DataFrame(rows), "tab_repaired_anatomy",
          "Architecture and activity on the repaired corpora. Arithmetic is SOPs per sample "
          "for the spiking models and MACs per sample for the dense ones; spike density and "
          "mean exit step apply to the spiking students only. Latency and energy are reported "
          "separately from the deployment measurement.", "tab:repaired:anatomy")


def fc1():
    f = pd.read_csv(REPO / "results/diagnostics/fc1_ratio_repair.csv")
    g = f[f.layer == "fc1"]
    piv = g.pivot_table(index=["model", "seed"], columns="arm",
                        values=["max_over_p99", "frac_rounded_to_zero"]).reset_index()
    piv.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in piv.columns]
    piv = piv.rename(columns={"model": "Model", "seed": "Seed"})
    for c in [c for c in piv.columns if c.startswith("frac")]:
        piv[c] = piv[c] * 100
    write(piv, "tab_fc1_ratio",
          "First-layer weight-scale ratio $\\max|W|/p_{99}|W|$ and the percentage of "
          "first-layer weights the per-tensor int8 scale rounds to zero, on CSE-CIC-IDS2018 "
          "before and after the dataset repair.", "tab:fc1ratio")


def m6():
    s = pd.read_csv(REPO / "results/tables/m3_quant_per_seed.csv")
    d = s[(s.dataset == "cse-cic-ids2018") & (s.model == "DNN-Tree")]
    out = d[["seed", "fp", "int8", "d"]].rename(
        columns={"seed": "Seed", "fp": "fp32", "int8": "int8 grid", "d": "$\\Delta$"})
    write(out, "tab_m6_per_seed",
          "DNN-Tree on repaired CSE-CIC-IDS2018: test macro-$F_1$ in floating point, after the "
          "int8-grid projection, and their difference, per seed. Before the repair the same "
          "cell spanned 48.57--76.44 with a mean $\\Delta$ of $-11.46$.", "tab:m6:perseed")


def envelope():
    d = pd.read_csv(REPO / "results/deployment/gbdt_compression.csv")
    d["gain_prune"] = d["gain_prune"].astype(str)
    d = d[~((d.dataset == "edgeiiot") & (d.n_classes == 15))]   # pre-repair task
    B = 256 * 1024
    rows = []
    for ds, g in d.groupby("dataset"):
        full = g[(g.n_estimators == 200) & (g.max_depth == 6) &
                 (g.gain_prune == "none") & (g.leaf_bits == 16)]
        fit = g[g.deploy_bytes <= B]
        best = fit.loc[fit.macro_f1.idxmax()] if len(fit) else None
        rows.append(dict(Dataset=LABEL.get(ds, ds),
                         FullKB=full.deploy_bytes.max() / 1024,
                         FullF1=full.macro_f1.max(),
                         FullFits="yes" if full.deploy_bytes.max() <= B else "no",
                         BestFitF1=best.macro_f1 if best is not None else np.nan,
                         BestFitKB=best.deploy_bytes / 1024 if best is not None else np.nan,
                         BestFitCfg=(f"{int(best.n_estimators)}/{int(best.max_depth)}/"
                                     f"{best.gain_prune}/{int(best.leaf_bits)}b")
                         if best is not None else "--",
                         NFitting=int(len(fit))))
    write(pd.DataFrame(rows), "tab_gbdt_envelope",
          "GBDT teacher against a 256\\,KB budget (seed 42): the full ensemble "
          "(200 rounds, depth 6, 16-bit leaves), and the highest-scoring compressed "
          "configuration that fits, given as rounds/depth/gain-prune/leaf bits.",
          "tab:dep:envelope")


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    predictive(); anatomy(); fc1(); m6(); envelope()
