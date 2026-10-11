#!/usr/bin/env python3
"""M3 table: fp32 vs int8-grid test macro-F1 for all four models x 11 datasets.

Sources (all under results/diagnostics/):
  DNN-Tree, Tree-to-Spike : val_curves/<model>/<ds>/runs/seed_<S>/test_fp32_int8.json
                            (deterministic reruns; fp32 weights captured before the
                            pipeline's own int8 projection; int8 side must reproduce
                            the stored results.json exactly)
  Vanilla SNN, TASNN      : m3_fp32_models.csv (evaluation of the stored fp32
                            checkpoints; int8 is a counterfactual)

Two int8 columns: "manuscript quantizer" (quantize_int8_, nn.Linear weights only)
and "all weight matrices" (same per-tensor rule on every weight tensor with
dim >= 2). They coincide for DNN-Tree and Tree-to-Spike (all weights Linear).
A cell is "pending" unless all 5 seeds are present.

    python scripts/make_m3_table.py
Outputs: results/tables/tab_m3_quant.{csv,tex}, results/tables/m3_quant_per_seed.csv
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DIAG = REPO / "results/diagnostics"
TAB = REPO / "results/tables"
DATASETS = ["kddcup99", "nslkdd", "unswnb15", "cicids2017", "cse-cic-ids2018",
            "iscxids2012", "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
LABEL = {"kddcup99": "KDD Cup 99", "nslkdd": "NSL-KDD", "unswnb15": "UNSW-NB15",
         "cicids2017": "CIC-IDS2017", "cse-cic-ids2018": "CSE-CIC-IDS2018",
         "iscxids2012": "ISCX-IDS2012", "ctu13": "CTU-13", "toniot": "TON\\_IoT",
         "edgeiiot": "Edge-IIoTset", "ciciomt2024": "CICIoMT2024", "ciciot2023": "CICIoT2023"}
MODELS = [("vanilla", "Vanilla SNN"), ("tasnn", "TASNN"),
          ("DNN-Tree", "DNN-Tree"), ("Tree-to-Spike", "Tree-to-Spike")]


def _provenance():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                      text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO).returncode != 0
    except Exception:
        sha, dirty = "unknown", False
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"generated {ts} by scripts/make_m3_table.py at git {sha}{'+dirty' if dirty else ''}"


def _load():
    rows = []
    for model in ("DNN-Tree", "Tree-to-Spike"):
        for f in (DIAG / "val_curves" / model).glob("*/runs/seed_*/test_fp32_int8.json"):
            r = json.load(open(f))
            rows.append(dict(model=model, dataset=r["dataset"], seed=r["seed"],
                             fp=r["fp_macro_f1"], int8=r["int8_macro_f1"],
                             int8_all=r["int8_macro_f1"],
                             fp_FAR=r["fp_FAR"], int8_FAR=r["int8_FAR"],
                             repro=r["reproduced"],
                             repro_abs_diff=abs(r["int8_macro_f1"] - r["stored_int8_macro_f1"])))
    p = DIAG / "m3_fp32_models.csv"
    if p.exists():
        for _, r in pd.read_csv(p).iterrows():
            rows.append(dict(model=r.model, dataset=r.dataset, seed=int(r.seed),
                             fp=r.fp_macro_f1, int8=r.int8_macro_f1,
                             int8_all=r.int8_all_macro_f1,
                             fp_FAR=r.fp_FAR, int8_FAR=r.int8_all_FAR,
                             repro=None, repro_abs_diff=r.repro_abs_diff))
    df = pd.DataFrame(rows)
    df["d"] = df.int8 - df.fp
    df["d_all"] = df.int8_all - df.fp
    return df


def main():
    df = _load().sort_values(["model", "dataset", "seed"])
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "m3_quant_per_seed.csv", index=False)

    cells = []
    for m, _ in MODELS:
        for d in DATASETS:
            g = df[(df.model == m) & (df.dataset == d)]
            done = len(g) == 5
            cells.append(dict(
                model=m, dataset=d, n_seeds=len(g), status="done" if done else "pending",
                fp_mean=g.fp.mean() if done else None,
                int8_mean=g.int8.mean() if done else None,
                int8_all_mean=g.int8_all.mean() if done else None,
                d_mean=g.d.mean() if done else None,
                d_all_mean=g.d_all.mean() if done else None,
                d_all_worst=g.d_all.min() if done else None,
                fp_sd=g.fp.std() if done else None,
                int8_all_sd=g.int8_all.std() if done else None,
                max_repro_abs_diff=g.repro_abs_diff.max() if done else None))
    c = pd.DataFrame(cells)
    prov = _provenance()
    with open(TAB / "tab_m3_quant.csv", "w") as f:
        f.write(f"# {prov}\n")
        c.round(4).to_csv(f, index=False)

    def fmt(x, signed=False):
        if x is None or pd.isna(x):
            return "--"
        return f"{x:+.2f}" if signed else f"{x:.2f}"

    lines = [f"% {prov}",
             "% source: results/tables/tab_m3_quant.csv",
             "\\begin{table*}[t]", "\\centering", "\\scriptsize",
             "\\caption{Cost of the int8-grid weight projection on test macro-F1 (\\%), mean over "
             "5 seeds. $\\Delta$ = int8 $-$ fp32; ``worst'' is the most negative per-seed "
             "$\\Delta$. \\emph{Manuscript quantizer}: per-tensor symmetric, "
             "$s=\\max|W|/127$, applied to \\texttt{nn.Linear} weights only. "
             "\\emph{All weight matrices}: the same rule on every weight tensor. The two "
             "coincide for DNN-Tree and Tree-to-Spike, whose weights are all linear; for Vanilla "
             "SNN the manuscript quantizer covers about 2\\% of the weights. DNN-Tree and "
             "Tree-to-Spike are deployed at int8 (their fp32 side comes from deterministic "
             "reruns that reproduce the reported int8 numbers exactly); Vanilla SNN and TASNN "
             "are reported in fp32, so their int8 column is a counterfactual. TASNN samples "
             "stochastic spikes at inference, so one evaluation varies by about 0.05 macro-F1 "
             "run to run; its fp32 and int8 passes use identical draws.}",
             "\\label{tab:m3:quant}",
             "\\begin{tabular}{l" + "rrr" * len(MODELS) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for _, n in MODELS) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(MODELS))),
             "Dataset & " + " & ".join(["fp32 & $\\Delta$ & worst"] * len(MODELS)) + " \\\\",
             "\\midrule"]
    for d in DATASETS:
        cols = []
        for m, _ in MODELS:
            r = c[(c.model == m) & (c.dataset == d)].iloc[0]
            if r.status != "done":
                cols += ["\\multicolumn{3}{c}{pending}"]
            else:
                cols += [fmt(r.fp_mean), fmt(r.d_all_mean, True), fmt(r.d_all_worst, True)]
        lines.append(f"{LABEL[d]} & " + " & ".join(cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (TAB / "tab_m3_quant.tex").write_text("\n".join(lines) + "\n")
    print(f"[wrote] {TAB/'tab_m3_quant.csv'} {TAB/'tab_m3_quant.tex'} ({(c.status=='done').sum()}/44 cells done)")


if __name__ == "__main__":
    main()
