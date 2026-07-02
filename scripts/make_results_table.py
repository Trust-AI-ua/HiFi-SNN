#!/usr/bin/env python3
"""Combined per-dataset results table (classification + binary IDS + efficiency)
for all 4 models, 11 datasets -> results/Journal_results/tables/table_results_full.tex (+ .md).

Energy methodology (consistent with benchmark_4way.py, NOT the template's raw-spike example):
  * Vanilla SNN / Tree-to-Spike: SOPs = measured synops_per_sample + input->layer-1 term
    (channels x hidden[0]); Energy = SOPs x 0.9 pJ.
  * DNN-Tree: measured MACs; Energy = MACs x 4.6 pJ.
  * TASNN: analytic transformer-MAC + spike-AC estimate (its raw spike count excludes the
    dominant transformer front-end); ops = MAC+AC, Energy = MAC*4.6 + AC*0.9 pJ.  (marked †)
Horowitz 45 nm. Latency: inference_ms_per_sample (spiking) / *_batch1 (DNN) — batching differs.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TAB = REPO / "results" / "Journal_results" / "tables"
E_AC, E_MAC = 0.9, 4.6

DS = [("kddcup99", "KDDCup99"), ("nslkdd", "NSL-KDD"), ("ctu13", "CTU-13"),
      ("iscxids2012", "ISCX2012"), ("cicids2017", "CIC-IDS2017"),
      ("cse-cic-ids2018", "CSE-CIC-2018"), ("unswnb15", "UNSW-NB15"),
      ("ciciot2023", "CICIoT2023"), ("ciciomt2024", "CICIoMT2024"),
      ("edgeiiot", "Edge-IIoTset"), ("toniot", "TON-IoT")]
MODELS = [("vanilla", "Vanilla SNN", "results/{d}/SNN_LeakyParallel/latency/seed_*/results.json"),
          ("tree", "Tree-to-Spike", "results/Tree-to-Spike/{d}/seed_*/results.json"),
          ("dnn", "DNN-Tree", "results/DNN-Tree/{d}/seed_*/results.json"),
          ("tasnn", "TASNN", "results/TASNN/{d}/seed_*/results.json")]


def _tasnn_ops_energy(r):
    c, F, C = r["config"], r["data"]["num_features"], r["data"]["num_classes"]
    d, nl, ff, T = c["d_model"], c["n_layers"], c["ff"], c["num_steps"]
    popM, Cstc, k = c["pop_M"], c["stc_channels"], c["stc_kernel"]
    h1, h2 = c["lif_hidden"]; L = F + 1; cs = F * (1 + popM)
    mac = F*d + nl*(4*L*d*d + 2*L*L*d + 2*L*d*ff) + d*Cstc + Cstc*h1*T + d*C + h2*C
    ac = Cstc*cs*k*T + h1*h1*T + h1*h2*T + h2*h2*T
    return mac + ac, mac * E_MAC + ac * E_AC


def cell(key, recs):
    if not recs:
        return None
    g = lambda fn: float(np.mean([fn(r) for r in recs]))
    M = lambda k: g(lambda r: r["test_plus"]["metrics"][k])
    B = lambda k: g(lambda r: r["test_plus"]["binary_ids_metrics"][k])
    o = {"acc": M("accuracy"), "bacc": M("balanced_accuracy"), "f1": M("f1_macro"),
         "mcc": M("matthews_corrcoef"), "dr": B("detection_rate"), "far": B("false_alarm_rate")}
    if key == "dnn":
        o["spar"] = None
        o["ops"] = g(lambda r: r["test_plus"]["compute"]["macs_per_sample"]); o["unit"] = "MACs"
        o["energy"] = o["ops"] * E_MAC
        o["lat"] = g(lambda r: r["test_plus"].get("inference_ms_per_sample_batch1") or 0)
        o["tasnn"] = False
    elif key == "tasnn":
        o["spar"] = g(lambda r: r["test_plus"]["spike_stats"]["spike_sparsity"])
        ops = [_tasnn_ops_energy(r) for r in recs]
        o["ops"] = float(np.mean([x[0] for x in ops])); o["unit"] = "MACs"
        o["energy"] = float(np.mean([x[1] for x in ops]))
        o["lat"] = g(lambda r: r["test_plus"].get("inference_ms_per_sample") or 0)
        o["tasnn"] = True
    else:  # vanilla / tree
        spv = [r["test_plus"]["spike_stats"].get("spike_sparsity") for r in recs]
        spv = [v for v in spv if isinstance(v, (int, float))]
        o["spar"] = float(np.mean(spv)) if spv else None
        def sops(r):
            syn = r["test_plus"]["spike_stats"].get("synops_per_sample")
            if syn is None:
                return None
            cfg, data = r["config"], r["data"]
            if key == "tree":
                h = cfg["hidden"]; inp = data["encoder_channels"] * (h[0] if isinstance(h, list) else 64)
            else:
                inp = data["num_features"] * cfg.get("hidden_size", 128)
            return syn + inp
        sv = [sops(r) for r in recs]; sv = [v for v in sv if v is not None]
        if not sv:
            print(f"  [warn] {key}: 0/{len(recs)} records have synops_per_sample")
        o["ops"] = float(np.mean(sv)) if sv else None
        o["unit"] = "SOPs"
        o["energy"] = o["ops"] * E_AC if o["ops"] is not None else None
        o["lat"] = g(lambda r: r["test_plus"].get("inference_ms_per_sample") or 0)
        o["tasnn"] = False
    return o


def fnum(x, unit=None):
    s = f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:,.0f}"
    return f"{s} {unit}" if unit else s


def build():
    data = {}
    for dk, _ in DS:
        data[dk] = {}
        for mk, _, patt in MODELS:
            recs = [json.load(open(f)) for f in glob.glob(str(REPO / patt.format(d=dk)))]
            data[dk][mk] = cell(mk, recs)

    # columns: (label, key, direction)  dir: +1 higher better, -1 lower better, 'E' by energy
    COLS = [("acc", +1), ("bacc", +1), ("f1", +1), ("mcc", +1), ("dr", +1),
            ("far", -1), ("energy", -1), ("lat", -1)]
    mkeys = [m[0] for m in MODELS]; mnames = {m[0]: m[1] for m in MODELS}

    def best(dk, col, direction):
        vals = {}
        for mk in mkeys:
            c = data[dk][mk]
            if c is None:
                continue
            v = c["energy"] if direction == "E" else c.get(col)
            if isinstance(v, (int, float)):
                vals[mk] = v
        if not vals:
            return None
        return min(vals, key=vals.get) if direction in (-1, "E") else max(vals, key=vals.get)

    def fmt(dk, mk, col, direction):
        c = data[dk][mk]
        v = c.get(col) if c else None
        if col == "spar" and (v is None):
            return "---"
        if v is None:
            return "---"
        if col in ("acc", "bacc", "f1", "dr", "far"):
            s = f"{v*100:.2f}"
        elif col == "mcc":
            s = f"{v:.4f}"
        elif col == "spar":
            s = f"{v*100:.1f}"
        elif col == "ops":
            s = fnum(v, c["unit"]) + (r"$^\dagger$" if c.get("tasnn") else "")
        elif col == "energy":
            s = fnum(v) + (r"$^\dagger$" if c.get("tasnn") else "")
        elif col == "lat":
            s = f"{v:.3f}"
        is_best = (best(dk, col, direction) == mk)
        return rf"\cellcolor{{bestcell}}\textbf{{{s}}}" if is_best else s

    lines = [
        r"% Requires: \usepackage{multirow,booktabs,colortbl,xcolor}",
        r"% \definecolor{bestcell}{HTML}{C6EFCE}  % light forest-green; for light blue use {HTML}{BBDEFB}",
        r"\begin{table*}[!ht]", r"\centering", r"\renewcommand{\arraystretch}{1.1}",
        r"\caption{Classification, binary IDS, and absolute-energy results across all datasets "
        r"(mean over 5 seeds). Best per metric and dataset in \colorbox{bestcell}{\textbf{bold}}; "
        r"the arrow ($\uparrow$/$\downarrow$) next to each metric gives the better direction. "
        r"We report \emph{absolute} compute energy rather than spike sparsity: sparsity is a ratio "
        r"that ignores network size and dense (non-spiking) components, whereas absolute energy "
        r"counts every operation. SNN/Tree-to-Spike energy = SOPs (incl.\ input layer) $\times$ "
        r"0.9\,pJ; DNN-Tree = MACs $\times$ 4.6\,pJ (Horowitz 45\,nm). $^\dagger$TASNN energy is an "
        r"analytic transformer-MAC + spike-AC estimate (its raw spike count excludes the dominant "
        r"dense transformer). Latency is wall-clock (batching differs: batch-1 for DNN-Tree, batched "
        r"per-sample otherwise).}",
        r"\label{tab:results_full}", r"{\footnotesize",
        r"\begin{tabular}{l l r r r r r r r r}", r"\toprule",
        r"& & \multicolumn{6}{c}{\textbf{Classification \& Detection}} & "
        r"\multicolumn{2}{c}{\textbf{Efficiency}} \\",
        r"\cmidrule(lr){3-8} \cmidrule(lr){9-10}",
        r"\textbf{Dataset} & \textbf{Model} & \textbf{ACC}\,$\uparrow$ & \textbf{BACC}\,$\uparrow$ & "
        r"\textbf{F1-M}\,$\uparrow$ & \textbf{MCC}\,$\uparrow$ & \textbf{DR}\,$\uparrow$ & "
        r"\textbf{FAR}\,$\downarrow$ & \textbf{Energy}\,$\downarrow$ & \textbf{Latency}\,$\downarrow$ \\",
        r"& & \textbf{(\%)} & \textbf{(\%)} & \textbf{(\%)} & & \textbf{(\%)} & \textbf{(\%)} & "
        r"\textbf{(pJ)} & \textbf{(ms)} \\", r"\midrule",
    ]
    for dk, dn in DS:
        lines.append(rf"\multirow{{4}}{{*}}{{{dn}}}")
        for mk in mkeys:
            cells = " & ".join(fmt(dk, mk, col, d) for col, d in COLS)
            lines.append(rf"& {mnames[mk]:13} & {cells} \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table*}", ""]
    (TAB / "table_results_full.tex").write_text("\n".join(lines))
    print(f"[wrote] {TAB}/table_results_full.tex  ({len(DS)} datasets x {len(MODELS)} models)")


if __name__ == "__main__":
    build()
