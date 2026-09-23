#!/usr/bin/env python3
"""4-way comparison artifacts for the IEEE write-up.

Models: Vanilla SNN (LeakyParallel/latency), Tree-to-Spike, DNN-Tree, TASNN.
Datasets: all 11 (6 classic + 5 modern). 5 seeds each -> mean ± std.

Emits, under results/Journal_results/:
  tables/  *.tex (booktabs) + *.md   - per-metric quality, IDS, efficiency
  figures/ *.pdf + *.png             - grouped bars + heatmaps (per metric), radar,
                                       Pareto (acc vs params), iso-trio energy,
                                       mean-rank bars + critical-difference diagrams

Energy (Horowitz 45 nm): spiking op E_AC = 0.9 pJ (SOPs); dense MAC E_MAC = 4.6 pJ.
Compute energy is reported for the parameter-matched iso-trio (SNN / Tree-to-Spike /
DNN-Tree); TASNN is the heavyweight transformer competitor - its spike-SOP count excludes
its transformer front-end, so it is compared on quality + parameters, not SOP energy.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "Journal_results"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

E_AC, E_MAC = 0.9, 4.6          # 32-bit FP (Horowitz 45nm): AC (add) 0.9 pJ, MAC (mult+add) 4.6 pJ
E_AC_I8, E_MAC_I8 = 0.03, 0.23  # int8 (Horowitz 45nm): AC (8b add) 0.03 pJ, MAC (8b mult 0.2 + add 0.03) 0.23 pJ

# (key, pretty, results-glob)
MODELS = [
    ("vanilla", "Vanilla SNN",   "results/{ds}/SNN_LeakyParallel/latency/seed_*/results.json"),
    ("tree",    "Tree-to-Spike", "results/Tree-to-Spike/{ds}/seed_*/results.json"),
    ("dnn",     "DNN-Tree",      "results/DNN-Tree/{ds}/seed_*/results.json"),
    ("tasnn",   "TASNN",         "results/TASNN/{ds}/seed_*/results.json"),
]
MKEYS = [m[0] for m in MODELS]
MNAME = {m[0]: m[1] for m in MODELS}
# IEEE-friendly Paul Tol "bright" qualitative palette (bright + colorblind-safe):
# IEEE/colorblind-safe qualitative set - blue / green / red / orange (Okabe-Ito amber for
# TASNN, distinct from blue & red). Clean solid fills; scatter plots add distinct markers.
COLOR = {"vanilla": "#4477AA", "tree": "#228833", "dnn": "#EE6677", "tasnn": "#E69F00"}
HATCH = {"vanilla": "",        "tree": "//",      "dnn": "..",       "tasnn": "xx"}
MARKER = {"vanilla": "o",      "tree": "s",       "dnn": "^",        "tasnn": "D"}

DATASETS = [
    ("kddcup99", "KDDCup99"), ("nslkdd", "NSL-KDD"), ("ctu13", "CTU-13"),
    ("iscxids2012", "ISCX2012"), ("cicids2017", "CIC-IDS2017"),
    ("cse-cic-ids2018", "CSE-CIC-2018"), ("unswnb15", "UNSW-NB15"),
    ("ciciot2023", "CICIoT2023"), ("ciciomt2024", "CICIoMT2024"),
    ("edgeiiot", "Edge-IIoTset"), ("toniot", "TON-IoT"),
]
DS_KEYS = [d[0] for d in DATASETS]
DS_NAME = {d[0]: d[1] for d in DATASETS}

# metric -> (json path getter, higher_is_better, pretty)
def _m(k):  return lambda r: r["test_plus"]["metrics"].get(k)
def _b(k):  return lambda r: (r["test_plus"].get("binary_ids_metrics") or {}).get(k)
METRICS = {
    "accuracy":  (_m("accuracy"),          True,  "Accuracy"),
    "f1_macro":  (_m("f1_macro"),          True,  "Macro-F1"),
    "mcc":       (_m("matthews_corrcoef"), True,  "MCC"),
    "bal_acc":   (_m("balanced_accuracy"), True,  "Balanced Acc."),
    "roc_auc":   (_m("roc_auc_weighted"),  True,  "ROC-AUC"),
    "DR":        (_b("detection_rate"),    True,  "Detection Rate"),
    "FAR":       (_b("false_alarm_rate"),  False, "False-Alarm Rate"),
}


def _load(ds, glob_tpl):
    recs = []
    for f in glob.glob(str(REPO / glob_tpl.format(ds=ds))):
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    return recs


def _agg(recs, getter):
    vals = [getter(r) for r in recs]
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _sops_total(r, key):
    """Total SOPs/sample INCLUDING the input->layer-1 term, which the stored
    synops_per_sample omits. Under latency/TTFS each input channel fires once and
    drives hidden[0] synapses, so input SOPs = channels * hidden[0]. The DNN's MACs
    already include its input layer, so counting this makes the comparison consistent."""
    syn = (r["test_plus"].get("spike_stats") or {}).get("synops_per_sample")
    if not isinstance(syn, (int, float)):
        return None
    cfg, data = r.get("config", {}), r.get("data", {})
    if key == "tree":
        ch = data.get("encoder_channels")
        h = cfg.get("hidden"); h0 = h[0] if isinstance(h, (list, tuple)) else 64
        inp = ch * h0 if ch else 0
    elif key == "vanilla":
        # NOTHING to add. The Vanilla SNN's stored value already counts its input
        # layer (src/models/snn.py: input_spikes * hidden + hidden_spikes * classes),
        # and only from the features that actually fire. Adding num_features *
        # hidden here counted that layer twice, as if every feature fired, and
        # overstated the model 2.6x to 3.6x. See results/tables/
        # vanilla_sops_double_count.md and scripts/guard_vanilla_ceiling.py.
        inp = 0
    else:
        inp = 0
    return syn + inp


def _energy_pj(recs, key):
    """Compute energy pJ/sample (mean). Spiking: total-SOPs (incl. input layer) * E_AC;
    DNN: measured MACs * E_MAC (already includes the input layer)."""
    out = []
    for r in recs:
        if key == "dnn":
            e = (r["test_plus"].get("compute") or {}).get("energy_pj_horowitz")
        else:
            sops = _sops_total(r, key)
            e = sops * E_AC if sops is not None else None
        if isinstance(e, (int, float)):
            out.append(e)
    return float(np.mean(out)) if out else None


def tasnn_energy_pj(rec):
    """Analytic per-sample energy for TASNN (its forward isn't SOP-instrumented).

    Transformer/dense MACs @ E_MAC + spiking-core ACs (operations on binary spike
    inputs) @ E_AC. Counts, per sample over T steps:
      DENSE (MAC): feature tokeniser F*d; transformer n_l*(4Ld^2 + 2L^2 d + 2Ld*ff)
                   [in/out proj + QK^T + AV + FFN]; XMG d*Cstc; LIF1.fc Cstc*h1*T
                   (real current input); heads d*C + h2*C.
      SPIKE (AC):  STC conv Cstc*c_spike*k*T (binary spike input); LIF1 lateral h1^2*T;
                   LIF2.fc h1*h2*T (spike input); LIF2 lateral h2^2*T.
    L=F+1 (CLS+features); c_spike=F*(1+pop_M). LayerNorm/softmax negligible (excluded).
    """
    c = rec["config"]; F = rec["data"]["num_features"]; C = rec["data"]["num_classes"]
    d, nl, ff, T = c["d_model"], c["n_layers"], c["ff"], c["num_steps"]
    popM, Cstc, k = c["pop_M"], c["stc_channels"], c["stc_kernel"]
    h1, h2 = c["lif_hidden"]
    L = F + 1
    c_spike = F * (1 + popM)
    mac = (F * d                                              # tokeniser
           + nl * (4 * L * d * d + 2 * L * L * d + 2 * L * d * ff)  # transformer
           + d * Cstc                                        # XMG gate
           + Cstc * h1 * T                                   # LIF1.fc (real input)
           + d * C + h2 * C)                                 # dual heads
    ac = (Cstc * c_spike * k * T                             # STC conv on spikes
          + h1 * h1 * T                                      # LIF1 lateral
          + h1 * h2 * T                                      # LIF2.fc (spike input)
          + h2 * h2 * T)                                     # LIF2 lateral
    return mac * E_MAC + ac * E_AC, mac, ac                  # (32b energy, MAC count, AC count)


# ---------------------------------------------------------------- load all
DATA = {}   # DATA[ds][mkey] = dict of aggregated stats
for ds in DS_KEYS:
    DATA[ds] = {}
    for mk, _, tpl in MODELS:
        recs = _load(ds, tpl)
        if not recs:
            DATA[ds][mk] = None
            continue
        cell = {"n": len(recs)}
        for met, (getter, _, _) in METRICS.items():
            cell[met] = _agg(recs, getter)             # (mean, std)
        cell["params"] = _agg(recs, lambda r: r["model_size"]["trainable_parameters"])[0]
        cell["sparsity"] = _agg(recs, lambda r: (r["test_plus"].get("spike_stats") or {}).get("spike_sparsity"))[0]
        cell["exit"] = _agg(recs, lambda r: (r["test_plus"].get("early_exit") or {}).get("mean_exit_step"))[0]
        # latency key differs by model (DNN stores *_batch1); fall back across names
        def _lat(r):
            tp = r["test_plus"]
            for key in ("inference_ms_per_sample", "inference_ms_per_sample_batch1"):
                if isinstance(tp.get(key), (int, float)):
                    return tp[key]
            return None
        cell["lat_ms"] = _agg(recs, _lat)[0]
        # 32-bit energy + matched-precision int8 energy (B2: reviewer-proof "an int8 DNN
        # is also cheap" - re-priced from the same op counts, no re-run).
        if mk == "tasnn":
            e32, ei8 = [], []
            for r in recs:
                if r.get("config") and r.get("data"):
                    e, mac, ac = tasnn_energy_pj(r)
                    e32.append(e); ei8.append(mac * E_MAC_I8 + ac * E_AC_I8)
            cell["E_pj"] = float(np.mean(e32)) if e32 else None
            cell["E_pj_i8"] = float(np.mean(ei8)) if ei8 else None
        else:
            cell["E_pj"] = _energy_pj(recs, mk)
            scale = (E_MAC_I8 / E_MAC) if mk == "dnn" else (E_AC_I8 / E_AC)
            cell["E_pj_i8"] = cell["E_pj"] * scale if cell["E_pj"] is not None else None
        DATA[ds][mk] = cell


def fmt(ms, pct=True):
    if ms is None or ms[0] is None:
        return "--"
    m, s = ms
    return f"{m*100:.1f}\\,$\\pm$\\,{s*100:.1f}" if pct else f"{m:.3f}$\\pm${s:.3f}"


def fmt_md(ms, pct=True):
    if ms is None or ms[0] is None:
        return "--"
    m, s = ms
    return f"{m*100:.1f}±{s*100:.1f}" if pct else f"{m:.3f}±{s:.3f}"


# ================================================================ TABLES
def per_metric_table(met):
    getter, hib, pretty = METRICS[met]
    # LaTeX
    tex = [r"\begin{table}[t]", r"\centering",
           rf"\caption{{{pretty} (\%, mean\,$\pm$\,std over 5 seeds) across datasets. "
           rf"Best per row in \textbf{{bold}}.}}",
           rf"\label{{tab:{met}}}",
           r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{l" + "c" * len(MKEYS) + "}", r"\toprule",
           "Dataset & " + " & ".join(MNAME[k] for k in MKEYS) + r" \\", r"\midrule"]
    md = ["| Dataset | " + " | ".join(MNAME[k] for k in MKEYS) + " |",
          "|" + "---|" * (len(MKEYS) + 1)]
    for ds in DS_KEYS:
        means = {k: (DATA[ds][k][met][0] if DATA[ds][k] and DATA[ds][k][met] else None) for k in MKEYS}
        valid = {k: v for k, v in means.items() if v is not None}
        best = (max if hib else min)(valid, key=valid.get) if valid else None
        cells_tex, cells_md = [], []
        for k in MKEYS:
            s = fmt(DATA[ds][k][met] if DATA[ds][k] else None)
            smd = fmt_md(DATA[ds][k][met] if DATA[ds][k] else None)
            if k == best:
                s = rf"\textbf{{{s}}}"; smd = f"**{smd}**"
            cells_tex.append(s); cells_md.append(smd)
        tex.append(f"{DS_NAME[ds]} & " + " & ".join(cells_tex) + r" \\")
        md.append(f"| {DS_NAME[ds]} | " + " | ".join(cells_md) + " |")
    # mean row
    tex.append(r"\midrule")
    mrow_tex, mrow_md = [], []
    avgs = {}
    for k in MKEYS:
        vals = [DATA[ds][k][met][0] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k][met] and DATA[ds][k][met][0] is not None]
        avgs[k] = float(np.mean(vals)) if vals else None
    best_avg = (max if hib else min)({k: v for k, v in avgs.items() if v is not None},
                                     key=lambda k: avgs[k]) if any(avgs.values()) else None
    for k in MKEYS:
        v = avgs[k]
        s = f"{v*100:.1f}" if v is not None else "--"
        smd = s
        if k == best_avg:
            s = rf"\textbf{{{s}}}"; smd = f"**{smd}**"
        mrow_tex.append(s); mrow_md.append(smd)
    tex.append(r"\textit{Mean} & " + " & ".join(mrow_tex) + r" \\")
    md.append(f"| **Mean** | " + " | ".join(mrow_md) + " |")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / f"table_{met}.tex").write_text("\n".join(tex))
    (TAB / f"table_{met}.md").write_text(f"### {pretty} (%, mean±std; best bold)\n\n" + "\n".join(md) + "\n")


for met in ["accuracy", "f1_macro", "mcc", "DR", "FAR"]:
    per_metric_table(met)


def efficiency_table():
    rows = []
    for k in MKEYS:
        params = [DATA[ds][k]["params"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["params"]]
        spars = [DATA[ds][k]["sparsity"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["sparsity"] is not None]
        lat = [DATA[ds][k]["lat_ms"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["lat_ms"] is not None]
        epj = [DATA[ds][k]["E_pj"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["E_pj"] is not None]
        epj8 = [DATA[ds][k]["E_pj_i8"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k].get("E_pj_i8") is not None]
        rows.append((k, MNAME[k],
                     int(np.mean(params)) if params else None,
                     np.mean(epj) if epj else None,
                     np.mean(epj8) if epj8 else None,
                     np.mean(spars) if spars else None,
                     np.mean(lat) if lat else None))
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Efficiency (mean across datasets). Compute energy is reported at "
           r"\textbf{matched precision}: 32-bit FP (MAC 4.6\,pJ / AC 0.9\,pJ) and int8 "
           r"(MAC 0.23\,pJ / AC 0.03\,pJ), Horowitz 45\,nm. Spiking models pay ACs (SOPs "
           r"\textbf{including the input$\to$layer-1 term}, consistent with the DNN's input-layer "
           r"MACs), DNN-Tree MACs; TASNN$^{\dagger}$ is an analytic transformer-MAC + spike-AC "
           r"estimate. Latency is indicative (batching differs).}",
           r"\label{tab:efficiency}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Model & Params & Energy@32b & Energy@int8 & Spike sparsity & Latency (ms) \\",
           r"& & (pJ/sample) & (pJ/sample) & & \\", r"\midrule"]
    md = ["| Model | Params | Energy@32b (pJ) | Energy@int8 (pJ) | Spike sparsity | Latency (ms) |",
          "|---|--:|--:|--:|--:|--:|"]
    for k, name, p, e, e8, sp, l in rows:
        ps = f"{p:,}" if p else "--"
        mark = r"$^{\dagger}$" if k == "tasnn" else ""
        mark_md = "†" if k == "tasnn" else ""
        es = (f"{e:,.0f}{mark}" if e else "--")
        es_md = (f"{e:,.0f}{mark_md}" if e else "--")
        e8s = (f"{e8:,.0f}{mark}" if e8 else "--")
        e8s_md = (f"{e8:,.0f}{mark_md}" if e8 else "--")
        sps = f"{sp:.3f}" if sp is not None else "--"
        ls = f"{l:.3f}" if l is not None else "--"
        tex.append(f"{name} & {ps} & {es} & {e8s} & {sps} & {ls} " + r"\\")
        md.append(f"| {name} | {ps} | {es_md} | {e8s_md} | {sps} | {ls} |")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / "table_efficiency.tex").write_text("\n".join(tex))
    md.append("\n† TASNN energy is an analytic FLOP estimate (transformer MACs×4.6pJ + spiking-core ACs×0.9pJ); the other models report measured SOPs/MACs.")
    (TAB / "table_efficiency.md").write_text("### Efficiency (mean across datasets)\n\n" + "\n".join(md) + "\n")


efficiency_table()

# ================================================================ FIGURES
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


def grouped_bar(met, fname):
    getter, hib, pretty = METRICS[met]
    x = np.arange(len(DS_KEYS)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    for i, k in enumerate(MKEYS):
        ms = [DATA[ds][k][met][0] if DATA[ds][k] and DATA[ds][k][met] else np.nan for ds in DS_KEYS]
        es = [DATA[ds][k][met][1] if DATA[ds][k] and DATA[ds][k][met] else 0 for ds in DS_KEYS]
        ax.bar(x + (i - 1.5) * w, [v * 100 if v == v else 0 for v in ms], w,
               yerr=[e * 100 for e in es], capsize=2, label=MNAME[k],
               color=COLOR[k], edgecolor="black", linewidth=0.4, error_kw={"lw": 0.6})
    ax.set_xticks(x); ax.set_xticklabels([DS_NAME[d] for d in DS_KEYS], rotation=40, ha="right")
    ax.set_ylabel(f"{pretty} (%)")
    ax.set_ylim(0, None if met == "FAR" else 100)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    save(fig, fname)


for _met in ["accuracy", "f1_macro", "mcc", "DR", "FAR"]:
    grouped_bar(_met, f"bar_{_met}")


def radar():
    axes = ["accuracy", "f1_macro", "mcc", "DR", "FAR"]
    labels = ["Accuracy", "Macro-F1", "MCC", "Detection", "Low FAR"]
    # aggregate mean across datasets; FAR -> 1-FAR (so higher=better); normalise each axis to best=1
    agg = {k: [] for k in MKEYS}
    for k in MKEYS:
        for met in axes:
            vals = [DATA[ds][k][met][0] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k][met] and DATA[ds][k][met][0] is not None]
            v = float(np.mean(vals)) if vals else 0.0
            if met == "FAR":
                v = 1.0 - v
            agg[k].append(v)
    arr = np.array([agg[k] for k in MKEYS])           # [models, axes]
    norm = arr / np.clip(arr.max(axis=0), 1e-9, None)  # best per axis = 1
    ang = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(4.4, 4.8)); ax = fig.add_subplot(111, polar=True)
    for i, k in enumerate(MKEYS):
        vals = norm[i].tolist(); vals += vals[:1]
        ax.plot(ang, vals, marker=MARKER[k], ms=4, lw=1.5, label=MNAME[k], color=COLOR[k])
        ax.fill(ang, vals, alpha=0.08, color=COLOR[k])
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_yticklabels([])
    fig.suptitle("Aggregate profile (each axis normalised; best = 1)", fontsize=9, y=0.99)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    save(fig, "radar_4way")


radar()


def radar_grid():
    """2x2 small-multiples: one radar per model (the other three shown faint grey for
    context), so each model's profile is readable without the 4-way overlap."""
    axes = ["accuracy", "f1_macro", "mcc", "DR", "FAR"]
    labels = ["Accuracy", "Macro-F1", "MCC", "Detection", "Low FAR"]
    agg = {k: [] for k in MKEYS}
    for k in MKEYS:
        for met in axes:
            vals = [DATA[ds][k][met][0] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k][met] and DATA[ds][k][met][0] is not None]
            v = float(np.mean(vals)) if vals else 0.0
            if met == "FAR":
                v = 1.0 - v
            agg[k].append(v)
    arr = np.array([agg[k] for k in MKEYS])
    norm = arr / np.clip(arr.max(axis=0), 1e-9, None)        # best per axis = 1
    ang = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist(); ang += ang[:1]
    nm = {k: norm[i].tolist() + [norm[i][0]] for i, k in enumerate(MKEYS)}
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.8), subplot_kw=dict(polar=True))
    for ax, k in zip(axs.ravel(), MKEYS):
        for j in MKEYS:                                       # faint context = other models
            if j != k:
                ax.plot(ang, nm[j], lw=0.8, color="0.78", alpha=0.8, zorder=1)
        ax.plot(ang, nm[k], marker=MARKER[k], ms=4, lw=2.2, color=COLOR[k], zorder=3)
        ax.fill(ang, nm[k], alpha=0.25, color=COLOR[k], zorder=2)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_yticklabels([])
        ax.set_title(MNAME[k], color=COLOR[k], fontsize=11, fontweight="bold", pad=14)
    fig.suptitle("Per-model aggregate profile (each axis normalised, best = 1; "
                 "grey = the other three models)", fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "radar_grid_4way")


radar_grid()


def _tol_blue_cmap():
    """White -> IEEE-scheme blue (#4477AA, the 4-way palette blue) sequential map,
    so heatmaps share the bright Tol scheme of the categorical figures."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("tol_blue", ["#FFFFFF", "#4477AA", "#26456E"])


def heatmap(met, fname):
    getter, hib, pretty = METRICS[met]
    mat = np.full((len(DS_KEYS), len(MKEYS)), np.nan)
    for i, ds in enumerate(DS_KEYS):
        for j, k in enumerate(MKEYS):
            if DATA[ds][k] and DATA[ds][k][met] and DATA[ds][k][met][0] is not None:
                mat[i, j] = DATA[ds][k][met][0] * 100
    fig, ax = plt.subplots(figsize=(4.2, 5.0))
    # darker blue = better everywhere: normal map when higher is better, reversed otherwise
    base = _tol_blue_cmap()
    cmap = base if hib else base.reversed()
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(MKEYS))); ax.set_xticklabels([MNAME[k] for k in MKEYS], rotation=30, ha="right")
    ax.set_yticks(range(len(DS_KEYS))); ax.set_yticklabels([DS_NAME[d] for d in DS_KEYS])
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    span = (vmax - vmin) or 1.0
    for i in range(len(DS_KEYS)):
        for j in range(len(MKEYS)):
            if mat[i, j] == mat[i, j]:
                frac = (mat[i, j] - vmin) / span          # 0..1 along the data range
                dark = frac if hib else 1.0 - frac        # fraction into the dark end
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if dark > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"{pretty} (%)")
    ax.set_title(f"{pretty} - dataset $\\times$ model")
    save(fig, fname)


for _met in ["accuracy", "f1_macro", "mcc", "DR", "FAR"]:
    heatmap(_met, f"heatmap_{_met}")


def energy_heatmap(fname="heatmap_energy"):
    """Compute-energy (pJ/sample) heatmap, dataset x model - same layout as the
    per-metric heatmaps but on a LOG color scale, because energy spans ~10^3
    (spiking students) to ~10^8 pJ (TASNN transformer) and a linear scale would
    collapse every spiking cell into one colour. Lower is better -> darker blue
    (reversed white->Tol-blue map, matching the other heatmaps)."""
    from matplotlib.colors import LogNorm

    def _fmt(v):
        if v >= 1e7:  return f"{v/1e6:.0f}M"
        if v >= 1e6:  return f"{v/1e6:.1f}M"
        if v >= 1e4:  return f"{v/1e3:.0f}k"
        if v >= 1e3:  return f"{v/1e3:.1f}k"
        return f"{v:.0f}"

    mat = np.full((len(DS_KEYS), len(MKEYS)), np.nan)
    for i, ds in enumerate(DS_KEYS):
        for j, k in enumerate(MKEYS):
            if DATA[ds][k] and DATA[ds][k].get("E_pj") is not None:
                mat[i, j] = DATA[ds][k]["E_pj"]
    fig, ax = plt.subplots(figsize=(4.2, 5.0))
    vmin = max(np.nanmin(mat), 1e-9); vmax = np.nanmax(mat)
    im = ax.imshow(mat, aspect="auto", cmap=_tol_blue_cmap().reversed(),
                   norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xticks(range(len(MKEYS))); ax.set_xticklabels([MNAME[k] for k in MKEYS], rotation=30, ha="right")
    ax.set_yticks(range(len(DS_KEYS))); ax.set_yticklabels([DS_NAME[d] for d in DS_KEYS])
    lo, hi = np.log10(vmin), np.log10(vmax)
    for i in range(len(DS_KEYS)):
        for j in range(len(MKEYS)):
            if mat[i, j] == mat[i, j]:
                # reversed map: low energy -> dark blue (white text); high -> near-white (black text)
                dark = 1.0 - (np.log10(mat[i, j]) - lo) / ((hi - lo) or 1.0)
                ax.text(j, i, _fmt(mat[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if dark > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Compute energy (pJ/sample, log)")
    ax.set_title("Compute energy - dataset $\\times$ model\n(lower is better; TASNN$^{\\dagger}$ analytic)")
    save(fig, fname)


energy_heatmap()


def pareto():
    """Mean Macro-F1 vs mean parameters (log) - upper-left = better."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for k in MKEYS:
        f1 = [DATA[ds][k]["f1_macro"][0] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["f1_macro"] and DATA[ds][k]["f1_macro"][0] is not None]
        pr = [DATA[ds][k]["params"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["params"]]
        if not f1 or not pr:
            continue
        ax.scatter(np.mean(pr), np.mean(f1) * 100, s=110, marker=MARKER[k],
                   color=COLOR[k], edgecolor="black", zorder=3, label=MNAME[k])
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log) - fewer is better")
    ax.set_ylabel("Macro-F1 (%, mean over datasets)")
    ax.set_title("Accuracy-size Pareto (upper-left = better)")
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    save(fig, "pareto_f1_vs_params")


pareto()


def energy_bar():
    """Compute energy pJ/sample for the parameter-matched iso-trio (log)."""
    trio = ["vanilla", "tree", "dnn"]
    vals = {}
    for k in trio:
        es = [DATA[ds][k]["E_pj"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["E_pj"] is not None]
        vals[k] = float(np.mean(es)) if es else np.nan
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    xs = range(len(trio))
    ax.bar(xs, [vals[k] for k in trio], color=[COLOR[k] for k in trio],
           edgecolor="black", width=0.6)
    ax.set_yscale("log"); ax.set_xticks(list(xs)); ax.set_xticklabels([MNAME[k] for k in trio])
    ax.set_ylabel("Compute energy (pJ/sample, log)")
    ax.set_title("Operation energy - iso-architecture trio\n(SOPs$\\times$0.9pJ vs MACs$\\times$4.6pJ)")
    if vals["dnn"] == vals["dnn"] and vals["tree"]:
        ax.text(0.5, 0.95, f"Tree-to-Spike: {vals['dnn']/vals['tree']:.0f}$\\times$ less than DNN-Tree",
                transform=ax.transAxes, ha="center", va="top", fontsize=8)
    save(fig, "energy_iso_trio")


energy_bar()


def energy_4way():
    """All 4 models on one log energy axis (TASNN via analytic FLOPs)."""
    vals = {}
    for k in MKEYS:
        es = [DATA[ds][k]["E_pj"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["E_pj"] is not None]
        vals[k] = float(np.mean(es)) if es else np.nan
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    xs = range(len(MKEYS))
    ax.bar(xs, [vals[k] for k in MKEYS], color=[COLOR[k] for k in MKEYS],
           edgecolor="black", width=0.62)
    ax.set_yscale("log"); ax.set_xticks(list(xs))
    ax.set_xticklabels([MNAME[k] for k in MKEYS], rotation=15)
    ax.set_ylabel("Compute energy (pJ/sample, log)")
    ax.set_title("Compute energy per inference\n(TASNN$^{\\dagger}$ = analytic transformer+spike FLOPs)")
    for i, k in enumerate(MKEYS):
        if vals[k] == vals[k]:
            ax.text(i, vals[k], f"{vals[k]:,.0f}", ha="center", va="bottom", fontsize=7)
    save(fig, "energy_4way")


def pareto_energy():
    """Mean Macro-F1 vs mean compute energy (log) - upper-left = better."""
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for k in MKEYS:
        f1 = [DATA[ds][k]["f1_macro"][0] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["f1_macro"] and DATA[ds][k]["f1_macro"][0] is not None]
        en = [DATA[ds][k]["E_pj"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["E_pj"] is not None]
        if not f1 or not en:
            continue
        ax.scatter(np.mean(en), np.mean(f1) * 100, s=110, marker=MARKER[k],
                   color=COLOR[k], edgecolor="black", zorder=3, label=MNAME[k])
    ax.set_xscale("log")
    ax.set_xlabel("Compute energy (pJ/sample, log) - lower is better")
    ax.set_ylabel("Macro-F1 (%, mean over datasets)")
    ax.set_title("Accuracy-energy Pareto (upper-left = better)")
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    save(fig, "pareto_f1_vs_energy")


energy_4way()
pareto_energy()


def energy_precision():
    """B2: 32-bit FP vs int8 compute energy per model (matched precision), log scale."""
    e32, ei8 = {}, {}
    for k in MKEYS:
        a = [DATA[ds][k]["E_pj"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["E_pj"] is not None]
        b = [DATA[ds][k]["E_pj_i8"] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k].get("E_pj_i8") is not None]
        e32[k] = float(np.mean(a)) if a else np.nan
        ei8[k] = float(np.mean(b)) if b else np.nan
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(MKEYS)); w = 0.38
    ax.bar(x - w / 2, [e32[k] for k in MKEYS], w, label="32-bit FP",
           color="#9e9e9e", edgecolor="black")
    ax.bar(x + w / 2, [ei8[k] for k in MKEYS], w, label="int8",
           color=[COLOR[k] for k in MKEYS], edgecolor="black")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([MNAME[k] for k in MKEYS], rotation=15)
    ax.set_ylabel("Compute energy (pJ/sample, log)")
    ax.set_title("Matched-precision compute energy (32-bit vs int8)")
    ax.legend(frameon=False, ncol=2)
    save(fig, "energy_precision")


energy_precision()


# ---- critical-difference / mean-rank across datasets ----
_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


def _avg_ranks(met):
    """Average rank per model across datasets where all models are present (1=best)."""
    hib = METRICS[met][1]
    ranks = {k: [] for k in MKEYS}
    N = 0
    for ds in DS_KEYS:
        vals = {k: (DATA[ds][k][met][0] if DATA[ds][k] and DATA[ds][k][met]
                    and DATA[ds][k][met][0] is not None else None) for k in MKEYS}
        if any(v is None for v in vals.values()):
            continue
        N += 1
        arr = np.array([vals[k] for k in MKEYS], float)
        s = -arr if hib else arr                      # smaller s = better
        order = np.argsort(s, kind="mergesort")
        rk = np.empty(len(MKEYS)); i = 0
        while i < len(MKEYS):                          # tie-aware average ranks
            j = i
            while j + 1 < len(MKEYS) and s[order[j + 1]] == s[order[i]]:
                j += 1
            r = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                rk[order[t]] = r
            i = j + 1
        for idx, k in enumerate(MKEYS):
            ranks[k].append(rk[idx])
    avg = {k: float(np.mean(ranks[k])) for k in MKEYS if ranks[k]}
    km = len(MKEYS)
    cd = _NEMENYI_Q05.get(km, 2.569) * np.sqrt(km * (km + 1) / (6.0 * max(N, 1)))
    return avg, cd, N


def mean_rank_bar(met, fname):
    avg, cd, N = _avg_ranks(met)
    order = sorted(avg, key=avg.get)                   # best first
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ys = list(range(len(order)))
    ax.barh(ys, [avg[k] for k in order], color=[COLOR[k] for k in order],
            edgecolor="black")
    ax.set_yticks(ys); ax.set_yticklabels([MNAME[k] for k in order]); ax.invert_yaxis()
    ax.set_xlabel("Mean rank across datasets (1 = best)")
    ax.set_title(f"{METRICS[met][2]} - mean rank (N={N}, CD={cd:.2f})")
    for i, k in enumerate(order):
        ax.text(avg[k] + 0.03, i, f"{avg[k]:.2f}", va="center", fontsize=8)
    ax.set_xlim(1, len(MKEYS)); ax.grid(axis="x", alpha=0.3)
    save(fig, fname)


def cd_diagram(met, fname):
    avg, cd, N = _avg_ranks(met)
    lo, hi = 1, len(MKEYS)
    fig, ax = plt.subplots(figsize=(6.2, 2.3)); ax.set_xlim(lo - 0.4, hi + 0.4)
    ax.set_ylim(0, 1.7); ax.axis("off")
    ax.plot([lo, hi], [1.3, 1.3], "k-", lw=1.2)
    # ranks are drawn best-to-the-RIGHT (xpos below), so the tick LABELS must be
    # reversed too; otherwise a rank-1.7 model sits under a tick reading "3".
    for r in range(lo, hi + 1):
        ax.plot([r, r], [1.3, 1.37], "k-", lw=1.0)
        ax.text(r, 1.43, str(hi - (r - lo)), ha="center", fontsize=8)
    ax.text((lo + hi) / 2, 1.55, "average rank (1 = best, right)", ha="center", fontsize=8)
    xpos = lambda r: hi - (r - lo)                     # best rank -> right
    items = sorted(avg.items(), key=lambda kv: kv[1])  # best first
    n = len(items); right = items[: (n + 1) // 2]; left = items[(n + 1) // 2:]
    for i, (kk, r) in enumerate(right):
        y = 1.12 - i * 0.30
        ax.plot([xpos(r), xpos(r)], [1.3, y], "k-", lw=0.8)
        ax.plot([xpos(r), hi + 0.25], [y, y], "k-", lw=0.8)
        ax.text(hi + 0.3, y, f"{MNAME[kk]} ({r:.2f})", va="center", ha="left", fontsize=8, color=COLOR[kk])
    for i, (kk, r) in enumerate(left):
        y = 1.12 - i * 0.30
        ax.plot([xpos(r), xpos(r)], [1.3, y], "k-", lw=0.8)
        ax.plot([xpos(r), lo - 0.25], [y, y], "k-", lw=0.8)
        ax.text(lo - 0.3, y, f"{MNAME[kk]} ({r:.2f})", va="center", ha="right", fontsize=8, color=COLOR[kk])
    # CD reference bar
    ax.plot([hi, hi - cd], [0.30, 0.30], "k-", lw=2.5)
    ax.text(hi - cd / 2, 0.40, f"CD = {cd:.2f}", ha="center", fontsize=8)
    # cliques: maximal groups whose rank span < CD (not significantly different)
    rs = [r for _, r in items]
    groups = []
    for i in range(len(rs)):
        j = i
        while j + 1 < len(rs) and rs[j + 1] - rs[i] < cd:
            j += 1
        if j > i:
            groups.append((rs[i], rs[j]))
    groups = [g for g in set(groups)
              if not any(h != g and h[0] <= g[0] and g[1] <= h[1] for h in groups)]
    ylev = 1.0
    for a, b in sorted(groups):
        ax.plot([xpos(a) + 0.04, xpos(b) - 0.04], [ylev, ylev], "k-", lw=3.5, solid_capstyle="round")
        ylev -= 0.07
    ax.set_title(f"Critical-difference diagram - {METRICS[met][2]} "
                 f"(Nemenyi $\\alpha$=0.05, N={N})", fontsize=9)
    save(fig, fname)


for _met in ["accuracy", "f1_macro", "mcc"]:
    mean_rank_bar(_met, f"meanrank_{_met}")
cd_diagram("f1_macro", "cd_diagram_f1_macro")
cd_diagram("mcc", "cd_diagram_mcc")

# ---- console summary
print(f"[benchmark_4way] wrote tables -> {TAB}")
for p in sorted(TAB.glob("*.md")):
    print(f"   {p.name}")
print(f"[benchmark_4way] wrote figures -> {FIG}")
for p in sorted(FIG.glob("*.pdf")):
    print(f"   {p.name}")
print("\nMean Macro-F1 across datasets:")
for k in MKEYS:
    vals = [DATA[ds][k]["f1_macro"][0] for ds in DS_KEYS if DATA[ds][k] and DATA[ds][k]["f1_macro"] and DATA[ds][k]["f1_macro"][0] is not None]
    print(f"   {MNAME[k]:16} {np.mean(vals)*100:.1f}%  (n_datasets={len(vals)})")

print("\nB2 matched-precision compute energy - DNN-Tree / Tree-to-Spike ratio:")
for prec, key in [("32-bit", "E_pj"), ("int8", "E_pj_i8")]:
    dnn = np.mean([DATA[ds]["dnn"][key] for ds in DS_KEYS if DATA[ds]["dnn"] and DATA[ds]["dnn"].get(key) is not None])
    tree = np.mean([DATA[ds]["tree"][key] for ds in DS_KEYS if DATA[ds]["tree"] and DATA[ds]["tree"].get(key) is not None])
    print(f"   {prec:7} DNN-Tree={dnn:>10,.0f} pJ   Tree-to-Spike={tree:>9,.0f} pJ   -> {dnn/tree:.1f}x less")
