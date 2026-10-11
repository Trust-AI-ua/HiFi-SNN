#!/usr/bin/env python3
"""4-way architecture/size/spikes table -> results/Journal_results/tables/table_architecture.{tex,md}.

Per dataset x model: class count, network architecture (layer widths), total
parameter count, hidden-layer spikes per sample, encoder mean time-to-first-spike
(TTFS), mean early-exit timestep, and relative energy vs the Tree-to-Spike
baseline (all mean +/- std over the 5 seeds where applicable). DNN-Tree is dense
(ReLU twin): its "spikes" cell reports activated ReLU neurons (marked), and it
has no TTFS/early-exit. TASNN has neither a TTFS input code nor early exit.
Architecture strings use the seed-42 encoder channel count (the tree-guided
encoder width varies by a few channels across seeds because the teacher refit
differs; params show the across-seed mean for the same reason). Energy reuses
cell() from make_results_table so the two tables can never disagree.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from make_results_table import cell as _energy_cell   # shared energy accounting

TAB = REPO / "results" / "Journal_results" / "tables"

DS_GENERAL = [("kddcup99", "KDDCup99"), ("nslkdd", "NSL-KDD"), ("ctu13", "CTU-13"),
              ("iscxids2012", "ISCX2012"), ("cicids2017", "CIC-IDS2017"),
              ("cse-cic-ids2018", "CSE-CIC-2018"), ("unswnb15", "UNSW-NB15")]
DS_IOT = [("ciciot2023", "CICIoT2023"), ("ciciomt2024", "CICIoMT2024"),
          ("edgeiiot", "Edge-IIoTset"), ("toniot", "TON-IoT")]
DS = DS_GENERAL + DS_IOT
GROUPS = [(DS_IOT, "iot", "IoT, IIoT, and IoMT"),
          (DS_GENERAL, "general", "general-purpose network"),
          (DS, "full", "all")]
MODELS = [("vanilla", "Vanilla SNN", "results/{d}/SNN_LeakyParallel/latency/seed_*/results.json"),
          ("tree", "Tree-to-Spike", "results/Tree-to-Spike/{d}/seed_*/results.json"),
          ("dnn", "DNN-Tree", "results/DNN-Tree/{d}/seed_*/results.json"),
          ("tasnn", "TASNN", "results/TASNN/{d}/seed_*/results.json")]


def _recs(patt):
    return [json.load(open(f)) for f in sorted(glob.glob(str(REPO / patt)))]


# Frozen tree-guided encoder tables (thresholds + channel map) that must ship
# with the deployed artifact. Tree-to-Spike AND its DNN-Tree twin both consume
# this encoder; Vanilla SNN (closed-form latency code) and TASNN (internal
# encoding) store no tables. Measured by scripts/measure_encoder_storage.py.
_ENC_PATH = REPO / "results" / "Journal_results" / "encoder_storage.json"
_ENC = json.load(open(_ENC_PATH)) if _ENC_PATH.exists() else {}


def _encoder_bytes(mk, dk):
    return float(_ENC.get(dk, {}).get("bytes", 0)) if mk in ("tree", "dnn") else 0.0


def _arch(mk, r):
    """Human-readable layer-width string from one (seed-42) record."""
    c, F, K = r["config"], r["data"]["num_features"], r["data"]["num_classes"]
    if mk == "vanilla":
        return f"{F} - {c['hidden_size']} - {K} (LIF, T={c['num_steps']})"
    if mk in ("tree", "dnn"):
        C = r["data"]["encoder_channels"]; h = c["hidden"]
        kind = "LIF, T=" + str(c["T"]) if mk == "tree" else "ReLU"
        return f"{C} - {h[0]} - {h[1]} - {K} ({kind})"
    # TASNN: transformer front-end + STC + LIF core
    h1, h2 = c["lif_hidden"]
    return (f"{F} - Tr(d={c['d_model']}$\\times${c['n_layers']}) - "
            f"STC{c['stc_channels']} - {h1} - {h2} - {K} (T={c['num_steps']})")


def _arch_md(mk, r):
    return _arch(mk, r).replace("$\\times$", "x")


def _ms(recs, getter):
    """(mean, std) of getter over records; None if absent everywhere."""
    vs = []
    for r in recs:
        try:
            v = getter(r)
        except (KeyError, TypeError):
            v = None
        if isinstance(v, (int, float)):
            vs.append(v)
    return (float(np.mean(vs)), float(np.std(vs, ddof=1))) if vs else (None, None)   # sample SD


def build(DS=DS, suffix='full', group='all'):
    rows = []
    for dk, dn in DS:
        recs_by = {mk: _recs(patt.format(d=dk)) for mk, _, patt in MODELS}
        # Rel.E baseline: Tree-to-Spike energy from the shared accounting
        e = {mk: (_energy_cell(mk, recs_by[mk]) or {}).get("energy") for mk, _, _ in MODELS}
        base = e.get("tree")
        for mk, mn, patt in MODELS:
            recs = recs_by[mk]
            if not recs:
                continue
            r0 = recs[0]  # sorted -> seed_42
            K = r0["data"]["num_classes"]
            params = float(np.mean([r["model_size"]["total_parameters"] for r in recs]))
            if mk == "dnn":
                # dense twin: no spikes; report ACTIVATED ReLU neurons per sample instead
                spk = _ms(recs, lambda r: r["test_plus"]["activation_stats"]["active_neurons_per_sample"])
            else:
                spk = _ms(recs, lambda r: r["test_plus"]["spike_stats"]["energy_spikes_per_sample"])
            ttfs = _ms(recs, lambda r: r["test_plus"]["spike_stats"]["mean_spike_time"])
            ttfs_T = next((r["test_plus"]["spike_stats"].get("ttfs_T") for r in recs
                           if r["test_plus"].get("spike_stats", {}).get("ttfs_T")), None)
            ee = _ms(recs, lambda r: r["test_plus"]["early_exit"]["mean_exit_step"])
            ee_T = next((r["test_plus"]["early_exit"].get("max_steps") for r in recs
                         if r["test_plus"].get("early_exit")), None)
            efac = (e[mk] / base) if (base and e.get(mk)) else None
            rows.append((dn, dk, K, mk, mn, _arch(mk, r0), _arch_md(mk, r0), params,
                         spk, (ttfs, ttfs_T), (ee, ee_T), efac))

    # ---------- LaTeX ----------
    L = [
        r"% Requires: \usepackage{multirow,booktabs}",
        r"\begin{table*}[!ht]", r"\centering", r"\renewcommand{\arraystretch}{1.05}",
        rf"\caption{{4-way model anatomy per dataset over the {group} intrusion detection "
        r"datasets: class count, network architecture "
        r"(layer widths), total parameters, and hidden-layer spikes generated per sample "
        r"(mean$_{\pm\text{std}}$ over 5 seeds). DNN-Tree is the dense iso-architecture "
        r"twin of Tree-to-Spike (identical widths and parameters, ReLU instead of LIF), "
        r"so it emits no spikes. Architecture strings use the seed-42 tree-guided encoder "
        r"width; the encoder (and hence Tree-to-Spike/DNN-Tree input width and parameter "
        r"count) varies slightly across seeds with the teacher refit, so parameters are "
        r"the across-seed mean. TASNN: Tr = transformer encoder (d\_model$\times$layers), "
        r"STC = spatio-temporal conv channels, then the LIF core. "
        r"$^\ddagger$For the dense DNN-Tree the cell reports \emph{activated neurons} per "
        r"sample (nonzero hidden ReLU outputs, measured on the test set of the "
        r"int8-quantised model): a ReLU neuron does work once per input, whereas a spiking "
        r"neuron does work once per emitted spike, so the two counts are the per-paradigm "
        r"analogs but not interchangeable (spikes accumulate over $T$ timesteps). "
        r"\textbf{TTFS} = mean encoder time-to-first-spike over spiking channels $\times$ "
        r"test samples, shown as steps$/T$ (Vanilla SNN: latency code over raw features, "
        r"sub-threshold inputs never fire; Tree-to-Spike: tree-guided code; DNN-Tree and "
        r"TASNN have no TTFS input code). Note TTFS is the \emph{mean arrival time of all "
        r"emitted spikes} (the code's centre of mass), not the first-arrival time: under "
        r"latency coding the most informative (highest-level) channels fire first, and the "
        r"first spike arrives at $t \approx 0$ for virtually every sample, which is why the "
        r"early-exit step can be far smaller than the mean TTFS. \textbf{Exit} = mean confidence-triggered "
        r"early-exit timestep$/T$ (DNN-Tree is one-shot; TASNN has no early exit). "
        r"\textbf{Rel.\,E} = compute energy relative to Tree-to-Spike on the same dataset "
        r"($n\times$ = $n$ times our energy; $^\dagger$TASNN analytic). "
        r"\textbf{Storage} = projected deployment footprint at FP32\,/\,int8, in KB: "
        r"model weights (4 / 1 bytes per parameter) plus, for the models that use the "
        r"tree-guided encoder, the frozen encoder tables it must ship with "
        r"($^{\S}$Tree-to-Spike and its DNN-Tree twin: split thresholds at 4 B plus a "
        r"3 B channel-map entry each; the Vanilla SNN latency code and TASNN store no "
        r"tables). These are analytical projections from parameter and table counts, "
        r"not measured serialized artifacts: our int8 stage is a per-tensor fake-quant "
        r"stand-in that quantizes Linear weights while retaining fp32 tensors, so the "
        r"int8 column is a lower bound (biases and encoder thresholds remain fp32).}",
        rf"\label{{tab:architecture_{suffix}}}", r"{\scriptsize",
        r"\begin{tabular}{l c l l r r r r r r}", r"\toprule",
        r"\textbf{Dataset} & \textbf{Classes} & \textbf{Model} & "
        r"\textbf{Architecture (in - hidden - out)} & \textbf{Params} & "
        r"\textbf{Storage (KB)} & "
        r"\textbf{Spikes/sample} & \textbf{TTFS} & \textbf{Exit} & \textbf{Rel.\,E} \\",
        r"\midrule",
    ]
    md = ["### 4-way model anatomy - classes, architecture, parameters, spikes, timing, energy\n",
          "Mean +/- std over 5 seeds. DNN-Tree is the dense iso-architecture twin: it emits no "
          "spikes, so its cell reports **activated neurons** per sample (nonzero hidden ReLU "
          "outputs of the int8 model, marked `(act.)`) - the dense analog, but not "
          "interchangeable with spikes (spikes accumulate over T steps). **TTFS** = mean encoder "
          "time-to-first-spike (steps/T; spiking channels only). **Exit** = mean early-exit "
          "timestep (steps/T; DNN-Tree is one-shot, TASNN has no early exit). **Rel.E** = "
          "compute energy relative to Tree-to-Spike on the same dataset (n x = n times ours). "
          "Architecture uses the seed-42 encoder width; params are the across-seed mean (the "
          "tree-guided encoder width shifts by a few channels per seed).\n",
          "| Dataset | Classes | Model | Architecture (in - hidden - out) | Params | Storage KB (fp32/int8) | Spikes/sample | TTFS | Exit | Rel.E |",
          "|---|--:|---|---|--:|--:|--:|--:|--:|--:|"]

    def _frac(pair, tex=True):
        (m, s), T = pair
        if m is None or T is None:
            return "---" if tex else "--"
        return (rf"{m:.1f}$_{{\pm{s:.1f}}}$/{T}" if tex else f"{m:.1f} +/- {s:.1f} /{T}")

    cur = None
    for dn, dk, K, mk, mn, arch, arch_md, params, spk, ttfs, ee, efac in rows:
        p = f"{params/1000:.1f}K" if params >= 1000 else f"{params:.0f}"
        # deployment storage = weights (fp32 4 B / int8 1 B per parameter)
        # + frozen tree-guided encoder tables (tree and dnn only; fp32 thresholds)
        eb = _encoder_bytes(mk, dk)
        kb32, kb8 = (params * 4 + eb) / 1024, (params * 1 + eb) / 1024
        mark = r"$^{\S}$" if eb else ""
        store_tex = f"{kb32:,.1f} / {kb8:,.1f}{mark}"
        store_md = f"{kb32:,.1f} / {kb8:,.1f}" + (" (incl. enc.)" if eb else "")
        sm, ss = spk
        if sm is None:
            spk_tex, spk_md = "---", "---"
        elif mk == "dnn":   # activated ReLU neurons, not spikes (see caption)
            spk_tex = rf"{sm:,.0f}$_{{\pm{ss:.1f}}}$$^\ddagger$"
            spk_md = f"{sm:,.0f} +/- {ss:.1f} (act.)"
        else:
            spk_tex = rf"{sm:,.0f}$_{{\pm{ss:.1f}}}$"
            spk_md = f"{sm:,.0f} +/- {ss:.1f}"
        ttfs_tex, ttfs_md = _frac(ttfs), _frac(ttfs, tex=False)
        ee_tex, ee_md = _frac(ee), _frac(ee, tex=False)
        if efac is None:
            ef_tex, ef_md = "---", "--"
        else:
            s = f"{efac/1000:,.0f}k" if efac >= 1000 else (f"{efac:.1f}" if efac >= 10 else f"{efac:.2f}")
            dag = r"$^\dagger$" if mk == "tasnn" else ""
            ef_tex = rf"{s}$\times${dag}"
            ef_md = f"{s}x" + (" (analytic)" if mk == "tasnn" else "")
        cells_tex = f"{arch} & {p} & {store_tex} & {spk_tex} & {ttfs_tex} & {ee_tex} & {ef_tex}"
        if dn != cur:
            if cur is not None:
                L.append(r"\midrule")
            L.append(rf"\multirow{{4}}{{*}}{{{dn}}} & \multirow{{4}}{{*}}{{{K}}}")
            cur = dn
            L.append(rf"  & {mn:13} & {cells_tex} \\")
        else:
            L.append(rf"& & {mn:13} & {cells_tex} \\")
        md.append(f"| {dn} | {K} | {mn} | {arch_md} | {p} | {store_md} | {spk_md} | {ttfs_md} | {ee_md} | {ef_md} |")

    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    (TAB / f"table_architecture_{suffix}.tex").write_text("\n".join(L))
    (TAB / f"table_architecture_{suffix}.md").write_text("\n".join(md) + "\n")
    print(f"[wrote] {TAB}/table_architecture_{suffix}.tex + .md  ({len(rows)} rows)")


if __name__ == "__main__":
    for _ds, _sfx, _grp in GROUPS:
        build(_ds, _sfx, _grp)
