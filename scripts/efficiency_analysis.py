#!/usr/bin/env python3
"""Post-hoc energy / efficiency analysis — the "four green additions".

Reads every model family's ``results.json`` (no retraining) and computes, per run:

  1. Horowitz inference energy   E_SNN  = N_AC  x 0.9 pJ          (AC = accumulate)
  2. Iso-architecture MAC count  N_MAC  = same net run densely    (-> E_ANN = N_MAC x 4.6 pJ)
  3. Energy-efficiency gain      alpha  = E_ANN / E_SNN           (SNN vs its dense-ANN twin)
  4. Energy-Latency Product      ELP    = E_SNN x inference_ms ;  FPS = 1000 / inference_ms

Constants are the Horowitz 45 nm CMOS model (MAC = 4.6 pJ, AC = 0.9 pJ), the
standard SNN-energy proxy. N_AC is the run's ``spike_stats.synops_per_sample``
(total synaptic operations over all layers). The encoders here feed *binary
spikes* into the first linear layer, so that layer is AC (not an analog-input MAC
layer) and no analog-encoding correction is applied.

Outputs (under results/efficiency/):
  * efficiency_metrics.csv       — one row per run
  * efficiency_summary.md        — mean +/- std per (model, dataset, variant)
  * pareto_energy_vs_accuracy.{pdf,png} — Horowitz energy vs accuracy frontier

Honest scope: energy/alpha require ``synops_per_sample`` (total SOPs) AND a MAC
formula for the architecture. Both are currently available for the **SNN
ablation** (synops backfilled, uniform MLP MAC formula). Hybrid families get
MACs/energy once they record total SOPs and gain a MAC formula (see TODO.md
item 1). Runs lacking either get FPS only, with energy columns left blank.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Horowitz 45 nm CMOS energy per operation (picojoules)
E_MAC_PJ = 4.6
E_AC_PJ = 0.9

OUT_DIR = REPO_ROOT / "results" / "efficiency"


# ----------------------------------------------------------------------------
# Iso-architecture dense-ANN MAC count per sample (the same topology run densely)
# ----------------------------------------------------------------------------
def iso_ann_macs(record: dict):
    """Return iso-architecture dense-ANN MACs/sample, or None if no formula yet.

    The reference is the SAME F->H->C topology run as a non-spiking ANN in a
    SINGLE forward pass (NO T factor): a standard ANN processes the input once,
    whereas the T timesteps are an SNN artifact and the SNN's SOP count already
    includes firing over T (``SOPs = f_r * T * FLOPs``). So:
        N_MAC = F*H + H*C        (single dense pass)
    Recurrent/conv gate MACs are NOT added, so alpha is *conservative* (an
    underestimate of the true gain) for SLSTM/SConv2dLSTM/recurrent neurons —
    documented in TODO.md / the paper.
    """
    model = record.get("model", "SNN")
    cfg = record.get("config", {})
    data = record.get("data", {})
    F = data.get("num_features")
    C = data.get("num_classes")
    if model == "SNN":
        H = cfg.get("hidden_size")
        if None in (F, C, H):
            return None
        return float(F * H + H * C)
    # Hybrid MAC formulas: TODO (see TODO.md item 1). Energy/alpha pending.
    return None


def _get(record, *path, default=None):
    cur = record
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def analyse_run(rj_path: Path) -> dict:
    r = json.load(open(rj_path))
    model = r.get("model", "SNN")
    cfg = r.get("config", {})
    tp = r.get("test_plus", {})
    m = tp.get("metrics", {})
    acc = m.get("accuracy")
    f1m = m.get("f1_macro")
    infer_ms = tp.get("inference_ms_per_sample")
    sops = _get(tp, "spike_stats", "synops_per_sample")
    macs = iso_ann_macs(r)

    e_snn = (sops * E_AC_PJ) if sops is not None else None
    e_ann = (macs * E_MAC_PJ) if macs is not None else None
    alpha = (e_ann / e_snn) if (e_snn not in (None, 0) and e_ann is not None) else None
    fps = (1000.0 / infer_ms) if infer_ms not in (None, 0) else None
    elp = (e_snn * infer_ms) if (e_snn is not None and infer_ms is not None) else None

    # T may be stored under different keys across families
    T = cfg.get("num_steps", cfg.get("seq_len", cfg.get("T")))
    return {
        "model": model,
        "dataset": r.get("dataset"),
        "neuron": r.get("neuron"),
        "encoding": r.get("encoding"),
        "seed": cfg.get("seed"),
        "accuracy": acc,
        "f1_macro": f1m,
        "num_steps": T,
        "sops_per_sample": sops,
        "macs_iso_ann": macs,
        "energy_snn_pj": e_snn,
        "energy_ann_pj": e_ann,
        "alpha_gain": alpha,
        "inference_ms_per_sample": infer_ms,
        "fps": fps,
        "elp_pj_ms": elp,
    }


def collect(only_models, only_datasets):
    """Walk every results.json under results/ (SNN ablation + hybrid trees)."""
    rows = []
    patterns = [
        "results/*/SNN_*/*/seed_*/results.json",         # SNN ablation
        "results/S-xLSTM/*/seed_*/results.json",
        "results/Spiking-Griffin/*/seed_*/results.json",
        "results/Spiking-SSM/*/seed_*/results.json",
        "results/Tree-to-Spike/*/seed_*/results.json",
    ]
    seen = set()
    for pat in patterns:
        for f in glob.glob(str(REPO_ROOT / pat)):
            if f in seen:
                continue
            seen.add(f)
            try:
                row = analyse_run(Path(f))
            except Exception as e:
                print(f"  warn: skip {f}: {e}")
                continue
            if only_models and row["model"] not in only_models:
                continue
            if only_datasets and row["dataset"] not in only_datasets:
                continue
            rows.append(row)
    return rows


def write_csv(rows, out_csv):
    cols = ["model", "dataset", "neuron", "encoding", "seed", "accuracy", "f1_macro",
            "num_steps", "sops_per_sample", "macs_iso_ann", "energy_snn_pj",
            "energy_ann_pj", "alpha_gain", "inference_ms_per_sample", "fps", "elp_pj_ms"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["dataset"] or "",
                                             r["neuron"] or "", r["encoding"] or "",
                                             r["seed"] or 0)):
            w.writerow(r)


def _agg(rows, key_fields, metric):
    """mean over seeds of `metric`, grouped by key_fields tuple."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is None:
            continue
        buckets[tuple(r[k] for k in key_fields)].append(v)
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in buckets.items()}


def write_summary_md(rows, out_md):
    have_energy = [r for r in rows if r["energy_snn_pj"] is not None]
    lines = ["# Energy / efficiency summary (Horowitz 45 nm: MAC=4.6 pJ, AC=0.9 pJ)\n"]
    lines.append(f"Runs analysed: {len(rows)}  |  with energy (synops present): "
                 f"{len(have_energy)}  |  FPS-only (energy pending): {len(rows)-len(have_energy)}\n")
    if have_energy:
        agg_acc = _agg(have_energy, ("model", "dataset", "neuron", "encoding"), "accuracy")
        agg_e = _agg(have_energy, ("model", "dataset", "neuron", "encoding"), "energy_snn_pj")
        agg_a = _agg(have_energy, ("model", "dataset", "neuron", "encoding"), "alpha_gain")
        lines.append("\n## Per-variant means (energy-capable runs)\n")
        lines.append("| model | dataset | neuron | encoding | acc | E_SNN (pJ/sample) | alpha (x) | n |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for k in sorted(agg_acc, key=lambda k: (k[1] or "", -(agg_a.get(k, (0,))[0]))):
            acc = agg_acc[k][0]; e = agg_e.get(k, (float('nan'),))[0]
            a = agg_a.get(k, (float('nan'),))[0]; n = agg_acc[k][2]
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {acc:.4f} | {e:,.1f} | {a:.1f} | {n} |")
    out_md.write_text("\n".join(lines) + "\n")


def plot_pareto(rows, out_path):
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from src.plotting import apply_ieee_style, WONG_PALETTE, save_figure
    import matplotlib.pyplot as plt
    apply_ieee_style()

    pts = [r for r in rows if r["energy_snn_pj"] and r["accuracy"] is not None]
    if not pts:
        print("  (no energy-capable runs yet — skipping Pareto figure)")
        return
    datasets = sorted({r["dataset"] for r in pts})
    enc_color = {"rate": WONG_PALETTE[1], "latency": WONG_PALETTE[2],
                 "delta": WONG_PALETTE[3]}
    enc_marker = {"rate": "o", "latency": "s", "delta": "^"}

    n = len(datasets)
    ncols = min(n, 2); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows),
                             squeeze=False)
    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        sub = [r for r in pts if r["dataset"] == ds]
        for enc in ["rate", "latency", "delta"]:
            e = [r for r in sub if r["encoding"] == enc]
            if not e:
                continue
            ax.scatter([r["energy_snn_pj"] for r in e], [r["accuracy"] for r in e],
                       c=enc_color.get(enc, "#888"), marker=enc_marker.get(enc, "o"),
                       edgecolor="black", linewidth=0.3, s=28, alpha=0.8, label=enc)
        # any runs with a non-standard encoding (hybrids)
        other = [r for r in sub if r["encoding"] not in enc_color]
        if other:
            ax.scatter([r["energy_snn_pj"] for r in other], [r["accuracy"] for r in other],
                       c="#444", marker="D", edgecolor="black", linewidth=0.3, s=30,
                       alpha=0.85, label="hybrid")
        ax.set_xscale("log")
        ax.set_xlabel("Horowitz energy (pJ/sample, log)", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.set_title(ds, fontsize=9)
        ax.legend(fontsize=6, frameon=False)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Accuracy vs Horowitz inference energy — upper-left is best",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    save_figure(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[],
                    help="restrict to these model names (e.g. SNN S-xLSTM)")
    ap.add_argument("--datasets", nargs="+", default=[],
                    help="restrict to these datasets")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect(set(args.models), set(args.datasets))
    if not rows:
        print("no results.json found (has anything been run / backfilled yet?)")
        return
    write_csv(rows, OUT_DIR / "efficiency_metrics.csv")
    write_summary_md(rows, OUT_DIR / "efficiency_summary.md")
    plot_pareto(rows, OUT_DIR / "pareto_energy_vs_accuracy")

    have_e = sum(1 for r in rows if r["energy_snn_pj"] is not None)
    print(f"analysed {len(rows)} runs ({have_e} energy-capable, {len(rows)-have_e} FPS-only)")
    print(f"  -> {OUT_DIR}/efficiency_metrics.csv")
    print(f"  -> {OUT_DIR}/efficiency_summary.md")
    print(f"  -> {OUT_DIR}/pareto_energy_vs_accuracy.(pdf|png)")


if __name__ == "__main__":
    main()
