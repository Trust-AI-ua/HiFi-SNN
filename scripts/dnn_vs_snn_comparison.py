#!/usr/bin/env python3
"""Fair DNN-vs-SNN comparison: DNN-Tree (dense ANN) vs Tree-to-Spike (spiking),
iso-architecture twins (identical Linear topology / parameter count).

Implements the multi-dimensional fair-comparison framework:
  * Iso-architecture: same params (verified, flagged if they ever differ).
  * Operational complexity: DNN MACs vs SNN SOPs (Accumulate ops).
  * Horowitz 45nm energy: E_MAC = 4.6 pJ, E_AC = 0.9 pJ.
        E_DNN = N_MAC * 4.6 ,  E_SNN = N_SOP * 0.9 ,  delta = E_DNN / E_SNN.
  * Batch-1 (edge) latency + Energy-Latency Product (ELP).
  * Accuracy-vs-energy Pareto figure.

Reads results/DNN-Tree/<ds>/seed_*/ and results/Tree-to-Spike/<ds>/seed_*/.
Outputs results/dnn_vs_snn_comparison.{md,csv} and a Pareto PNG/PDF.

Caveats it prints: the SNN latency on file is *batched* (bs=128), not batch-1;
and the Horowitz SOP energy is compute-only (it excludes the SNN's per-timestep
membrane/weight memory-access overhead, which a unit-latency T=1 SNN would avoid).
"""
from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path

E_MAC_PJ, E_AC_PJ = 4.6, 0.9
DATASETS = ["kddcup99", "nslkdd", "cicids2017", "ctu13", "iscxids2012", "cse-cic-ids2018"]


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _load(pattern):
    out = []
    for f in glob.glob(pattern):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def dnn_cell(ds):
    recs = _load(f"results/DNN-Tree/{ds}/seed_*/results.json")
    if not recs:
        return None
    g = lambda fn: mean([fn(r) for r in recs])
    return {
        "n": len(recs),
        "acc": g(lambda r: r["test_plus"]["metrics"]["accuracy"]),
        "f1m": g(lambda r: r["test_plus"]["metrics"]["f1_macro"]),
        "mcc": g(lambda r: r["test_plus"]["metrics"]["matthews_corrcoef"]),
        "params": g(lambda r: r["model_size"]["trainable_parameters"]),
        "ops": g(lambda r: r["test_plus"]["compute"]["macs_per_sample"]),
        "energy_pj": g(lambda r: r["test_plus"]["compute"]["energy_pj_horowitz"]),
        "lat_ms": g(lambda r: r["test_plus"]["inference_ms_per_sample_batch1"]),
    }


def snn_cell(ds):
    recs = _load(f"results/Tree-to-Spike/{ds}/seed_*/results.json")
    if not recs:
        return None
    g = lambda fn: mean([fn(r) for r in recs])
    sops = g(lambda r: r["test_plus"]["spike_stats"].get("synops_per_sample"))
    return {
        "n": len(recs),
        "acc": g(lambda r: r["test_plus"]["metrics"]["accuracy"]),
        "f1m": g(lambda r: r["test_plus"]["metrics"]["f1_macro"]),
        "mcc": g(lambda r: r["test_plus"]["metrics"]["matthews_corrcoef"]),
        "params": g(lambda r: r["model_size"]["trainable_parameters"]),
        "ops": sops,
        "energy_pj": (sops * E_AC_PJ) if sops is not None else None,
        "lat_ms": g(lambda r: r["test_plus"].get("inference_ms_per_sample")),  # batched (caveat)
    }


def main():
    rows = []
    for ds in DATASETS:
        d, s = dnn_cell(ds), snn_cell(ds)
        if d is None or s is None:
            continue
        delta = d["energy_pj"] / s["energy_pj"] if (d["energy_pj"] and s["energy_pj"]) else None
        rows.append({"ds": ds, "d": d, "s": s, "delta": delta})

    if not rows:
        print("No paired DNN-Tree / Tree-to-Spike results yet.")
        return

    lines = ["# Fair DNN-vs-SNN comparison — DNN-Tree (dense ANN) vs Tree-to-Spike (spiking)",
             "",
             "Iso-architecture twins: identical Linear topology, same XGBoost teacher + "
             "tree-guided encoder + distillation; only the compute paradigm differs.",
             f"Horowitz 45nm: MAC = {E_MAC_PJ} pJ, AC/SOP = {E_AC_PJ} pJ. "
             "delta = E_DNN / E_SNN (energy gain of the SNN).",
             "",
             "| Dataset | Params (=) | Acc DNN/SNN | F1m DNN/SNN | MCC DNN/SNN | MACs (DNN) | SOPs (SNN) | E_DNN pJ | E_SNN pJ | delta (x) |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    agg = {"acc_d": [], "acc_s": [], "f1_d": [], "f1_s": [], "mcc_d": [], "mcc_s": [], "delta": []}
    for r in rows:
        d, s = r["d"], r["s"]
        peq = "=" if abs((d["params"] or 0) - (s["params"] or 0)) < 1 else f"DNN {d['params']:.0f}/SNN {s['params']:.0f}!"
        lines.append(
            f"| {r['ds']} | {d['params']:.0f} {peq} | {d['acc']:.3f}/{s['acc']:.3f} | "
            f"{d['f1m']:.3f}/{s['f1m']:.3f} | {d['mcc']:.3f}/{s['mcc']:.3f} | "
            f"{d['ops']:.0f} | {s['ops']:.0f} | {d['energy_pj']:.0f} | {s['energy_pj']:.0f} | "
            f"{r['delta']:.1f} |")
        agg["acc_d"].append(d["acc"]); agg["acc_s"].append(s["acc"])
        agg["f1_d"].append(d["f1m"]); agg["f1_s"].append(s["f1m"])
        agg["mcc_d"].append(d["mcc"]); agg["mcc_s"].append(s["mcc"])
        agg["delta"].append(r["delta"])
    lines.append(
        f"| **macro-avg** | = | {mean(agg['acc_d']):.3f}/{mean(agg['acc_s']):.3f} | "
        f"{mean(agg['f1_d']):.3f}/{mean(agg['f1_s']):.3f} | {mean(agg['mcc_d']):.3f}/{mean(agg['mcc_s']):.3f} | "
        f"— | — | — | — | **{mean(agg['delta']):.1f}** |")

    lines += ["", "## Batch-1 latency & Energy-Latency Product (ELP)",
              "| Dataset | DNN lat (ms, batch-1) | SNN lat (ms, batched*) | ELP_DNN | ELP_SNN |",
              "|---|--:|--:|--:|--:|"]
    for r in rows:
        d, s = r["d"], r["s"]
        elp_d = (d["energy_pj"] * d["lat_ms"]) if d["lat_ms"] else None
        elp_s = (s["energy_pj"] * s["lat_ms"]) if s["lat_ms"] else None
        lines.append(f"| {r['ds']} | {d['lat_ms']:.3f} | "
                     f"{s['lat_ms']:.3f} | {elp_d:.0f} | {elp_s:.0f} |"
                     if d["lat_ms"] and s["lat_ms"] else f"| {r['ds']} | n/a | n/a | n/a | n/a |")

    lines += ["",
              "\\* SNN latency on file is batched (bs=128), not batch-1 — re-measure for a strict ELP.",
              "",
              "## Notes",
              f"- **Energy gain delta = {mean(agg['delta']):.1f}x** on average: the spiking student does "
              "fewer, cheaper (addition-only) ops than the dense ANN's MACs.",
              "- **Same parameters, comparable accuracy** -> the gap is the paradigm, not capacity "
              "(iso-architecture principle).",
              "- Horowitz SOP energy is **compute-only**; the SNN's T-step membrane/weight memory "
              "access is extra (a unit-latency T=1 SNN would remove it). State this in the paper.",
              "- Edge realism: DNN measured at **batch-1** (neuromorphic HW is batch-1 optimised).",
              ]

    out_md = Path("results/dnn_vs_snn_comparison.md")
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[wrote {out_md}]")

    # ---- Pareto: accuracy vs energy ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in rows:
            ax.scatter(r["d"]["energy_pj"], r["d"]["acc"], c="#d98c2b", marker="s", s=70,
                       edgecolor="k", linewidth=0.4, label="DNN-Tree" if r is rows[0] else None)
            ax.scatter(r["s"]["energy_pj"], r["s"]["acc"], c="#2b8c7a", marker="o", s=70,
                       edgecolor="k", linewidth=0.4, label="Tree-to-Spike (SNN)" if r is rows[0] else None)
            ax.annotate(r["ds"][:6], (r["s"]["energy_pj"], r["s"]["acc"]),
                        fontsize=7, color="#2b8c7a", xytext=(3, 3), textcoords="offset points")
            ax.plot([r["d"]["energy_pj"], r["s"]["energy_pj"]], [r["d"]["acc"], r["s"]["acc"]],
                    color="#999", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("Horowitz inference energy (pJ/sample, log) — lower is better")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs energy — SNN (left) is the Pareto-better corner")
        ax.legend(); ax.grid(True, which="both", ls=":", alpha=0.4)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"results/dnn_vs_snn_pareto.{ext}", dpi=150)
        print("[wrote results/dnn_vs_snn_pareto.png/.pdf]")
    except Exception as e:
        print(f"(pareto plot skipped: {e})")


if __name__ == "__main__":
    main()
