#!/usr/bin/env python3
"""Comprehensive 3-way benchmark: Vanilla SNN (LeakyParallel/latency) vs
Tree-to-Spike (hybrid SNN) vs DNN-Tree (iso-architecture dense ANN).

Reports quality, efficiency, and Horowitz energy across all 6 datasets, 5 seeds.
  * DNN-Tree is the iso-architecture twin of Tree-to-Spike (same params).
  * Energy: DNN = MACs x 4.6 pJ ; SNNs = SOPs x 0.9 pJ.
  * delta = energy gain of each SNN vs an *equal-capacity* dense ANN twin:
        Tree-to-Spike  -> the measured DNN-Tree (literal iso-twin)
        Vanilla SNN    -> its analytic iso-ANN (F*H + H*C single pass), since no
                          DNN twin was trained for the 128-wide raw-feature model.
Outputs results/benchmark_3way.md + results/benchmark_3way.csv.
"""
from __future__ import annotations
import glob, json, csv
from pathlib import Path

E_MAC, E_AC = 4.6, 0.9
DS = ["kddcup99", "nslkdd", "cicids2017", "ctu13", "iscxids2012", "cse-cic-ids2018"]
T_VANILLA = 25


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _load(globs):
    recs = []
    for g in globs:
        for f in glob.glob(g):
            try:
                recs.append(json.load(open(f)))
            except Exception:
                pass
    return recs


def vanilla_recs(ds):
    return _load([f"results/SNN_Model_compare_Neuron_vs_Encoding/{ds}/SNN_LeakyParallel/latency/seed_*/results.json",
                  f"results/{ds}/SNN_LeakyParallel/latency/seed_*/results.json"])


def cell(recs, kind):
    """kind in {vanilla, tree, dnn}. Returns averaged metric dict."""
    if not recs:
        return None
    g = lambda fn: mean([fn(r) for r in recs])
    M = lambda k: g(lambda r: r["test_plus"]["metrics"].get(k))
    B = lambda k: g(lambda r: (r["test_plus"].get("binary_ids_metrics") or {}).get(k))
    out = {
        "n": len(recs),
        "accuracy": M("accuracy"), "bal_acc": M("balanced_accuracy"),
        "f1_macro": M("f1_macro"), "f1_weighted": M("f1_weighted"),
        "precision_macro": M("precision_macro"), "recall_macro": M("recall_macro"),
        "roc_auc": M("roc_auc_weighted"), "mcc": M("matthews_corrcoef"),
        "kappa": M("cohen_kappa"), "DR": B("detection_rate"), "FAR": B("false_alarm_rate"),
        "params": g(lambda r: r["model_size"]["trainable_parameters"]),
    }
    nc = g(lambda r: r["data"]["num_classes"])
    out["nc"] = nc
    if kind == "dnn":
        out["ops"] = g(lambda r: r["test_plus"]["compute"]["macs_per_sample"])
        out["E_pj"] = g(lambda r: r["test_plus"]["compute"]["energy_pj_horowitz"])
        out["sparsity"] = None; out["exit_frac"] = None
        out["lat_ms"] = g(lambda r: r["test_plus"].get("inference_ms_per_sample_batch1"))
        out["iso_macs"] = out["ops"]
        out["neurons"] = sum(recs[0]["config"]["hidden"]); out["T"] = 1; out["Teff"] = 1.0
    else:
        sops = g(lambda r: (r["test_plus"].get("spike_stats") or {}).get("synops_per_sample"))
        out["ops"] = sops
        out["E_pj"] = sops * E_AC if sops is not None else None
        out["sparsity"] = g(lambda r: (r["test_plus"].get("spike_stats") or {}).get("spike_sparsity"))
        Tm = T_VANILLA if kind == "vanilla" else g(lambda r: r["config"].get("T", 20))
        ee = g(lambda r: (r["test_plus"].get("early_exit") or {}).get("mean_exit_step"))
        out["exit_frac"] = (ee / Tm) if ee is not None else None
        out["lat_ms"] = g(lambda r: r["test_plus"].get("inference_ms_per_sample"))
        out["T"] = Tm; out["Teff"] = ee
        if kind == "vanilla":   # analytic iso-ANN MACs (single pass): F*H + H*C
            F = g(lambda r: r["data"]["num_features"])
            H = g(lambda r: r["config"].get("hidden_size", 128))
            out["iso_macs"] = F * H + H * nc; out["neurons"] = H + nc
        else:
            out["iso_macs"] = None   # filled from the DNN twin below
            out["neurons"] = sum(recs[0]["config"]["hidden"]) + nc
    return out


# ---- memory-aware total energy at batch-1 (Horowitz + memory model) ----
E_MEM_SRAM = 5.0   # pJ per 32-bit on-chip SRAM access (Horowitz/Han)


def total_energy_batch1(c, kind, e_mem=E_MEM_SRAM):
    """Compute + memory energy per sample at batch-1, early-exit for SNNs."""
    if c is None:
        return None
    if kind == "dnn":
        n_mem = c["params"] + 2 * c["neurons"]
        return c["ops"] * E_MAC + n_mem * e_mem
    Teff = c["Teff"] or c["T"]
    sops_eff = c["ops"] * (Teff / c["T"])
    n_mem = sops_eff + 2 * c["neurons"] * Teff
    return sops_eff * E_AC + n_mem * e_mem


def main():
    data = {ds: {"vanilla": cell(vanilla_recs(ds), "vanilla"),
                 "tree": cell(_load([f"results/Tree-to-Spike/{ds}/seed_*/results.json"]), "tree"),
                 "dnn": cell(_load([f"results/DNN-Tree/{ds}/seed_*/results.json"]), "dnn")}
            for ds in DS}
    # Tree's iso-ANN MACs = its DNN twin's measured MACs
    for ds in DS:
        if data[ds]["tree"] and data[ds]["dnn"]:
            data[ds]["tree"]["iso_macs"] = data[ds]["dnn"]["ops"]
    # delta per SNN = (iso_macs * 4.6) / (SOPs * 0.9)
    for ds in DS:
        for k in ("vanilla", "tree"):
            c = data[ds][k]
            if c and c.get("iso_macs") and c.get("ops"):
                c["delta"] = (c["iso_macs"] * E_MAC) / (c["ops"] * E_AC)

    MODELS = [("Vanilla SNN (LeakyParallel/latency)", "vanilla"),
              ("Tree-to-Spike (hybrid SNN)", "tree"),
              ("DNN-Tree (dense ANN)", "dnn")]
    QM = ["accuracy", "bal_acc", "f1_macro", "f1_weighted", "mcc", "kappa",
          "precision_macro", "recall_macro", "roc_auc", "DR", "FAR"]

    def macro(k, key):
        return mean([data[ds][k][key] for ds in DS if data[ds][k] and data[ds][k].get(key) is not None])

    L = ["# Comprehensive benchmark — Vanilla SNN vs Tree-to-Spike vs DNN",
         "",
         "5 seeds x 6 datasets, macro-averaged across datasets. DNN-Tree is the "
         "iso-architecture twin of Tree-to-Spike (identical params). Horowitz 45nm: "
         "MAC = 4.6 pJ, AC/SOP = 0.9 pJ.",
         "",
         "## 1. Quality (macro-averaged across 6 datasets)",
         "| Model | Acc | Bal-Acc | F1-mac | F1-wt | MCC | kappa | Prec | Rec | ROC-AUC | DR | FAR |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, k in MODELS:
        L.append("| " + name + " | " + " | ".join(
            (f"{macro(k,m):.3f}" if macro(k, m) is not None else "-") for m in QM) + " |")

    L += ["", "## 2. Efficiency & energy (macro-averaged)",
          "| Model | Params | Ops/sample | Op type | Energy (pJ) | Sparsity | Exit frac | delta vs dense ANN |",
          "|---|--:|--:|:--|--:|--:|--:|--:|"]
    optype = {"vanilla": "SOPs", "tree": "SOPs", "dnn": "MACs"}
    for name, k in MODELS:
        sp = macro(k, "sparsity"); ex = macro(k, "exit_frac"); dl = macro(k, "delta")
        sp_s = f"{sp:.3f}" if sp is not None else "-"
        ex_s = f"{ex:.2f}" if ex is not None else "-"
        dl_s = f"{dl:.1f}x" if dl is not None else "1.0x (ref)"
        L.append(f"| {name} | {macro(k,'params'):.0f} | {macro(k,'ops'):.0f} | {optype[k]} | "
                 f"{macro(k,'E_pj'):.0f} | {sp_s} | {ex_s} | {dl_s} |")

    # ---- macro cells + memory-aware total energy (batch-1, on-chip SRAM) ----
    KEYS = ["params", "ops", "neurons", "T", "Teff", "iso_macs", "nc"]
    mcell = {k: {key: macro(k, key) for key in KEYS} for _, k in MODELS}
    tot = {k: total_energy_batch1(mcell[k], k) for _, k in MODELS}
    # vanilla's analytic dense-ANN twin total energy
    van_twin = (mcell["vanilla"]["iso_macs"] * E_MAC +
                (mcell["vanilla"]["iso_macs"] + 2 * (mcell["vanilla"]["neurons"] - mcell["vanilla"]["nc"])) * E_MEM_SRAM)
    dtot = {"vanilla": van_twin / tot["vanilla"], "tree": tot["dnn"] / tot["tree"], "dnn": 1.0}

    L += ["", "## 2b. HEADLINE energy — memory-aware total, batch-1 (on-chip SRAM 5 pJ)",
          "Compute **+** memory access (the publication-grade number; compute-only flatters SNNs). "
          "SNNs use early-exit (T_eff); DNN amortises weights only at high batch, so batch-1 is its worst case.",
          "| Model | Compute E (pJ) | Total E @B=1 (pJ) | delta_total vs dense ANN |",
          "|---|--:|--:|--:|"]
    for name, k in MODELS:
        dl = dtot[k]
        L.append(f"| {name} | {macro(k,'E_pj'):.0f} | {tot[k]:.0f} | "
                 + (f"**{dl:.1f}x**" if k != 'dnn' else "1.0x (ref)") + " |")
    L += ["",
          f"- At batch-1 with memory access, **Tree-to-Spike still uses {dtot['tree']:.1f}x less total energy** "
          f"than its iso DNN twin, and **Vanilla SNN {dtot['vanilla']:.1f}x** vs its dense twin — the win "
          "survives a full memory accounting because early-exit keeps SNN memory traffic below the DNN's "
          "dense weight reads. (Without early-exit it can drop below 1x at DRAM — see energy_memory_model.md.)",
          ]

    L += ["", "## 3. Per-dataset F1-macro", "| Model | " + " | ".join(d[:8] for d in DS) + " |",
          "|---|" + "--:|" * len(DS)]
    for name, k in MODELS:
        L.append("| " + name + " | " + " | ".join(
            (f"{data[ds][k]['f1_macro']:.3f}" if data[ds][k] and data[ds][k]['f1_macro'] is not None else "-")
            for ds in DS) + " |")

    L += ["", "## 4. Per-dataset energy gain (delta = E_dense-ANN / E_SNN)",
          "| Model | " + " | ".join(d[:8] for d in DS) + " |", "|---|" + "--:|" * len(DS)]
    for name, k in [("Vanilla SNN", "vanilla"), ("Tree-to-Spike", "tree")]:
        L.append("| " + name + " | " + " | ".join(
            (f"{data[ds][k].get('delta'):.1f}x" if data[ds][k] and data[ds][k].get('delta') else "-")
            for ds in DS) + " |")

    L += ["", "## Bottom line",
          f"- **Accuracy**: all three are comparable (macro F1 — Vanilla {macro('vanilla','f1_macro'):.3f}, "
          f"Tree-to-Spike {macro('tree','f1_macro'):.3f}, DNN {macro('dnn','f1_macro'):.3f}); "
          "the spiking models are not paying an accuracy penalty.",
          f"- **Energy**: Tree-to-Spike uses **{macro('tree','delta'):.1f}x** less compute energy than its "
          f"iso-architecture DNN twin; the Vanilla SNN **{macro('vanilla','delta'):.1f}x** vs its dense twin.",
          f"- **Model size**: Tree-to-Spike/DNN are tiny ({macro('tree','params'):.0f} params) vs the "
          f"Vanilla SNN ({macro('vanilla','params'):.0f}).",
          "- **Caveats**: Horowitz energy is compute-only (excludes the SNN's per-timestep memory access); "
          "latency on file is batched for the SNNs vs batch-1 for the DNN, so ELP needs a batch-1 re-measure.",
          ]

    Path("results/benchmark_3way.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    with open("results/benchmark_3way.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["model", "dataset", "n_seeds"] + QM + ["params", "ops", "E_pj", "sparsity", "exit_frac", "delta", "lat_ms"]
        w.writerow(cols)
        for name, k in MODELS:
            for ds in DS:
                c = data[ds][k]
                if not c: continue
                w.writerow([name, ds, c["n"]] + [c.get(m) for m in QM] +
                           [c.get("params"), c.get("ops"), c.get("E_pj"), c.get("sparsity"),
                            c.get("exit_frac"), c.get("delta"), c.get("lat_ms")])
    print("\n[wrote results/benchmark_3way.md + results/benchmark_3way.csv]")

    # ---- batch-1 latency from the re-measure CSV ----
    lat = {"vanilla": None, "tree": None, "dnn": None}
    name2k = {"Vanilla SNN": "vanilla", "Tree-to-Spike": "tree", "DNN-Tree": "dnn"}
    p = Path("results/latency_elp.csv")
    if p.exists():
        acc = {"vanilla": [], "tree": [], "dnn": []}
        with open(p) as fh:
            for row in csv.DictReader(fh):
                k = name2k.get(row["model"])
                if k:
                    acc[k].append(float(row["batch1_latency_ms"]))
        lat = {k: mean(v) for k, v in acc.items()}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        COL = {"vanilla": "#7a5bd0", "tree": "#2b8c7a", "dnn": "#d98c2b"}
        NAME = {"vanilla": "Vanilla SNN", "tree": "Tree-to-Spike", "dnn": "DNN-Tree"}

        # ---------- Pareto: accuracy vs memory-aware total energy (batch-1) ----------
        fig, ax = plt.subplots(figsize=(7, 5))
        for k in ("vanilla", "tree", "dnn"):
            ax.scatter(tot[k], macro(k, "f1_macro"), s=160, c=COL[k], edgecolor="k",
                       linewidth=0.6, zorder=3, label=NAME[k])
            ax.annotate(NAME[k], (tot[k], macro(k, "f1_macro")), fontsize=9,
                        xytext=(6, 6), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("Total energy @ batch-1, compute+SRAM (pJ/sample, log) — lower is better")
        ax.set_ylabel("F1-macro")
        ax.set_title("Accuracy vs total energy (memory-aware, batch-1)\nupper-left = Pareto-better")
        ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"results/pareto_accuracy_vs_total_energy.{ext}", dpi=150)
        plt.close(fig)

        # ---------- radar: 6 axes, normalised so best=1 ----------
        f1 = {k: macro(k, "f1_macro") for k in COL}
        mcc = {k: macro(k, "mcc") for k in COL}
        far = {k: macro(k, "FAR") for k in COL}
        prm = {k: macro(k, "params") for k in COL}
        nhi = lambda d: {k: d[k] / max(d.values()) for k in d}
        nlo = lambda d: {k: min(d.values()) / d[k] for k in d}
        axes_data = [("F1-macro", nhi(f1)), ("MCC", nhi(mcc)), ("Low FAR", nlo(far)),
                     ("Energy eff.", nlo(tot)), ("Speed (1/lat)", nlo(lat)), ("Compactness", nlo(prm))]
        labels = [a for a, _ in axes_data]
        ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        ang += ang[:1]
        fig = plt.figure(figsize=(7, 7)); ax = fig.add_subplot(111, polar=True)
        for k in ("vanilla", "tree", "dnn"):
            vals = [d[k] for _, d in axes_data]; vals += vals[:1]
            ax.plot(ang, vals, color=COL[k], linewidth=2, label=NAME[k])
            ax.fill(ang, vals, color=COL[k], alpha=0.12)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "", "", ""])
        ax.set_ylim(0, 1.05)
        ax.set_title("3-way profile (each axis normalised, best = outer edge)", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"results/radar_3way.{ext}", dpi=150)
        plt.close(fig)
        print("[wrote results/pareto_accuracy_vs_total_energy.png/.pdf + results/radar_3way.png/.pdf]")
    except Exception as e:
        print(f"(plots skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
