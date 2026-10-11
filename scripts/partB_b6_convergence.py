#!/usr/bin/env python3
"""B6: convergence evidence from the stored per-epoch logs (pre-registered, Part B).

Arms on the IoT four, seeds 42-46: tree (Tree-to-Spike), entropy, quantile, no-KD, raw,
DNN-Tree, Vanilla SNN. The metric is validation macro-F1 where it is stored (Tree-to-Spike
and DNN-Tree, from the M3 diagnostic reruns in results/diagnostics/val_curves/, whose final
test score must equal the stored run's) and validation accuracy elsewhere, the declared
proxy (D8). Both in percentage points.

Trigger (D8): for each arm and dataset, the mean over seeds of (value at epoch 10 minus
value at epoch 8), excluding the three stalled entropy runs. If it exceeds 1.0 point for
any arm, the full affected block is rerun with a 20-epoch budget and validation checkpoint
selection; otherwise the curves go into the appendix and the fixed-budget results stand.

    python scripts/partB_b6_convergence.py
Output: results/journal_partB/B6/<run_id>/{curves.csv,trigger.csv,summary.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

STALLED = {("entropy", "ciciot2023", 44), ("entropy", "ciciot2023", 45), ("entropy", "ciciomt2024", 46)}
ACC_ARMS = {"entropy": "results/encoder_ablation/{d}/entropy/seed_{s}/results.json",
            "quantile": "results/encoder_ablation/{d}/quantile/seed_{s}/results.json",
            "nokd": "results/encoder_ablation/{d}/nokd/seed_{s}/results.json",
            "raw": "results/encoder_ablation/{d}/m8_raw/seed_{s}/results.json",
            "vanilla": "results/{d}/SNN_LeakyParallel/latency/seed_{s}/results.json"}
F1_ARMS = {"tree": "Tree-to-Spike", "dnn": "DNN-Tree"}
THRESHOLD = 1.0


def main():
    st = C.stamp("B6")
    out = C.run_dir(st)
    rows = []
    for arm, model in F1_ARMS.items():
        for ds in C.IOT:
            for s in C.SEEDS:
                v = pd.read_csv(C.REPO / f"results/diagnostics/val_curves/{model}/{ds}/seed_{s}.csv")
                last = v[v.epoch == v.epoch.max()].iloc[0]
                rec = json.loads((C.REPO / f"results/{model}/{ds}/seed_{s}/results.json").read_text())
                repro = abs(last.test_f1_rerun - 100 * rec["test_plus"]["metrics"]["f1_macro"]) < 1e-3   # CSV has 4 decimals
                for r in v.itertuples():
                    rows.append(dict(arm=arm, dataset=ds, seed=s, epoch=int(r.epoch), metric="val_macro_f1",
                                     value=float(r.val_macro_f1), rerun_reproduces_stored=repro))
                # cross-check: the stored run's own per-epoch validation accuracy
                for e in rec["training"]["training_curve"]:
                    rows.append(dict(arm=arm, dataset=ds, seed=s, epoch=int(e["epoch"]),
                                     metric="val_accuracy (stored run, cross-check)", value=100 * float(e["val_accuracy"]),
                                     rerun_reproduces_stored=True))
    for arm, patt in ACC_ARMS.items():
        for ds in C.IOT:
            for s in C.SEEDS:
                c = json.loads((C.REPO / patt.format(d=ds, s=s)).read_text())["training"]["training_curve"]
                for e in c:
                    rows.append(dict(arm=arm, dataset=ds, seed=s, epoch=int(e["epoch"]), metric="val_accuracy (proxy)",
                                     value=100 * float(e["val_accuracy"]), rerun_reproduces_stored=True))
    d = pd.DataFrame(rows)
    d["stalled"] = [(a, ds, s) in STALLED for a, ds, s in zip(d.arm, d.dataset, d.seed)]
    d.to_csv(out / "curves.csv", index=False)
    if set(d.groupby(["arm", "dataset", "seed"]).epoch.max()) != {10}:
        sys.exit("not every run has ten epochs")
    w = d[~d.stalled].pivot_table(index=["arm", "dataset", "seed", "metric"], columns="epoch", values="value").reset_index()
    w["gain_8_to_10"] = w[10] - w[8]
    t = w.groupby(["arm", "dataset", "metric"]).agg(n_seeds=("seed", "size"), mean_gain=("gain_8_to_10", "mean"),
                                                   max_seed_gain=("gain_8_to_10", "max")).reset_index()
    t["fires"] = t.mean_gain > THRESHOLD
    t.to_csv(out / "trigger.csv", index=False)
    nonrep = sorted({(a, ds, int(sd)) for a, ds, sd, rp in zip(d.arm, d.dataset, d.seed, d.rerun_reproduces_stored) if not rp})
    xc = "val_accuracy (stored run, cross-check)"
    decisive = t[t.metric != xc]
    summary = dict(stamp=st, threshold_points=THRESHOLD,
                   fired=decisive[decisive.fires][["arm", "dataset", "metric", "mean_gain"]].to_dict("records"),
                   cross_check_fired=t[(t.metric == xc) & t.fires][["arm", "dataset", "mean_gain"]].to_dict("records"),
                   val_curve_reruns_not_reproducing_stored=nonrep,
                   excluded_stalled=sorted(map(list, STALLED)),
                   largest=t.sort_values("mean_gain", ascending=False).head(5)[["arm", "dataset", "metric", "mean_gain"]].to_dict("records"))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(t.round(3).to_string(index=False))
    print("\nnon-reproducing val-curve reruns:", nonrep)
    print("TRIGGER (pre-registered metrics):", ("FIRES for " + str(summary["fired"])) if summary["fired"] else "does not fire")
    print("cross-check on the stored runs' accuracy:", summary["cross_check_fired"] or "does not fire")


if __name__ == "__main__":
    main()
