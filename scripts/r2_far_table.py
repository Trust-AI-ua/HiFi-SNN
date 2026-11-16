#!/usr/bin/env python3
"""Macro-F1, false-alarm rate and detection rate for every system on the IoT four
(revision 2, step 1; results/preregistration/2026-11-16_r2_integer_gbdt.md).

Fills the gaps in the guide tables: the compact and full GBDT (refit teachers,
results/tables/tab_teacher_per_dataset.csv; TON-IoT recomputed independently on all five
seeds and equal to the table) and PLE-MLP at its three precisions (the B3 records). The four
main models come from their run records through make_results_table.cell, the published path.
Five-seed mean and sample SD.

    python scripts/r2_far_table.py
Output: results/journal_r2/far_table.{csv,md}
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/journal_r2"
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
NAME = {"ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024", "edgeiiot": "Edge-IIoTset", "toniot": "TON-IoT"}
SEEDS = range(42, 47)


def main():
    import make_results_table as MRT
    rows = []
    for key, label, patt in MRT.MODELS:
        for ds in IOT:
            recs = [json.loads(Path(x).read_text()) for x in sorted(glob.glob(str(REPO / patt.format(d=ds))))
                    if int(Path(x).parent.name.split("_")[1]) in SEEDS]
            c = MRT.cell(key, recs)
            rows.append(dict(dataset=NAME[ds], system=label, precision="int8 weights" if key in ("tree", "dnn") else "fp32",
                             macro_f1=100 * c["f1"], macro_f1_sd=100 * c["f1_sd"], far=100 * c["far"], far_sd=100 * c["far_sd"],
                             dr=100 * c["dr"], dr_sd=100 * c["dr_sd"], source=patt.format(d=ds)))
    t = pd.read_csv(REPO / "results/tables/tab_teacher_per_dataset.csv")
    for arm, label in (("full", "Full GBDT"), ("compact", "Compact GBDT (validation-selected rounds)")):
        for ds in IOT:
            r = t[(t.dataset == ds) & (t.arm == arm)].iloc[0]
            if int(r.n_seeds) != 5:
                raise SystemExit(f"{ds} {arm}: {r.n_seeds} seeds")
            rows.append(dict(dataset=NAME[ds], system=label, precision="native", macro_f1=r.F1M, macro_f1_sd=r.F1M_sd,
                             far=r.FAR, far_sd=r.FAR_sd, dr=r.DR, dr_sd=r.DR_sd, source="results/tables/tab_teacher_per_dataset.csv"))
    sel = json.loads((REPO / "results/journal_partB/B3/selection.json").read_text())["selection"]
    for k, lab in (("fp32", "fp32"), ("per_tensor_int8", "per-tensor int8"), ("per_channel_int8", "per-channel int8")):
        for ds in IOT:
            v = []
            for sd in SEEDS:
                d = REPO / f"results/journal_partB/B3/final/{ds}/seed_{sd}"
                if (d / "REUSED_FROM.txt").exists():
                    d = REPO / (d / "REUSED_FROM.txt").read_text().strip()
                r = json.loads((d / "results.json").read_text())
                if r["config"] != sel[ds]["config"]:
                    raise SystemExit(f"B3 {ds}/{sd}: not the selected configuration")
                m, bm = r["test"][k]["metrics"], r["test"][k]["binary_ids_metrics"]
                v.append((100 * m["f1_macro"], 100 * bm["false_alarm_rate"], 100 * bm["detection_rate"]))
            v = np.array(v)
            rows.append(dict(dataset=NAME[ds], system="PLE-MLP", precision=lab, macro_f1=v[:, 0].mean(),
                             macro_f1_sd=v[:, 0].std(ddof=1), far=v[:, 1].mean(), far_sd=v[:, 1].std(ddof=1),
                             dr=v[:, 2].mean(), dr_sd=v[:, 2].std(ddof=1), source="results/journal_partB/B3/final"))
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "far_table.csv", index=False)
    fmt = lambda m, s: f"{m:.2f} ± {s:.2f}"
    md = ["# Macro-F1, FAR and DR, IoT four (five seeds)", "",
          "Mean ± sample SD over seeds 42-46, test split, in percent. FAR = benign flows flagged as attacks; "
          "DR = attack flows flagged as attacks. CICIoT2023 neural models are the 20-epoch block (A1).", ""]
    for ds in IOT:
        md += [f"## {NAME[ds]}", "", "| System | Precision | Macro-F1 | FAR | DR |", "|---|---|--:|--:|--:|"]
        for r in d[d.dataset == NAME[ds]].itertuples():
            md.append(f"| {r.system} | {r.precision} | {fmt(r.macro_f1, r.macro_f1_sd)} | {fmt(r.far, r.far_sd)} | {fmt(r.dr, r.dr_sd)} |")
        md.append("")
    (OUT / "far_table.md").write_text("\n".join(md))
    print(d.pivot_table(index=["system", "precision"], columns="dataset", values="far").round(2).to_string())


if __name__ == "__main__":
    main()
