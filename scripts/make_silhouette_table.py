#!/usr/bin/env python3
"""Build the silhouette table (cluster cleanliness of each model's penultimate
representation) from results/Journal_results/embeddings_silhouette.csv.

Run after scripts/embed_clusters.py has populated the CSV for the datasets.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TAB = REPO / "results" / "Journal_results" / "tables"
CSVP = REPO / "results" / "Journal_results" / "embeddings_silhouette.csv"

DS = [("kddcup99", "KDDCup99"), ("nslkdd", "NSL-KDD"), ("ctu13", "CTU-13"),
      ("iscxids2012", "ISCX2012"), ("cicids2017", "CIC-IDS2017"),
      ("cse-cic-ids2018", "CSE-CIC-2018"), ("unswnb15", "UNSW-NB15"),
      ("ciciot2023", "CICIoT2023"), ("ciciomt2024", "CICIoMT2024"),
      ("edgeiiot", "Edge-IIoTset"), ("toniot", "TON-IoT")]
MODELS = ["Vanilla SNN", "Tree-to-Spike", "DNN-Tree", "TASNN"]


def load():
    d = {}
    for r in csv.reader(open(CSVP)):
        if not r or r[0] == "dataset":
            continue
        ds, m, s = r[0], r[1], r[2]
        d.setdefault(ds, {})[m] = float(s) if s not in ("nan", "") else float("nan")
    return d


def main():
    d = load()
    md = ["### Cluster cleanliness - silhouette of each model's penultimate representation (UMAP)\n",
          "Higher = better-separated classes. Best per row in **bold** (`n/a` = degenerate "
          "embedding). Companion to `figures/embeddings_<dataset>.pdf`.\n",
          "| Dataset | " + " | ".join(MODELS) + " |",
          "|---|" + "--:|" * len(MODELS)]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Cluster cleanliness: silhouette score of each model's penultimate "
           r"representation (UMAP, identical settings). Higher = better-separated classes; "
           r"best per row in \textbf{bold}.}", r"\label{tab:silhouette}", r"\footnotesize",
           r"\begin{tabular}{l" + "r" * len(MODELS) + "}", r"\toprule",
           "Dataset & " + " & ".join(MODELS) + r" \\", r"\midrule"]
    agg = {m: [] for m in MODELS}
    for key, name in DS:
        row = d.get(key, {})
        vals = {m: row.get(m, float("nan")) for m in MODELS}
        finite = {m: v for m, v in vals.items() if np.isfinite(v)}
        best = max(finite, key=finite.get) if finite else None
        cells_md, cells_tex = [], []
        for m in MODELS:
            v = vals[m]
            s = f"{v:.3f}" if np.isfinite(v) else "n/a"
            if np.isfinite(v):
                agg[m].append(v)
            cells_md.append(f"**{s}**" if m == best else s)
            cells_tex.append(rf"\textbf{{{s}}}" if m == best else s)
        md.append(f"| {name} | " + " | ".join(cells_md) + " |")
        tex.append(f"{name} & " + " & ".join(cells_tex) + r" \\")
    # mean row
    means = {m: (np.mean(agg[m]) if agg[m] else float("nan")) for m in MODELS}
    bestm = max((m for m in MODELS if np.isfinite(means[m])), key=lambda m: means[m])
    md.append("| **Mean** | " + " | ".join(
        (f"**{means[m]:.3f}**" if m == bestm else f"{means[m]:.3f}") for m in MODELS) + " |")
    tex.append(r"\midrule")
    tex.append(r"\textit{Mean} & " + " & ".join(
        (rf"\textbf{{{means[m]:.3f}}}" if m == bestm else f"{means[m]:.3f}") for m in MODELS) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / "table_silhouette.md").write_text("\n".join(md) + "\n")
    (TAB / "table_silhouette.tex").write_text("\n".join(tex))
    print(f"[wrote] {TAB}/table_silhouette.md + .tex")
    print("Mean silhouette:", {m: round(means[m], 3) for m in MODELS})


if __name__ == "__main__":
    main()
