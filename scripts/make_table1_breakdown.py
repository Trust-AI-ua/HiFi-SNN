#!/usr/bin/env python3
"""Per-component artifact breakdown for the two int8 systems (Table I).

Every column is a single-run seed-42 deployment measurement. The components are
what the serialization rule of src/serialize.py actually stores:

  int8 weights    one byte per 2-D weight value
  scales          one float32 per weight tensor
  float values    biases and 1-D buffers, which are not on the grid
  encoder tables  bin edges, channel list and config; float32 and UNCHANGED by
                  quantization, since requantizing a threshold would move a bin
                  boundary and change the model
  scaler          the preprocessing metadata common to every system

The encoder tables are worth reading before the network: on several datasets they
are the larger half of the artifact, and DNN-Tree carries exactly the same ones,
which is why the two systems sit at footprint parity.

    python scripts/make_table1_breakdown.py
Output: results/tables/tab_artifact_breakdown.{csv,md}
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
COMP = [("model_int8_bytes", "int8 weights"), ("model_scale_bytes", "scales"),
        ("model_float_bytes", "float values"), ("encoder_edges_bytes", "encoder edges"),
        ("encoder_channels_bytes", "encoder channels"),
        ("encoder_config_bytes", "encoder config"), ("scaler_bytes", "scaler")]


def main():
    d = pd.read_csv(REPO / "results/deployment/deployment_measurements.csv", comment="#")
    out, md = [], []
    for sysname, label in (("t2s_full", "Tree-to-Spike"), ("dnn_tree", "DNN-Tree")):
        s = d[d.system == sysname].set_index("dataset")
        for ds in s.index:
            row = {"system": label, "dataset": ds}
            for col, name in COMP:
                row[name] = int(s.loc[ds, col]) if not pd.isna(s.loc[ds, col]) else 0
            row["total"] = int(s.loc[ds, "total_bytes"])
            row["encoder share %"] = round(
                100 * (row["encoder edges"] + row["encoder channels"] + row["encoder config"])
                / row["total"], 1)
            out.append(row)
    t = pd.DataFrame(out)
    t.to_csv(REPO / "results/tables/tab_artifact_breakdown.csv", index=False)
    for label in ("Tree-to-Spike", "DNN-Tree"):
        g = t[t.system == label].set_index("dataset").drop(columns="system")
        print(f"\n== {label} (bytes, seed 42)")
        print(g.to_string())
        print(f"   IoT mean total {g.loc[IOT, 'total'].mean()/1024:.1f} KB | "
              f"encoder share {g.loc[IOT, 'encoder share %'].mean():.1f}%")
    cols = list(t.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.values]
    (REPO / "results/tables/tab_artifact_breakdown.md").write_text("\n".join(md) + "\n")
    print(f"\n[wrote] {REPO/'results/tables/tab_artifact_breakdown.csv'} and .md")


if __name__ == "__main__":
    main()
