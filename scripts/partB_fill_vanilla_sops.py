#!/usr/bin/env python3
"""Apply deployment_measure.py's Vanilla SNN operation-count rule to the CICIoT2023 row.

deployment_measure.py takes the Vanilla SNN's SOPs from its run record
(test_plus.spike_stats.synops_per_sample) and falls back to counting on the first 2,000
latency inputs only when the record lacks the field. The CICIoT2023 block records (A1)
lacked it when the 2026-10-07 deployment table was measured, so that one row carries the
2,000-input fallback (3,715.18) while the other ten carry full-test counts. The records were
then completed by the established full-test replay (scripts/backfill_snn_metrics.py, as in
58b1dbd2; block seed 42, identical to the fixed-budget seed-42 model, reproduced its stored
3,739.65 exactly). This applies the code's own rule to the completed record: sops and
sops_paper_convention of the ciciot2023 vanilla_snn row. No timing field changes.

Guard: the same rule applied to the other ten datasets' rows changes nothing.
Adds one header line to results/deployment/deployment_measurements.csv.

    python scripts/partB_fill_vanilla_sops.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results/deployment/deployment_measurements.csv"


def rule(r):
    """deployment_measure.py: sops from the record; paper convention adds F * hidden."""
    rec = json.loads((REPO / f"results/{r.dataset}/SNN_LeakyParallel/latency/seed_{int(r.seed)}/results.json").read_text())
    syn = rec["test_plus"]["spike_stats"].get("synops_per_sample")
    if syn is None:
        return None
    hid = yaml.safe_load(open(REPO / f"configs/{r.dataset}/SNN_LeakyParallel/latency/config.yaml"))["hidden_size"]
    return float(syn), float(syn) + float(r.n_channels) * float(hid)


def main():
    lines = CSV.read_text().splitlines()
    header = [l for l in lines if l.startswith("#")]
    if any("vanilla_sops_fill" in l for l in header):
        sys.exit("already applied")
    d = pd.read_csv(CSV, comment="#", float_precision="round_trip")
    v = d[d.system == "vanilla_snn"]
    for i, r in v.iterrows():
        new = rule(r)
        if r.dataset == "ciciot2023":
            continue
        if new is None or new != (r.sops, r.sops_paper_convention):
            sys.exit(f"guard: the rule changes {r.dataset} ({r.sops}, {r.sops_paper_convention}) -> {new}")
    i = v.index[v.dataset == "ciciot2023"]
    if len(i) != 1:
        sys.exit("expected one ciciot2023 vanilla_snn row")
    i = i[0]
    old = (d.at[i, "sops"], d.at[i, "sops_paper_convention"])
    d.at[i, "sops"], d.at[i, "sops_paper_convention"] = rule(d.loc[i])
    header.append(f"# vanilla_sops_fill source=scripts/partB_fill_vanilla_sops.py ciciot2023 vanilla_snn sops "
                  f"{float(old[0])!r} -> {float(d.at[i, 'sops'])!r} (2,000-input fallback -> run record, completed by "
                  f"scripts/backfill_snn_metrics.py); no timing field changed")
    CSV.write_text("\n".join(header) + "\n" + d.to_csv(index=False))
    print(f"ciciot2023 vanilla_snn: sops {old[0]} -> {d.at[i, 'sops']}, "
          f"paper convention {old[1]} -> {d.at[i, 'sops_paper_convention']}; other ten rows unchanged by the rule")


if __name__ == "__main__":
    main()
