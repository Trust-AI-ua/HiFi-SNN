#!/usr/bin/env python3
"""Derive the appendix-level tables the manuscript needs, from committed artifacts.

Everything here is a reshaping or a completion of something already measured. No
new experiment, and nothing recomputed from raw runs that an existing script
already owns: this reads those scripts' outputs so a number cannot disagree with
its source.

Four outputs:

  dedup_per_protocol.csv     per-protocol before/after macro-F1 under exact
                             duplicate removal. `dedup_analysis.py` reports the
                             leader verdict and the paired margin; the absolute
                             per-protocol means were only ever inside its JSON.
  sparsity_per_protocol.csv  zero / gate-band / active feature counts per
                             protocol, all five, with the band interval named.
  e1_arms_complete.csv       the three E1 arms *plus* the confirmation-sweep
                             baseline they are measured against. Without the
                             baseline row the timing premium cannot be read off
                             the table it is quoted from.
  covariate_shift_appendix.csv  the shift summary reduced to the columns the
                             appendix table uses.

    python scripts/appendix_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

A = ROOT / "results/analysis"
OUT = A / "appendix"

#: The two gates. Latency fires on x > THETA_LAT; delta, on a static vector,
#: reduces exactly to x > THETA_DELTA. The half-open band between them is the
#: set of features one code sees and the other does not, and it is the reason
#: the two encodings are not comparable at their own thresholds.
THETA_LAT, THETA_DELTA = 0.01, 0.05

#: E1 ran three seeds; the baseline must be taken over the same three.
E1_SEEDS = (42, 43, 44)
NEURON = "LeakyParallel"

PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
             "ctu13_v2_f0", "ctu13_v2_f1"]


def dedup_per_protocol() -> pd.DataFrame:
    """Per-protocol absolute macro-F1, official vs duplicate-free."""
    rows = []
    for pop in ("encoding_axis_lp", "neuron_axis_latency"):
        f = A / f"dedup/{pop}/summary.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d.insert(0, "population", pop)
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Name the delta explicitly rather than leaving it to be subtracted by hand.
    if {"f1_official", "f1_dedup"} <= set(out.columns):
        out["delta_pp"] = 100.0 * (out.f1_dedup - out.f1_official)
    return out


def sparsity_per_protocol() -> pd.DataFrame:
    """Zero, gate-band and active feature counts, per protocol.

    `feature_sparsity.csv` already carries the zero and band counts. What it
    does not carry is the active count, which is the denominator every ratio in
    Section IV-C is implicitly taken over, and the band interval is not named in
    the file so a reader cannot tell which two thresholds it lies between.
    """
    d = pd.read_csv(A / "feature_sparsity.csv")
    d = d.rename(columns={"zero_per_sample": "zero_count",
                          "band_per_sample": "gate_band_count"})
    d["active_count"] = d.d - d.zero_count
    d["above_delta_gate_count"] = d.active_count - d.gate_band_count
    d["gate_band_interval"] = f"({THETA_LAT}, {THETA_DELTA}]"
    d["active_pct"] = 100.0 * d.active_count / d.d
    d["gate_band_pct_of_active"] = 100.0 * d.gate_band_count / d.active_count
    return d[["protocol", "d", "zero_count", "zero_pct", "active_count",
              "active_pct", "gate_band_interval", "gate_band_count",
              "gate_band_pct_of_active", "above_delta_gate_count"]]


def e1_arms_complete() -> pd.DataFrame:
    """The three E1 arms plus the baseline arm they are compared against.

    The timing premium is latency@0.01 minus delta@0.01. Arm C supplies the
    second term; the first comes from the confirmation sweep and has never
    appeared in the ablation table, which is why that table reads as three arms
    with an unexplained premium column attached.
    """
    abl = pd.read_csv(A / "e1_threshold_ablation.csv")

    # The baseline is built from the same three seeds the E1 arms use, not from
    # the five-seed confirmation means. e1_timing_isolation.csv was verified to
    # be a 42-44 mean; taking the five-seed figure instead would silently change
    # the premium (on KDDCup99 by 0.85 pp) and break the pairing that makes the
    # comparison valid.
    from src.populations import macro_f1_pair
    from src.readout import discover
    rows = []
    for r in discover():
        if r.variant != f"{NEURON}/latency" or r.seed not in E1_SEEDS:
            continue
        rec, pre = r.record(), r.prefix()
        m = macro_f1_pair(pre["y_true"].astype(np.int64),
                          pre["y_pred"].astype(np.int64),
                          rec["data"]["num_classes"])
        tp = rec["test_plus"]
        rows.append({
            "protocol": r.protocol, "seed": r.seed,
            "f1": m["macro_f1_fixed_universe"],
            "mcc": tp["metrics"].get("matthews_corrcoef"),
            "dr": tp["binary_ids_metrics"].get("detection_rate"),
            "far": tp["binary_ids_metrics"].get("false_alarm_rate"),
            "sops": tp["spike_accounting"]["sops_per_sample"],
            "in_spk": tp["spike_accounting"]["input_spikes_per_sample"],
        })
    b = pd.DataFrame(rows)
    base = (b.groupby("protocol")
              .agg(n=("seed", "nunique"), f1=("f1", "mean"),
                   f1sd=("f1", lambda v: v.std(ddof=1)), mcc=("mcc", "mean"),
                   dr=("dr", "mean"), far=("far", "mean"),
                   sops=("sops", "mean"), in_spk=("in_spk", "mean"))
              .reset_index())
    base.insert(0, "arm", "BASELINE_latency_thr001")

    # Cross-check against the committed isolation table rather than trusting the
    # recomputation: if these disagree the premium in the paper is wrong.
    ti_chk = pd.read_csv(A / "e1_timing_isolation.csv", index_col=0)
    err = (base.set_index("protocol").f1 - ti_chk.f1_timing).abs().max()
    if err > 1e-9:
        raise SystemExit(
            f"baseline macro-F1 disagrees with e1_timing_isolation.csv by "
            f"{err:.3e}; the timing premium would be wrong.")

    out = pd.concat([base, abl], ignore_index=True)
    out["is_baseline"] = out.arm.str.startswith("BASELINE")
    order = {"BASELINE_latency_thr001": 0, "A_latency_thr000": 1,
             "B_latency_thr005": 2, "C_delta_thr001": 3}
    out["_o"] = out.arm.map(order)
    out = out.sort_values(["protocol", "_o"]).drop(columns="_o")

    # The premium, recomputed here from the two arms so the table is
    # self-contained and cannot disagree with the figure.
    prem = {}
    for proto in out.protocol.unique():
        g = out[out.protocol == proto].set_index("arm")
        if {"BASELINE_latency_thr001", "C_delta_thr001"} <= set(g.index):
            prem[proto] = 100.0 * (g.loc["BASELINE_latency_thr001", "f1"]
                                   - g.loc["C_delta_thr001", "f1"])
    out["timing_premium_pp"] = out.protocol.map(prem)
    return out


def covariate_shift_appendix() -> pd.DataFrame:
    d = pd.read_csv(A / "covariate_shift/summary.csv")
    cols = ["protocol", "d", "ks_median", "ks_p90", "w_p90", "domain_auc",
            "w_top5_share_pct", "n_features_w_gt_0p1", "timing_premium_pp"]
    return d[[c for c in cols if c in d.columns]].sort_values(
        "domain_auc", ascending=False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fn in (("dedup_per_protocol", dedup_per_protocol),
                     ("sparsity_per_protocol", sparsity_per_protocol),
                     ("e1_arms_complete", e1_arms_complete),
                     ("covariate_shift_appendix", covariate_shift_appendix)):
        try:
            df = fn()
        except Exception as e:                                    # noqa: BLE001
            print(f"  {name}: SKIPPED ({type(e).__name__}: {e})")
            continue
        if df is None or df.empty:
            print(f"  {name}: no data")
            continue
        p = OUT / f"{name}.csv"
        df.to_csv(p, index=False, float_format="%.8f")
        made.append(p)
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))

    print(f"\nwrote {len(made)} file(s) to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
