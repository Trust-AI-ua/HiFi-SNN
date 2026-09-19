#!/usr/bin/env python3
"""Operation-priced and memory-inclusive energy for the seven deployed systems.

Runs 1, 2, 3 and 5 of the deployment analysis. Nothing is trained and nothing is
re-measured: this prices the operation and byte counts already measured by
scripts/deployment_measure.py.

Run 1  arithmetic energy under the paper's existing Horowitz accounting, with tree
       comparisons priced -- not withheld. A comparison is a subtract-and-branch and
       sits in the same Horowitz cost class as an add (0.9 pJ fp32, 0.03 pJ int8).
       Refusing to price it protects a number that does not need protecting.
Run 2  memory-inclusive total energy, tiered by artifact size, plus a sensitivity
       sweep over the SRAM and DRAM per-access costs.
Run 3  SRAM-budget feasibility: which artifacts fit entirely on chip.
Run 5  spiking systems repriced at Loihi-class per-SOP energy.

Every constant is declared here and swept where it matters. The point estimates are
illustrations; the sensitivity bands are the result.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "deployment"

# --- Horowitz 45 nm arithmetic, as used throughout the paper -------------------
E_MAC_FP32, E_ADD_FP32 = 4.6, 0.9      # multiply-accumulate / add
E_MAC_INT8, E_ADD_INT8 = 0.23, 0.03
# A tree comparison is a subtract-and-branch: add cost class, NOT a MAC.
E_CMP_FP32, E_CMP_INT8 = E_ADD_FP32, E_ADD_INT8
# Loihi-class measured per-SOP energy, for Run 5. 0.9 pJ is an arithmetic-only
# paper number; real neuromorphic silicon pays routing and state on top.
E_SOP_LOIHI = 23.6

NODE_RECORD_BYTES = 16                 # feat idx + threshold + 2 child pointers
HIDDEN_STATE = 128                     # 64+64 LIF/ReLU units

def mem_tier(total_bytes, sram_small, sram_mid, sram_large, dram):
    """Per-access cost keyed to whether the artifact fits on chip."""
    b = total_bytes
    if b <= 8 * 1024:      return sram_small
    if b <= 32 * 1024:     return sram_mid
    if b <= 1024 * 1024:   return sram_large
    return dram

def profile(df):
    """Per-system access and operation profile, averaged over datasets/seeds."""
    g = df.groupby("system").mean(numeric_only=True)
    rows = []
    for sysname, r in g.iterrows():
        macs = 0.0 if pd.isna(r.get("macs")) else float(r["macs"])
        sops = 0.0 if pd.isna(r.get("sops")) else float(r["sops"])
        cmps = 0.0 if pd.isna(r.get("tree_comparisons")) else float(r["tree_comparisons"])
        chans = 0.0 if pd.isna(r.get("n_channels")) else float(r["n_channels"])
        exit_t = r.get("mean_exit")
        is_snn = sops > 0
        is_tree = sysname.endswith("gbdt")
        # weight / node reads
        w_reads = sops if is_snn else (cmps if is_tree else macs)
        # membrane state read+write per timestep; dense activations written once
        state = (HIDDEN_STATE * float(exit_t) * 2) if (is_snn and not pd.isna(exit_t)) \
                else (0.0 if is_tree else HIDDEN_STATE)
        inputs = chans if chans > 0 else 78.0     # raw feature vector, mean width
        rows.append(dict(system=sysname, macs=macs, sops=sops, cmps=cmps,
                         total_bytes=float(r["total_bytes"]),
                         w_reads=w_reads, state=state, inputs=inputs,
                         accesses=w_reads + state + inputs))
    return pd.DataFrame(rows).set_index("system")

def e_arith(p, int8=False):
    mac = E_MAC_INT8 if int8 else E_MAC_FP32
    add = E_ADD_INT8 if int8 else E_ADD_FP32
    cmp_ = E_CMP_INT8 if int8 else E_CMP_FP32
    return p.macs * mac + p.sops * add + p.cmps * cmp_

def e_total(p, sram_s=10, sram_m=20, sram_l=100, dram=1300, int8=False):
    per = p.total_bytes.map(lambda b: mem_tier(b, sram_s, sram_m, sram_l, dram))
    return e_arith(p, int8) + p.accesses * per, per

if __name__ == "__main__":
    d = pd.read_csv(OUT / "deployment_measurements.csv")
    q = pd.read_csv(OUT / "deployment_summary.csv").set_index("system")
    p = profile(d)
    order = ["full_gbdt", "compact_gbdt", "raw_mlp_fp", "raw_mlp_int8",
             "dnn_tree", "t2s_full", "t2s_exit"]
    p = p.loc[order]; f1 = q.loc[order, "macro_f1"]

    print("=" * 78)
    print("RUN 1 — arithmetic energy under the paper's Horowitz accounting")
    print("=" * 78)
    a32, a8 = e_arith(p, False), e_arith(p, True)
    t1 = pd.DataFrame({"macro_F1": f1.round(2), "ops": p.macs + p.sops + p.cmps,
                       "E_arith_fp32_pJ": a32.round(0), "E_arith_int8_pJ": a8.round(1)})
    print(t1.to_string())
    print(f"\n  compact_gbdt vs t2s_exit, fp32: "
          f"{a32['compact_gbdt']:.0f} vs {a32['t2s_exit']:.0f} pJ "
          f"({a32['compact_gbdt']/a32['t2s_exit']:.2f}x)")

    print("\n" + "=" * 78)
    print("RUN 2 — memory-inclusive total energy (SRAM 10/20/100 pJ, DRAM 1300 pJ)")
    print("=" * 78)
    tot, per = e_total(p)
    t2 = pd.DataFrame({"artifact_KB": (p.total_bytes/1024).round(1),
                       "pJ_per_access": per.astype(int), "accesses": p.accesses.round(0),
                       "E_arith": a32.round(0), "E_mem": (p.accesses*per).round(0),
                       "E_total_nJ": (tot/1000).round(1),
                       "mem_share": (p.accesses*per/tot).round(3)})
    print(t2.to_string())
    print(f"\n  t2s_exit vs dnn_tree total: {tot['t2s_exit']/1000:.1f} vs "
          f"{tot['dnn_tree']/1000:.1f} nJ  -> ratio {tot['dnn_tree']/tot['t2s_exit']:.2f}x")
    print(f"  (arithmetic-only ratio was {a32['dnn_tree']/a32['t2s_exit']:.2f}x)")

    print("\n  sensitivity: SRAM 5-100 pJ x DRAM 640-2600 pJ")
    recs = []
    for s in (5, 10, 20, 50, 100):
        for dr in (640, 1300, 2600):
            t, _ = e_total(p, sram_s=s, sram_m=s, sram_l=s, dram=dr)
            recs.append(dict(sram=s, dram=dr,
                             t2s_vs_dnn=round(t["dnn_tree"]/t["t2s_exit"], 2),
                             t2s_vs_cgbdt=round(t["compact_gbdt"]/t["t2s_exit"], 2),
                             t2s_vs_fgbdt=round(t["full_gbdt"]/t["t2s_exit"], 2),
                             winner=t.idxmin()))
    sens = pd.DataFrame(recs)
    print(sens.to_string(index=False))
    print(f"\n  lowest-energy system across all {len(sens)} settings: "
          f"{sorted(sens.winner.unique())}")

    print("\n" + "=" * 78)
    print("RUN 3 — SRAM-budget feasibility (does the whole artifact fit on chip?)")
    print("=" * 78)
    budgets = [("128 KB", 128*1024), ("256 KB", 256*1024), ("1 MB", 1024*1024),
               ("8 MB", 8*1024*1024), ("unlimited", np.inf)]
    fit = pd.DataFrame({name: (p.total_bytes <= b).map({True: "fits", False: "--"})
                        for name, b in budgets})
    fit.insert(0, "artifact_KB", (p.total_bytes/1024).round(1))
    print(fit.to_string())

    print("\n" + "=" * 78)
    print("RUN 5 — spiking systems repriced at Loihi-class per-SOP energy")
    print("=" * 78)
    loihi = p.macs*E_MAC_FP32 + p.sops*E_SOP_LOIHI + p.cmps*E_CMP_FP32
    t5 = pd.DataFrame({"E_arith_0.9pJ_per_SOP": a32.round(0),
                       f"E_arith_{E_SOP_LOIHI}pJ_per_SOP": loihi.round(0),
                       "rank_at_0.9": a32.rank().astype(int),
                       "rank_at_loihi": loihi.rank().astype(int)})
    print(t5.to_string())

    t1.join(t2).to_csv(OUT / "energy_model.csv")
    sens.to_csv(OUT / "energy_sensitivity.csv", index=False)
    print(f"\n[wrote] {OUT/'energy_model.csv'}, {OUT/'energy_sensitivity.csv'}")
