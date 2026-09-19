#!/usr/bin/env python3
"""Memory-inclusive energy, per dataset, under two reuse assumptions.

Supersedes the aggregate model in deployment_energy.py on two counts.

1. Energy is computed PER DATASET and then averaged. The previous version averaged
   mean_exit across datasets first, which is wrong: 5 of 11 datasets have early exit
   disabled by the validation gate and are scored at the full 20-step horizon, so the
   13.29 aggregate is a mean effective horizon, not a mean exit step (enabled-only
   mean is 7.69).

2. Two memory variants, reported side by side rather than one chosen:

   cold   one tier-priced access per operation. Pessimistic for every system, but
          NOT equally: it is the only model under which a 5 MB tree walk and a 53 KB
          weight matrix pay the same per-access price, which flatters the tree.
   reuse  first touch of a distinct item at the tier price, subsequent touches at L1.
          A 53 KB working set is L2-resident after the first pass; 76k tree nodes
          visited 7.6k times are almost all cold.

The truth is between them. The band is the result; neither point estimate is.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]; OUT = REPO/"results"/"deployment"
E_MAC, E_ADD = 4.6, 0.9            # Horowitz 45 nm fp32; comparison = add class
E_L1 = 1.0                         # warm hit
HIDDEN = 128

def tier(b, sram, dram):
    return sram if b <= 1024*1024 else dram

def row_energy(r, sram, dram, reuse):
    macs = 0.0 if pd.isna(r.macs) else r.macs
    sops = 0.0 if pd.isna(r.sops) else r.sops
    cmps = 0.0 if pd.isna(r.tree_comparisons) else r.tree_comparisons
    is_snn, is_tree = sops > 0, str(r.system).endswith("gbdt")
    e_arith = macs*E_MAC + sops*E_ADD + cmps*E_ADD

    w_ops = sops if is_snn else (cmps if is_tree else macs)
    exit_t = r.mean_exit if not pd.isna(r.mean_exit) else 1.0
    state = HIDDEN*exit_t*2 if is_snn else (0.0 if is_tree else HIDDEN)
    inputs = r.n_channels if not pd.isna(r.n_channels) else 78.0
    per = tier(r.total_bytes, sram, dram)

    if not reuse:
        e_mem = (w_ops + state + inputs) * per
    else:
        # distinct items: int8 weights ~ 1 B each; tree nodes counted directly
        distinct = r.n_nodes if is_tree and r.n_nodes > 0 else max(r.model_bytes, 1)
        cold = min(w_ops, distinct); warm = max(w_ops - cold, 0.0)
        e_mem = cold*per + warm*E_L1 + (state + inputs)*E_L1
    return e_arith, e_mem

def table(d, sram=100, dram=1300, reuse=False):
    out = []
    for _, r in d.iterrows():
        a, m = row_energy(r, sram, dram, reuse)
        out.append(dict(dataset=r.dataset, system=r.system, e_arith=a, e_mem=m, e_tot=a+m))
    return pd.DataFrame(out).groupby("system").e_tot.mean()

if __name__ == "__main__":
    d = pd.read_csv(OUT/"deployment_measurements.csv")
    order = ["full_gbdt","compact_gbdt","raw_mlp_fp","dnn_tree","t2s_full","t2s_exit"]
    print("="*74); print("Per-dataset total energy (nJ), two reuse assumptions"); print("="*74)
    for reuse in (False, True):
        t = table(d, reuse=reuse).reindex(order)/1000
        print(f"\n-- {'REUSE-AWARE' if reuse else 'COLD (one access per op)'} @ SRAM 100 / DRAM 1300 pJ")
        print(t.round(1).to_string())
        print(f"   dnn_tree / t2s_exit = {t['dnn_tree']/t['t2s_exit']:.2f}x")

    print("\n" + "="*74); print("Crossover sweep: dnn_tree / t2s_exit ratio (>1 = SNN wins)"); print("="*74)
    rec = []
    for reuse in (False, True):
        for s in (5, 10, 20, 50, 100):
            t = table(d, sram=s, reuse=reuse)
            rec.append(dict(model="reuse" if reuse else "cold", sram_pJ=s,
                            ratio=round(t["dnn_tree"]/t["t2s_exit"], 2),
                            vs_compact_gbdt=round(t["compact_gbdt"]/t["t2s_exit"], 1)))
    s = pd.DataFrame(rec)
    print(s.pivot(index="sram_pJ", columns="model", values="ratio").to_string())
    print("\nvs compact_gbdt:")
    print(s.pivot(index="sram_pJ", columns="model", values="vs_compact_gbdt").to_string())
    s.to_csv(OUT/"energy_crossover.csv", index=False)
    print(f"\n[wrote] {OUT/'energy_crossover.csv'}")
