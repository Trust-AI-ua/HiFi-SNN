#!/usr/bin/env python3
"""How far does the screening protocol displace the design-space ranking?

This is the paper's own question reduced to one number. Both rankings cover the
same 27 configurations; only the evaluation protocol differs. Kendall's tau
between them, with the permutation test registered in
PREREGISTRATION_ADDENDUM.md §4 (10,000 permutations, RNG seed 20260814),
measures the displacement directly.

Runs against whatever confirmation protocols are complete, and says which those
are. A partial answer computed on four protocols is useful and clearly labelled;
a silent one computed on three would not be.

Emits:
  tau.json                 tau, permutation p, movers, protocols covered
  rank_comparison.csv      per configuration: screening rank, confirmation rank
  fig_rank_shift.{pdf,png} slopegraph, screening -> confirmation

    python scripts/protocol_shift.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.populations import macro_f1_pair, members, require_population  # noqa: E402
from src.readout import discover  # noqa: E402

OUT = ROOT / "results/analysis/protocol_shift"
FIG = ROOT / "paper/figures"
CONF_PROTOCOLS = ["nslkdd_v2", "kddcup99_v2", "cicids2017_v2",
                  "ctu13_v2_f0", "ctu13_v2_f1"]
N_PERM, PERM_SEED = 10_000, 20260814


def mean_ranks(wide: pd.DataFrame) -> pd.Series:
    r = np.apply_along_axis(lambda v: rankdata(-v, method="average"), 1,
                            wide.to_numpy())
    return pd.Series(r.mean(axis=0), index=wide.columns)


def screening_ranks() -> pd.Series:
    d = pd.read_csv(ROOT / "results/v1_submitted/sweep_540.csv")
    d["variant"] = d.neuron + "/" + d.encoding
    require_population(d.variant.unique(), "full_27_screening")
    return mean_ranks(d.pivot_table(index=["dataset", "seed"], columns="variant",
                                    values="f1_macro"))


def confirmation_ranks():
    """Mean ranks over whatever confirmation protocols are complete.

    A protocol enters only when all 27 configurations are present at 5 seeds;
    a partially-filled protocol would rank an incomplete field.
    """
    rows = []
    for r in discover():
        if r.protocol not in CONF_PROTOCOLS:
            continue
        rec = r.record()
        pre = r.prefix()
        m = macro_f1_pair(pre["y_true"].astype(np.int64),
                          pre["y_pred"].astype(np.int64),
                          rec["data"]["num_classes"])
        rows.append({"protocol": r.protocol, "variant": r.variant,
                     "seed": r.seed, "f1": m["macro_f1_fixed_universe"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.Series(dtype=float), [], df
    full = [p for p, g in df.groupby("protocol")
            if g.variant.nunique() == 27 and g.groupby("variant").seed.nunique().min() == 5]
    if not full:
        return pd.Series(dtype=float), [], df
    sub = df[df.protocol.isin(full)]
    require_population(sub.variant.unique(), "full_27_confirmation")
    wide = sub.pivot_table(index=["protocol", "seed"], columns="variant",
                           values="f1").dropna()
    return mean_ranks(wide), full, df


def slopegraph(cmp: pd.DataFrame, protocols, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": None,
        "savefig.pad_inches": 0.0, "font.family": "serif", "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
    })
    fig, ax = plt.subplots(figsize=(3.487, 4.6), layout="constrained")
    for _, r in cmp.iterrows():
        lat = r["variant"].endswith("/latency")
        ax.plot([0, 1], [r.rank_screening, r.rank_confirmation],
                color="#0072B2" if lat else "#BBBBBB",
                lw=1.4 if lat else 0.7, alpha=1.0 if lat else 0.55,
                zorder=3 if lat else 1)
    for side, col, x, ha in ((("rank_screening"), "screening", -0.03, "right"),
                             (("rank_confirmation"), "confirmation", 1.03, "left")):
        for _, r in cmp.iterrows():
            ax.text(x, r[side], r["variant"], fontsize=5.0, ha=ha,
                    va="center",
                    color="#0072B2" if r["variant"].endswith("/latency")
                    else "#777777")
    ax.set_xlim(-0.72, 1.72)
    ax.set_ylim(cmp[["rank_screening", "rank_confirmation"]].max().max() + 1, 0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["screening", "confirmation"], fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel("mean rank (1 = best, top)")
    ax.set_title(f"{len(protocols)} confirmation protocols", fontsize=8,
                 loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext))
    plt.close(fig)


def main() -> int:
    scr = screening_ranks()
    conf, protocols, raw = confirmation_ranks()
    OUT.mkdir(parents=True, exist_ok=True)

    if conf.empty:
        have = (raw.groupby("protocol").variant.nunique().to_dict()
                if not raw.empty else {})
        print("no confirmation protocol has all 27 configurations at 5 seeds yet.")
        print(f"  configurations present per protocol: {have}")
        return 0

    cmp = pd.DataFrame({"rank_screening": scr, "rank_confirmation": conf}).dropna()
    cmp.index.name = "variant"
    cmp = cmp.reset_index()
    cmp["pos_screening"] = rankdata(cmp.rank_screening, method="average")
    cmp["pos_confirmation"] = rankdata(cmp.rank_confirmation, method="average")
    cmp["movement"] = cmp.pos_screening - cmp.pos_confirmation   # + = improved
    cmp = cmp.sort_values("rank_confirmation")
    cmp.to_csv(OUT / "rank_comparison.csv", index=False, float_format="%.6f")

    tau, _ = kendalltau(cmp.rank_screening, cmp.rank_confirmation)
    rng = np.random.default_rng(PERM_SEED)
    a = cmp.rank_screening.to_numpy()
    b = cmp.rank_confirmation.to_numpy()
    null = np.array([kendalltau(a, rng.permutation(b))[0] for _ in range(N_PERM)])
    p = float((np.abs(null) >= abs(tau)).mean())

    up = cmp.nlargest(5, "movement")[["variant", "pos_screening",
                                      "pos_confirmation", "movement"]]
    dn = cmp.nsmallest(5, "movement")[["variant", "pos_screening",
                                       "pos_confirmation", "movement"]]
    payload = {
        "protocols_included": protocols,
        "n_protocols": len(protocols),
        "complete": len(protocols) == len(CONF_PROTOCOLS),
        "n_configurations": int(len(cmp)),
        "kendall_tau": float(tau),
        "permutation_p": p,
        "n_permutations": N_PERM, "rng_seed": PERM_SEED,
        "leader_screening": cmp.sort_values("rank_screening").variant.iloc[0],
        "leader_confirmation": cmp.variant.iloc[0],
        "biggest_gainers": up.to_dict("records"),
        "biggest_losers": dn.to_dict("records"),
    }
    (OUT / "tau.json").write_text(json.dumps(payload, indent=2))
    slopegraph(cmp, protocols, FIG / "fig_rank_shift")

    status = "COMPLETE" if payload["complete"] else "PARTIAL"
    print(f"=== protocol shift [{status}] over {len(protocols)} protocols: "
          f"{', '.join(protocols)} ===")
    print(f"Kendall tau = {tau:.4f}  (permutation p = {p:.4f}, "
          f"{N_PERM} permutations, seed {PERM_SEED})")
    print(f"leader screening    : {payload['leader_screening']}")
    print(f"leader confirmation : {payload['leader_confirmation']}")
    print("\ntop 10 by confirmation rank:")
    print(cmp.head(10)[["variant", "rank_screening", "rank_confirmation",
                        "movement"]].round(3).to_string(index=False))
    print("\nlargest upward movement:"); print(up.to_string(index=False))
    print("largest downward movement:"); print(dn.to_string(index=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
