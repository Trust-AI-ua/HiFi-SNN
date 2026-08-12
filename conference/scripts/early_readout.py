#!/usr/bin/env python3
"""Anytime-inference analysis: how fast does each encoding accumulate evidence?

This is the mechanism result the encoding claim rests on. Terminal macro-F1 for
latency and rate is close and its dataset-level bootstrap interval includes
zero, so "latency is better" cannot carry the paper on quality alone. The
defensible question is instead whether latency coding reaches near-terminal
quality at a smaller temporal budget — and that is what T95 and T99 measure.

Reads only saved artifacts; trains nothing. Emits per (protocol, variant, seed):

    F1(t), MCC(t), DR(t), cumulative output spikes(t) for t = 1..T
    T95, T99 as fixed in PREREGISTRATION.md §11

and per (protocol, variant) the seed mean and SD.

    python scripts/early_readout.py --protocols nslkdd_v2 kddcup99_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.readout import (convergence_step, discover, prefix_predictions)  # noqa: E402

OUT = ROOT / "results/analysis/early_readout"
ENC_ORDER = ["latency", "rate", "delta"]


def curves_for(run) -> pd.DataFrame:
    pre = run.prefix()
    cum, y = pre["cum_out"], pre["y_true"].astype(np.int64)
    preds = prefix_predictions(cum)
    T = cum.shape[1]
    # Output spikes emitted up to t, per sample, averaged — the temporal cost of
    # deciding at t. Cumulative by construction, so no running sum needed.
    spikes_t = cum.astype(np.int64).sum(axis=2).mean(axis=0)

    rows = []
    for t in range(T):
        p = preds[:, t]
        rows.append({
            "t": t + 1,
            "f1_macro": f1_score(y, p, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y, p) if len(np.unique(p)) > 1 else 0.0,
            "DR": float(((p != 0) & (y != 0)).sum() / max((y != 0).sum(), 1)),
            "FAR": float(((p != 0) & (y == 0)).sum() / max((y == 0).sum(), 1)),
            "output_spikes_cum": float(spikes_t[t]),
        })
    df = pd.DataFrame(rows)
    df.insert(0, "seed", run.seed)
    df.insert(0, "encoding", run.encoding)
    df.insert(0, "neuron", run.neuron)
    df.insert(0, "protocol", run.protocol)
    return df


def plot_curves(curves: pd.DataFrame, protocol: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = curves[curves.protocol == protocol]
    if sub.empty:
        return
    colors = {"latency": "#4C72B0", "rate": "#DD8452", "delta": "#55A868"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for enc in ENC_ORDER:
        g = sub[sub.encoding == enc]
        if g.empty:
            continue
        m = g.groupby("t")[["f1_macro", "output_spikes_cum"]].mean()
        s = g.groupby("t")["f1_macro"].std().fillna(0)
        axes[0].plot(m.index, m.f1_macro, color=colors[enc], label=enc, lw=1.4)
        axes[0].fill_between(m.index, m.f1_macro - s, m.f1_macro + s,
                             color=colors[enc], alpha=0.18, linewidth=0)
        axes[1].plot(m.output_spikes_cum, m.f1_macro, color=colors[enc],
                     label=enc, lw=1.4)
    axes[0].set_xlabel("timestep $t$")
    axes[0].set_ylabel("macro-F1 at $t$")
    axes[0].set_title(f"{protocol}: evidence accumulation", fontsize=9)
    axes[1].set_xlabel("cumulative output spikes per sample")
    axes[1].set_ylabel("macro-F1")
    axes[1].set_title("quality vs temporal cost", fontsize=9)
    for ax in axes:
        ax.legend(fontsize=7, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=None)
    ap.add_argument("--neurons", nargs="+", default=["LeakyParallel"])
    args = ap.parse_args()

    runs = discover(protocols=args.protocols, neurons=args.neurons)
    if not runs:
        print("no runs with artifacts found yet — nothing to analyse.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)

    curves, summary = [], []
    for r in runs:
        c = curves_for(r)
        curves.append(c)
        f1 = c.f1_macro.to_numpy()
        t95, t99 = convergence_step(f1, 0.95), convergence_step(f1, 0.99)
        summary.append({
            "protocol": r.protocol, "neuron": r.neuron, "encoding": r.encoding,
            "seed": r.seed, "T": len(f1),
            "f1_terminal": f1[-1], "f1_peak": f1.max(),
            "t_peak": int(f1.argmax() + 1),
            "T95": t95, "T99": t99,
            "spikes_at_T99": (float(c.output_spikes_cum.iloc[t99 - 1])
                              if t99 else None),
            "spikes_terminal": float(c.output_spikes_cum.iloc[-1]),
        })
        print(f"  {r.protocol:24s} {r.variant:26s} s{r.seed}  "
              f"F1(T)={f1[-1]:.4f}  T95={t95}  T99={t99}", flush=True)

    curves_df = pd.concat(curves, ignore_index=True)
    summ = pd.DataFrame(summary)
    curves_df.to_csv(OUT / "curves.csv", index=False)
    summ.to_csv(OUT / "per_seed.csv", index=False)

    agg = (summ.groupby(["protocol", "neuron", "encoding"])
               .agg(n=("seed", "size"),
                    f1_terminal_mean=("f1_terminal", "mean"),
                    f1_terminal_sd=("f1_terminal", "std"),
                    T95_mean=("T95", "mean"), T95_sd=("T95", "std"),
                    T95_median=("T95", "median"),
                    T99_mean=("T99", "mean"), T99_sd=("T99", "std"),
                    T99_median=("T99", "median"),
                    # The across-seed distribution of T99 is skewed: most seeds
                    # converge in a few steps and an occasional one does not, so
                    # the mean sits well above the typical run. Both are
                    # reported; the median is the honest headline. The T95/T99
                    # definitions themselves are frozen (PREREGISTRATION.md 11)
                    # -- this is a choice of summary statistic, not of metric.
                    T99_min=("T99", "min"), T99_max=("T99", "max"),
                    spikes_at_T99_mean=("spikes_at_T99", "mean"),
                    spikes_terminal_mean=("spikes_terminal", "mean"))
               .reset_index())
    agg["spike_saving_at_T99_pct"] = 100 * (
        1 - agg.spikes_at_T99_mean / agg.spikes_terminal_mean)
    agg.to_csv(OUT / "summary.csv", index=False)

    for proto in sorted(curves_df.protocol.unique()):
        plot_curves(curves_df, proto, OUT / f"readout_{proto}")

    print("\n=== T95 / T99 by protocol and encoding (seed mean ± SD) ===")
    show = agg.copy()
    for c in ("T95", "T99"):
        show[c] = (show[f"{c}_median"].round(1).astype(str) + " (mean "
                   + show[f"{c}_mean"].round(1).astype(str) + " ± "
                   + show[f"{c}_sd"].fillna(0).round(1).astype(str) + ")")
    show["F1(T)"] = (show.f1_terminal_mean.round(4).astype(str) + " ± "
                     + show.f1_terminal_sd.fillna(0).round(4).astype(str))
    print(show[["protocol", "neuron", "encoding", "n", "F1(T)", "T95", "T99",
                "spike_saving_at_T99_pct"]].to_string(index=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
