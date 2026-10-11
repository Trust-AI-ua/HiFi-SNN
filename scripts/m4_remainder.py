#!/usr/bin/env python3
"""M4 remainder, from the regenerated (input-inclusive) prefix artifacts.

Per cell (dataset x seed), at the validation-selected exit threshold from
results/early_exit/selected_test.csv (full window where the gate rejected exit):

  1. theta_unc flag rate: share of test samples that never reach theta_exit AND end
     with final confidence below theta_unc = 0.5, i.e. the samples the deployed
     model would flag as uncertain. For rejected cells every sample runs to T, so
     the flag is final confidence < 0.5.
  2. exits before the first input spike: share of test samples whose exit step
     comes before any input channel has fired (cum_in_spikes == 0 at exit), so the
     decision rests on biases alone.
  3. accuracy-versus-operations curve: test macro-F1 against mean input-inclusive
     SOPs for every theta in the sweep grid plus the full window. The curve is
     descriptive; thresholds are SELECTED on validation, never on this curve.

The exit rule, confidence and scoring are imported from scripts/early_exit_sweep.py
so this cannot diverge from the published exit tables.

    python scripts/m4_remainder.py
Output: results/tables/tab_m4_flags.{csv,md}, results/tables/m4_accuracy_vs_ops.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from early_exit_sweep import THETAS, load, replay, full_window, score, softmax  # noqa: E402

THETA_UNC = 0.5
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]


def main():
    sel = pd.read_csv(REPO / "results/early_exit/selected_test.csv")
    flags, curve = [], []
    for r in sel.itertuples():
        pt = load(r.dataset, r.seed, "test")
        y = pt["labels"]
        n = len(y)
        final_conf = softmax(pt["prefix_logits"][:, -1, :].astype(np.float32)).max(axis=1)
        if r.early_exit_disabled:
            exited = np.zeros(n, bool)
            step = np.full(n, pt["prefix_logits"].shape[1])
        else:
            _, step, exited, _, _ = replay(pt, float(r.selected_threshold))
        unc = (~exited) & (final_conf < THETA_UNC)
        in_at_exit = pt["cum_in_spikes"][np.arange(n), step - 1]
        before_input = exited & (in_at_exit == 0)
        flags.append(dict(dataset=r.dataset, seed=r.seed, exit_admitted=not r.early_exit_disabled,
                          theta=r.selected_threshold, unc_flag_rate=100 * unc.mean(),
                          exit_before_first_input=100 * before_input.mean(),
                          exit_before_first_input_of_exiting=(100 * before_input.sum() / exited.sum())
                          if exited.any() else np.nan,
                          exited=100 * exited.mean()))
        for th in THETAS:
            p, st, _, sops, _ = replay(pt, th)
            curve.append(dict(dataset=r.dataset, seed=r.seed, theta=th, mean_exit=float(st.mean()),
                              mean_sops=float(sops.mean()), **score(y, p)))
        p, st, sops, _ = full_window(pt)
        curve.append(dict(dataset=r.dataset, seed=r.seed, theta="full", mean_exit=float(st.mean()),
                          mean_sops=float(sops.mean()), **score(y, p)))
    f = pd.DataFrame(flags)
    c = pd.DataFrame(curve)
    c.to_csv(REPO / "results/tables/m4_accuracy_vs_ops.csv", index=False)
    g = f.groupby("dataset").agg(admitted=("exit_admitted", "sum"), unc=("unc_flag_rate", "mean"),
                                 unc_sd=("unc_flag_rate", "std"),
                                 before=("exit_before_first_input", "mean"),
                                 before_of_exiting=("exit_before_first_input_of_exiting", "mean"),
                                 exited=("exited", "mean"))
    g["iot"] = g.index.isin(IOT)
    g = g.sort_values("iot", ascending=False)
    g.to_csv(REPO / "results/tables/tab_m4_flags.csv")
    print("per dataset, five-seed means, test split (%)")
    print(g.round(2).to_string())
    print("\naccuracy vs operations (IoT, five-seed mean): theta -> macro-F1 / mean SOPs")
    ci = c[c.dataset.isin(IOT)].copy()
    ci["theta"] = ci.theta.astype(str)
    piv = ci.groupby(["dataset", "theta"]).agg(f1=("macro_f1", "mean"), sops=("mean_sops", "mean"))
    for ds in IOT:
        s = piv.loc[ds]
        print(f"  {ds:12s} " + "  ".join(f"{t}:{s.loc[t,'f1']:.1f}/{s.loc[t,'sops']:.0f}"
                                          for t in [str(x) for x in THETAS] + ["full"]))
    md = ["| Dataset | Exit admitted (of 5) | θ_unc flag rate | Exit before first input spike (all samples) | … of exiting samples | Samples exiting |",
          "|---|---|---|---|---|---|"]
    for ds, r in g.iterrows():
        md.append(f"| {ds}{' (IoT)' if r.iot else ''} | {int(r.admitted)} | {r.unc:.2f}% ± {r.unc_sd:.2f} | "
                  f"{r.before:.2f}% | {r.before_of_exiting:.2f}% | {r.exited:.1f}% |")
    (REPO / "results/tables/tab_m4_flags.md").write_text("\n".join(md) + "\n")
    print("\n[wrote] results/tables/tab_m4_flags.{csv,md}, results/tables/m4_accuracy_vs_ops.csv")


if __name__ == "__main__":
    main()
