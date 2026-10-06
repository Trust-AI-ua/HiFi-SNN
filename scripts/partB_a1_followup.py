#!/usr/bin/env python3
"""A1 follow-up for the CICIoT2023 block: early exit, flags and exit audit from the
selected checkpoints of the block's Tree-to-Spike runs (pre-registration amendment A1).

Inputs: prefix outputs of the block's tree runs (scripts/extract_prefix.py with
--results-root results/journal_partB/B6/block/tree, gated on exact reconstruction of each
block record) under results/journal_partB/B6/followup/prefix_outputs, and the early-exit
selection on them (scripts/early_exit_sweep.py with --prefix-root/--out-root, validation
only, same eight thresholds and one-point rule).

Computed per seed with the published definitions, imported or copied unchanged:
  flags        scripts/m4_remainder.py: theta_unc = 0.5 flag rate at the selected exit
               threshold (full window where the gate rejects exit), exits before the first
               input spike, exit share
  exit audit   scripts/exit_audit_iot.py: inclusion, t' median and p90 (pooled over seeds),
               wrong exits at the fixed diagnostic theta = 0.90
Guard, before any CICIoT2023 value: the same code on the production prefix outputs
reproduces the published rows of results/tables/tab_exit_audit_iot.csv (all IoT cells) and
results/tables/tab_m4_flags.csv (IoT datasets) exactly.

    python scripts/partB_a1_followup.py
Output: results/journal_partB/B6/followup/{flags.csv,exit_audit.csv,summary.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402
import early_exit_sweep as ES  # noqa: E402

FU = C.REPO / "results/journal_partB/B6/followup"
THETA_UNC, TH_AUDIT = 0.5, 0.90


def flags_cell(pt, disabled, theta):
    y = pt["labels"]
    n = len(y)
    final_conf = ES.softmax(pt["prefix_logits"][:, -1, :].astype(np.float32)).max(axis=1)
    if disabled:
        exited = np.zeros(n, bool)
        step = np.full(n, pt["prefix_logits"].shape[1])
    else:
        _, step, exited, _, _ = ES.replay(pt, float(theta))
    unc = (~exited) & (final_conf < THETA_UNC)
    in_at_exit = pt["cum_in_spikes"][np.arange(n), step - 1]
    before = exited & (in_at_exit == 0)
    return dict(unc_flag_rate=100 * unc.mean(), exit_before_first_input=100 * before.mean(),
                exited=100 * exited.mean())


def audit_cell(pt):
    q = ES.softmax(pt["prefix_logits"].astype(np.float32), axis=-1)
    conf, pred, y = q.max(-1), q.argmax(-1), pt["labels"]
    T = pred.shape[1]
    ok = pred[:, -1] == y
    tp = ((pred == y[:, None]) & ok[:, None]).argmax(1)[ok] + 1
    fired = conf >= TH_AUDIT
    ex = fired.any(1)
    t_star = np.where(ex, fired.argmax(1), T - 1)
    wrong = ex & (pred[np.arange(len(y)), t_star] != y)
    return dict(inclusion=100 * ok.mean(), t_median=float(np.median(tp)), t_p90=float(np.percentile(tp, 90)),
                wrong_exit=100 * wrong.mean(), exit_share=100 * ex.mean()), tp


def guard():
    ES.PREFIX = C.REPO / "results/prefix_outputs"
    pub = pd.read_csv(C.REPO / "results/tables/tab_exit_audit_iot.csv")
    for r in pub.itertuples():
        a, _ = audit_cell(ES.load(r.dataset, r.seed, "test"))
        for k in ("inclusion", "t_median", "t_p90", "wrong_exit", "exit_share"):
            if abs(a[k] - getattr(r, k)) > 1e-9:
                sys.exit(f"guard: exit audit {r.dataset}/{r.seed} {k} {a[k]} vs published {getattr(r, k)}")
    sel = pd.read_csv(C.REPO / "results/early_exit/selected_test.csv")
    sel = sel[sel.dataset.isin(C.IOT)]
    rows = []
    for r in sel.itertuples():
        f = flags_cell(ES.load(r.dataset, r.seed, "test"), r.early_exit_disabled, r.selected_threshold)
        rows.append(dict(dataset=r.dataset, **f))
    g = pd.DataFrame(rows).groupby("dataset").unc_flag_rate.mean()
    pubf = pd.read_csv(C.REPO / "results/tables/tab_m4_flags.csv", index_col=0)
    for ds in C.IOT:
        if abs(g[ds] - pubf.loc[ds, "unc"]) > 1e-9:
            sys.exit(f"guard: flag rate {ds} {g[ds]} vs published {pubf.loc[ds, 'unc']}")
    return f"exit audit ({len(pub)} cells) and IoT flag rates reproduced exactly from production prefixes"


def main():
    st = C.stamp("A1_followup")
    g = guard()
    print("GUARD:", g, flush=True)
    ES.PREFIX = FU / "prefix_outputs"
    sel = pd.read_csv(FU / "early_exit/selected_test.csv")
    if sorted(sel.seed) != list(C.SEEDS) or set(sel.dataset) != {"ciciot2023"}:
        sys.exit(f"follow-up selection incomplete: {sel[['dataset', 'seed']].values.tolist()}")
    flags, audit, tps = [], [], []
    for r in sel.itertuples():
        pt = ES.load("ciciot2023", r.seed, "test")
        flags.append(dict(dataset="ciciot2023", seed=r.seed, exit_admitted=not r.early_exit_disabled,
                          theta=r.selected_threshold, **flags_cell(pt, r.early_exit_disabled, r.selected_threshold)))
        a, tp = audit_cell(pt)
        audit.append(dict(dataset="ciciot2023", seed=r.seed, n=len(pt["labels"]), **a))
        tps.append(tp)
    F, A = pd.DataFrame(flags), pd.DataFrame(audit)
    F.to_csv(FU / "flags.csv", index=False)
    A.to_csv(FU / "exit_audit.csv", index=False)
    tp = np.concatenate(tps)
    summary = dict(stamp=st, guard=g,
                   admitted=int(F.exit_admitted.sum()), flag_rate_mean=float(F.unc_flag_rate.mean()),
                   flag_rate_sd=float(F.unc_flag_rate.std(ddof=1)),
                   wrong_exit_mean=float(A.wrong_exit.mean()), wrong_exit_min=float(A.wrong_exit.min()),
                   wrong_exit_max=float(A.wrong_exit.max()), inclusion_mean=float(A.inclusion.mean()),
                   tprime_median_pooled=float(np.median(tp)), tprime_p90_pooled=float(np.percentile(tp, 90)),
                   selected=sel[["seed", "selected_threshold", "early_exit_disabled", "full_macro_f1",
                                 "exit_macro_f1", "d_macro_f1", "mean_exit", "mean_sop_reduction"]].to_dict("records"))
    (FU / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "stamp"}, indent=1, default=float))


if __name__ == "__main__":
    main()
