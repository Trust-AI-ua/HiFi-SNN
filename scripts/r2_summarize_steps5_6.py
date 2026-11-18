#!/usr/bin/env python3
"""Revision-2 steps 5 and 6: variance decomposition and factorial ablation summaries
(results/preregistration/2026-11-17_r2_steps4_6.md). Descriptive; no decision rule.

Step 5. For CICIoT2023 and CICIoMT2024: every run's test macro-F1 and FAR; arm D (seed 42,
data seeds 42..46) and arm S (data seed 42, seeds 42..46) with their sample SD, beside the
published five-seed SD (results/Tree-to-Spike, both seeds varying). The shared cell (42, 42) is
compared with the stored seed-42 record; any difference is run-to-run nondeterminism.

Step 6. Encoder (tree/raw) x distillation (kd/nokd) x window (T20/T25), Edge-IIoTset and
CICIoMT2024, five seeds. T = 20 cells tree/kd, tree/nokd and raw/nokd are reused from
results/Tree-to-Spike, results/encoder_ablation/<ds>/nokd and .../m8_raw if their configuration
equals the new cells' apart from the factor keys (defaults filled in: encoder_mode 'tree',
class_weighted False; 'device' ignored) and their recorded data (split sizes, training class
counts) equals the new cells'. A reused cell that fails is reported and the factorial is not
computed for that dataset until it is rerun. Effects per seed from that seed's eight cells,
coded -1/+1 (raw/tree, nokd/kd, T20/T25): an effect is the mean of the cells where the product
of its factors' codes is +1 minus the mean where it is -1. Reported as the mean over seeds with
a t-based 95% interval (4 degrees of freedom).

    .venv/bin/python scripts/r2_summarize_steps5_6.py
Output: results/journal_r2/variance_summary.{csv,md}, factorial_cells.csv, factorial_effects.csv,
        factorial_summary.md
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
R2 = REPO / "results/journal_r2"
SEEDS = list(range(42, 47))
FACTOR_KEYS = {"encoder_mode", "a_kd", "a_rank", "T"}
RUN_KEYS = {"seed", "results_dir", "summary_csv", "data_seed", "device"}
DEFAULTS = {"encoder_mode": "tree", "class_weighted": False}
REUSE = {("tree", "kd"): "results/Tree-to-Spike/{ds}", ("tree", "nokd"): "results/encoder_ablation/{ds}/nokd",
         ("raw", "nokd"): "results/encoder_ablation/{ds}/m8_raw"}


def metrics(p: Path):
    r = json.loads(p.read_text())
    m = r["test_plus"]
    return 100 * m["metrics"]["f1_macro"], 100 * m["binary_ids_metrics"]["false_alarm_rate"], r


def variance():
    rows, md = [], ["# Step 5: variance decomposition (test split, percent)", ""]
    for ds in ("ciciot2023", "ciciomt2024"):
        cells = [("D", 42, d) for d in SEEDS] + [("S", s, 42) for s in SEEDS if s != 42]
        got = {}
        for arm, s, d in cells:
            p = R2 / f"variance/{ds}/data_{d}/seed_{s}/results.json"
            if p.exists():
                f1, far, _ = metrics(p)
                got[(s, d)] = (f1, far)
                rows.append(dict(dataset=ds, arm="D+S" if (s, d) == (42, 42) else arm, seed=s, data_seed=d,
                                 macro_f1=f1, far=far))
        pub = [metrics(REPO / f"results/Tree-to-Spike/{ds}/seed_{s}/results.json")[:2] for s in SEEDS]
        armD = [got[(42, d)] for d in SEEDS if (42, d) in got]
        armS = [got[(s, 42)] for s in SEEDS if (s, 42) in got]
        sd = lambda xs, i: float(np.std([x[i] for x in xs], ddof=1)) if len(xs) > 1 else float("nan")
        mean = lambda xs, i: float(np.mean([x[i] for x in xs])) if xs else float("nan")
        md += [f"## {ds}", "", "| Set | Runs | Macro-F1 mean | Macro-F1 SD | FAR mean | FAR SD |", "|---|--:|--:|--:|--:|--:|"]
        for name, xs in (("Arm D (data seed varies)", armD), ("Arm S (training seed varies)", armS),
                         ("Published (both vary)", pub)):
            md.append(f"| {name} | {len(xs)} | {mean(xs, 0):.2f} | {sd(xs, 0):.2f} | {mean(xs, 1):.2f} | {sd(xs, 1):.2f} |")
        if (42, 42) in got:
            d_f1, d_far = got[(42, 42)][0] - pub[0][0], got[(42, 42)][1] - pub[0][1]
            md += ["", f"Shared cell (42, 42) against the stored seed-42 record: macro-F1 {d_f1:+.4f}, FAR {d_far:+.4f}"
                   + (" (identical)." if d_f1 == 0 and d_far == 0 else " (run-to-run nondeterminism).")]
        done = len(armD) == 5 and len(armS) == 5
        md += ["", "" if done else f"**Incomplete: arm D {len(armD)}/5, arm S {len(armS)}/5.**", ""]
    pd.DataFrame(rows).to_csv(R2 / "variance_summary.csv", index=False)
    (R2 / "variance_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def norm_cfg(c):
    c = {**DEFAULTS, **c}
    return {k: v for k, v in c.items() if k not in RUN_KEYS | FACTOR_KEYS}


def factorial():
    import r2_train_cell as C
    cells, issues = [], []
    for ds in ("edgeiiot", "ciciomt2024"):
        new_ref = {}
        for enc, kd, T in itertools.product(("tree", "raw"), ("kd", "nokd"), (20, 25)):
            reused = T == 20 and (enc, kd) in REUSE
            base = REUSE[(enc, kd)].format(ds=ds) if reused else f"results/journal_r2/factorial/{ds}/{enc}_{kd}_T{T}"
            for s in SEEDS:
                p = REPO / base / f"seed_{s}/results.json"
                if not p.exists():
                    continue
                f1, far, r = metrics(p)
                if not reused and s not in new_ref:
                    new_ref[s] = r                     # the data split depends on the seed: compare like with like
                cells.append(dict(dataset=ds, encoder=enc, distillation=kd, T=T, seed=s, macro_f1=f1, far=far,
                                  reused=reused, source=base, _rec=r))
        # reuse check
        want = norm_cfg(C.production(ds))
        for c in cells:
            if c["dataset"] != ds or not c["reused"]:
                continue
            got = norm_cfg(c["_rec"]["config"])
            diff = {k: (got.get(k), want.get(k)) for k in set(got) | set(want)
                    if k not in ("subsample_fraction", "label_mode", "dataset", "test_fraction") and got.get(k) != want.get(k)}
            ref = new_ref.get(c["seed"])
            if ref is not None:
                for k in ("splits", "class_counts_train"):
                    if c["_rec"]["data"].get(k) != ref["data"].get(k):
                        diff[f"data.{k}"] = "differs from the new cells of the same seed"
            else:
                diff.setdefault("data", "no new cell yet to compare the recorded data with")
            c["reuse_ok"] = not diff
            if diff:
                issues.append(f"{ds} {c['encoder']}/{c['distillation']}/T20 seed {c['seed']}: {diff}")
    df = pd.DataFrame([{k: v for k, v in c.items() if k != "_rec"} for c in cells])
    df.to_csv(R2 / "factorial_cells.csv", index=False)
    md = ["# Step 6: factorial ablation (test macro-F1 and FAR, percent)", ""]
    eff_rows = []
    codes = {"E": ("encoder", {"raw": -1, "tree": 1}), "K": ("distillation", {"nokd": -1, "kd": 1}), "W": ("T", {20: -1, 25: 1})}
    for ds in ("edgeiiot", "ciciomt2024"):
        d = df[df.dataset == ds] if len(df) else df
        md += [f"## {ds}", "", "| Encoder | Distillation | T | Seeds | Macro-F1 mean | SD | FAR mean | SD | Source |",
               "|---|---|--:|--:|--:|--:|--:|--:|---|"]
        for (enc, kd, T), g in d.groupby(["encoder", "distillation", "T"]) if len(d) else []:
            md.append(f"| {enc} | {kd} | {T} | {len(g)} | {g.macro_f1.mean():.2f} | {g.macro_f1.std(ddof=1):.2f} | "
                      f"{g.far.mean():.2f} | {g.far.std(ddof=1):.2f} | {'reused' if g.reused.all() else 'new'} |")
        complete = len(d) == 40
        bad = [i for i in issues if i.startswith(ds) and "no new cell yet" not in i]
        pending = [i for i in issues if i.startswith(ds) and "no new cell yet" in i]
        if pending:
            md += ["", f"Reuse check: configuration equal for the reused cells; the data comparison is pending "
                   f"({len(pending)} cell-seeds) until a new cell exists."]
        if bad:
            md += ["", f"**Reuse check failed for {len(bad)} cell-seeds; those cells must be rerun before effects are computed:**", ""]
            md += [f"- {i}" for i in bad]
        if not complete or bad or pending:
            md += ["", f"**Effects not computed: {len(d)}/40 cell-seeds present{', reuse check failing' if bad else ''}{', reuse data check pending' if pending else ''}.**", ""]
            continue
        per_seed = []
        for s in SEEDS:
            g = d[d.seed == s].copy()
            for f, (col, mp) in codes.items():
                g[f] = g[col].map(mp)
            for r in range(1, 4):
                for combo in itertools.combinations("EKW", r):
                    sign = np.prod([g[f].values for f in combo], axis=0)
                    for m in ("macro_f1", "far"):
                        per_seed.append(dict(dataset=ds, effect="x".join(combo), metric=m, seed=s,
                                             value=g[m].values[sign > 0].mean() - g[m].values[sign < 0].mean()))
        ps = pd.DataFrame(per_seed)
        t = stats.t.ppf(0.975, 4)
        md += ["", "Effects (+1 minus -1; E: tree vs raw, K: kd vs nokd, W: T25 vs T20), mean over seeds, 95% t interval:", "",
               "| Effect | Macro-F1 | 95% CI | FAR | 95% CI |", "|---|--:|---|--:|---|"]
        for eff, g in ps.groupby("effect", sort=False):
            cells_ = []
            for m in ("macro_f1", "far"):
                v = g[g.metric == m].value
                h = t * v.std(ddof=1) / np.sqrt(len(v))
                cells_ += [f"{v.mean():+.2f}", f"[{v.mean() - h:+.2f}, {v.mean() + h:+.2f}]"]
                eff_rows.append(dict(dataset=ds, effect=eff, metric=m, mean=v.mean(), sd=v.std(ddof=1), ci_low=v.mean() - h,
                                     ci_high=v.mean() + h, n_seeds=len(v)))
            md.append(f"| {eff} | " + " | ".join(cells_) + " |")
        md.append("")
    pd.DataFrame(eff_rows).to_csv(R2 / "factorial_effects.csv", index=False)
    (R2 / "factorial_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    variance()
    factorial()


if __name__ == "__main__":
    main()
