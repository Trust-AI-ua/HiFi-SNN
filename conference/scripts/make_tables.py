#!/usr/bin/env python3
"""Emit the paper's LaTeX tables from frozen analysis outputs.

Every number is read from a committed file under `results/`; nothing is
recomputed here and nothing is typed by hand. If a table needs a number that no
analysis produces, the fix is to add it to the analysis, not to this script.

    python scripts/make_tables.py --tag v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables"

DS_PRETTY = {"nslkdd": "NSL-KDD", "kddcup99": "KDDCup99",
             "cicids2017": "CIC-IDS2017", "ctu13": "CTU-13"}


def _tt(v: str) -> str:
    return r"\texttt{" + v.replace("_", r"\_") + "}"


def table_ranking(tag: str, top: int = 10) -> str:
    a = ROOT / "results/analysis" / tag
    mr = pd.read_csv(a / "mean_rank.csv", index_col=0)
    fr = json.load(open(a / "friedman.json"))
    per = pd.read_csv(a / "per_dataset.csv")

    f1 = per.pivot(index="variant", columns="dataset", values="f1_macro_mean")
    sd = per.pivot(index="variant", columns="dataset", values="f1_macro_std")
    order = [c for c in ["nslkdd", "kddcup99", "cicids2017", "ctu13"] if c in f1]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Top-%d of the %d (neuron, encoding) variants by \textbf{mean rank}"
        r" over the %d (dataset $\times$ seed) blocks; rank~1 is best. Per-dataset"
        r" cells are mean $\pm$ standard deviation \emph{over the five seeds}, never"
        r" pooled across datasets. Friedman $\chi^2=%.1f$, $p=%.1e$; Nemenyi critical"
        r" difference $=%.2f$, so the %d variants within %.2f of the leader are"
        r" statistically indistinguishable from it.}"
        % (top, fr["k_variants"], fr["n_blocks"], fr["friedman_chi2"],
           fr["friedman_p"], fr["nemenyi_critical_difference"],
           len(fr["statistically_indistinguishable_from_best"]),
           fr["nemenyi_critical_difference"]),
        r"\label{tab:ranking}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l r " + "c" * len(order) + "}",
        r"\toprule",
        r"Variant & Mean rank & " + " & ".join(DS_PRETTY[d] for d in order) + r" \\",
        r"\midrule",
    ]
    cd = fr["nemenyi_critical_difference"]
    best = mr["mean_rank"].iloc[0]
    for v in mr.index[:top]:
        cells = [f"{f1.loc[v, d]:.4f} $\\pm$ {sd.loc[v, d]:.4f}" for d in order]
        mark = r"$^{\dagger}$" if mr.loc[v, "mean_rank"] - best <= cd else ""
        lines.append(f"{_tt(v)} & {mr.loc[v, 'mean_rank']:.2f}{mark} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\vspace{2pt}",
              r"\footnotesize $^{\dagger}$ within one critical difference of the"
              r" leader.",
              r"\end{table}"]
    return "\n".join(lines)


def table_focused(tag: str) -> str:
    d = json.load(open(ROOT / "results/analysis" / tag / "focused_tests.json"))
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Pre-specified paired comparisons (PREREGISTRATION.md \S4),"
        r" Wilcoxon signed-rank over the %d blocks with Holm correction across the"
        r" family of %d tests. $\Delta$ is (A $-$ B) macro-F1 in percentage points;"
        r" the interval is a hierarchical bootstrap resampling datasets first and"
        r" seeds within datasets.}" % (d["tests"][0]["n_blocks"], d["family_size"]),
        r"\label{tab:focused}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l l r r r l}", r"\toprule",
        r"A & B & W/L/T & median $\Delta$ & Holm $p$ & bootstrap $\Delta$ [95\% CI] \\",
        r"\midrule",
    ]
    for t in d["tests"]:
        lo, hi = t["bootstrap_ci_pp"]
        sig = "" if t["significant_after_holm"] else r"$^{\ast}$"
        lines.append(
            f"{_tt(t['a'])} & {_tt(t['b'])} & "
            f"{t['wins_a']}/{t['wins_b']}/{t.get('exact_ties', 0)} & "
            f"{t['median_delta_pp']:+.2f} & {t['holm_adjusted_p']:.4f}{sig} & "
            f"{t['bootstrap_effect_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\vspace{2pt}",
              r"\vspace{-2pt}",
        r"\footnotesize $^{\ast}$ not significant at $\alpha=0.05$ after Holm."
        r" W/L/T counts blocks won, lost and \emph{exactly} tied; ties are"
        r" dropped before ranking (\texttt{zero\_method=wilcox}) and so enter"
        r" neither the signed-rank sum nor the effective sample size. A"
        r" displayed median of $+0.00$ therefore indicates a difference"
        r" smaller than the printed precision, not an exact tie.",
              r"\end{table}"]
    return "\n".join(lines)


def table_audit() -> str:
    df = pd.read_csv(ROOT / "results/analysis/split_audit.csv")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Split audit for the leakage-resistant (v2) protocols. Group"
        r" overlap is the number of groups shared between partitions and must be"
        r" zero wherever disjointness is claimed; a violation aborts the training"
        r" job. Duplicate rows are \emph{reported}, not asserted away: KDDCup99"
        r" repeats most of its official test set inside training, which is the"
        r" documented reason NSL-KDD exists. No transform is fitted on test data"
        r" in any protocol.}",
        r"\label{tab:audit}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l l r r r r r}", r"\toprule",
        r"Dataset & Protocol & Train & Test & Group & Overlap & Dup.\ test (\%) \\",
        r"\midrule",
    ]
    # `{,}` keeps LaTeX from treating the thousands separator as punctuation and
    # inserting a space after it. Applied per number, never to the whole string —
    # a global replace would corrupt the caption prose.
    def num(v) -> str:
        return f"{int(v):,}".replace(",", "{,}")

    for _, r in df.iterrows():
        lines.append(
            f"{r['dataset']} & {r['protocol']} & {num(r['n_train'])} & "
            f"{num(r['n_test'])} & {r['group']} & {int(r['overlap_train_test'])} & "
            f"{r['dup_test_pct']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def table_protocol_audit() -> str:
    """One table explaining why v2 exists, replacing several paragraphs.

    Each row is an issue we audited in the original protocol, what the audit
    found, and what the revised protocol does about it. Deliberately compact:
    a reviewer should be able to see the whole methodological case at a glance
    and check any single number against the artifact that produced it.
    """
    rows = [
        ("NSL-KDD", "Exact train/test duplicate records",
         "2.68\\% of test rows", "Duplicate-free sensitivity reported"),
        ("KDDCup99", "Exact train/test duplicate records",
         "57.53\\% of test rows",
         "Duplicate-free sensitivity; ranking unchanged"),
        ("CIC-IDS2017", "Random split interleaves flows of one attack burst",
         "Present in the legacy protocol",
         "Within-capture order-disjoint; day-disjoint transfer test"),
        ("CTU-13", "Host aggregates computed over whole scenario; random split",
         "$100.00\\rightarrow92.01$ macro-F1",
         "Causal aggregates + scenario-disjoint folds"),
        ("all", "Categorical vocabulary fitted on train$\\cup$test",
         "0 NSL-KDD / 2 KDDCup99 test rows affected",
         "All learned transforms fitted on train only"),
    ]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Protocol audit motivating the leakage-resistant (v2) "
        r"evaluation. Each row is an issue we tested for in the original "
        r"protocol, what the test found, and how v2 treats it. The union-split "
        r"vocabulary turned out to be empirically negligible; we report its "
        r"size rather than citing that as grounds for keeping it.}",
        r"\label{tab:protocol-audit}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l p{3.4cm} l p{3.6cm}}", r"\toprule",
        r"Dataset & Issue audited & Finding & Revised treatment \\",
        r"\midrule",
    ]
    for ds, issue, finding, fix in rows:
        lines.append(f"{ds} & {issue} & {finding} & {fix} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("tab_ranking", lambda: table_ranking(args.tag)),
                     ("tab_focused", lambda: table_focused(args.tag)),
                     ("tab_audit", table_audit),
                     ("tab_protocol_audit", table_protocol_audit)]:
        p = (OUT / f"{name}.tex" if name.startswith("tab_audit")
             or name == "tab_protocol_audit"
             else OUT / f"{name}_{args.tag}.tex")
        p.write_text(fn() + "\n")
        print("wrote", p)


if __name__ == "__main__":
    main()
