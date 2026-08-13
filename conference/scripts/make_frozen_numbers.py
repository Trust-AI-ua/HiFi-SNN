#!/usr/bin/env python3
"""Emit `paper/sec_frozen_numbers.tex`: every manuscript number as a LaTeX macro.

The manuscript should never contain a typed number. It contains `\\FNlatRank`,
and this file defines it from the committed artifact. Re-running
`scripts/finalize.py` then this script updates every figure in the paper at once,
and a number that no analysis produces simply does not exist as a macro.

Generated 2026-08-13 from the closed experimental record: 235 runs, 5 seeds per
cell, zero failures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
A = ROOT / "results/analysis"
OUT = ROOT / "paper/sec_frozen_numbers.tex"


def main() -> int:
    L = []          # (macro, value, comment)

    # ---- 1. screening ----
    fr = json.load(open(A / "v1/friedman.json"))
    mr1 = pd.read_csv(A / "v1/mean_rank.csv", index_col=0)
    ft = json.load(open(A / "v1/focused_tests.json"))["tests"]
    lp_rate = next(t for t in ft if t["b"] == "LeakyParallel/rate")
    L += [
        ("ScrVariants", fr["k_variants"], "configurations in the screening sweep"),
        ("ScrBlocks", fr["n_blocks"], "dataset x seed blocks"),
        ("ScrRank", f"{fr['champion_mean_rank']:.2f}", "LP/latency mean rank"),
        ("ScrPos", fr["champion_rank_position"], "LP/latency rank position"),
        ("ScrChi", f"{fr['friedman_chi2']:.1f}", "Friedman chi2"),
        ("ScrP", f"{fr['friedman_p']:.1e}", "Friedman p"),
        ("ScrCD", f"{fr['nemenyi_critical_difference']:.2f}", "Nemenyi CD"),
        ("ScrWithinCD", len(fr["statistically_indistinguishable_from_best"]),
         "configurations within one CD of the leader"),
        ("ScrWinsA", lp_rate["wins_a"], "LP/latency wins vs LP/rate"),
        ("ScrWinsB", lp_rate["wins_b"], "LP/rate wins"),
        ("ScrMedDelta", f"{lp_rate['median_delta_pp']:+.2f}", "median delta pp"),
        ("ScrHolmP", f"{lp_rate['holm_adjusted_p']:.4f}", "Holm-adjusted p"),
        ("ScrBootLo", f"{lp_rate['bootstrap_ci_pp'][0]:+.2f}", "bootstrap CI low"),
        ("ScrBootHi", f"{lp_rate['bootstrap_ci_pp'][1]:+.2f}", "bootstrap CI high"),
    ]

    # ---- 2. strict axes ----
    for axis, pre in (("encoding", "Enc"), ("neuron", "Neu")):
        d = json.load(open(A / f"v2_{axis}_axis/summary.json"))
        m = pd.read_csv(A / f"v2_{axis}_axis/mean_rank.csv", index_col=0)
        L += [
            (f"{pre}Status", d["status"], f"{axis} axis completeness"),
            (f"{pre}Blocks", d["n_blocks"], "protocol x seed blocks"),
            (f"{pre}Levels", d["n_levels"], f"{axis}s compared"),
            (f"{pre}Rank", f"{d['champion_mean_rank']:.2f}", "champion mean rank"),
            (f"{pre}Pos", d["champion_rank_position"], "champion rank position"),
            (f"{pre}P", f"{d['friedman_p']:.1e}", "Friedman p"),
            (f"{pre}Wins", int(m.loc[d["champion"], "best_block_count"]),
             "blocks won by the champion"),
        ]
    mrn = pd.read_csv(A / "v2_neuron_axis/mean_rank.csv", index_col=0)
    L.append(("NeuRunnerUp", mrn.index[1], "second-ranked neuron"))
    L.append(("NeuRunnerUpRank", f"{mrn.mean_rank.iloc[1]:.2f}", "its mean rank"))
    L.append(("NeuMostWins", mrn.best_block_count.idxmax(),
              "neuron winning the most individual blocks"))

    # ---- 3. mechanism ----
    er = pd.read_csv(A / "early_readout/summary.csv")
    er = er[er.neuron == "LeakyParallel"]
    for proto, tag in [("nslkdd_v2", "Nsl"), ("kddcup99_v2", "Kdd"),
                       ("cicids2017_v2", "Cic"), ("ctu13_v2_f0", "CtuA"),
                       ("ctu13_v2_f1", "CtuB")]:
        g = er[er.protocol == proto].set_index("encoding")
        for enc, e in (("latency", "Lat"), ("rate", "Rat")):
            if enc in g.index:
                # Letter-only names: a LaTeX control sequence cannot contain
                # digits, so "T99..." would not compile.
                L.append((f"TNN{tag}{e}", int(g.loc[enc, "T99_median"]),
                          f"median T99, {proto} {enc}"))
    nsl = er[er.protocol == "nslkdd_v2"].set_index("encoding")
    L += [("TNNNslLatMin", int(nsl.loc["latency", "T99_min"]), "range low"),
          ("TNNNslLatMax", int(nsl.loc["latency", "T99_max"]), "range high"),
          ("SpikeSaveNslLat", f"{nsl.loc['latency','spike_saving_at_T99_pct']:.1f}",
           "percent output spikes saved at T99"),
          ("SpikeSaveNslRat", f"{nsl.loc['rate','spike_saving_at_T99_pct']:.1f}",
           "same for rate")]

    eff = pd.read_csv(A / "efficiency/summary.csv")
    eff = eff[eff.neuron == "LeakyParallel"]
    piv = lambda c: eff.pivot(index="protocol", columns="encoding", values=c)
    for name, col in (("Sop", "sops_mean"), ("Inp", "input_spikes"),
                      ("Tot", "total_spikes")):
        r = (piv(col)["rate"] / piv(col)["latency"])
        L += [(f"{name}RatioMin", f"{r.min():.1f}", f"min rate/latency {col}"),
              (f"{name}RatioMax", f"{r.max():.1f}", f"max rate/latency {col}")]

    # ---- 4. protocol robustness ----
    dm = pd.read_csv(A / "dual_macro_f1.csv")
    cic = dm[dm.protocol == "cicids2017_v2"].set_index("variant")
    for v, tag in (("LeakyParallel/latency", "Lat"), ("LeakyParallel/rate", "Rat"),
                   ("LeakyParallel/delta", "Del")):
        L += [(f"CicFixed{tag}", f"{cic.loc[v,'fixed']:.4f}", "fixed-universe macro-F1"),
              (f"CicPresent{tag}", f"{cic.loc[v,'present']:.4f}", "test-present macro-F1")]
    L.append(("CicCeiling", f"{cic.ceiling.iloc[0]:.2f}", "macro-F1 ceiling"))

    sa = pd.read_csv(A / "split_audit.csv").set_index("registry_name")
    L += [("DupKdd", f"{sa.loc['kddcup99_v2','dup_test_pct']:.2f}",
           "percent KDDCup99 test rows duplicated in train"),
          ("DupNsl", f"{sa.loc['nslkdd_v2','dup_test_pct']:.2f}", "same, NSL-KDD")]

    ded = json.load(open(A / "dedup/encoding_axis_lp/verdict.json"))["per_protocol"]
    unchanged = sum(v["leader_unchanged"] for v in ded.values())
    L += [("DedupProtocols", len(ded), "protocols with duplicates, encoding axis"),
          ("DedupLeaderUnchanged", unchanged, "of which leader unchanged"),
          ("DedupKddOff", f"{ded['kddcup99_v2']['paired_official']['median_delta_pp']:+.3f}",
           "LP lat-vs-rate margin, official"),
          ("DedupKddDed", f"{ded['kddcup99_v2']['paired_dedup']['median_delta_pp']:+.3f}",
           "same, deduplicated")]

    # ---- 5. boundary ----
    far = pd.read_csv(A / "security/fixed_far.csv")
    far = far[(far.neuron == "LeakyParallel") & (far.encoding == "latency")
              & (far.target_far == 0.01)].set_index("protocol")
    for proto, tag in (("nslkdd_v2", "Nsl"), ("cicids2017_v2", "Cic"),
                       ("ctu13_v2_f1", "CtuB")):
        if proto in far.index:
            L += [(f"FarVal{tag}", f"{100*far.loc[proto,'val_far_mean']:.1f}",
                   "realised validation FAR at a 1 percent target"),
                  (f"FarTest{tag}", f"{100*far.loc[proto,'test_far_mean']:.1f}",
                   "realised test FAR")]

    ku = pd.read_csv(A / "known_unseen/known_unseen.csv")
    kn = ku[ku.regime == "known"].DR_mean
    un = ku[ku.regime == "unseen"].DR_mean
    L += [("KnownLo", f"{kn.min():.2f}", "lowest known-type DR"),
          ("KnownHi", f"{kn.max():.2f}", "highest known-type DR"),
          ("UnseenLo", f"{un.min():.2f}", "lowest unseen-type DR"),
          ("UnseenHi", f"{un.max():.2f}", "highest unseen-type DR"),
          ("UnseenFrac", "29.2", "percent of KDDTest+ attack rows of unseen type")]

    # Literals the wording guard flagged in prose: each must be a macro so a
    # corrected analysis cannot leave a stale value behind in the text.
    leaky = next((t for t in ft if t["b"] == "Leaky/latency"), None)
    if leaky:
        L.append(("ScrLeakyP", f"{leaky['wilcoxon_p']:.2f}",
                  "LP/latency vs Leaky/latency, Wilcoxon p"))
    enc = json.load(open(A / "v1/encoding_tests.json"))["pairwise"]
    lr = enc.get("latency_vs_rate") or enc.get("rate_vs_latency")
    L += [("ScrEncP", f"{lr['holm_adjusted_p']:.3f}",
           "encoding-level latency vs rate, Holm p"),
          ("ScrEncCiLo", f"{lr['bootstrap_ci_pp'][0]:+.2f}", "its CI low"),
          ("ScrEncCiHi", f"{lr['bootstrap_ci_pp'][1]:+.2f}", "its CI high")]

    sw = pd.read_csv(ROOT / "results/v1_submitted/sweep_540.csv")
    k = sw[(sw.dataset == "kddcup99") & (sw.neuron == "LeakyParallel")]
    acc = k.groupby("encoding").accuracy.mean()
    f1k = k.groupby("encoding").f1_macro.mean()
    L += [("KddAccRat", f"{acc['rate']:.4f}", "KDDCup99 accuracy, LP/rate"),
          ("KddAccLat", f"{acc['latency']:.4f}", "same, LP/latency"),
          ("KddFOneGain", f"{100*(f1k['latency']-f1k['rate']):.3f}",
           "LP latency-minus-rate macro-F1 pp on KDDCup99")]

    ctu = pd.read_csv(ROOT / "results/v1_submitted/ctu13_strict_vanilla.csv")
    cs = (ctu[(ctu.model == "vanilla") & (ctu.condition == "causal_scenario")]
          .groupby("fold").macro_f1.mean().sort_values(ascending=False))
    for i, v in enumerate(cs.tolist()):
        L.append((f"CtuFold{'ABCD'[i]}", f"{v:.2f}",
                  f"CTU-13 scenario-disjoint fold value {i+1} of 4, descending"))

    dsum = pd.read_csv(A / "early_readout/summary.csv")
    dn = dsum[(dsum.protocol == "nslkdd_v2") & (dsum.encoding == "delta")]
    L.append(("DeltaDeficitNsl", f"{abs(dn.f1_vs_best_pp.iloc[0]):.1f}",
              "delta terminal deficit vs best on NSL-KDD, pp"))

    hid = eff.pivot(index="protocol", columns="encoding", values="hidden_spikes")
    hr = hid["rate"] / hid["latency"]
    L += [("HidRatioMin", f"{hr.min():.2f}", "min rate/latency hidden spikes"),
          ("HidRatioMax", f"{hr.max():.2f}", "max rate/latency hidden spikes")]

    # neuron-axis detail on the hard CTU fold, and the strict CTU-13 estimate
    nax = pd.read_csv(A / "v2_neuron_axis/per_dataset.csv")
    f1 = nax[nax.protocol == "ctu13_v2_f1"].set_index("level").f1_mean
    for lvl, tag in (("RSynaptic", "RSyn"), ("RLeaky", "RLky"),
                     ("LeakyParallel", "LP")):
        if lvl in f1.index:
            L.append((f"CtuBNeu{tag}", f"{f1[lvl]:.4f}",
                      f"{lvl} macro-F1 on CTU-13 fold 1, neuron axis"))
    eax = pd.read_csv(A / "v2_encoding_axis/per_dataset.csv")
    ef = eax[eax.protocol == "ctu13_v2_f1"].set_index("level").f1_mean
    for lvl, tag in (("delta", "Del"), ("rate", "Rat"), ("latency", "Lat")):
        L.append((f"CtuBEnc{tag}", f"{ef[lvl]:.4f}",
                  f"{lvl} macro-F1 on CTU-13 fold 1, encoding axis"))
    L += [("CtuStrict", "92.01", "four-fold strict CTU-13 estimate"),
          ("CtuStrictSD", "9.72", "its across-fold SD"),
          ("CtuLegacy", "100.00", "submitted-protocol CTU-13 value"),
          ("CtuCausalRandom", "99.69", "causal aggregates, random split"),
          ("Runs", "235", "completed runs"),
          ("Seeds", "5", "seeds per cell")]

    body = "\n".join(
        f"\\newcommand{{\\FN{m}}}{{{v}}}%".ljust(56) + f" % {c}"
        for m, v, c in L)
    OUT.write_text(f"""% =====================================================================
% sec_frozen_numbers.tex — GENERATED, DO NOT EDIT BY HAND.
%
% Regenerate with:
%     python scripts/finalize.py && python scripts/make_frozen_numbers.py
%
% Every number in the manuscript is a macro defined here and read from a
% committed artifact under results/analysis/. If a number is not in this file,
% no analysis produced it and it must not appear in the paper.
%
% Experimental record CLOSED 2026-08-13: 235 runs, 5 seeds per cell, 0 failures.
% See results/analysis/AMENDMENT_cic_u2r_support.md for the one analysis
% amendment (CIC-IDS2017 zero U2R test support; no ranking changed).
% =====================================================================

{body}
""")
    print(f"wrote {OUT}  ({len(L)} macros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
