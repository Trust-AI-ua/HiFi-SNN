#!/usr/bin/env python3
"""Generate results/tables/numbers.tex: every number the manuscript quotes, as a macro.

The manuscript \\input{}s this file and never types a result. Each macro is read
from a committed artifact at generation time, and results/tables/numbers_provenance.json
records, per macro, its value, source file and meaning. scripts/check_numbers.py
then scans a manuscript for typed numbers that should have been macros.

Conventions
  * prefix \\JN (journal numbers), letters only: LaTeX macro names cannot contain
    digits, so 2023 is spelled into the dataset code and 42 into the word
  * signed quantities carry an explicit sign in math mode (\\ensuremath{+0.56})
  * SDs are sample SDs (ddof = 1), five seeds, unless the name says otherwise
  * every macro states its aggregation in the provenance file: five-seed mean,
    single seed-42 run, or pooled

    python scripts/make_numbers.py
"""
from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/tables/numbers.tex"
PROV = REPO / "results/tables/numbers_provenance.json"
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
DS = {"ciciot2023": "CicIoT", "ciciomt2024": "CicIoMT", "edgeiiot": "EdgeIIoT", "toniot": "TonIoT",
      "cse-cic-ids2018": "CseCic", "cicids2017": "CicIds", "ctu13": "Ctu", "kddcup99": "Kdd",
      "nslkdd": "NslKdd", "unswnb15": "Unsw", "iscxids2012": "Iscx"}
MOD = {"treespike": "Tts", "dnn": "Dnn", "vanilla": "Van", "tasnn": "Tasnn"}

MACROS: list[tuple[str, str, str, str, str]] = []   # name, value, source, aggregation, meaning


def add(name, value, source, agg, meaning):
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise SystemExit(f"macro name {name!r} is not letters-only")
    if any(m[0] == name for m in MACROS):
        raise SystemExit(f"duplicate macro {name}")
    MACROS.append((name, value, source, agg, meaning))


def f(v, d=2):
    return f"{v:.{d}f}"


def sgn(v, d=2):
    return rf"\ensuremath{{{'+' if v >= 0 else '-'}{abs(v):.{d}f}}}"


def pct(v, d=1):
    return f"{v:.{d}f}"


def rel(p):
    return str(p.relative_to(REPO))


# ------------------------------------------------------------------ quality
def quality():
    src = REPO / "results/diagnostics/per_seed_all_cells.csv"
    c = pd.read_csv(src)
    ci = c[c.dataset.isin(IOT)]
    mean = ci.pivot_table(index="dataset", columns="model", values="f1", aggfunc="mean")
    sd = ci.pivot_table(index="dataset", columns="model", values="f1", aggfunc=lambda x: x.std(ddof=1))
    for ds in IOT:
        for m, mc in MOD.items():
            add(f"JNFone{mc}{DS[ds]}", f(mean.loc[ds, m]), rel(src), "five-seed mean",
                f"test macro-F1, {m}, {ds}")
            add(f"JNFone{mc}{DS[ds]}Sd", f(sd.loc[ds, m]), rel(src), "five-seed sample SD",
                f"SD of test macro-F1, {m}, {ds}")
    for m, mc in MOD.items():
        add(f"JNFone{mc}IotMean", f(mean.loc[IOT, m].mean()), rel(src), "mean of four five-seed means",
            f"IoT mean test macro-F1, {m}")
    diff = mean.loc[IOT, "treespike"] - mean.loc[IOT, "dnn"]
    for ds in IOT:
        add(f"JNDiffTtsDnn{DS[ds]}", sgn(diff[ds]), rel(src), "difference of five-seed means",
            f"Tree-to-Spike minus DNN-Tree macro-F1 (int8, deployed), {ds}")
    add("JNDiffTtsDnnIotMean", sgn(diff.mean()), rel(src), "IoT mean", "Tree-to-Spike minus DNN-Tree, IoT mean, int8")
    add("JNDiffTtsDnnIotMedian", sgn(diff.median()), rel(src), "IoT median", "median of the four differences")
    for other, oc in (("vanilla", "Van"), ("tasnn", "Tasnn")):
        dd = mean.loc[IOT, "treespike"] - mean.loc[IOT, other]
        add(f"JNDiffTts{oc}IotMean", sgn(dd.mean()), rel(src), "IoT mean", f"Tree-to-Spike minus {other}")
        add(f"JNWinsTts{oc}", str(int((dd > 0).sum())), rel(src), "count of four", f"datasets where Tree-to-Spike beats {other}")


def precision():
    src = REPO / "results/tables/tab_m3_quant.csv"
    d = pd.read_csv(src, comment="#")
    d = d[d.dataset.isin(IOT) & d.model.isin(["Tree-to-Spike", "DNN-Tree"])]
    fp = d.pivot(index="dataset", columns="model", values="fp_mean")
    q = d.pivot(index="dataset", columns="model", values="int8_all_mean")
    fpd = fp["Tree-to-Spike"] - fp["DNN-Tree"]
    for ds in IOT:
        add(f"JNFpDiffTtsDnn{DS[ds]}", sgn(fpd[ds]), rel(src), "M3 reruns, five-seed mean",
            f"Tree-to-Spike minus DNN-Tree macro-F1 at fp32, {ds}")
        add(f"JNQuantDropDnn{DS[ds]}", sgn(q.loc[ds, 'DNN-Tree'] - fp.loc[ds, 'DNN-Tree']), rel(src),
            "M3 reruns, five-seed mean", f"DNN-Tree int8 minus fp32 macro-F1, {ds}")
        add(f"JNQuantDropTts{DS[ds]}", sgn(q.loc[ds, 'Tree-to-Spike'] - fp.loc[ds, 'Tree-to-Spike']), rel(src),
            "M3 reruns, five-seed mean", f"Tree-to-Spike int8 minus fp32 macro-F1, {ds}")
    add("JNFpDiffTtsDnnIotMean", sgn(fpd.mean()), rel(src), "IoT mean of M3 reruns", "fp32 difference, IoT mean")
    add("JNFpDnnWins", str(int((fpd < 0).sum())), rel(src), "count of four", "IoT datasets where DNN-Tree leads at fp32")
    import math
    add("JNFpGapIotBound", f"{math.ceil(abs(fpd.mean()) * 10) / 10:.1f}", rel(src),
        "IoT mean of M3 reruns, rounded UP to one decimal",
        "bound on the largest Tree-to-Spike vs DNN-Tree gap (fp32); abstract: within this many points")
    add("JNIntDiffTtsDnnRerunIotMean", sgn((q["Tree-to-Spike"] - q["DNN-Tree"]).mean()), rel(src),
        "IoT mean of M3 reruns", "int8 difference on the M3 reruns (headline runs: JNDiffTtsDnnIotMean)")
    add("JNFpTtsIotMean", f(fp["Tree-to-Spike"].mean()), rel(src), "IoT mean of M3 reruns", "Tree-to-Spike fp32 IoT mean")
    add("JNFpDnnIotMean", f(fp["DNN-Tree"].mean()), rel(src), "IoT mean of M3 reruns", "DNN-Tree fp32 IoT mean")


def statistics():
    import io
    from contextlib import redirect_stdout
    import m9_iot
    with redirect_stdout(io.StringIO()):
        _, _, R, _, fr = m9_iot.main()
    src = "scripts/m9_iot.py (results/diagnostics/per_seed_all_cells.csv)"
    add("JNFriedmanChi", f(fr["chi2"]), src, "IoT four, five-seed means", "Friedman chi-square, df 3")
    add("JNFriedmanPExact", f(fr["p_exact"], 4), src, "exact permutation", "Friedman exact p-value")
    add("JNFriedmanPAsym", f(fr["p_asym"], 3), src, "asymptotic", "Friedman asymptotic p (unreliable at N=4)")
    add("JNImanDavenportF", f(fr["Ff"]), src, "IoT four", "Iman-Davenport F")
    add("JNImanDavenportP", f(fr["p_F"], 4), src, "IoT four", "Iman-Davenport p")
    for j, m in enumerate(m9_iot.MODELS):
        add(f"JNMeanRank{MOD[m]}", f(R[:, j].mean()), src, "IoT four", f"mean Friedman rank, {m}")


# ------------------------------------------------------------------ energy
def energy():
    src = REPO / "results/tables/tab_iot_arithmetic.csv"
    t = pd.read_csv(src, index_col=0)
    for ds in IOT:
        add(f"JNArithRatio{DS[ds]}", f(t.loc[ds, "DNN / T2S"]), rel(src), "five-seed mean, full window",
            f"DNN-Tree / Tree-to-Spike arithmetic energy, {ds}")
        add(f"JNSops{DS[ds]}", f"{t.loc[ds, 'T2S SOPs']:,.0f}", rel(src), "five-seed mean, full window",
            f"Tree-to-Spike SOPs per sample incl. input layer, {ds}")
        add(f"JNMacs{DS[ds]}", f"{t.loc[ds, 'DNN MACs']:,.0f}", rel(src), "five-seed mean", f"DNN-Tree MACs, {ds}")
        add(f"JNVanRatio{DS[ds]}", f(t.loc[ds, "Vanilla / T2S"]) if not np.isnan(t.loc[ds, "Vanilla / T2S"]) else "--",
            rel(src), "five-seed mean", f"Vanilla SNN / Tree-to-Spike arithmetic energy, {ds}")
    add("JNArithRatioIot", f((t["DNN MACs"] * 4.6).mean() / (t["T2S SOPs"] * 0.9).mean()), rel(src),
        "ratio of IoT means, 32-bit", "DNN-Tree / Tree-to-Spike arithmetic energy, IoT")
    add("JNArithRatioIotInt", f((t["DNN MACs"] * 0.23).mean() / (t["T2S SOPs"] * 0.03).mean()), rel(src),
        "ratio of IoT means, int8 pricing", "same at int8 Horowitz prices")
    eff = REPO / "results/Journal_results/tables/table_efficiency.md"
    rows = {l.split("|")[1].strip(): l.split("|") for l in eff.read_text().splitlines() if l.startswith("| ")}
    num = lambda s: float(s.strip().replace(",", "").replace("†", ""))
    e32 = num(rows["DNN-Tree"][3]) / num(rows["Tree-to-Spike"][3])
    e8 = num(rows["DNN-Tree"][4]) / num(rows["Tree-to-Spike"][4])
    add("JNArithRatioPooled", f(e32), rel(eff), "pooled over eleven datasets, 32-bit", "pooled arithmetic ratio")
    add("JNArithRatioPooledInt", f(e8), rel(eff), "pooled over eleven datasets, int8", "pooled arithmetic ratio, int8")

    src = REPO / "results/deployment/energy_crossover_iot.csv"
    b = pd.read_csv(src)
    word = {5: "Five", 10: "Ten", 20: "Twenty", 50: "Fifty", 100: "Hundred"}
    for r in b.itertuples():
        cfgc = "Dep" if r.config == "deployed" else "Full"
        add(f"JNBand{cfgc}{r.model.capitalize()}{word[r.sram_pJ]}", f(r.ratio), rel(src),
            f"single seed-42 run, IoT four, {r.config}",
            f"DNN-Tree / Tree-to-Spike total energy, {r.model}, {r.sram_pJ} pJ SRAM")


def deployment():
    import make_dep_systems_table as M
    src = REPO / "results/deployment/deployment_measurements.csv"
    d = M.deployed(pd.read_csv(src, comment="#"))
    i = d[d.dataset.isin(IOT)]
    names = {"t2s_deployed": "Tts", "dnn_tree": "Dnn", "vanilla_snn": "Van", "tasnn": "Tasnn",
             "compact_gbdt": "GbdtCompact", "full_gbdt": "GbdtFull"}
    for s, n in names.items():
        g = i[i.system == s]
        add(f"JNKb{n}", f(g.total_bytes.mean() / 1024, 1), rel(src), "single seed-42 run, IoT mean",
            f"artifact size KB, {s}")
        add(f"JNLatMs{n}", f(g.p50_ms.mean()), rel(src), "single seed-42 run, IoT mean, batch-1 CPU",
            f"p50 latency ms, {s}")
        add(f"JNFitsKb{n}", str(int((g.total_bytes <= 131072).sum())), rel(src), "count of four",
            f"IoT datasets where {s} fits 128 KB")
    # Encoder threshold tables inside the deployed Tree-to-Spike artifact (int8
    # serialization: int8 weights + float32 scales/biases; bin edges, channel list and
    # config stay float32). Same rows as JNKbTts, so size and share describe one artifact.
    t = i[i.system == "t2s_deployed"]
    enc = t.encoder_edges_bytes + t.encoder_channels_bytes + t.encoder_config_bytes
    share = 100 * enc / t.total_bytes
    add("JNEncKbTts", f(enc.mean() / 1024, 1), rel(src), "single seed-42 run, IoT mean",
        "encoder threshold tables in the Tree-to-Spike artifact, KB")
    add("JNEncShareTts", pct(share.mean()), rel(src), "single seed-42 run, mean of the four per-dataset shares",
        "percent of the Tree-to-Spike artifact taken by the encoder threshold tables")
    add("JNEncShareTtsMin", pct(share.min()), rel(src), "single seed-42 run, IoT min", "lowest per-dataset share")
    add("JNEncShareTtsMax", pct(share.max()), rel(src), "single seed-42 run, IoT max", "highest per-dataset share")
    full = pd.read_csv(src, comment="#")
    add("JNLatMsTtsFull", f(full[full.dataset.isin(IOT) & (full.system == "t2s_full")].p50_ms.mean()),
        rel(src), "single seed-42 run, IoT mean", "Tree-to-Spike full-window p50 latency")


# ------------------------------------------------------------------ other analyses
def others():
    src = REPO / "results/tables/tab_teacher_vs_student.csv"
    t = pd.read_csv(src)
    code = {"CICIoT2023": "CicIoT", "CICIoMT2024": "CicIoMT", "Edge-IIoTset": "EdgeIIoT", "TON-IoT": "TonIoT"}
    for r in t.itertuples():
        add(f"JNStudentMinusTeacher{code[r.dataset]}", sgn(r.diff_f1), rel(src), "paired by seed, five seeds",
            f"student minus teacher macro-F1, {r.dataset}")
        add(f"JNStudentMinusTeacher{code[r.dataset]}Sd", f(r.sd_f1), rel(src), "sample SD of paired difference", "")
        add(f"JNStudentSeedWins{code[r.dataset]}", r.seed_wins_f1.split("/")[0], rel(src), "count of five", "")

    src = REPO / "results/tables/tab_m8_decomposition.csv"
    t = pd.read_csv(src)
    stepname = ["Arch", "Precision", "Encoder", "Distill"]
    for k, r in enumerate(t.itertuples()):
        for ds in IOT:
            add(f"JNStep{stepname[k]}{DS[ds]}", sgn(getattr(r, ds)), rel(src), "five-seed mean", f"M8 step {k+1}, {ds}")
        add(f"JNStep{stepname[k]}IotMean", sgn(r._6), rel(src), "IoT mean", f"M8 step {k+1}, IoT mean")
    ps = pd.read_csv(REPO / "results/tables/tab_m8_decomposition_per_seed.csv")
    step = (ps.nokd - ps.arm_int8).groupby(ps.dataset)
    for ds in IOT:
        add(f"JNStepEncoder{DS[ds]}Sd", f(step.std()[ds]), "results/tables/tab_m8_decomposition_per_seed.csv",
            "sample SD, paired by seed", f"encoder-step SD, {ds}")
        add(f"JNStepEncoder{DS[ds]}Snr", f(abs(step.mean()[ds]) / step.std()[ds], 1),
            "results/tables/tab_m8_decomposition_per_seed.csv", "mean / SD", f"encoder step in SDs, {ds}")

    src = REPO / "results/tables/tab_m5_edges.csv"
    g = pd.read_csv(src, index_col=0)
    for ds in IOT:
        add(f"JNViolation{DS[ds]}", pct(100 * g.loc[ds, "frac"]), rel(src), "five-seed mean",
            f"percent of retained features with |E_j| >= T, {ds}")
    import m5_edges  # noqa: F401  (pooled numbers recomputed from the saved tables)
    import glob
    rows = []
    for fpath in glob.glob(str(REPO / "results/diagnostics/val_curves/Tree-to-Spike/*/runs/seed_*/encoder_channels.json")):
        ds = Path(fpath).parts[-4]
        if ds in IOT:
            E = {c["feature"]: c["n_edges"] for c in json.load(open(fpath))["channels"]}
            rows += list(E.values())
    e = np.array(rows)
    for T, w in ((10, "Ten"), (20, "Twenty"), (40, "Forty")):
        add(f"JNViolationPooledT{w}", pct(100 * (e >= T).mean()), "saved encoder tables", "pooled IoT, five seeds",
            f"percent of retained features violating the bound at T = {T}")
    add("JNExactShare", pct(100 * (e < 20).mean(), 0), "saved encoder tables", "pooled IoT", "percent transmitted exactly at T = 20")
    add("JNMaxEdges", str(int(e.max())), "saved encoder tables", "pooled IoT, max over cells",
        "largest |E_j| of any retained feature")
    add("JNBoundWindow", str(int(e.max()) + 1), "saved encoder tables", "pooled IoT",
        "window meeting T >= |E_j| + 1 for every retained feature")
    add("JNBoundWindowFactor", f((int(e.max()) + 1) / 20, 1), "saved encoder tables", "pooled IoT",
        "that window as a multiple of T = 20 ('roughly thirteenfold')")
    add("JNViolationPerCellMean", pct(100 * g.loc[IOT, "frac"].mean()), rel(src), "mean over IoT datasets of five-seed means",
        "percent violating at T = 20, per-dataset mean (pooled: JNViolationPooledTTwenty)")
    for ds in IOT:
        add(f"JNMedianEdges{DS[ds]}", pct(g.loc[ds, "median_E"], 0), rel(src), "five-seed mean of per-cell medians",
            f"median |E_j| of retained features, {ds}")
    add("JNViolatingCount", str(int((e >= 20).sum())), "saved encoder tables", "pooled IoT", "violating retained features")
    add("JNRetainedCount", str(len(e)), "saved encoder tables", "pooled IoT", "retained features")

    src = REPO / "results/early_exit/selected_test.csv"
    s = pd.read_csv(src)
    add("JNExitRejected", str(int(s.early_exit_disabled.sum())), rel(src), "count of 55 runs", "runs rejecting early exit")
    add("JNExitRuns", str(len(s)), rel(src), "", "runs in the exit chain")

    src = REPO / "results/deployment/gbdt_compression.csv"
    c = pd.read_csv(src)
    c["gain_prune"] = c.gain_prune.astype(str)
    e = c[(c.dataset == "edgeiiot") & (c.n_classes == 14)]
    full = e[(e.n_estimators == 200) & (e.max_depth == 6) & (e.gain_prune == "none") & (e.leaf_bits == 16)].macro_f1.iloc[0]
    add("JNEdgeSweepBeatFull", str(int((e.macro_f1 > full).sum())), rel(src), "seed 42, all configurations",
        "Edge-IIoTset compressed configurations exceeding the full teacher")
    add("JNEdgeSweepTotal", str(len(e)), rel(src), "", "Edge-IIoTset sweep configurations")
    add("JNEdgeSweepBest", f(e.macro_f1.max()), rel(src), "seed 42, test-selected", "best configuration macro-F1")
    add("JNEdgeTeacherFull", f(full), rel(src), "seed 42", "full teacher macro-F1 in the sweep")


def tsweep():
    """Window sweep from the ladder: T = 10, 40, 80 against production T = 20.

    The four IoT datasets. The first Edge-IIoTset arms were run on the pre-repair corpus
    and are retired to results/archive/superseded/; its arms here are the 2026-10-08 rerun on the
    repaired corpus (all 175 ladder runs: production teacher and split). The arms differ
    from production only in T (checked: encoder_mode and class_weighted resolve to the
    same defaults; the Edge-IIoTset rerun's configs differ only in T).
    Pre-committed criterion, not met: every T within 1.0 point of T = 20.
    The ladder arms are 10-epoch runs, so CICIoT2023's T = 20 reference is the
    fixed-budget run (results/fixed_budget_ciciot2023), not the 20-epoch block (A1).
    """
    word = {10: "Ten", 40: "Forty", 80: "Eighty"}
    def f1s(paths):
        return np.array([100 * json.loads(Path(x).read_text())["test_plus"]["metrics"]["f1_macro"] for x in paths])
    for ds in IOT:
        root = "results/fixed_budget_ciciot2023" if ds == "ciciot2023" else "results"
        recs = [json.loads((REPO / f"{root}/Tree-to-Spike/{ds}/seed_{s}/results.json").read_text()) for s in range(42, 47)]
        if any(r["training"].get("selection", {}).get("enabled") or len(r["training"]["training_curve"]) != 10 for r in recs):
            raise SystemExit(f"T sweep: {ds} reference is not the 10-epoch fixed-budget run")
        base = np.mean([100 * r["test_plus"]["metrics"]["f1_macro"] for r in recs])
        defaults = {"encoder_mode": "tree", "class_weighted": False}
        for T, w in word.items():
            for s, r in zip(range(42, 47), recs):
                c, p = json.loads((REPO / f"results/ladder/{ds}/T{T}/seed_{s}/results.json").read_text())["config"], r["config"]
                extra = {k for k in set(c) | set(p) if c.get(k, "absent") != p.get(k, "absent")} - {"T"}
                if c["T"] != T or any(k in p or c[k] != defaults.get(k, "no default") for k in extra):
                    raise SystemExit(f"T sweep: {ds} T{T} seed {s} differs from production in more than T")
            v = f1s([REPO / f"results/ladder/{ds}/T{T}/seed_{s}/results.json" for s in range(42, 47)])
            add(f"JNTsweep{DS[ds]}{w}", sgn(v.mean() - base), f"results/ladder/{ds}/T{T}; {root}/Tree-to-Spike/{ds}",
                "five-seed mean difference vs the 10-epoch T = 20 run", f"macro-F1 at T = {T} minus T = 20, {ds}")


def compact():
    src = REPO / "results/deployment/compact_gbdt.csv"
    c = pd.read_csv(src)
    e = c[c.dataset == "edgeiiot"].selected_rounds
    add("JNCompactRoundsEdgeIIoTMin", str(int(e.min())), rel(src), "five seeds", "fewest rounds kept, Edge-IIoTset")
    add("JNCompactRoundsEdgeIIoTMax", str(int(e.max())), rel(src), "five seeds", "most rounds kept, Edge-IIoTset")
    # JNKbGbdtCompactEdgeIIoT is defined with the other per-dataset artifact sizes
    # in results_section(); same source, same format.
    add("JNTimingWarmup", "500", "scripts/deployment_measure.py (N_WARMUP)", "protocol constant", "warm-up calls")
    add("JNTimingN", "2{,}000", "scripts/deployment_measure.py (N_TIMED)", "protocol constant", "timed samples")


def results_section():
    """Values for section_V_results.tex (IoT four, five seeds unless stated)."""
    import glob as _glob
    import make_results_table as MRT
    import deployment_energy2 as E2
    from make_dep_systems_table import deployed

    # 1. results table: every metric through make_results_table.cell(), the path the
    #    published results tables use, so the section cannot drift from them
    code = {"vanilla": "Van", "tree": "Tts", "dnn": "Dnn", "tasnn": "Tasnn"}
    energy = {}
    for key, _, patt in MRT.MODELS:
        for ds in IOT:
            recs = [json.loads(Path(x).read_text()) for x in sorted(_glob.glob(str(REPO / patt.format(d=ds))))
                    if int(Path(x).parent.name.split("_")[1]) in range(42, 47)]
            if len(recs) != 5:
                raise SystemExit(f"{key}/{ds}: {len(recs)} seeds")
            c = MRT.cell(key, recs)
            for m, mc in (("acc", "Acc"), ("bacc", "Bacc"), ("mcc", "Mcc"), ("dr", "Dr"), ("far", "Far")):
                add(f"JN{mc}{code[key]}{DS[ds]}", f(100 * c[m]), patt.format(d=ds), "five-seed mean",
                    f"{m} x100, {key}, {ds}")
                add(f"JN{mc}{code[key]}{DS[ds]}Sd", f(100 * c[m + "_sd"]), patt.format(d=ds), "five-seed sample SD", "")
            energy[(key, ds)] = c["energy"]

    # 4a. energy table: 32-bit, full-window arithmetic, pJ per sample (same cell())
    for (key, ds), e in energy.items():
        add(f"JNEnergy{code[key]}{DS[ds]}", f"{e:,.0f}".replace(",", "{,}"),
            "scripts/make_results_table.py cell()", "five-seed mean, 32-bit, full window",
            f"arithmetic energy pJ per sample, {key}, {ds}")

    # 2. encoder ablation: macro-F1 and input-inclusive full-window SOPs (thousands)
    variants = {"Tree": "results/Tree-to-Spike/{d}/seed_{s}/results.json",
                "Quantile": "results/encoder_ablation/{d}/quantile/seed_{s}/results.json",
                "Uniform": "results/encoder_ablation/{d}/uniform/seed_{s}/results.json",
                "All": "results/encoder_ablation/{d}/all/seed_{s}/results.json",
                "Nokd": "results/encoder_ablation/{d}/nokd/seed_{s}/results.json"}
    fone = {}
    for v, patt in variants.items():
        for ds in IOT:
            rs = [json.loads((REPO / patt.format(d=ds, s=sd)).read_text()) for sd in range(42, 47)]
            f1v = float(np.mean([100 * r["test_plus"]["metrics"]["f1_macro"] for r in rs]))
            sops = float(np.mean([r["test_plus"]["spike_stats"]["synops_per_sample"]
                                  + r["data"]["encoder_channels"] * r["config"]["hidden"][0] for r in rs]))
            fone[(v, ds)] = f1v
            add(f"JNAbl{v}{DS[ds]}Fone", f(f1v), patt.format(d=ds, s="*"), "five-seed mean", f"macro-F1, {v}, {ds}")
            add(f"JNAbl{v}{DS[ds]}Ksops", f(sops / 1000, 1), patt.format(d=ds, s="*"),
                "five-seed mean, full window, input-inclusive", f"SOPs per sample (thousands), {v}, {ds}")
    m8 = pd.read_csv(REPO / "results/diagnostics/m8_input_spikes.csv")
    for ds in IOT:
        g = m8[m8.dataset == ds]
        if len(g) != 5:
            raise SystemExit(f"m8 input spikes: {ds} has {len(g)} seeds")
        add(f"JNAblRaw{DS[ds]}Fone", f(g.macro_f1.mean()), "results/encoder_ablation/{ds}/m8_raw", "five-seed mean",
            f"macro-F1, raw latency (M8 arm: raw-feature latency code with the no-KD loss), {ds}")
        add(f"JNAblRaw{DS[ds]}Ksops", f(g.total_sops.mean() / 1000, 1), "results/diagnostics/m8_input_spikes.csv",
            "five-seed mean, full window, input-inclusive (spikes actually emitted)",
            f"SOPs per sample (thousands), raw latency, {ds}")
    add("JNAblQuantileLoses", str(sum(fone[("Quantile", ds)] < fone[("Tree", ds)] for ds in IOT)),
        "encoder ablation", "count of four", "IoT datasets where the quantile variant loses macro-F1 to tree")

    # 3. per-channel quantization: delta = int8 minus fp32, M3 reruns
    pc = pd.read_csv(REPO / "results/tables/tab_perchannel.csv")
    pm = pc.groupby(["model", "dataset"])[["fp32_f1", "per_tensor_f1", "per_channel_f1"]].mean()
    for model, mc in (("Tree-to-Spike", "Tts"), ("DNN-Tree", "Dnn")):
        for ds in IOT:
            r = pm.loc[(model, ds)]
            add(f"JNPcDelta{mc}{DS[ds]}", sgn(r.per_channel_f1 - r.fp32_f1), "results/tables/tab_perchannel.csv",
                "five-seed mean, M3 reruns", f"per-channel int8 minus fp32 macro-F1, {model}, {ds}")
            add(f"JNPtDelta{mc}{DS[ds]}", sgn(r.per_tensor_f1 - r.fp32_f1), "results/tables/tab_perchannel.csv",
                "five-seed mean, M3 reruns", f"per-tensor int8 minus fp32 macro-F1, {model}, {ds}")

    # 4b. memory share of total energy, band systems, 100 pJ SRAM, both memory models,
    #     both student configurations (deployed and full window)
    raw = pd.read_csv(REPO / "results/deployment/deployment_measurements.csv", comment="#")
    shares = []
    for cfgname, frame in (("deployed", deployed(raw)), ("full", raw.assign(system=raw.system.replace({"t2s_full": "t2s_deployed"})))):
        fr = frame[frame.dataset.isin(IOT) & frame.system.isin(["tasnn", "vanilla_snn", "dnn_tree", "t2s_deployed"])]
        for reuse in (False, True):
            for _, r in fr.iterrows():
                a, mem = E2.row_energy(r, 100, 1300, reuse)
                shares.append(100 * mem / (a + mem))
    add("JNMemShareMin", pct(min(shares)), "scripts/deployment_energy2.py row_energy",
        "single seed-42 run, IoT four, 100 pJ SRAM, both memory models, both configurations",
        "lowest memory share of total energy across TASNN, Vanilla SNN, DNN-Tree, Tree-to-Spike")
    add("JNMemShareMax", pct(max(shares)), "scripts/deployment_energy2.py row_energy", "same", "highest memory share")
    # 4c. Vanilla SNN / Tree-to-Spike memory-inclusive ratio, deployed
    band = pd.read_csv(REPO / "results/deployment/energy_crossover_iot.csv")
    word = {5: "Five", 10: "Ten", 20: "Twenty", 50: "Fifty", 100: "Hundred"}
    for r in band[band.config == "deployed"].itertuples():
        add(f"JNVanBandDep{r.model.capitalize()}{word[r.sram_pJ]}", f(r.vs_vanilla_snn),
            "results/deployment/energy_crossover_iot.csv", "single seed-42 run, IoT four, deployed",
            f"Vanilla SNN / Tree-to-Spike total energy, {r.model}, {r.sram_pJ} pJ SRAM")

    # 5. deployment extras
    di = raw[raw.dataset.isin(IOT)]
    for ds in IOT:
        for sysname, n in (("vanilla_snn", "Van"), ("compact_gbdt", "GbdtCompact"), ("full_gbdt", "GbdtFull")):
            r = di[(di.dataset == ds) & (di.system == sysname)].iloc[0]
            add(f"JNKb{n}{DS[ds]}", f(r.total_bytes / 1024, 1), "results/deployment/deployment_measurements.csv",
                "single seed-42 run", f"artifact KB, {sysname}, {ds}")
    tt = pd.read_csv(REPO / "results/tables/tab_teacher_per_dataset.csv")
    for arm, n in (("full", "GbdtFull"), ("compact", "GbdtCompact")):
        g = tt[(tt.arm == arm) & tt.dataset.isin(IOT)]
        add(f"JNFone{n}IotMean", f(g.F1M.mean()), "results/tables/tab_teacher_per_dataset.csv",
            "IoT mean of five-seed means", f"GBDT {arm} macro-F1")
        for r in g.itertuples():
            add(f"JNFone{n}{DS[r.dataset]}", f(r.F1M), "results/tables/tab_teacher_per_dataset.csv",
                "five-seed mean", f"GBDT {arm} macro-F1, {r.dataset}")

    # 6. envelope: repriced sweep (verified packed format), 128 KiB
    env = pd.read_csv(REPO / "results/deployment/gbdt_compression_repriced.csv")
    env["gain_prune"] = env.gain_prune.astype(str)
    ei = env[env.dataset.isin(IOT)]
    fits = ei[ei.deploy_bytes_packed_fp32_thresholds <= 131072]
    add("JNEnvFitsIot", str(fits.dataset.nunique()), "results/deployment/gbdt_compression_repriced.csv",
        "seed 42, count of four", "IoT datasets with at least one sweep configuration within 128 KB")
    full = ei[(ei.n_estimators == 200) & (ei.max_depth == 6) & (ei.gain_prune == "none") & (ei.leaf_bits == 16)] \
        .set_index("dataset").macro_f1
    near = fits[fits.apply(lambda r: r.macro_f1 >= full[r.dataset] - 1.0, axis=1)]
    add("JNEnvFitsNearFullIot", str(near.dataset.nunique()), "results/deployment/gbdt_compression_repriced.csv",
        "seed 42, count of four, test-selected (disclosed)",
        "IoT datasets with a configuration within 128 KB AND within one point of the full ensemble")

    # 7. early-exit audit
    sel = pd.read_csv(REPO / "results/early_exit/selected_test.csv")
    ton = sel[(sel.dataset == "toniot") & (~sel.early_exit_disabled)].seed.tolist()
    add("JNExitTonAdmittedCount", str(len(ton)), "results/early_exit/selected_test.csv", "of five seeds",
        "TON-IoT seeds on which early exit is admitted")
    add("JNExitTonAdmittedSeeds", " and ".join(str(x) for x in ton), "results/early_exit/selected_test.csv", "",
        "which TON-IoT seeds admit early exit")
    ea = pd.read_csv(REPO / "results/tables/tab_exit_audit_iot.csv")
    for ds in IOT:
        g = ea[ea.dataset == ds]
        r42 = g[g.seed == 42].iloc[0]
        tp = np.concatenate([_tprime(ds, sd) for sd in range(42, 47)])
        for stat, col in (("Wrong", "wrong_exit"), ("Inclusion", "inclusion")):
            add(f"JNExit{stat}{DS[ds]}SeedFortyTwo", f(r42[col]), "results/tables/tab_exit_audit_iot.csv",
                "seed 42 (as in evidence_arrival.json)", f"{col} %, {ds}")
            add(f"JNExit{stat}{DS[ds]}Five", f(g[col].mean()), "results/tables/tab_exit_audit_iot.csv",
                "mean over five seeds", f"{col} %, {ds}")
            add(f"JNExit{stat}{DS[ds]}Min", f(g[col].min()), "results/tables/tab_exit_audit_iot.csv", "min over seeds", "")
            add(f"JNExit{stat}{DS[ds]}Max", f(g[col].max()), "results/tables/tab_exit_audit_iot.csv", "max over seeds", "")
        add(f"JNExitTMedian{DS[ds]}SeedFortyTwo", f(r42.t_median, 0), "results/tables/tab_exit_audit_iot.csv", "seed 42", "t' median")
        add(f"JNExitTPninety{DS[ds]}SeedFortyTwo", f(r42.t_p90, 0), "results/tables/tab_exit_audit_iot.csv", "seed 42", "t' p90")
        add(f"JNExitTMedian{DS[ds]}Five", f(float(np.median(tp)), 0), "prefix artifacts", "pooled over five seeds", "t' median")
        add(f"JNExitTPninety{DS[ds]}Five", f(float(np.percentile(tp, 90)), 0), "prefix artifacts", "pooled over five seeds", "t' p90")


def pooled():
    """Appendix: pooled eleven-dataset statistics, parsed from scripts/m9_stats.py's output."""
    src = REPO / "results/diagnostics/m9_stats.txt"
    t = src.read_text()
    code = {"DNN-Tree": "Dnn", "Tree-to-Spike": "Tts", "Vanilla SNN": "Van", "TASNN": "Tasnn"}
    # Ranks recomputed EXACTLY from the per-seed cells: m9_stats.txt prints them to three
    # decimals, and re-rounding 1.545 (really 17/11 = 1.5454...) to two gives 1.54, not 1.55.
    from scipy.stats import rankdata
    c = pd.read_csv(REPO / "results/diagnostics/per_seed_all_cells.csv")
    mm = c.groupby(["dataset", "model"]).f1.mean().unstack()
    names = {"dnn": "DNN-Tree", "treespike": "Tree-to-Spike", "vanilla": "Vanilla SNN", "tasnn": "TASNN"}
    R = pd.DataFrame([rankdata(-row) for row in mm.values], columns=mm.columns).mean()
    for m_, name in names.items():
        add(f"JNPooledRank{code[name]}", f(R[m_]), "results/diagnostics/per_seed_all_cells.csv",
            "N = 11 datasets, exact", f"pooled mean rank, {name}")
    chi, pchi = re.search(r"Friedman chi2_F=([\d.]+) df=3 p=([\d.e+-]+)", t).groups()
    ff, pff = re.search(r"Iman-Davenport F_F=([\d.]+) df=\(3,30\) p=([\d.e+-]+)", t).groups()
    cd = re.search(r"Nemenyi CD=([\d.]+)", t).group(1)
    sci = lambda x: (lambda m, e: rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}")(*f"{float(x):.1e}".split("e"))
    add("JNPooledChi", f(float(chi)), rel(src), "N = 11", "pooled Friedman chi-square, df 3")
    add("JNPooledChiP", sci(pchi), rel(src), "N = 11", "pooled Friedman p")
    add("JNPooledIdF", f(float(ff)), rel(src), "N = 11", "pooled Iman-Davenport F, df (3, 30)")
    add("JNPooledIdP", sci(pff), rel(src), "N = 11", "pooled Iman-Davenport p")
    add("JNPooledCD", f(float(cd), 3), rel(src), "N = 11, k = 4, alpha 0.05", "Nemenyi critical difference")
    for a, b_, gap, sig in re.findall(r"  (DNN-Tree|Tree-to-Spike|Vanilla SNN|TASNN) vs (DNN-Tree|Tree-to-Spike|Vanilla SNN|TASNN): ([\d.]+) (SIG|ns)", t):
        add(f"JNPooledGap{code[a]}{code[b_]}", f(float(gap), 3), rel(src), "N = 11", f"mean-rank gap {a} vs {b_} ({sig})")
    md, med, sdv = re.search(r"Tree-to-Spike - DNN-Tree: mean diff ([-\d.]+)\s+median ([-\d.]+)\s+sd ([\d.]+)", t).groups()
    add("JNPooledDiffTtsDnnMean", sgn(float(md)), rel(src), "N = 11", "pooled T2S - DNN mean difference")
    add("JNPooledDiffTtsDnnMedian", sgn(float(med)), rel(src), "N = 11", "pooled T2S - DNN median difference")
    add("JNPooledTostP", f(float(re.search(r"TOST \(margin \+/-1.0\): p=([\d.]+)", t).group(1)), 3), rel(src),
        "N = 11", "TOST p at a one-point margin")
    lo, hi = re.search(r"bootstrap 95% CI \[([-\d.]+), ([-\d.]+)\]", t).groups()
    add("JNPooledCiLo", sgn(float(lo)), rel(src), "N = 11", "bootstrap 95% CI lower")
    add("JNPooledCiHi", sgn(float(hi)), rel(src), "N = 11", "bootstrap 95% CI upper")


def general_purpose():
    """Appendix: the seven general-purpose benchmarks (macro-F1, five-seed means)."""
    src = REPO / "results/diagnostics/per_seed_all_cells.csv"
    c = pd.read_csv(src)
    gp = c[~c.dataset.isin(IOT)]
    if gp.dataset.nunique() != 7 or (gp.groupby(["dataset", "model"]).seed.nunique() != 5).any():
        raise SystemExit("general-purpose cells incomplete")
    means = gp.groupby(["dataset", "model"]).f1.mean().groupby("model").mean()
    for m, mc in MOD.items():
        add(f"JNFone{mc}GpMean", f(means[m]), rel(src), "mean over seven GP datasets of five-seed means",
            f"general-purpose macro-F1, {m}")
    add("JNDiffTtsVanGpMean", sgn(means["treespike"] - means["vanilla"]), rel(src), "GP mean", "T2S minus Vanilla SNN, GP")
    add("JNDiffTtsDnnGpMean", sgn(means["treespike"] - means["dnn"]), rel(src), "GP mean", "T2S minus DNN-Tree, GP")
    cse = gp[gp.dataset == "cse-cic-ids2018"].groupby("model").f1
    for m, mc in MOD.items():
        add(f"JNFone{mc}CseCic", f(cse.mean()[m]), rel(src), "five-seed mean", f"CSE-CIC-IDS2018 macro-F1, {m}")


def results_extras():
    """Flagged column, the CICIoMT2024 accuracy-versus-operations point, and ranges
    the text derives, computed from unrounded values (ratios of rounded macros drift)."""
    fl = pd.read_csv(REPO / "results/tables/tab_m4_flags.csv", index_col=0)
    for ds in IOT:
        add(f"JNExitFlag{DS[ds]}", f(fl.loc[ds, "unc"]), "results/tables/tab_m4_flags.csv",
            "mean over five seeds, each at its validation-selected threshold (full window where rejected)",
            f"theta_unc = 0.5 flag rate %, {ds}")
        add(f"JNExitFlag{DS[ds]}Sd", f(fl.loc[ds, "unc_sd"]), "results/tables/tab_m4_flags.csv", "sample SD over seeds", "")
    add("JNExitBeforeInputMax", f(fl.loc[IOT, "before"].max()), "results/tables/tab_m4_flags.csv",
        "five-seed mean, max over IoT", "percent of samples exiting before their first input spike")
    # accuracy-versus-operations curve, CICIoMT2024, theta 0.99 (descriptive, test)
    c = pd.read_csv(REPO / "results/tables/m4_accuracy_vs_ops.csv")
    c["theta"] = c.theta.astype(str)
    g = c[c.dataset == "ciciomt2024"]
    a, fw = g[g.theta == "0.99"].set_index("seed"), g[g.theta == "full"].set_index("seed")
    src = "results/tables/m4_accuracy_vs_ops.csv"
    add("JNCurveCicIoMTFoneNinetyNine", f(a.macro_f1.mean()), src, "five-seed mean, test, descriptive", "macro-F1 at theta 0.99")
    add("JNCurveCicIoMTFoneFull", f(fw.macro_f1.mean()), src, "five-seed mean", "full-window macro-F1")
    add("JNCurveCicIoMTSavedNinetyNine", pct(100 * (1 - a.mean_sops.mean() / fw.mean_sops.mean())), src,
        "ratio of five-seed means", "percent fewer operations at theta 0.99 than the full window")
    per = 100 * (1 - a.mean_sops / fw.mean_sops)
    add("JNCurveCicIoMTSavedMin", pct(per.min()), src, "min over seeds", "")
    add("JNCurveCicIoMTSavedMax", pct(per.max()), src, "max over seeds", "")
    sel = pd.read_csv(REPO / "results/early_exit/selected_test.csv")
    sm = sel[sel.dataset == "ciciomt2024"]
    add("JNCurveCicIoMTSelectedCount", str(int((sm.selected_threshold == 0.99).sum())),
        "results/early_exit/selected_test.csv", "of five seeds", "seeds on which validation selected theta 0.99")
    add("JNExitDepCicIoMTDiff", sgn(sm.d_macro_f1.mean()), "results/early_exit/selected_test.csv",
        "five-seed mean, validation-selected threshold", "deployed exit minus full-window macro-F1, CICIoMT2024")
    add("JNExitDepCicIoMTSaved", pct(sm.mean_sop_reduction.mean()), "results/early_exit/selected_test.csv",
        "five-seed mean, validation-selected threshold", "percent fewer operations at the deployed exit, CICIoMT2024")
    # ablation ranges, unrounded
    def arm(ds, patt):
        rs = [json.loads((REPO / patt.format(d=ds, s=sd)).read_text()) for sd in range(42, 47)]
        return (np.mean([100 * r["test_plus"]["metrics"]["f1_macro"] for r in rs]),
                np.mean([r["test_plus"]["spike_stats"]["synops_per_sample"]
                         + r["data"]["encoder_channels"] * r["config"]["hidden"][0] for r in rs]))
    m8 = pd.read_csv(REPO / "results/diagnostics/m8_input_spikes.csv").groupby("dataset").total_sops.mean()
    loss, extra, enc = [], [], []
    for ds in IOT:
        (tf, ts), (qf, qs) = arm(ds, "results/Tree-to-Spike/{d}/seed_{s}/results.json"), \
            arm(ds, "results/encoder_ablation/{d}/quantile/seed_{s}/results.json")
        _, ns = arm(ds, "results/encoder_ablation/{d}/nokd/seed_{s}/results.json")
        loss.append(tf - qf); extra.append(100 * (qs / ts - 1)); enc.append(100 * (ns / m8[ds] - 1))
    for nm, v, d_, mng in (("JNAblQuantileLossMin", min(loss), 2, "smallest quantile loss, points"),
                           ("JNAblQuantileLossMax", max(loss), 2, "largest quantile loss, points"),
                           ("JNAblQuantileExtraOpsMin", min(extra), 0, "least extra operations of quantile, percent"),
                           ("JNAblQuantileExtraOpsMax", max(extra), 0, "most extra operations of quantile, percent"),
                           ("JNAblEncoderExtraOpsMin", min(enc), 0, "tree-guided vs raw latency, like-for-like (no-KD loss), percent"),
                           ("JNAblEncoderExtraOpsMax", max(enc), 0, "same, max")):
        add(nm, f(v, d_), "encoder ablation + results/diagnostics/m8_input_spikes.csv", "five-seed means, unrounded", mng)
    # Vanilla SNN against Tree-to-Spike: arithmetic range and total-energy multiple
    t = pd.read_csv(REPO / "results/tables/tab_iot_arithmetic.csv", index_col=0)
    r = t["Vanilla / T2S"]
    if r.isna().any():
        raise SystemExit("tab_iot_arithmetic.csv has an empty Vanilla SNN ratio; regenerate it")
    add("JNVanArithRatioMin", f(r.min()), "results/tables/tab_iot_arithmetic.csv", "five-seed means, IoT min", "")
    add("JNVanArithRatioMax", f(r.max()), "results/tables/tab_iot_arithmetic.csv", "five-seed means, IoT max", "")
    add("JNVanArithRatioIot", f(t["Vanilla pJ"].mean() / t["T2S pJ"].mean()), "results/tables/tab_iot_arithmetic.csv",
        "ratio of IoT means", "Vanilla SNN / Tree-to-Spike arithmetic")
    b = pd.read_csv(REPO / "results/deployment/energy_crossover_iot.csv")
    dv = b[b.config == "deployed"].vs_vanilla_snn
    add("JNTtsOverVanTotalMin", f(1 / dv.max(), 1), "results/deployment/energy_crossover_iot.csv",
        "single seed-42 run, deployed, over swept SRAM prices", "Tree-to-Spike total energy as a multiple of the Vanilla SNN's, min")
    add("JNTtsOverVanTotalMax", f(1 / dv.min(), 1), "results/deployment/energy_crossover_iot.csv", "same", "max")


def _reported_val_acc(r):
    """Validation accuracy of the reported checkpoint: the selected epoch where checkpoint
    selection is on (the CICIoT2023 block, amendment A1), else the last epoch."""
    tc = r["training"]["training_curve"]
    sel = r["training"].get("selection") or {}
    if sel.get("enabled"):
        e = sel["selected_epoch"]
        if tc[e - 1]["epoch"] != e:
            raise SystemExit("training curve is not indexed by epoch")
        return tc[e - 1]["val_accuracy"]
    return tc[-1]["val_accuracy"]


def _tprime(ds, seed):
    z = np.load(REPO / f"results/prefix_outputs/{ds}/treespike/seed_{seed}/test_prefix.npz", allow_pickle=True)
    pl = z["prefix_logits"].astype(np.float32)
    pred, y = pl.argmax(-1), z["labels"]
    ok = pred[:, -1] == y
    return ((pred == y[:, None]) & ok[:, None]).argmax(1)[ok] + 1


def perchannel():
    src = REPO / "results/tables/tab_perchannel.csv"
    if not src.exists():
        print("[skip] per-channel results not aggregated yet")
        return
    d = pd.read_csv(src)
    if (d.groupby("model").size() < 20).any():
        print("[skip] per-channel results incomplete:", d.groupby("model").size().to_dict())
        return
    cols = [("fp32_f1", "Fp"), ("per_tensor_f1", "PerTensor"), ("per_channel_f1", "PerChannel")]
    m = d.groupby(["model", "dataset"])[[c for c, _ in cols]].mean()
    for model, mc in (("Tree-to-Spike", "Tts"), ("DNN-Tree", "Dnn")):
        for col, cc in cols:
            add(f"JNPc{mc}{cc}IotMean", f(m.loc[model][col].mean()), rel(src), "IoT mean of five-seed means, M3 reruns",
                f"{model} macro-F1, {cc}")
    diff = m.loc["Tree-to-Spike"] - m.loc["DNN-Tree"]
    for col, cc in cols:
        add(f"JNPcDiff{cc}IotMean", sgn(diff[col].mean()), rel(src), "IoT mean, M3 reruns",
            f"Tree-to-Spike minus DNN-Tree, {cc}")
        for ds in IOT:
            add(f"JNPcDiff{cc}{DS[ds]}", sgn(diff.loc[ds, col]), rel(src), "five-seed mean, M3 reruns",
                f"Tree-to-Spike minus DNN-Tree, {cc}, {ds}")
    dn = m.loc["DNN-Tree"]
    loss = dn.fp32_f1 - dn.per_tensor_f1
    gain = dn.per_channel_f1 - dn.per_tensor_f1
    add("JNPcDnnRecoveredIot", pct(100 * gain.mean() / loss.mean(), 0), rel(src), "IoT means",
        "percent of DNN-Tree's per-tensor loss recovered by per-channel scaling")
    add("JNPcDnnRecoveredEdgeIIoT", pct(100 * gain["edgeiiot"] / loss["edgeiiot"], 0), rel(src), "five-seed means",
        "same, Edge-IIoTset")
    add("JNPcDnnLossEdgeIIoT", f(loss["edgeiiot"]), rel(src), "five-seed mean", "DNN-Tree fp32 minus per-tensor, Edge-IIoTset")
    add("JNPcDnnLossPerChannelEdgeIIoT", f(dn.fp32_f1["edgeiiot"] - dn.per_channel_f1["edgeiiot"]), rel(src),
        "five-seed mean", "DNN-Tree fp32 minus per-channel, Edge-IIoTset")
    one = d[(d.model == "DNN-Tree") & (d.dataset == "toniot") & (d.seed == 43)].iloc[0]
    add("JNPcTonSeedFp", f(one.fp32_f1), rel(src), "TON-IoT seed 43", "DNN-Tree fp32")
    add("JNPcTonSeedPerTensor", f(one.per_tensor_f1), rel(src), "TON-IoT seed 43", "DNN-Tree per-tensor int8")
    add("JNPcTonSeedPerChannel", f(one.per_channel_f1), rel(src), "TON-IoT seed 43", "DNN-Tree per-channel int8")


# ------------------------------------------------------------------ E1 / E2 (pre-registered)
def e1_e2():
    """E1 (dense twin on spike-time input, "Tq") and E2 (entropy edges), read from the
    per-run records; results/preregistration/2026-09-24_matched_info_and_entropy_edges.md.
    Refuses unless every cell exists and shares its split and teacher with the stored
    Tree-to-Spike run (the registered post-run check)."""
    seeds = range(42, 47)
    rj = lambda p: json.loads((REPO / p).read_text())
    t2s = {(ds, sd): rj(f"results/Tree-to-Spike/{ds}/seed_{sd}/results.json") for ds in IOT for sd in seeds}
    sig = lambda r: (r["test_plus"]["n_samples"], json.dumps(r["data"]["class_counts_train"], sort_keys=True),
                     r["test_plus"]["teacher_upper_bound"]["f1_macro"])

    def load(patt):
        out = {}
        for ds in IOT:
            for sd in seeds:
                p = patt.format(d=ds, s=sd)
                if not (REPO / p).exists():
                    raise SystemExit(f"E1/E2: {p} missing")
                r = rj(p)
                if sig(r) != sig(t2s[(ds, sd)]):
                    raise SystemExit(f"E1/E2: {p} split or teacher differs from Tree-to-Spike")
                out[(ds, sd)] = r
        return out

    # E1: fp32, per-tensor int8, per-channel int8
    E1 = "results/DNN-TQ/{d}/seed_{s}/results.json"
    tq = load(E1)
    # The CICIoT2023 block records (amendment A1) store the three precisions under
    # test_plus.precisions; the 10-epoch records under fp32_before_projection.
    def prec(r, k):
        t = r["test_plus"]
        if "precisions" in t:
            return 100 * t["precisions"][k]["metrics"]["f1_macro"]
        return 100 * {"fp32": t["fp32_before_projection"], "per_tensor_int8": t,
                      "per_channel_int8": t["fp32_before_projection"].get("per_channel_int8")}[k]["metrics"]["f1_macro"]
    for r in tq.values():
        if "precisions" in r["test_plus"] and prec(r, "per_tensor_int8") != 100 * r["test_plus"]["metrics"]["f1_macro"]:
            raise SystemExit("E1: a block record's per-tensor precision is not its deployed score")
    get = {"Fp": lambda r: prec(r, "fp32"), "PerTensor": lambda r: prec(r, "per_tensor_int8"),
           "PerChannel": lambda r: prec(r, "per_channel_int8")}
    pc = pd.read_csv(REPO / "results/tables/tab_perchannel.csv")
    ref = {"Fp": "fp32_f1", "PerTensor": "per_tensor_f1", "PerChannel": "per_channel_f1"}
    src1 = E1.format(d="*", s="*")
    mean, gap, lvl_gap, chg = {}, {}, {}, {}
    for cc, g in get.items():
        for ds in IOT:
            v = np.array([g(tq[(ds, sd)]) for sd in seeds])
            mean[(cc, ds)] = v.mean()
            add(f"JNPcTq{cc}{DS[ds]}", f(v.mean()), src1, "five-seed mean", f"DNN-TQ macro-F1, {cc}, {ds}")
            add(f"JNPcTq{cc}{DS[ds]}Sd", f(v.std(ddof=1)), src1, "five-seed sample SD", f"DNN-TQ, {cc}, {ds}")
            t = pc[(pc.model == "Tree-to-Spike") & (pc.dataset == ds)][ref[cc]].mean()
            l = pc[(pc.model == "DNN-Tree") & (pc.dataset == ds)][ref[cc]].mean()
            gap[(cc, ds)], lvl_gap[(cc, ds)], chg[(cc, ds)] = v.mean() - t, l - t, v.mean() - l
            add(f"JNTqGap{cc}{DS[ds]}", sgn(gap[(cc, ds)]), f"{src1}; results/tables/tab_perchannel.csv",
                "five-seed means", f"DNN-TQ minus Tree-to-Spike, {cc}, {ds}")
        add(f"JNPcTq{cc}IotMean", f(np.mean([mean[(cc, ds)] for ds in IOT])), src1,
            "IoT mean of five-seed means", f"DNN-TQ macro-F1, {cc}")
        add(f"JNTqGap{cc}IotMean", sgn(np.mean([gap[(cc, ds)] for ds in IOT])),
            f"{src1}; results/tables/tab_perchannel.csv", "IoT mean", f"DNN-TQ minus Tree-to-Spike, {cc}")
    for cc, nm in (("PerTensor", "Pt"), ("PerChannel", "Pc")):
        for ds in IOT:
            add(f"JN{nm}DeltaTq{DS[ds]}", sgn(mean[(cc, ds)] - mean[("Fp", ds)]), src1, "five-seed mean",
                f"{cc} int8 minus fp32 macro-F1, DNN-TQ, {ds}")
        add(f"JN{nm}DeltaTqIotMean", sgn(np.mean([mean[(cc, ds)] - mean[("Fp", ds)] for ds in IOT])), src1,
            "IoT mean", f"{cc} int8 minus fp32 macro-F1, DNN-TQ")
    g_lv = np.mean([lvl_gap[("Fp", ds)] for ds in IOT])
    g_tq = np.mean([gap[("Fp", ds)] for ds in IOT])
    add("JNTqGapLevelFpIotMean", sgn(g_lv), "results/tables/tab_perchannel.csv", "IoT mean, M3 reruns",
        "DNN-Tree (level input) minus Tree-to-Spike, fp32 (E1 reference gap)")
    add("JNTqClosedPct", pct(100 * (1 - g_tq / g_lv), 0), f"{src1}; results/tables/tab_perchannel.csv", "IoT means",
        "E1: percent of the fp32 gap closed by spike-time input (registered outcome C)")
    for cc, nm in (("Fp", "Fp"), ("PerTensor", "PerTensor")):
        tot = sum(chg[(cc, ds)] for ds in IOT)
        add(f"JNTqEdgeShare{nm}", pct(100 * chg[(cc, "edgeiiot")] / tot, 0), f"{src1}; results/tables/tab_perchannel.csv",
            "five-seed means",
            f"E1: Edge-IIoTset's percent of the four-dataset {cc} change (spike-time minus level input)")
    t2s_fp = pc[(pc.model == "Tree-to-Spike") & (pc.dataset == "toniot")].set_index("seed").fp32_f1
    ton = [get["Fp"](tq[("toniot", sd)]) - t2s_fp[sd] for sd in seeds]
    add("JNTqTonGapMin", f(min(ton)), f"{src1}; results/tables/tab_perchannel.csv", "TON-IoT, per seed", "DNN-TQ minus Tree-to-Spike fp32, smallest seed")
    add("JNTqTonGapMax", f(max(ton)), f"{src1}; results/tables/tab_perchannel.csv", "TON-IoT, per seed", "DNN-TQ minus Tree-to-Spike fp32, largest seed")

    # E2: entropy edges vs tree edges
    E2p = "results/encoder_ablation/{d}/entropy/seed_{s}/results.json"
    ent = load(E2p)
    src2 = E2p.format(d="*", s="*")
    fone = lambda r: 100 * r["test_plus"]["metrics"]["f1_macro"]
    ksops = lambda r: (r["test_plus"]["spike_stats"]["synops_per_sample"]
                       + r["data"]["encoder_channels"] * r["config"]["hidden"][0]) / 1000
    quant = {(ds, sd): rj(f"results/encoder_ablation/{ds}/quantile/seed_{sd}/results.json") for ds in IOT for sd in seeds}
    d_ds, fill, tot_e, vs_q = {}, 0, 0, {}
    for ds in IOT:
        e = np.array([fone(ent[(ds, sd)]) for sd in seeds])
        t = np.array([fone(t2s[(ds, sd)]) for sd in seeds])
        q = np.array([fone(quant[(ds, sd)]) for sd in seeds])
        d = t - e
        d_ds[ds], vs_q[ds] = d.mean(), e.mean() - q.mean()
        add(f"JNAblEntropy{DS[ds]}Fone", f(e.mean()), src2, "five-seed mean", f"macro-F1, Entropy, {ds}")
        add(f"JNAblEntropy{DS[ds]}FoneSd", f(e.std(ddof=1)), src2, "five-seed sample SD", f"macro-F1, Entropy, {ds}")
        add(f"JNAblEntropy{DS[ds]}Ksops", f(np.mean([ksops(ent[(ds, sd)]) for sd in seeds]), 1), src2,
            "five-seed mean, full window, input-inclusive", f"SOPs per sample (thousands), Entropy, {ds}")
        add(f"JNEntD{DS[ds]}", sgn(d.mean()), src2, "five-seed means", f"E2: tree minus entropy macro-F1, {ds}")
        add(f"JNEntDMin{DS[ds]}", sgn(d.min()), src2, "per seed, paired", f"E2: smallest seed tree minus entropy, {ds}")
        add(f"JNEntDMax{DS[ds]}", sgn(d.max()), src2, "per seed, paired", f"E2: largest seed tree minus entropy, {ds}")
        add(f"JNEntSeedsAhead{DS[ds]}", str(int((d > 0).sum())), src2, "count of five seeds",
            f"E2: seeds where tree beats entropy, {ds}")
        add(f"JNEntVsQuantile{DS[ds]}", sgn(vs_q[ds]), src2, "five-seed means", f"entropy minus quantile macro-F1, {ds}")
        for sd in seeds:
            g = ent[(ds, sd)]["data"]["entropy_edges"]
            fill += g["filler_total"]; tot_e += g["edges_entropy_total"]
    dsum = sum(d_ds.values())
    add("JNEntDIotMean", sgn(dsum / 4), src2, "IoT mean", "E2: tree minus entropy macro-F1 (registered outcome A)")
    add("JNEntTreeAhead", str(sum(v > 0 for v in d_ds.values())), src2, "count of four", "E2: datasets where tree beats entropy")
    add("JNEntBeatsQuantile", str(sum(v > 0 for v in vs_q.values())), src2, "count of four",
        "datasets where entropy beats quantile")
    for ds in IOT:
        add(f"JNEntShare{DS[ds]}", pct(100 * d_ds[ds] / dsum, 0), src2, "five-seed means",
            f"E2: {ds}'s percent of the four-dataset tree-minus-entropy sum")
    shares = []
    for ds in IOT:
        gs = [ent[(ds, sd)]["data"]["entropy_edges"] for sd in seeds]
        shares.append(sum(g["filler_total"] for g in gs) / sum(g["edges_entropy_total"] for g in gs))
    add("JNEntFallbackMaxPct", pct(100 * max(shares), 1), src2, "per dataset, pooled over seeds",
        "E2: largest per-dataset share of median fallback cuts (percent)")
    # Stalled runs (post hoc, disclosed; not a registered analysis): final validation
    # accuracy more than 5 points below the median final validation accuracy of the ten
    # runs (tree and entropy arms) on the same dataset. "Final" is the reported
    # checkpoint: the selected epoch in the CICIoT2023 block, the last epoch elsewhere.
    fva = _reported_val_acc
    stalled = {"tree": set(), "entropy": set()}
    for ds in IOT:
        med = np.median([fva(t2s[(ds, sd)]) for sd in seeds] + [fva(ent[(ds, sd)]) for sd in seeds])
        for arm, recs in (("tree", t2s), ("entropy", ent)):
            stalled[arm] |= {(ds, sd) for sd in seeds if fva(recs[(ds, sd)]) < med - 0.05}
    rule = "post hoc: final val. accuracy > 5 points below the dataset's ten-run median"
    add("JNEntStalled", str(len(stalled["entropy"])), src2, rule, "E2: stalled entropy runs (of 20)")
    add("JNEntStalledTree", str(len(stalled["tree"])), "results/Tree-to-Spike/*/seed_*/results.json", rule,
        "E2: stalled tree runs (of 20)")
    d_ns = {}
    for ds in IOT:
        keep = [sd for sd in seeds if (ds, sd) not in stalled["entropy"] | stalled["tree"]]
        d_ns[ds] = np.mean([fone(t2s[(ds, sd)]) - fone(ent[(ds, sd)]) for sd in keep])
        add(f"JNEntD{DS[ds]}NoStall", sgn(d_ns[ds]), src2, f"mean over non-stalled seeds ({rule})",
            f"E2: tree minus entropy without stalled runs, {ds}")
    add("JNEntDIotMeanNoStall", sgn(np.mean(list(d_ns.values()))), src2, f"IoT mean ({rule})",
        "E2: tree minus entropy without stalled runs")
    add("JNEntTreeAheadNoStall", str(sum(v > 0 for v in d_ns.values())), src2, f"count of four ({rule})",
        "E2: datasets where tree beats entropy without stalled runs")
    # Tree edge counts per kept feature (the per-dataset pattern in Section V-B)
    for ds in IOT:
        pfs = [ent[(ds, sd)]["data"]["entropy_edges"]["per_feature"] for sd in seeds]
        meds = [np.median([v[0] for v in pf.values()]) for pf in pfs]
        ones = [np.mean([v[0] == 1 for v in pf.values()]) for pf in pfs]
        add(f"JNEntMedianEdges{DS[ds]}", f(np.mean(meds), 0), src2, "mean over seeds of the per-seed median",
            f"tree edges per kept feature, median, {ds}")
        add(f"JNEntSingleEdgePct{DS[ds]}", pct(100 * np.mean(ones), 0), src2, "mean over seeds",
            f"percent of kept features with a single tree edge, {ds}")
    # Single-threshold features: do tree and entropy pick the same cut? (seed 42 spot check)
    sc_src = REPO / "results/diagnostics/single_edge_partition.csv"
    sc = pd.read_csv(sc_src)
    for ds in ("edgeiiot", "toniot"):
        g = sc[sc.dataset == ds]
        add(f"JNEntSingleCutSame{DS[ds]}", str(int(g.same_partition.sum())), rel(sc_src), "seed 42",
            f"single-edge features where the entropy cut splits the training values as the tree's does, {ds}")
        add(f"JNEntSingleCutTotal{DS[ds]}", str(len(g)), rel(sc_src), "seed 42", f"single-edge kept features, {ds}")
    add("JNEntSingleCutSame", str(int(sc.same_partition.sum())), rel(sc_src), "seed 42, Edge-IIoTset and TON-IoT",
        "single-edge features where the entropy cut matches the tree's partition")
    add("JNEntSingleCutTotal", str(len(sc)), rel(sc_src), "seed 42, Edge-IIoTset and TON-IoT", "single-edge kept features")
    for v, patt in (("Tree", "results/Tree-to-Spike/{d}/seed_{s}/results.json"),
                    ("Quantile", "results/encoder_ablation/{d}/quantile/seed_{s}/results.json"), ("Entropy", E2p)):
        m = np.mean([np.mean([fone(rj(patt.format(d=ds, s=sd))) for sd in seeds]) for ds in IOT])
        add(f"JNAbl{v}IotMeanFone", f(m), patt.format(d="*", s="*"), "IoT mean of five-seed means", f"macro-F1, {v}")


# ------------------------------------------------------------------ Part B (pre-registered)
# results/preregistration/2026-09-28_journal_partB.md with amendment 1. Each item reads
# the committed output of its own script. The run folders are fixed here, so adopting a
# rerun means editing this table, and the change shows in the diff.
PB = REPO / "results/journal_partB"
PB_RUN = {"B1": "B1/20260928T052115Z-228a55", "B2": "B2/20260928T051838Z-70742c",
          "B4": "B4/20260928T051907Z-3f9ad4", "B4gbdt": "B4_gbdt/20260928T053446Z-0364e8",
          "B5": "B5/20260928T051420Z-5888c1", "B6": "B6/20260928T051655Z-c9f47f",
          "A6": "A6/20261006T155717Z-0a37dd", "B8": "B8/20261007T041558Z-66406a"}
SEEDS = range(42, 47)
FIXED = "results/fixed_budget_ciciot2023"
# the twelve arms of the CICIoT2023 block (A1): code, record path under results/
BLOCK_ARMS = {"tree": ("Tts", "Tree-to-Spike/{d}/seed_{s}/results.json"),
              "dnn": ("Dnn", "DNN-Tree/{d}/seed_{s}/results.json"),
              "vanilla": ("Van", "{d}/SNN_LeakyParallel/latency/seed_{s}/results.json"),
              "tasnn": ("Tasnn", "TASNN/{d}/seed_{s}/results.json"),
              "dnn_tq": ("Tq", "DNN-TQ/{d}/seed_{s}/results.json"),
              "raw": ("Raw", "encoder_ablation/{d}/m8_raw/seed_{s}/results.json"),
              **{a: (c, f"encoder_ablation/{{d}}/{a}/seed_{{s}}/results.json") for a, c in
                 (("quantile", "Quantile"), ("uniform", "Uniform"), ("all", "All"), ("nokd", "Nokd"),
                  ("entropy", "Entropy"), ("entropy_tree", "EntTree"))}}
CLS = {"dos": "Dos", "r2l": "RtwoL", "probe": "Probe"}


def _pb(item):
    return PB / PB_RUN[item]


def _rj(p):
    return json.loads(Path(p).read_text())


def _int(v):
    return f"{int(round(v)):,}".replace(",", "{,}")


def _sci(v, d=1):
    if v == 0:
        return "0"
    m, e = f"{v:.{d}e}".split("e")
    return rf"\ensuremath{{{m}\times10^{{{int(e)}}}}}"


def _f1(r):
    return 100 * r["test_plus"]["metrics"]["f1_macro"]


def _far(r):
    return 100 * r["test_plus"]["binary_ids_metrics"]["false_alarm_rate"]


def _ksops(r):
    return (r["test_plus"]["spike_stats"]["synops_per_sample"]
            + r["data"]["encoder_channels"] * r["config"]["hidden"][0]) / 1000


def _split(r):
    n = r["test_plus"]["n_samples"] if "test_plus" in r else r["data"]["n_test"]
    return n, json.dumps(r["data"]["class_counts_train"], sort_keys=True)


def _arm(patt, datasets=IOT, root="results", teacher=True):
    """Five seeds of one arm. Refuses unless each run shares its split, and its teacher
    where it records one, with the Tree-to-Spike run under the same root."""
    out = {}
    for ds in datasets:
        for sd in SEEDS:
            p = f"{root}/{patt.format(d=ds, s=sd)}"
            r, t = _rj(REPO / p), _rj(REPO / f"{root}/Tree-to-Spike/{ds}/seed_{sd}/results.json")
            if _split(r) != _split(t):
                raise SystemExit(f"{p}: split differs from Tree-to-Spike")
            tb = r["test_plus"].get("teacher_upper_bound")
            if teacher and tb is not None and tb["f1_macro"] != t["test_plus"]["teacher_upper_bound"]["f1_macro"]:
                raise SystemExit(f"{p}: teacher differs from Tree-to-Spike")
            out[(ds, sd)] = r
    return out


def partB_b1():
    """B1: Proposition 1 on 55 cells (eleven datasets, seeds 42-46) and the encoder audit."""
    d = _pb("B1")
    s, c = _rj(d / "summary.json"), pd.read_csv(d / "cells.csv")
    if len(c) != 55 or c.groupby("dataset").seed.nunique().ne(5).any() or set(c.dataset) != set(DS) \
            or not s["guard_352_767"] or not c.table_match.all():
        raise SystemExit("B1: not the complete 55-cell run, or its guard failed")
    src = rel(d / "cells.csv")
    agree = min(c.agree_dump.min(), c.agree_exact.min())
    add("JNPropCells", str(len(c)), src, "count", "B1: dataset-seed cells checked")
    add("JNPropFlows", _int(c.n_test.sum()), src, "sum over the 55 cells", "B1: test flows compared with their surrogates")
    add("JNPropAgreePct", "100" if agree == 1.0 else f(100 * agree, 3), src,
        "minimum over cells and both threshold sources", "B1: percent of test flows whose teacher label equals the surrogate's")
    add("JNPropDisagree", str(int(c.disagree_dump.sum() + c.disagree_exact.sum())), src, "sum over cells",
        "B1: flows whose label differs from the surrogate's")
    add("JNPropMaxDproba", f"{max(c.max_dproba_dump.max(), c.max_dproba_exact.max()):g}", src, "max over cells",
        "B1: largest absolute difference in a predicted probability")
    add("JNPropThresholdMismatch", str(int(c.thresholds_dump_vs_exact_mismatch.sum())), src, "sum over cells",
        "B1: thresholds where the text dump differs from the model JSON's float32 value")
    for ds in DS:
        g = c[c.dataset == ds]
        over, gain = 100 * g.violating.sum() / g.retained.sum(), 100 * g.pruned_gain_share.mean()
        if abs(over - s["share_exceeding_bound_by_dataset"][ds]) > 0.006 or abs(gain - s["pruned_gain_share_by_dataset"][ds]) > 0.0006:
            raise SystemExit(f"B1: cells.csv and summary.json disagree on {ds}")
        add(f"JNPropOver{DS[ds]}", pct(over), src, "pooled over five seeds",
            f"B1: percent of retained features with |E_j| >= T = 20, {ds}")
        add(f"JNPropMedianEdges{DS[ds]}", f(s["median_E_by_dataset"][ds], 1), rel(d / "summary.json"),
            "five-seed mean of per-seed medians", f"B1: median |E_j| of retained features, {ds}")
        add(f"JNPropPrunedGain{DS[ds]}", f(gain), src, "five-seed mean", f"B1: percent of total gain removed by pruning, {ds}")
    add("JNPropOverAll", pct(100 * c.violating.sum() / c.retained.sum()), src, "pooled over 55 cells",
        "B1: percent of retained features over the bound, eleven datasets")
    add("JNPropViolatingAll", _int(c.violating.sum()), src, "pooled over 55 cells", "B1: retained features over the bound")
    add("JNPropRetainedAll", _int(c.retained.sum()), src, "pooled over 55 cells", "B1: retained features")


def partB_b2():
    """B2: entropy cutter vs per-feature entropy tree (seed 42), and the fourth arm (A2)."""
    d = _pb("B2")
    s, pf = _rj(d / "summary.json"), pd.read_csv(d / "per_feature.csv")
    if set(pf.seed) != {42} or set(pf.dataset) != set(IOT):
        raise SystemExit("B2: per_feature.csv is not seed 42 on the IoT four")
    src = rel(d / "per_feature.csv")
    co = 100 * pf.coincide.sum() / pf.ig_cuts.sum()
    if abs(co - s["pooled_coincide_pct"]) > 1e-9:
        raise SystemExit("B2: per_feature.csv and summary.json disagree")
    add("JNCutsRule", f"{s['rule_pct']:.0f}", rel(d / "summary.json"), "registered rule", "B2: coincidence share that ends B2")
    add("JNCutsCoincidePooled", f(co), src, "seed 42, pooled IoT", "B2: percent of information-gain cuts that coincide")
    add("JNCutsExactPooled", f(100 * pf.exact.sum() / pf.ig_cuts.sum()), src, "seed 42, pooled IoT", "B2: percent matching exactly")
    add("JNCutsCoincideCount", _int(pf.coincide.sum()), src, "seed 42, pooled IoT", "B2: coinciding cuts")
    add("JNCutsTotal", _int(pf.ig_cuts.sum()), src, "seed 42, pooled IoT", "B2: information-gain cuts compared")
    add("JNCutsFillers", str(int(pf.fillers.sum())), src, "seed 42, pooled IoT", "B2: the cutter's median filler cuts (not compared, D3)")
    for ds in IOT:
        g = pf[pf.dataset == ds]
        add(f"JNCutsCoincide{DS[ds]}", pct(100 * g.coincide.sum() / g.ig_cuts.sum()), src, "seed 42", f"B2: percent coinciding, {ds}")
        add(f"JNCutsExact{DS[ds]}", pct(100 * g.exact.sum() / g.ig_cuts.sum()), src, "seed 42", f"B2: percent exact, {ds}")
        add(f"JNCutsMedianOffset{DS[ds]}", _sci(s["per_dataset"][ds]["median_offset"]), rel(d / "summary.json"), "seed 42",
            f"B2: median absolute offset in scaled [0, 1] units, {ds}")
    # fourth arm (A2): per-feature entropy-tree cuts, E2 protocol; CICIoT2023 from the block
    t2s = _arm("Tree-to-Spike/{d}/seed_{s}/results.json")
    et = _arm("encoder_ablation/{d}/entropy_tree/seed_{s}/results.json")
    ent = _arm("encoder_ablation/{d}/entropy/seed_{s}/results.json")
    src = "results/encoder_ablation/*/entropy_tree/seed_*/results.json"
    rule = "post hoc (E2's): reported-checkpoint val. accuracy > 5 points below the dataset's ten-run median (tree and entropy tree)"
    st_et, st_tree = set(), set()
    for ds in IOT:
        med = np.median([_reported_val_acc(t2s[(ds, sd)]) for sd in SEEDS] + [_reported_val_acc(et[(ds, sd)]) for sd in SEEDS])
        st_et |= {(ds, sd) for sd in SEEDS if _reported_val_acc(et[(ds, sd)]) < med - 0.05}
        st_tree |= {(ds, sd) for sd in SEEDS if _reported_val_acc(t2s[(ds, sd)]) < med - 0.05}
    d_ds, d_ns, over, fone = {}, {}, {}, {}
    for ds in IOT:
        e = np.array([_f1(et[(ds, sd)]) for sd in SEEDS])
        t = np.array([_f1(t2s[(ds, sd)]) for sd in SEEDS])
        n = np.array([_f1(ent[(ds, sd)]) for sd in SEEDS])
        dd = t - e
        d_ds[ds], over[ds], fone[ds] = dd.mean(), e.mean() - n.mean(), e.mean()
        add(f"JNAblEntTree{DS[ds]}Fone", f(e.mean()), src, "five-seed mean", f"macro-F1, entropy tree, {ds}")
        add(f"JNAblEntTree{DS[ds]}FoneSd", f(e.std(ddof=1)), src, "five-seed sample SD", f"macro-F1, entropy tree, {ds}")
        add(f"JNAblEntTree{DS[ds]}Ksops", f(np.mean([_ksops(et[(ds, sd)]) for sd in SEEDS]), 1), src,
            "five-seed mean, full window, input-inclusive", f"SOPs per sample (thousands), entropy tree, {ds}")
        add(f"JNAblEntTree{DS[ds]}Far", f(np.mean([_far(et[(ds, sd)]) for sd in SEEDS])), src, "five-seed mean",
            f"FAR %, entropy tree, {ds}")
        add(f"JNFourthD{DS[ds]}", sgn(dd.mean()), src, "five-seed means", f"A2: tree minus entropy tree macro-F1, {ds}")
        add(f"JNFourthSeedsAhead{DS[ds]}", str(int((dd > 0).sum())), src, "count of five seeds",
            f"A2: seeds where tree beats entropy tree, {ds}")
        add(f"JNFourthOverEnt{DS[ds]}", sgn(over[ds]), src, "five-seed means", f"A2: entropy tree minus entropy cutter, {ds}")
        keep = [i for i, sd in enumerate(SEEDS) if (ds, sd) not in st_et | st_tree]
        d_ns[ds] = dd[keep].mean()
        add(f"JNFourthD{DS[ds]}NoStall", sgn(d_ns[ds]), src, f"mean over non-stalled seeds ({rule})",
            f"A2: tree minus entropy tree without stalled runs, {ds}")
    add("JNFourthDIotMean", sgn(np.mean(list(d_ds.values()))), src, "IoT mean", "A2: tree minus entropy tree macro-F1")
    add("JNFourthAhead", str(sum(v > 0 for v in d_ds.values())), src, "count of four", "A2: datasets where tree beats entropy tree")
    add("JNAblEntTreeIotMeanFone", f(np.mean(list(fone.values()))), src, "IoT mean of five-seed means", "macro-F1, entropy tree")
    add("JNFourthStalled", str(len(st_et)), src, rule, "A2: stalled entropy-tree runs (of 20)")
    add("JNFourthStalledTree", str(len(st_tree)), "results/Tree-to-Spike/*/seed_*/results.json", rule,
        "A2: stalled tree runs under the same rule (of 20)")
    add("JNFourthDIotMeanNoStall", sgn(np.mean(list(d_ns.values()))), src, f"IoT mean ({rule})",
        "A2: tree minus entropy tree without stalled runs")
    add("JNFourthAheadNoStall", str(sum(v > 0 for v in d_ns.values())), src, f"count of four ({rule})",
        "A2: datasets where tree beats entropy tree without stalled runs")
    add("JNFourthOverEntMin", sgn(min(over.values())), src, "five-seed means, IoT min", "A2: entropy tree minus entropy cutter, smallest")
    add("JNFourthOverEntMax", sgn(max(over.values())), src, "five-seed means, IoT max", "A2: entropy tree minus entropy cutter, largest")


def partB_b3():
    """B3: PLE-MLP (rtdl_num_embeddings), selected configuration, five seeds; latency (B8 run)."""
    sel = _rj(PB / "B3/selection.json")["selection"]
    recs = {}
    for ds in IOT:
        for sd in SEEDS:
            d = PB / f"B3/final/{ds}/seed_{sd}"
            if (d / "REUSED_FROM.txt").exists():
                d = REPO / (d / "REUSED_FROM.txt").read_text().strip()
            r = _rj(d / "results.json")
            if r["config"] != sel[ds]["config"] or r["seed"] != sd or r["dataset"] != ds:
                raise SystemExit(f"B3: {d} is not the selected configuration's seed-{sd} run")
            if _split(r) != _split(_rj(REPO / f"results/Tree-to-Spike/{ds}/seed_{sd}/results.json")):
                raise SystemExit(f"B3: {d} split differs from Tree-to-Spike")
            recs[(ds, sd)] = r
    src = "results/journal_partB/B3/final/*/seed_*/results.json"
    P = {"Fp": "fp32", "PerTensor": "per_tensor_int8", "PerChannel": "per_channel_int8"}
    means = {}
    for cc, k in P.items():
        for ds in IOT:
            v = np.array([100 * recs[(ds, sd)]["test"][k]["metrics"]["f1_macro"] for sd in SEEDS])
            means[(cc, ds)] = v.mean()
            add(f"JNPle{cc}{DS[ds]}", f(v.mean()), src, "five-seed mean", f"B3: PLE-MLP macro-F1, {k}, {ds}")
            add(f"JNPle{cc}{DS[ds]}Sd", f(v.std(ddof=1)), src, "five-seed sample SD", f"B3: PLE-MLP, {k}, {ds}")
        add(f"JNPle{cc}IotMean", f(np.mean([means[(cc, ds)] for ds in IOT])), src, "IoT mean of five-seed means",
            f"B3: PLE-MLP macro-F1, {k}")
    for ds in IOT:
        v = np.array([100 * recs[(ds, sd)]["test"]["fp32"]["binary_ids_metrics"]["false_alarm_rate"] for sd in SEEDS])
        add(f"JNPleFar{DS[ds]}", f(v.mean()), src, "five-seed mean, fp32", f"B3: PLE-MLP FAR %, {ds}")
        add(f"JNPleFar{DS[ds]}Sd", f(v.std(ddof=1)), src, "five-seed sample SD, fp32", f"B3: PLE-MLP FAR %, {ds}")
        m = recs[(ds, 42)]["model"]
        add(f"JNPleParams{DS[ds]}", _int(m["parameters"]), src, "seed 42", f"B3: PLE-MLP parameters, {ds}")
        add(f"JNPleEncBytes{DS[ds]}", _int(m["encoder_storage_bytes"]), src, "seed 42", f"B3: bin edges at 4 B, {ds}")
        add(f"JNPleMacs{DS[ds]}", _int(m["macs_total"]), src, "seed 42", f"B3: MACs incl. encoding (features x bins), {ds}")
        add(f"JNPleNj{DS[ds]}", _int(m["energy_pj_32bit"] / 1000), src, "seed 42, 4.6 pJ per MAC", f"B3: arithmetic nJ, 32-bit, {ds}")
        add(f"JNPleNjInt{DS[ds]}", f(m["energy_pj_int8"] / 1000, 1), src, "seed 42, 0.23 pJ per MAC", f"B3: arithmetic nJ, int8, {ds}")
    pc = pd.read_csv(REPO / "results/tables/tab_perchannel.csv")
    ref = {"Fp": "fp32_f1", "PerTensor": "per_tensor_f1", "PerChannel": "per_channel_f1"}
    for cc in P:
        ple = np.mean([means[(cc, ds)] for ds in IOT])
        for model, mc in (("Tree-to-Spike", "Tts"), ("DNN-Tree", "Dnn")):
            o = np.mean([pc[(pc.model == model) & (pc.dataset == ds)][ref[cc]].mean() for ds in IOT])
            add(f"JNPleMinus{mc}{cc}IotMean", sgn(ple - o), f"{src}; results/tables/tab_perchannel.csv",
                "IoT means of five-seed means (M3 reruns)", f"B3: PLE-MLP minus {model}, {P[cc]}")
    tt = pd.read_csv(REPO / "results/tables/tab_teacher_per_dataset.csv")
    gf = tt[(tt.arm == "full") & tt.dataset.isin(IOT)].F1M.mean()
    add("JNPleMinusGbdtFullIotMean", sgn(np.mean([means[("Fp", ds)] for ds in IOT]) - gf),
        f"{src}; results/tables/tab_teacher_per_dataset.csv", "IoT means", "B3: PLE-MLP (fp32) minus the full GBDT")
    # selection (D4, D5)
    c0 = {k: {sel[ds]["config"][k] for ds in IOT} for k in ("n_bins", "width", "epochs")}
    if any(len(v) != 1 for v in c0.values()):
        raise SystemExit("B3: datasets differ in bins, width or epochs; the macros below assume one value")
    add("JNPleBins", str(c0["n_bins"].pop()), "results/journal_partB/B3/selection.json", "selected on every dataset", "B3: bins")
    add("JNPleWidth", str(c0["width"].pop()), "results/journal_partB/B3/selection.json", "selected on every dataset", "B3: width")
    add("JNPleEpochs", str(c0["epochs"].pop()), "results/journal_partB/B3/selection.json", "selected on every dataset", "B3: epoch budget")
    for ds in IOT:
        add(f"JNPleLr{DS[ds]}", _sci(sel[ds]["config"]["lr"], 0), "results/journal_partB/B3/selection.json", "selected", f"B3: learning rate, {ds}")
    n_search = len(list((PB / "B3/search").glob("*/cfg_*/results.json")))
    add("JNPleSearchRuns", str(n_search), "results/journal_partB/B3/search", "count", "B3: search runs (seed 42)")
    be = [r["training"]["best_epoch"] for r in recs.values()]
    add("JNPleBestEpochMin", str(min(be)), src, "min over 20 runs", "B3: earliest selected epoch")
    add("JNPleBestEpochMax", str(max(be)), src, "max over 20 runs", "B3: latest selected epoch")
    lat = pd.read_csv(_pb("B8") / "ple_latency.csv").set_index("dataset")
    lsrc = rel(_pb("B8") / "ple_latency.csv")
    for ds in IOT:
        add(f"JNLatMsPle{DS[ds]}", f(lat.loc[ds, "p50_ms"], 3), lsrc, "single seed-42 run, batch-1 CPU", f"B3: PLE-MLP p50 latency ms, {ds}")
    add("JNLatMsPle", f(lat.loc[IOT, "p50_ms"].mean(), 3), lsrc, "single seed-42 run, IoT mean, batch-1 CPU", "B3: PLE-MLP p50 latency ms")


def partB_b4():
    """B4 and A3: CICIoMT2024 provider split with the validation-selected GBDT; A4 feasibility."""
    ds = "ciciomt2024"
    g = pd.read_csv(_pb("B4gbdt") / "selected.csv").set_index("seed").loc[list(SEEDS)]
    pcl = pd.read_csv(_pb("B4gbdt") / "per_class.csv")
    src = rel(_pb("B4gbdt") / "selected.csv")
    t2s = _arm("Tree-to-Spike/{d}/seed_{s}/results.json", [ds])
    dnn = _arm("DNN-Tree/{d}/seed_{s}/results.json", [ds])
    ent = _arm("encoder_ablation/{d}/entropy/seed_{s}/results.json", [ds])
    f1 = lambda recs: np.array([_f1(recs[(ds, sd)]) for sd in SEEDS])
    T, D, E, G = f1(t2s), f1(dnn), f1(ent), g.test_macro_f1.values
    add("JNNrGbdtFone", f(G.mean()), src, "five-seed mean", "A3: validation-selected GBDT macro-F1, CICIoMT2024")
    add("JNNrGbdtFoneSd", f(G.std(ddof=1)), src, "five-seed sample SD", "")
    add("JNNrGbdtFar", f(g.FAR.mean()), src, "five-seed mean", "A3: GBDT FAR %")
    add("JNNrGbdtFarSd", f(g.FAR.std(ddof=1)), src, "five-seed sample SD", "")
    add("JNNrGbdtDr", f(g.DR.mean()), src, "five-seed mean", "A3: GBDT detection rate %")
    modal = g.groupby(["rounds", "depth"]).size().idxmax()
    add("JNNrGbdtRounds", str(int(modal[0])), src, "modal selection", "A3: selected boosting rounds")
    add("JNNrGbdtDepth", str(int(modal[1])), src, "modal selection", "A3: selected depth")
    add("JNNrGbdtSameConfig", str(int(((g.rounds == modal[0]) & (g.depth == modal[1])).sum())), src, "count of five seeds",
        "A3: seeds selecting that configuration")
    for nm, a, b in (("TtsMinusEnt", T, E), ("TtsMinusDnn", T, D), ("GbdtMinusTts", G, T)):
        add(f"JNNr{nm}", sgn((a - b).mean()), f"{src}; stored CICIoMT2024 runs", "paired by seed, five-seed mean",
            f"B4: {nm}, macro-F1, CICIoMT2024 provider split")
        add(f"JNNr{nm}Seeds", str(int(((a - b) > 0).sum())), f"{src}; stored CICIoMT2024 runs", "count of five seeds",
            f"B4: seeds where {nm} is positive")
    for nm, recs in (("Tts", t2s), ("Dnn", dnn), ("Ent", ent)):
        add(f"JNNrFar{nm}", f(np.mean([_far(recs[(ds, sd)]) for sd in SEEDS])), "stored CICIoMT2024 runs", "five-seed mean",
            f"B4: FAR %, {nm}")
    for cls, code in (("Benign", "Benign"), ("DDoS", "Ddos"), ("DoS", "Dos"), ("MQTT", "Mqtt"), ("Recon", "Recon"),
                      ("Spoofing", "Spoofing")):
        c = pcl[pcl.cls == cls]
        if c.seed.nunique() != 5:
            raise SystemExit(f"B4: per-class {cls} incomplete")
        add(f"JNNrGbdtRecall{code}", pct(c.recall.mean()), rel(_pb("B4gbdt") / "per_class.csv"), "five-seed mean",
            f"A3: GBDT recall %, {cls}")
        add(f"JNNrSupport{code}", _int(c.support_test.mean()), rel(_pb("B4gbdt") / "per_class.csv"), "five-seed mean",
            f"A3: test support, {cls}")
    # A4: TON-IoT host-disjoint split, feasibility only
    s = _rj(_pb("B4") / "summary.json")["ton_host_disjoint"]
    h = pd.read_csv(_pb("B4") / "ton_host_split.csv")
    ssrc = rel(_pb("B4") / "summary.json")
    hosts = {k: v for k, v in s["hosts_per_class"].items() if k != "normal"}
    top = {k: v for k, v in s["top_host_share_per_class"].items() if k != "normal"}
    absent = pd.DataFrame(s["classes_absent_from_test"]).groupby("seed").size().reindex(list(SEEDS), fill_value=0)
    share = h.groupby("seed").test_rows_pct.first()
    add("JNTonHosts", str(s["n_hosts"]), ssrc, "TON-IoT", "A4: source hosts")
    add("JNTonHostsPerAttackMin", str(min(hosts.values())), ssrc, "min over attack classes", "A4: hosts carrying an attack class")
    add("JNTonHostsPerAttackMax", str(max(hosts.values())), ssrc, "max over attack classes", "")
    add("JNTonTopHostMin", pct(100 * min(top.values()), 0), ssrc, "min over attack classes", "A4: share of a class on its largest host, %")
    add("JNTonTopHostMax", pct(100 * max(top.values()), 0), ssrc, "max over attack classes", "")
    add("JNTonAbsentMin", str(int(absent.min())), ssrc, "min over seeds", "A4: classes with no test rows")
    add("JNTonAbsentMax", str(int(absent.max())), ssrc, "max over seeds", "")
    add("JNTonTestShareMin", pct(share.min(), 0), rel(_pb("B4") / "ton_host_split.csv"), "min over seeds", "A4: test share of rows, %")
    add("JNTonTestShareMax", pct(share.max(), 0), rel(_pb("B4") / "ton_host_split.csv"), "max over seeds", "")


def partB_b5():
    """B5: train-only vocabulary, cross-partition duplicates, Edge-IIoTset label conflicts."""
    d = _pb("B5")
    s, dup = _rj(d / "summary.json"), pd.read_csv(d / "duplicates.csv")
    ssrc, dsrc = rel(d / "summary.json"), rel(d / "duplicates.csv")
    add("JNVocabTrigger", f"{s['trigger_pct']:g}", ssrc, "registered rule", "B5: percent of test rows that triggers a rerun")
    add("JNVocabFired", str(len(s["vocabulary_trigger_fired_for"])), ssrc, "count", "B5: datasets triggering the vocabulary rerun")
    for ds in ("toniot", "edgeiiot"):
        add(f"JNVocabMax{DS[ds]}", f(s["max_affected_pct_by_dataset"][ds], 3), ssrc, "max over seeds",
            f"B5: percent of test rows with a category unseen in training, {ds}")
    for ds in DS:
        add(f"JNDup{DS[ds]}", pct(s["dup_pct_by_dataset"][ds]), ssrc, "pooled over seeds",
            f"B5: percent of test rows duplicated in training, {ds}")
    gp = [s["dup_pct_by_dataset"][ds] for ds in DS if ds not in IOT]
    add("JNDupGpMin", pct(min(gp)), ssrc, "min over GP datasets", "B5: duplicated test rows %, general purpose")
    add("JNDupGpMax", pct(max(gp)), ssrc, "max over GP datasets", "")
    cm = dup.groupby(["dataset", "cls"]).dup_pct.mean()
    for ds in IOT:
        top = cm[ds]
        add(f"JNDupClassMax{DS[ds]}", pct(top.max()), dsrc, "five-seed mean, max over classes",
            f"B5: most-duplicated class ({top.idxmax()}), % of its test rows, {ds}")
    dom = pd.DataFrame(s["dominated_classes"])
    add("JNDomRule", "50", "results/preregistration/2026-09-28_journal_partB.md (D7)", "registered rule",
        "B5: percent of a class's test rows duplicated in training that makes it dominated")
    add("JNDomIot", str(int(dom.dataset.isin(IOT).sum())), ssrc, "count", "B5: dominated IoT classes")
    add("JNDomGroups", str(dom.groupby(["dataset", "cls"]).ngroups), ssrc, "count", "B5: dominated dataset-class pairs")
    for (ds, cls), gg in dom.groupby(["dataset", "cls"]):
        a = dup[(dup.dataset == ds) & (dup.cls == cls)].dup_pct
        code = f"{DS[ds]}{CLS[cls]}"
        add(f"JNDomShare{code}", pct(a.mean()), dsrc, "five-seed mean", f"B5: duplicated share of {ds} {cls} test rows, %")
        add(f"JNDomShare{code}Min", pct(a.min()), dsrc, "min over seeds", "")
        add(f"JNDomShare{code}Max", pct(a.max()), dsrc, "max over seeds", "")
        add(f"JNDomSeeds{code}", str(len(gg)), ssrc, "count of five seeds", f"B5: seeds on which {ds} {cls} is dominated")
    e = dup[dup.dataset == "edgeiiot"].groupby("seed").agg(c=("n_label_conflict", "sum"), n=("n_test", "sum"))
    add("JNEdgeConflictRows", _int(e.c.mean()), dsrc, "five-seed mean",
        "B5: Edge-IIoTset test rows with an identical training vector of another label")
    add("JNEdgeConflictPct", pct(100 * (e.c / e.n).mean()), dsrc, "five-seed mean", "same, % of test rows")


def partB_a6():
    """A6: per-class F1 of the dominated classes, with and without the duplicated rows."""
    d = _pb("A6")
    pc, gd = pd.read_csv(d / "per_class.csv"), pd.read_csv(d / "guards.csv")
    if not gd.ok.all():
        raise SystemExit("A6: a stored model did not reproduce its score")
    code = {"Tree-to-Spike": "Tts", "DNN-Tree": "Dnn", "Vanilla SNN": "Van", "TASNN": "Tasnn"}
    src = rel(d / "per_class.csv")
    for (ds, cls, model), g in pc.groupby(["dataset", "cls", "model"]):
        if g.seed.nunique() != 5 or set(g.subset) != {"full", "without_duplicates"}:
            raise SystemExit(f"A6: {ds} {cls} {model} incomplete")
        nm = f"JNDom{DS[ds]}{CLS[cls]}{code[model]}"
        add(nm + "Full", f(g[g.subset == "full"].f1.mean()), src, "five-seed mean", f"A6: class F1, all test rows, {model}, {ds} {cls}")
        add(nm + "NoDup", f(g[g.subset == "without_duplicates"].f1.mean()), src, "five-seed mean",
            f"A6: class F1 without the rows duplicated in training, {model}, {ds} {cls}")


def partB_b6():
    """B6: convergence trigger, the CICIoT2023 block (A1) against the fixed-budget runs it
    replaces (appendix: "original fixed-budget protocol"), and the stalled-run diagnosis."""
    import make_results_table as MRT
    d = _pb("B6")
    s, tr = _rj(d / "summary.json"), pd.read_csv(d / "trigger.csv")
    ssrc = rel(d / "summary.json")
    if len(s["fired"]) != 1 or (s["fired"][0]["arm"], s["fired"][0]["dataset"]) != ("dnn", "ciciot2023"):
        raise SystemExit("B6: the trigger outcome differs from the one these macros describe")
    add("JNConvThreshold", f(s["threshold_points"], 1), ssrc, "registered rule", "B6: epoch-8-to-10 gain that fires the rerun")
    add("JNConvGainDnnCicIoT", f(s["fired"][0]["mean_gain"]), ssrc, "mean over five seeds", "B6: DNN-Tree, CICIoT2023, val. macro-F1 gain")
    prim = tr[~tr.metric.str.contains("cross-check") & ~tr.fires]
    nx = prim.loc[prim.mean_gain.idxmax()]
    add("JNConvGainNextMax", f(nx.mean_gain), rel(d / "trigger.csv"), "mean over seeds",
        f"B6: largest gain that did not fire ({nx.arm}, {nx.dataset}, {nx.metric})")
    add("JNConvExcludedStalled", str(len(s["excluded_stalled"])), ssrc, "count", "B6: stalled entropy runs excluded")
    # the block and the runs it replaces
    ds = "ciciot2023"
    blk, fix = {}, {}
    for arm, (_, patt) in BLOCK_ARMS.items():
        blk[arm] = _arm(patt, [ds])
        # the replaced m8_raw runs record the single-thread pool's teacher; the arm does not use it
        fix[arm] = _arm(patt, [ds], root=FIXED, teacher=arm != "raw")
        for r in blk[arm].values():
            if not r["training"]["selection"]["enabled"] or len(r["training"]["training_curve"]) != 20:
                raise SystemExit(f"B6: {arm} block record is not a 20-epoch selected run")
        for r in fix[arm].values():
            if (r["training"].get("selection") or {}).get("enabled") or len(r["training"]["training_curve"]) != 10:
                raise SystemExit(f"B6: {arm} fixed-budget record is not a 10-epoch run")
    bsrc = "results/journal_partB/B6/block (= the CICIoT2023 production records)"
    eps = [r["training"]["selection"]["selected_epoch"] for a in blk for r in blk[a].values()]
    add("JNBlockRuns", str(len(eps)), bsrc, "count", "A1: block runs")
    add("JNBlockArms", str(len(blk)), bsrc, "count", "A1: block arms")
    add("JNBlockSelEpochMedian", f(np.median(eps), 0), bsrc, "median over runs", "A1: selected epoch")
    add("JNBlockSelEpochMin", str(min(eps)), bsrc, "min over runs", "A1: selected epoch")
    add("JNBlockSelEpochMax", str(max(eps)), bsrc, "max over runs", "A1: selected epoch")
    add("JNBlockSelEpochAboveTen", str(sum(e > 10 for e in eps)), bsrc, "count of runs", "A1: runs selecting an epoch after 10")
    add("JNBlockSelEpochTwenty", str(sum(e == 20 for e in eps)), bsrc, "count of runs", "A1: runs selecting the last epoch")
    ten = [(a, sd) for a in blk for (_, sd), r in blk[a].items() if r["training"]["selection"]["selected_epoch"] == 10]
    same = sum(_f1(blk[a][(ds, sd)]) == _f1(fix[a][(ds, sd)]) for a, sd in ten)
    add("JNBlockEpochTenRuns", str(len(ten)), bsrc, "count of runs", "A1: runs selecting epoch 10")
    add("JNBlockEpochTenIdentical", str(same), f"{bsrc}; {FIXED}", "count of runs",
        "A1: of those, runs whose test macro-F1 equals the fixed-budget run's exactly")
    for arm, (code, patt) in BLOCK_ARMS.items():
        b = np.array([_f1(blk[arm][(ds, sd)]) for sd in SEEDS])
        x = np.array([_f1(fix[arm][(ds, sd)]) for sd in SEEDS])
        fsrc = f"{FIXED}/{patt.format(d=ds, s='*')}"
        add(f"JNBlockGain{code}", sgn(b.mean() - x.mean()), f"{bsrc}; {fsrc}", "five-seed means",
            f"A1: 20-epoch selected minus fixed-budget macro-F1, {arm}, CICIoT2023")
        add(f"JNFixedFone{code}CicIoT", f(x.mean()), fsrc, "five-seed mean, original fixed-budget protocol",
            f"macro-F1, {arm}, CICIoT2023")
        add(f"JNFixedFone{code}CicIoTSd", f(x.std(ddof=1)), fsrc, "five-seed sample SD", "")
        if arm not in ("dnn", "dnn_tq", "vanilla", "tasnn", "raw"):
            add(f"JNFixedKsops{code}CicIoT", f(np.mean([_ksops(r) for r in fix[arm].values()]), 1), fsrc,
                "five-seed mean, full window, input-inclusive", f"SOPs per sample (thousands), {arm}, fixed budget")
    mk = {"vanilla": "vanilla", "tree": "tree", "dnn": "dnn", "tasnn": "tasnn"}
    for arm in mk:
        c = MRT.cell(arm, [fix[arm][(ds, sd)] for sd in SEEDS])
        code = BLOCK_ARMS[arm][0]
        for m, mc in (("acc", "Acc"), ("bacc", "Bacc"), ("mcc", "Mcc"), ("dr", "Dr"), ("far", "Far")):
            add(f"JNFixed{mc}{code}CicIoT", f(100 * c[m]), f"{FIXED}/{BLOCK_ARMS[arm][1].format(d=ds, s='*')}",
                "five-seed mean, original fixed-budget protocol", f"{m} x100, {arm}, CICIoT2023")
            add(f"JNFixed{mc}{code}CicIoTSd", f(100 * c[m + "_sd"]), f"{FIXED}/{BLOCK_ARMS[arm][1].format(d=ds, s='*')}",
                "five-seed sample SD", "")
    for model, mc in (("Tree-to-Spike", "Tts"), ("DNN-Tree", "Dnn")):
        pcs = [_rj(REPO / f"{FIXED}/diagnostics/perchannel/{model}_{ds}_seed{sd}.json") for sd in SEEDS]
        for k, cc in (("fp32_f1", "Fp"), ("per_tensor_f1", "PerTensor"), ("per_channel_f1", "PerChannel")):
            add(f"JNFixedPc{mc}{cc}CicIoT", f(np.mean([p[k] for p in pcs])), f"{FIXED}/diagnostics/perchannel",
                "five-seed mean, M3 reruns, original fixed-budget protocol", f"{model} macro-F1, {cc}, CICIoT2023")
    fq = fix["dnn_tq"]
    for cc, g in (("Fp", lambda r: r["test_plus"]["fp32_before_projection"]["metrics"]["f1_macro"]),
                  ("PerChannel", lambda r: r["test_plus"]["fp32_before_projection"]["per_channel_int8"]["metrics"]["f1_macro"])):
        add(f"JNFixedTq{cc}CicIoT", f(100 * np.mean([g(r) for r in fq.values()])), f"{FIXED}/DNN-TQ/{ds}",
            "five-seed mean, original fixed-budget protocol", f"DNN-TQ macro-F1, {cc}, CICIoT2023 (per-tensor: JNFixedFoneTqCicIoT)")
    # stalled entropy runs (10-epoch protocol): reruns with an observer
    obs = {k: _rj(PB / f"B6/stalled/{k}/observer.json") for k in ("ciciot2023_seed44", "ciciot2023_seed45", "ciciomt2024_seed46")}
    osrc = "results/journal_partB/B6/stalled/*/observer.json"
    add("JNStallReruns", str(len(obs)), osrc, "count", "B6: stalled runs rerun with the observer")
    add("JNStallReproduced", str(sum(bool(o["reproduces_stored"] and o["curve_identical"] and o["test_identical"])
                                     for o in obs.values())), osrc, "count", "B6: reruns reproducing curve and test score exactly")
    for lay, key in (("One", "firing_rate_l1"), ("Two", "firing_rate_l2")):
        v = [e[key] for o in obs.values() for e in o["epochs"]]
        add(f"JNStallFire{lay}Min", pct(100 * min(v)), osrc, "min over runs and epochs", f"B6: layer-{lay.lower()} firing rate, %")
        add(f"JNStallFire{lay}Max", pct(100 * max(v)), osrc, "max over runs and epochs", "")
    for dsx in ("ciciot2023", "ciciomt2024"):
        os_ = [o for k, o in obs.items() if k.startswith(dsx)]
        for cls, cc in (("DDoS", "Ddos"), ("DoS", "Dos")):
            for key, kc in (("pred_dist", "Pred"), ("true_dist", "True")):
                v = [100 * e[key][o["class_names"].index(cls)] / sum(e[key]) for o in os_ for e in o["epochs"]]
                add(f"JNStall{kc}{cc}{DS[dsx]}", pct(np.mean(v), 0), osrc, "mean over all epochs of the stalled runs",
                    f"B6: {'predicted' if kc == 'Pred' else 'true'} {cls} share of validation flows, %, {dsx}")
        va = [e["val_accuracy"] for o in os_ for e in o["epochs"]]
        add(f"JNStallValAcc{DS[dsx]}Min", f(min(va)), osrc, "min over runs and epochs", f"B6: validation accuracy, {dsx}")
        add(f"JNStallValAcc{DS[dsx]}Max", f(max(va)), osrc, "max over runs and epochs", "")


def partB_b8():
    """B8: peak resident memory over the latency workload (VmHWM, seed 42, gated run)."""
    d = _pb("B8")
    m = pd.read_csv(d / "peak_memory.csv")
    names = {"t2s_exit": "Tts", "t2s_full": "TtsFull", "dnn_tree": "Dnn", "vanilla_snn": "Van", "tasnn": "Tasnn",
             "compact_gbdt": "GbdtCompact", "full_gbdt": "GbdtFull", "ple_mlp": "Ple"}
    if set(m.system) != set(names) or set(m.dataset) != set(IOT) or len(m) != len(names) * len(IOT) or (m.seed != 42).any():
        raise SystemExit("B8: peak_memory.csv is not one seed-42 row per IoT dataset and system")
    src = rel(d / "peak_memory.csv")
    for s, c in names.items():
        g = m[m.system == s].set_index("dataset")
        add(f"JNMemMb{c}", f(g.peak_minus_baseline_kb.mean() / 1024, 1), src, "single seed-42 run, IoT mean",
            f"B8: peak RSS above the common-import baseline, MB, {s}")
        for ds in IOT:
            add(f"JNMemMb{c}{DS[ds]}", f(g.loc[ds, "peak_minus_baseline_kb"] / 1024, 1), src, "single seed-42 run",
                f"B8: peak RSS above baseline, MB, {s}, {ds}")
    add("JNMemBaselineMb", f(m.baseline_rss_kb.mean() / 1024, 0), src, "mean over the 32 processes",
        "B8: resident set after the common imports (numpy, pandas, torch, xgboost, models), MB")


def main():
    for fn in (quality, precision, statistics, energy, deployment, others, tsweep, compact, perchannel,
               results_section, results_extras, general_purpose, pooled, e1_e2,
               partB_b1, partB_b2, partB_b3, partB_b4, partB_b5, partB_a6, partB_b6, partB_b8):
        fn()
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    # The generator's own two outputs are excluded: once tracked, the previous
    # generation is always "modified" at this point and would mark every run dirty.
    # conference/ is excluded too: a separate paper with its own numbers pipeline,
    # nothing here reads it, and its work in progress says nothing about these values.
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", ".",
                            ":!results/tables/numbers.tex",
                            ":!results/tables/numbers_provenance.json", ":!conference"], cwd=REPO).returncode != 0
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = [
        "% " + "=" * 70,
        "% numbers.tex -- GENERATED, DO NOT EDIT BY HAND.",
        "%   python scripts/make_numbers.py",
        f"% generated {ts} at git {sha}{' (+ uncommitted changes)' if dirty else ''}",
        "% Every result the manuscript quotes is a macro here, read from a committed",
        "% artifact. Provenance per macro: results/tables/numbers_provenance.json.",
        "% If a number is not in this file, no artifact produced it.",
        "% " + "=" * 70]
    body = [f"\\newcommand{{\\{n}}}{{{v}}}%".ljust(64) + f" % {mng or agg}" for n, v, _, agg, mng in MACROS]
    OUT.write_text("\n".join(head + body) + "\n")
    PROV.write_text(json.dumps({"generated": ts, "git": sha, "dirty": dirty,
                                "macros": {n: dict(value=v, source=s, aggregation=a, meaning=m)
                                           for n, v, s, a, m in MACROS}}, indent=1))
    print(f"[wrote] {OUT.relative_to(REPO)} ({len(MACROS)} macros) and {PROV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
