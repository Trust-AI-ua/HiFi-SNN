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


def main():
    for fn in (quality, precision, statistics, energy, deployment, others, perchannel):
        fn()
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO).returncode != 0
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
