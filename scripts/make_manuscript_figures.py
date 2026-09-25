#!/usr/bin/env python3
"""Figures and appendix tables for manuscript_tnnls.tex -> Figures/.

Every value comes from results/tables/numbers.tex or from the run records and
artifacts at tag tnnls-submission (2adfc77). Before any new (general-purpose) value is
written, the same code path must reproduce the published IoT macros exactly; a
mismatch stops the script (part-3 rule: a new measurement path reproduces the old one
on unchanged cells first). Guards are logged to Figures/README.md.

  Main text   pareto_iot.pdf, exit_anatomy_ciciot.pdf, tree-to-spike.pdf
  Appendix    cd_diagram.pdf, memory_band.pdf, margin_sweep.pdf,
              tab_results_general.tex, tab_encoder_ablation_general.tex,
              tab_envelope.tex, tab_exit_general.tex
  Data        Figures/data/*.csv, the plotted values

    python scripts/make_manuscript_figures.py [--preview DIR]   # DIR: PNG previews
"""
from __future__ import annotations

import argparse
import glob
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
OUT = REPO / "Figures"
DATA = OUT / "data"
TAG = "tnnls-submission"
SEEDS = range(42, 47)

IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
GP = ["kddcup99", "nslkdd", "ctu13", "iscxids2012", "cicids2017", "cse-cic-ids2018", "unswnb15"]
NAME = {"ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024", "edgeiiot": "Edge-IIoTset",
        "toniot": "TON-IoT", "kddcup99": "KDDCup99", "nslkdd": "NSL-KDD", "ctu13": "CTU-13",
        "iscxids2012": "ISCX-IDS2012", "cicids2017": "CIC-IDS2017", "cse-cic-ids2018": "CSE-CIC-IDS2018",
        "unswnb15": "UNSW-NB15"}
DS = {"ciciot2023": "CicIoT", "ciciomt2024": "CicIoMT", "edgeiiot": "EdgeIIoT", "toniot": "TonIoT"}

# models: key in make_results_table, macro code, deployment system, label
MODELS = [("tree", "Tts", "t2s_deployed", "Tree-to-Spike"), ("dnn", "Dnn", "dnn_tree", "DNN-Tree"),
          ("vanilla", "Van", "vanilla_snn", "Vanilla SNN"), ("tasnn", "Tasnn", "tasnn", "TASNN")]
# validated categorical palette (all-pairs, light): blue, orange, aqua, violet; shapes are the
# secondary encoding so identity never rests on color alone (grayscale print, CVD)
COLOR = {"Tts": "#2a78d6", "Dnn": "#eb6834", "Van": "#1baf7a", "Tasnn": "#4a3aa7"}
MARKER = {"Tts": "o", "Dnn": "s", "Van": "^", "Tasnn": "D"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COL_W, DBL_W = 3.5, 7.16          # IEEE column / page width, inches

GUARDS: list[str] = []
PREVIEW: Path | None = None


# ------------------------------------------------------------------ provenance
def git(*a):
    return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()


def check_tag():
    tag = git("rev-parse", "--short", f"{TAG}^{{commit}}")
    for p in ("results/tables/numbers.tex", "results", "scripts/make_results_table.py",
              "scripts/deployment_energy2.py", "scripts/make_dep_systems_table.py"):
        if subprocess.run(["git", "diff", "--quiet", TAG, "--", p], cwd=REPO).returncode != 0:
            sys.exit(f"{p} differs from {TAG}; values must come from the tag")
    GUARDS.append(f"sources unchanged from tag {TAG} ({tag}): numbers.tex, results/, energy and table code")
    return tag


def macros():
    out = {}
    for line in (REPO / "results/tables/numbers.tex").read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{", line)
        if not m:
            continue
        i, depth = m.end(), 1
        for j in range(i, len(line)):
            depth += {"{": 1, "}": -1}.get(line[j], 0)
            if depth == 0:
                out[m.group(1)] = line[i:j]
                break
    return out


M = {}


def num(name):
    v = M[name].replace(r"\ensuremath", "").replace("{,}", "")
    return float(v.strip("{}").replace("+", ""))


def f(v, d=2):
    return f"{v:.{d}f}"


def guard(label, pairs):
    """pairs: (macro name, recomputed formatted string); every one must match exactly."""
    bad = [(n, M.get(n), s) for n, s in pairs if M.get(n) != s]
    if bad:
        sys.exit(f"GUARD FAILED ({label}): " + "; ".join(f"{n}: macro {a!r} vs recomputed {b!r}" for n, a, b in bad[:6]))
    GUARDS.append(f"{label}: {len(pairs)} published macros reproduced exactly")


# ------------------------------------------------------------------ plotting setup
def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": BASE, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    return plt


def save(fig, name):
    fig.savefig(OUT / name)
    if PREVIEW:
        fig.savefig(PREVIEW / name.replace(".pdf", ".png"), dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[wrote] Figures/{name}")


def records(patt):
    return [json.loads(Path(x).read_text()) for x in sorted(glob.glob(str(REPO / patt)))
            if int(Path(x).parent.name.split("_")[1]) in SEEDS]


# ------------------------------------------------------------------ 1. Pareto (main text)
def energy_memory_seed42():
    """Memory-inclusive energy per sample, reuse-aware, SRAM 20 pJ (DRAM 1300), deployed,
    seed 42, IoT, from deployment_measurements.csv through deployment_energy2.row_energy."""
    import deployment_energy2 as E2
    from make_dep_systems_table import deployed
    d = deployed(pd.read_csv(REPO / "results/deployment/deployment_measurements.csv", comment="#"))
    d = d[d.dataset.isin(IOT)]
    rows = []
    for _, r in d.iterrows():
        a, m = E2.row_energy(r, 20, 1300, True)
        rows.append(dict(dataset=r.dataset, system=r.system, nJ=(a + m) / 1000))
    return pd.DataFrame(rows)


def pareto():
    plt = plt_setup()
    mem = energy_memory_seed42()
    rows = []
    for key, mc, sysname, lab in MODELS:
        for ds in IOT:
            e = mem[(mem.dataset == ds) & (mem.system == sysname)].nJ
            rows.append(dict(model=lab, code=mc, dataset=ds, f1=num(f"JNFone{mc}{DS[ds]}"),
                             arith_nJ=num(f"JNEnergy{mc}{DS[ds]}") / 1000,
                             mem_nJ=float(e.iloc[0]) if len(e) else np.nan))
    t = pd.DataFrame(rows)
    means = t.groupby(["model", "code"], sort=False)[["arith_nJ", "mem_nJ"]].mean().reset_index()
    means["f1"] = [num(f"JNFone{c}IotMean") for c in means.code]
    t.to_csv(DATA / "pareto_iot_points.csv", index=False)
    means.to_csv(DATA / "pareto_iot_means.csv", index=False)
    cells = pd.read_csv(REPO / "results/diagnostics/per_seed_all_cells.csv")
    mkey = {"Tts": "treespike", "Dnn": "dnn", "Van": "vanilla", "Tasnn": "tasnn"}
    iot_mean = lambda c: cells[cells.dataset.isin(IOT) & (cells.model == mkey[c])].groupby("dataset").f1.mean().mean()
    guard("pareto y = Table III macro-F1 (per dataset and IoT mean, unrounded source)",
          [(f"JNFone{c}IotMean", f(iot_mean(c))) for c in means.code]
          + [(f"JNFone{c}{DS[d]}", f(cells[(cells.dataset == d) & (cells.model == mkey[c])].f1.mean()))
             for c in means.code for d in IOT])

    def energy_label(v_nj):
        return f"{v_nj/1e6:.1f} mJ" if v_nj >= 1e6 else f"{v_nj/1e3:.0f} µJ"

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 4.4), sharey=True)
    panels = [("arith_nJ", "Arithmetic energy per sample (nJ), 32-bit, full window\n(Table VII(a); five-seed mean)", (2, 100),
               {"Tts": (-4, 7), "Dnn": (7, -3), "Van": (0, 9)}),
              ("mem_nJ", "Memory-inclusive energy per sample (nJ), reuse-aware,\n20 pJ SRAM, deployed (seed 42)", (50, 400),
               {"Tts": (-6, 8), "Dnn": (8, -9), "Van": (-8, 6)})]
    for ax, (col, xl, xlim, offs), tag in zip(axes, panels, "ab"):
        ax.set_xscale("log")
        for key, mc, sysname, lab in MODELS:
            g, mm = t[t.code == mc], means[means.code == mc].iloc[0]
            off = g[col] > xlim[1]
            xm = min(mm[col], xlim[1] * 0.93)
            ax.scatter([xm], [mm.f1], s=62, marker=MARKER[mc], color=COLOR[mc], edgecolors="white",
                       linewidths=1.0, zorder=3, clip_on=False, label=lab)
            ax.scatter(g[col].where(~off, xlim[1] * 0.93), g.f1, s=15, marker=MARKER[mc], color=COLOR[mc],
                       alpha=0.6, linewidths=0, zorder=4, clip_on=False)
            if off.any():
                ax.annotate(f"{lab} off scale:\n{energy_label(g[col].min())}–{energy_label(g[col].max())}",
                            xy=(xlim[1] * 0.99, mm.f1), xytext=(xlim[1] ** 0.62 * xlim[0] ** 0.38, 61.5),
                            fontsize=6.5, color=INK2, ha="center", va="center",
                            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7, connectionstyle="arc3,rad=0.25"))
            elif mc in offs:
                dx, dy = offs[mc]
                ax.annotate(lab, xy=(xm, mm.f1), xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                            color=INK, ha="right" if dx < 0 else ("left" if dx > 0 else "center"), va="center")
        ax.set_xlim(*xlim)
        ax.set_xlabel(xl, fontsize=7.5)
        ax.set_ylabel("Macro-F1 (%), five-seed mean")
        ax.text(0.01, 0.98, f"({tag})", transform=ax.transAxes, va="top", fontsize=8, color=INK)
    from matplotlib.ticker import FixedLocator, NullFormatter
    for ax, ticks in ((axes[0], [2, 5, 10, 20, 50, 100]), (axes[1], [50, 100, 200, 400])):
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([str(v) for v in ticks])
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
                   handletextpad=0.2, columnspacing=0.9)
    fig.text(0.99, 0.005, "small: one IoT dataset; large: IoT mean", ha="right", fontsize=6.5, color=MUTED)
    fig.tight_layout(h_pad=1.0)
    save(fig, "pareto_iot.pdf")


# ------------------------------------------------------------------ prefix helpers
def prefix(ds, seed, split="test"):
    z = np.load(REPO / f"results/prefix_outputs/{ds}/treespike/seed_{seed}/{split}_prefix.npz", allow_pickle=True)
    L = z["prefix_logits"].astype(np.float32)
    L = L - L.max(-1, keepdims=True)
    q = np.exp(L)
    q /= q.sum(-1, keepdims=True)
    return q, z["labels"], list(z["class_names"])


def margin_log_check(q, y):
    """The prefix outputs must reproduce scripts/exit_margin_test.py's logged numbers."""
    log = (REPO / "results/repaired_runs/queue_exit_margin2.log").read_text()
    pred = q.argmax(-1)
    s = np.sort(q, -1)
    c1, c2 = s[..., -1], s[..., -2]
    T, N = pred.shape[1], len(y)
    full_acc = (pred[:, -1] == y).mean() * 100
    fired = c1 >= 0.90
    ex = fired.any(1)
    t_star = np.where(ex, fired.argmax(1), T - 1)
    wrong = ex & (pred[np.arange(N), t_star] != y)
    cl = pred == y[:, None]
    has = cl[wrong].any(1)
    firstc = np.where(has, cl[wrong].argmax(1), -1)
    after = has & (firstc > t_star[wrong])
    ts, tp = t_star[wrong][after].mean() + 1, firstc[after].mean() + 1
    want = dict(acc=float(re.search(r"full acc ([\d.]+)", log).group(1)),
                ts=float(re.search(r"mean exit step t\*\s*: ([\d.]+)", log).group(1)),
                tp=float(re.search(r"mean recovery step t'\s*: ([\d.]+)", log).group(1)))
    got = dict(acc=round(full_acc, 2), ts=round(ts, 2), tp=round(tp, 2))
    if got != want:
        sys.exit(f"GUARD FAILED (prefix vs exit_margin_test log): {got} vs {want}")
    for m_, acc_log in re.findall(r"^\s+(0\.\d\d)\s+[\d.]+%\s+[\d.]+\s+([\d.]+)", log, re.M):
        f2 = (c1 - c2) >= float(m_)
        e2 = f2.any(1)
        t2 = np.where(e2, f2.argmax(1), T - 1)
        acc = (pred[np.arange(N), t2] == y).mean() * 100
        if f"{acc:.2f}" != acc_log:
            sys.exit(f"GUARD FAILED (margin gate m={m_}): {acc:.2f} vs log {acc_log}")
    GUARDS.append("CICIoT2023 seed 42 prefix outputs reproduce exit_margin_test log "
                  f"(full acc {want['acc']}, t* {want['ts']}, t' {want['tp']}, 7 margin-gate accuracies)")
    return ts, tp, wrong, t_star


# ------------------------------------------------------------------ 2. exit anatomy (main text)
def exit_anatomy():
    plt = plt_setup()
    q, y, names = prefix("ciciot2023", 42)
    ts, tp_rec, wrong, t_star = margin_log_check(q, y)
    pred = q.argmax(-1)
    T = pred.shape[1]
    steps = np.arange(1, T + 1)
    c1 = q.max(-1)
    fired = c1 >= 0.90
    ex = fired.any(1)
    first = np.where(ex, fired.argmax(1), -1)
    cum_exit = np.array([100 * ((first >= 0) & (first <= t)).mean() for t in range(T)])
    ddos = names.index("DDoS")
    dm = y == ddos
    ddos_ok = 100 * (pred[dm] == ddos).mean(0)
    ok = pred[:, -1] == y
    tprime = ((pred == y[:, None]) & ok[:, None]).argmax(1)[ok] + 1
    hist = np.array([100 * (tprime == s).mean() for s in steps])
    share_ddos_wrong = 100 * (y[wrong] == ddos).mean()
    pd.DataFrame(dict(step=steps, cum_exit_pct=cum_exit, ddos_running_correct_pct=ddos_ok,
                      tprime_hist_pct=hist)).to_csv(DATA / "exit_anatomy_ciciot_seed42.csv", index=False)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 3.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.35]))
    a1.plot(steps, cum_exit, color=COLOR["Tts"], lw=2, marker="o", ms=3)
    a1.set_ylabel("Exited by step (%)")
    a1.set_ylim(0, 103)
    a2.bar(steps, hist, width=0.72, color=BASE, edgecolor="none",
           label="t′ of full-window-correct samples (%)")
    a2.plot(steps, ddos_ok, color=COLOR["Tts"], lw=2, marker="o", ms=3,
            label="DDoS: running argmax correct (%)")
    a2.set_ylabel("Share of samples (%)")
    a2.set_xlabel("Timestep")
    a2.set_ylim(0, 103)
    a2.set_xticks([1, 5, 10, 15, 20])
    for ax in (a1, a2):
        ax.axvline(ts, color=INK, lw=0.9, ls="--")
        ax.axvline(tp_rec, color=INK, lw=0.9, ls=":")
    a1.text(ts + 0.25, 8, f"mean wrong-exit\nstep {ts:.2f}", fontsize=6.5, color=INK)
    a1.text(tp_rec + 0.25, 8, f"mean recovery\nstep t′ {tp_rec:.2f}", fontsize=6.5, color=INK)
    a2.text(20.3, 88, "line: DDoS samples whose\nrunning argmax is correct", ha="right", va="top",
            fontsize=6.5, color=INK)
    a2.text(20.3, 24, "bars: t′ of all full-window-\ncorrect samples (share)", ha="right", va="top",
            fontsize=6.5, color=INK2)
    a1.text(0.01, 1.02, r"CICIoT2023, seed 42, $\theta_{exit}=0.90$", transform=a1.transAxes, fontsize=7, color=INK2)
    fig.tight_layout(h_pad=0.6)
    save(fig, "exit_anatomy_ciciot.pdf")
    GUARDS.append(f"exit anatomy: DDoS is {share_ddos_wrong:.1f}% of CICIoT2023 seed-42 wrong exits at 0.90; "
                  f"median t' of full-window-correct samples {np.median(tprime):.0f}")
    return share_ddos_wrong


# ------------------------------------------------------------------ 3. schematic (main text, redrawn)
def schematic():
    plt = plt_setup()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(DBL_W, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(-4.5, 40)
    ax.axis("off")
    TEACH, SPIKE, IO = "#fde9df", "#e3eefb", "#f0efec"

    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc=fc, ec=BASE, lw=0.8))
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=7.5, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", fontsize=6.3, color=INK2, linespacing=1.15)

    STY = {"data": "-", "train": (0, (4, 2)), "params": (0, (1, 1.6))}

    def arrow(p, q, style, rad=0.0):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK, linestyle=STY[style],
                                                        shrinkA=1, shrinkB=1, mutation_scale=8,
                                                        connectionstyle=f"arc3,rad={rad}"))

    def label(x, y, t):
        ax.text(x, y, t, fontsize=6.2, color=INK2, ha="center", va="center", linespacing=1.1)

    h = 10
    y1, y2 = 22.0, 2.0
    ax.text(0.5, 35.5, "Training (teacher present)", fontsize=7.5, color=INK, weight="bold")
    ax.text(0.5, 14.3, "Deployment (teacher removed)", fontsize=7.5, color=INK, weight="bold")
    # training row
    box(1, y1, 13, h, "Flow features", "tabular, per flow", IO)
    box(18, y1, 16, h, "XGBoost teacher", "cost-weighted\n200 rounds, depth 6", TEACH)
    box(45, y1, 18, h, "Tree-guided encoder", "split thresholds → bins\nTTFS, T = 20, M ≤ 3", SPIKE)
    box(69, y1, 15, h, "LIF student", "C-64-64-K\nsurrogate gradients", SPIKE)
    arrow((14.4, y1 + 5), (17.6, y1 + 5), "data")
    arrow((34.4, y1 + 5), (44.6, y1 + 5), "params")
    label(39.5, y1 + 7.6, "thresholds,\nimportances")
    arrow((63.4, y1 + 5), (68.6, y1 + 5), "data")
    label(66.0, y1 + 6.6, "spikes")
    arrow((26, y1 + h + 0.4), (76.5, y1 + h + 0.4), "train", rad=-0.13)
    label(51.25, y1 + h + 5.9, "soft targets: KD + ranking (with CE, sparsity, FP terms)")
    # banner under the teacher: what survives into deployment
    ax.add_patch(FancyBboxPatch((11.5, 17.4), 41.5, 2.2, boxstyle="round,pad=0.15,rounding_size=0.8",
                                fc="white", ec=BASE, lw=0.6))
    ax.text(32.25, 18.5, "boosting structure discarded; split thresholds kept in the encoder",
            ha="center", va="center", fontsize=6.1, color=INK, style="italic")
    arrow((54, y1 - 0.3), (54, y2 + h + 0.4), "params")
    arrow((76.5, y1 - 0.3), (76.5, y2 + h + 0.4), "params")
    label(79.6, 17.0, "weights")
    # deployment row
    box(1, y2, 13, h, "Live flow", "features", IO)
    box(45, y2, 18, h, "Encoder (fixed)", "threshold lookup\nno trees evaluated", SPIKE)
    box(69, y2, 15, h, "LIF student", "int8-grid weights", SPIKE)
    box(86.5, y2, 13, h, "Verdict", "exit at $\\theta_{exit}$\n(validation-selected)\nflag if below $\\theta_{unc}$", IO)
    arrow((14.4, y2 + 5), (44.6, y2 + 5), "data")
    arrow((63.4, y2 + 5), (68.6, y2 + 5), "data")
    label(66.0, y2 + 6.6, "spikes")
    arrow((84.4, y2 + 5), (86.1, y2 + 5), "data")
    # legend: one row along the bottom
    for x0, (style, lab) in zip((14, 42, 70), (("data", "inference data path"), ("train", "training-only signal"),
                                               ("params", "fixed parameters carried over"))):
        arrow((x0, -3.0), (x0 + 4.5, -3.0), style)
        ax.text(x0 + 5.3, -3.0, lab, fontsize=6.4, color=INK2, va="center")
    save(fig, "tree-to-spike.pdf")


# ------------------------------------------------------------------ 4. CD diagram (appendix)
def cd_diagram():
    plt = plt_setup()
    names = [("Dnn", "DNN-Tree"), ("Tts", "Tree-to-Spike"), ("Van", "Vanilla SNN"), ("Tasnn", "TASNN")]
    R = {c: num(f"JNPooledRank{c}") for c, _ in names}
    cd = num("JNPooledCD")
    order = sorted(names, key=lambda x: R[x[0]])
    rk = [R[c] for c, _ in order]
    cliques = []
    for i in range(len(rk)):
        j = max(k for k in range(i, len(rk)) if rk[k] - rk[i] < cd)
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    lab = {c: n for c, n in names}
    joined = [" - ".join(lab[order[k][0]] for k in range(a, b + 1)) for a, b in cliques]
    GUARDS.append(f"CD diagram: ranks and CD from macros; cliques (gap < CD {cd}): {joined}")
    fig = plt.figure(figsize=(COL_W, 1.35))
    ax = fig.add_axes([0.25, 0.02, 0.50, 0.96])
    ax.set_xlim(1, 4)
    ax.set_ylim(-1.05, 0.95)
    ax.axis("off")
    ax.plot([1, 4], [0, 0], color=INK, lw=0.8, clip_on=False)
    for v in np.arange(1, 4.01, 0.25):
        big = abs(v - round(v)) < 1e-9
        ax.plot([v, v], [0, 0.09 if big else 0.05], color=INK, lw=0.8, clip_on=False)
        if big:
            ax.text(v, 0.13, f"{int(v)}", ha="center", va="bottom", fontsize=7.5)
    ax.plot([1, 1 + cd], [0.62, 0.62], color=INK, lw=1.1, clip_on=False)
    for v in (1, 1 + cd):
        ax.plot([v, v], [0.57, 0.67], color=INK, lw=1.1, clip_on=False)
    ax.text(1 + cd / 2, 0.68, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=7)
    half = len(order) // 2
    rows = {}
    for k, (c, n) in enumerate(order):
        left = k < half
        row = k if left else len(order) - 1 - k           # best and worst ranks on the first row
        y = -0.55 - 0.38 * row
        edge = 0.93 if left else 4.07
        ax.plot([R[c], R[c], edge], [0, y, y], color=INK2, lw=0.8, clip_on=False)
        ax.text(edge + (-0.04 if left else 0.04), y, n, ha="right" if left else "left", va="center", fontsize=7)
        ax.text(R[c] + (-0.04 if left else 0.04), y + 0.04, f"{R[c]:.2f}", ha="right" if left else "left",
                va="bottom", fontsize=6, color=INK2)
    for i, (a, b) in enumerate(cliques):
        y = -0.14 - 0.14 * i
        ax.plot([rk[a] - 0.04, rk[b] + 0.04], [y, y], color=INK, lw=2.4, solid_capstyle="round")
    save(fig, "cd_diagram.pdf")
    return joined


# ------------------------------------------------------------------ 5. memory band (appendix)
def memory_band():
    plt = plt_setup()
    import deployment_energy2 as E2
    from make_dep_systems_table import deployed
    raw = pd.read_csv(REPO / "results/deployment/deployment_measurements.csv", comment="#")
    dep = deployed(raw)
    dep = dep[dep.dataset.isin(IOT)]
    fw = raw[raw.dataset.isin(IOT) & raw.system.isin(["dnn_tree", "vanilla_snn", "t2s_full"])].copy()
    fw.loc[fw.system == "t2s_full", "system"] = "t2s_deployed"
    ref = pd.read_csv(REPO / "results/deployment/energy_crossover_iot.csv")
    grid = np.unique(np.concatenate([np.geomspace(5, 100, 60), [5, 10, 20, 50, 100]]))
    rows = []
    for cfg, d in (("deployed", dep), ("full_window", fw)):
        for reuse in (False, True):
            for s in grid:
                t = E2.table(d, sram=s, reuse=reuse)
                rows.append(dict(config=cfg, model="reuse" if reuse else "cold", sram_pJ=s,
                                 dnn=t["dnn_tree"] / t["t2s_deployed"], van=t["vanilla_snn"] / t["t2s_deployed"]))
    b = pd.DataFrame(rows)
    chk = b[b.sram_pJ.isin([5, 10, 20, 50, 100])].merge(ref, on=["config", "model", "sram_pJ"])
    bad = chk[(chk.dnn.round(2) != chk.ratio) | (chk.van.round(2) != chk.vs_vanilla_snn)]
    if len(chk) != 20 or len(bad):
        sys.exit(f"GUARD FAILED (memory band vs energy_crossover_iot.csv): {len(chk)} matched, {len(bad)} differ")
    GUARDS.append("memory band: recomputed ratios equal energy_crossover_iot.csv at all 20 tabulated points")
    b.to_csv(DATA / "memory_band_iot.csv", index=False)
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    lines = [("dnn", "deployed", "cold", "DNN-Tree, cold, deployed"), ("dnn", "deployed", "reuse", "DNN-Tree, reuse, deployed"),
             ("van", "deployed", "reuse", "Vanilla SNN, reuse, deployed"),
             ("dnn", "full_window", "cold", "DNN-Tree, cold, full window"), ("dnn", "full_window", "reuse", "DNN-Tree, reuse, full window")]
    for col, cfg, mdl, lab in lines:
        g = b[(b.config == cfg) & (b.model == mdl)]
        code = "Dnn" if col == "dnn" else "Van"
        ax.plot(g.sram_pJ, g[col], color=COLOR[code], lw=1.6 if mdl == "reuse" else 1.1,
                ls="-" if cfg == "deployed" else (0, (4, 2)), label=lab,
                marker="o" if mdl == "reuse" else None, markevery=[0, len(g) - 1], ms=3)
    ax.axhline(1.0, color=INK, lw=0.8)
    ax.text(98, 1.02, "parity", fontsize=6.3, color=INK2, va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.set_xlabel("SRAM access energy (pJ)")
    ax.set_ylabel("Baseline / Tree-to-Spike energy\n(above 1: Tree-to-Spike cheaper)")
    ax.set_ylim(0, max(1.5, b[["dnn", "van"]].max().max() * 1.08))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=6.3,
              handlelength=2.6, columnspacing=1.0)
    ax.text(0.99, 0.97, "IoT means, seed 42", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=MUTED)
    save(fig, "memory_band.pdf")


# ------------------------------------------------------------------ 6. margin sweep (appendix)
def margin_sweep():
    plt = plt_setup()
    from sklearn.metrics import f1_score
    q, y, _ = prefix("ciciot2023", 42)
    pred = q.argmax(-1)
    s = np.sort(q, -1)
    c1, marg = s[..., -1], s[..., -1] - s[..., -2]
    T, N = pred.shape[1], len(y)
    F1 = lambda p: 100 * f1_score(y, p, average="macro", zero_division=0)
    full = F1(pred[:, -1])
    stored = 100 * json.loads((REPO / "results/Tree-to-Spike/ciciot2023/seed_42/results.json").read_text())["test_plus"]["metrics"]["f1_macro"]
    if abs(full - stored) > 1e-9:
        sys.exit(f"GUARD FAILED (prefix full-window macro-F1 {full} vs stored {stored})")
    GUARDS.append(f"margin sweep: prefix full-window macro-F1 equals the stored CICIoT2023 seed-42 run ({stored:.2f})")
    grid = np.round(np.concatenate([np.arange(0.10, 0.951, 0.05), [0.97, 0.99]]), 2)
    rows = []
    for th in grid:
        for gate, stat in (("max-posterior", c1), ("margin", marg)):
            fired = stat >= th
            ex = fired.any(1)
            t = np.where(ex, fired.argmax(1), T - 1)
            rows.append(dict(gate=gate, threshold=th, gap_macro_f1=F1(pred[np.arange(N), t]) - full,
                             exit_pct=100 * ex.mean(), mean_exit_step=t.mean() + 1))
    d = pd.DataFrame(rows)
    d.to_csv(DATA / "margin_sweep_ciciot_seed42.csv", index=False)
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for gate, color, ls, mk in (("max-posterior", INK, "-", "o"), ("margin", MUTED, (0, (4, 2)), "s")):
        g = d[d.gate == gate]
        ax.plot(g.threshold, g.gap_macro_f1, color=color, ls=ls, lw=1.5, marker=mk, ms=3,
                label=f"{gate} gate" if gate == "margin" else "max-posterior gate (deployed)")
    ax.axhline(-1.0, color=INK, lw=0.9, ls="-.")
    ax.text(0.99, -1.6, "one-point admission bound", fontsize=6.3, color=INK2, va="top", ha="right")
    ax.axhline(0, color=BASE, lw=0.6)
    ax.set_xlabel("Gate threshold ($\\theta$ or margin $m$)")
    ax.set_ylabel("Macro-F1 at exit − full window\n(points)")
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    ax.text(0.01, 1.02, "CICIoT2023, seed 42, test", transform=ax.transAxes, fontsize=7, color=INK2)
    save(fig, "margin_sweep.pdf")
    return d


# ------------------------------------------------------------------ LaTeX helpers
def tex_table(name, caption, label, colspec, header, body, star=False, notes=None, tag=""):
    env = "table*" if star else "table"
    lines = [f"% generated by scripts/make_manuscript_figures.py from tag {TAG} ({tag}); typed values, no macros",
             f"\\begin{{{env}}}[t]\\centering\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\setlength{\\tabcolsep}{4pt}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *header, "\\midrule",
             *body, "\\bottomrule", "\\end{tabular}"]
    if notes:
        lines += [f"\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\scriptsize {notes}}}"]
    lines += [f"\\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"[wrote] Figures/{name}.tex")


def thousands(v):
    return f"{v:,.0f}".replace(",", "{,}")


# ------------------------------------------------------------------ 7. results table, GP (appendix)
def results_general(tag):
    import make_results_table as MRT
    cells = pd.read_csv(REPO / "results/diagnostics/per_seed_all_cells.csv")
    patt = {k: p for k, _, p in MRT.MODELS}
    mkey = {"tree": "treespike", "dnn": "dnn", "vanilla": "vanilla", "tasnn": "tasnn"}

    def row(key, ds):
        recs = records(patt[key].format(d=ds))
        if len(recs) != 5:
            sys.exit(f"{key}/{ds}: {len(recs)} seeds")
        c = MRT.cell(key, recs)
        g = cells[(cells.dataset == ds) & (cells.model == mkey[key])].f1
        if len(g) != 5:
            sys.exit(f"per_seed_all_cells {key}/{ds}: {len(g)} seeds")
        return c, g.mean(), g.std(ddof=1)

    pairs = []
    for key, mc, _, _ in MODELS:
        for ds in IOT:
            c, f1m, f1s = row(key, ds)
            for m, n in (("acc", "Acc"), ("bacc", "Bacc"), ("mcc", "Mcc"), ("dr", "Dr"), ("far", "Far")):
                pairs.append((f"JN{n}{mc}{DS[ds]}", f(100 * c[m])))
            pairs += [(f"JNEnergy{mc}{DS[ds]}", thousands(c["energy"])), (f"JNFone{mc}{DS[ds]}", f(f1m)),
                      (f"JNFone{mc}{DS[ds]}Sd", f(f1s))]
    guard("results table path (make_results_table.cell + per_seed_all_cells) on IoT", pairs)
    gp_pairs = []
    for key, mc, _, _ in MODELS:
        per = [cells[(cells.dataset == d) & (cells.model == mkey[key])].f1.mean() for d in GP]
        gp_pairs += [(f"JNFone{mc}GpMean", f(np.mean(per))), (f"JNFone{mc}CseCic", f(per[GP.index("cse-cic-ids2018")]))]
    guard("general-purpose macro-F1 already in numbers.tex (GP means, CSE-CIC-IDS2018)", gp_pairs)
    body, rows = [], []
    for i, ds in enumerate(GP):
        for j, (key, mc, _, lab) in enumerate(MODELS):
            c, f1m, f1s = row(key, ds)
            nj = c["energy"] / 1000
            rows.append(dict(dataset=ds, model=lab, f1=f1m, f1_sd=f1s, **{k: 100 * c[k] for k in ("acc", "bacc", "mcc", "dr", "far")},
                             arith_nJ=nj))
            e = f"{nj:,.1f}" if nj < 100 else f"{nj:,.0f}"
            body.append(f"{NAME[ds] if j == 0 else ''} & {lab}{'$^\\dagger$' if key == 'tasnn' else ''} & "
                        f"{f1m:.2f} $\\pm$ {f1s:.2f} & {100*c['acc']:.2f} & {100*c['bacc']:.2f} & {100*c['mcc']:.2f} & "
                        f"{100*c['dr']:.2f} & {100*c['far']:.2f} & {e.replace(',', '{,}')} \\\\")
        if i < len(GP) - 1:
            body.append("\\midrule")
    pd.DataFrame(rows).to_csv(DATA / "tab_results_general.csv", index=False)
    tex_table("tab_results_general",
              "Results on the seven general-purpose benchmarks. Predictive columns: five-seed mean (macro-F1 $\\pm$ "
              "sample SD); arithmetic energy: five-seed mean per sample at 32-bit Horowitz prices, full 20-step window, "
              "input layer included. CSE-CIC-IDS2018 is the repaired corpus at the shared 0.5 subsample.",
              "tab:results_general", "llrrrrrrr",
              ["& & \\multicolumn{6}{c}{Predictive, five-seed mean (\\%)} & Energy \\\\",
               "\\cmidrule(lr){3-8}",
               "Dataset & Model & Macro-F1 & ACC & BACC & MCC$\\times$100 & DR & FAR & nJ \\\\"],
              body, star=True, tag=tag,
              notes="$^\\dagger$TASNN energy is an analytic estimate (transformer MACs at 4.6\\,pJ plus spike "
                    "accumulates at 0.9\\,pJ); its recorded spike count omits the dense front-end. Vanilla SNN operations "
                    "are its recorded synaptic operations, which include the input layer and the recurrent term.")


# ------------------------------------------------------------------ 8. encoder ablation, GP (appendix)
def ablation_general(tag):
    variants = [("Tree", "results/Tree-to-Spike/{d}/seed_{s}/results.json"),
                ("Quantile", "results/encoder_ablation/{d}/quantile/seed_{s}/results.json"),
                ("Uniform", "results/encoder_ablation/{d}/uniform/seed_{s}/results.json"),
                ("All", "results/encoder_ablation/{d}/all/seed_{s}/results.json"),
                ("Nokd", "results/encoder_ablation/{d}/nokd/seed_{s}/results.json")]

    def arm(p, ds):
        rs = [json.loads((REPO / p.format(d=ds, s=s)).read_text()) for s in SEEDS]
        f1 = float(np.mean([100 * r["test_plus"]["metrics"]["f1_macro"] for r in rs]))
        ks = float(np.mean([r["test_plus"]["spike_stats"]["synops_per_sample"]
                            + r["data"]["encoder_channels"] * r["config"]["hidden"][0] for r in rs])) / 1000
        return f1, ks

    pairs = []
    for v, p in variants:
        for ds in IOT:
            f1, ks = arm(p, ds)
            pairs += [(f"JNAbl{v}{DS[ds]}Fone", f(f1)), (f"JNAbl{v}{DS[ds]}Ksops", f(ks, 1))]
    guard("encoder-ablation path on IoT (Table V)", pairs)
    body, rows = [], []
    for ds in GP:
        cells = [arm(p, ds) for _, p in variants]
        rows.append(dict(dataset=ds, **{f"{v}_f1": c[0] for (v, _), c in zip(variants, cells)},
                         **{f"{v}_ksop": c[1] for (v, _), c in zip(variants, cells)}))
        best = max(c[0] for c in cells)
        fmt = lambda c: (f"\\textbf{{{c[0]:.2f}}}" if f"{c[0]:.2f}" == f"{best:.2f}" else f"{c[0]:.2f}") + f" & {c[1]:.1f}"
        body.append(f"{NAME[ds]} & " + " & ".join(fmt(c) for c in cells) + " \\\\")
    pd.DataFrame(rows).to_csv(DATA / "tab_encoder_ablation_general.csv", index=False)
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{v}}}" for v in ("Tree", "Quantile", "Uniform", "All channels", "No KD"))
    tex_table("tab_encoder_ablation_general",
              "Encoder ablation on the seven general-purpose benchmarks (as Table V): macro-F1 (\\%) and SOPs per sample "
              "(thousands, full window, input layer included), five-seed means. Bold: best macro-F1 in the row.",
              "tab:encoder_ablation_general", "l" + "rr" * 5,
              [f"& {head} \\\\", " ".join(f"\\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(5)),
               "Dataset" + " & F1 & kSOP" * 5 + " \\\\"],
              body, star=True, tag=tag,
              notes="The raw-latency arm was run on the IoT benchmarks only. The All-channels CSE-CIC-IDS2018 seed-44 run "
                    "was fitted with a two-thread XGBoost teacher (test macro-F1 0.682122 vs 0.681954 for the other arms); "
                    "every other cell shares its teacher with the Tree arm.")


# ------------------------------------------------------------------ 9. envelope (appendix)
def envelope(tag):
    env = pd.read_csv(REPO / "results/deployment/gbdt_compression_repriced.csv")
    env["gain_prune"] = env.gain_prune.astype(str)
    B128, B256 = 128 * 1024, 256 * 1024
    ei = env[env.dataset.isin(IOT)]
    fits = ei[ei.deploy_bytes_packed_fp32_thresholds <= B128]
    full_i = ei[(ei.n_estimators == 200) & (ei.max_depth == 6) & (ei.gain_prune == "none") & (ei.leaf_bits == 16)] \
        .set_index("dataset").macro_f1
    near = fits[fits.apply(lambda r: r.macro_f1 >= full_i[r.dataset] - 1.0, axis=1)]
    guard("envelope path (repriced sweep) on IoT", [("JNEnvFitsIot", str(fits.dataset.nunique())),
                                                    ("JNEnvFitsNearFullIot", str(near.dataset.nunique()))])
    body, rows = [], []
    for i, ds in enumerate(IOT + GP):
        g = env[env.dataset == ds]
        full = g[(g.n_estimators == 200) & (g.max_depth == 6) & (g.gain_prune == "none") & (g.leaf_bits == 16)].iloc[0]
        f128, f256 = g[g.deploy_bytes_packed_fp32_thresholds <= B128], g[g.deploy_bytes_packed_fp32_thresholds <= B256]
        best = f128.loc[f128.macro_f1.idxmax()] if len(f128) else None
        cfg = (f"{int(best.n_estimators)}/{int(best.max_depth)}/{best.gain_prune}/{int(best.leaf_bits)}b"
               if best is not None else "--")
        rows.append(dict(dataset=ds, configs=len(g), full_KB=full.deploy_bytes_packed_fp32_thresholds / 1024,
                         full_f1=full.macro_f1, n_128=len(f128), n_256=len(f256), best_cfg=cfg,
                         best_KB=best.deploy_bytes_packed_fp32_thresholds / 1024 if best is not None else np.nan,
                         best_f1=best.macro_f1 if best is not None else np.nan))
        r = rows[-1]
        bf = f"{r['best_f1']:.2f}" if best is not None else "--"
        body.append(f"{NAME[ds]} & {thousands(r['full_KB'])} & {r['full_f1']:.2f} & {r['n_128']} & {r['n_256']} & "
                    f"{cfg.replace('none', '--')} & {bf} \\\\")
        if i == len(IOT) - 1:
            body.append("\\midrule")
    n = {r["configs"] for r in rows}
    if n != {48}:
        sys.exit(f"envelope: config counts {n}")
    pd.DataFrame(rows).to_csv(DATA / "tab_envelope.csv", index=False)
    tex_table("tab_envelope",
              "GBDT teacher against microcontroller memory budgets, seed 42, repriced packed format (float32 "
              "thresholds, 16-bit leaves unless the configuration says 8b, one value per leaf). Of 48 swept "
              "configurations per dataset: how many fit 128\\,KB and 256\\,KB, and the highest-scoring configuration "
              "within 128\\,KB (rounds/depth/gain-prune quantile/leaf bits). Configurations are selected on test "
              "macro-F1 (disclosed).",
              "tab:envelope", "lrrrrlr",
              ["& \\multicolumn{2}{c}{Full ensemble} & \\multicolumn{2}{c}{\\# fitting} & "
               "\\multicolumn{2}{c}{Best within 128\\,KB} \\\\",
               "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
               "Dataset & KB & Macro-F1 & 128\\,KB & 256\\,KB & Config & Macro-F1 \\\\"],
              body, star=False, tag=tag)


# ------------------------------------------------------------------ 10. exit audit, GP (appendix)
def exit_general(tag):
    fl = pd.read_csv(REPO / "results/tables/tab_m4_flags.csv", index_col=0)
    sel = pd.read_csv(REPO / "results/early_exit/selected_test.csv")
    ref = json.loads((REPO / "results/evidence_arrival.json").read_text())

    def stats(ds):
        per, tps = [], []
        for s in SEEDS:
            q, y, _ = prefix(ds, s)
            conf, pred = q.max(-1), q.argmax(-1)
            T = pred.shape[1]
            ok = pred[:, -1] == y
            tp = ((pred == y[:, None]) & ok[:, None]).argmax(1)[ok] + 1
            fired = conf >= 0.90
            ex = fired.any(1)
            t_star = np.where(ex, fired.argmax(1), T - 1)
            wrong = ex & (pred[np.arange(len(y)), t_star] != y)
            per.append(dict(seed=s, inclusion=100 * ok.mean(), wrong=100 * wrong.mean(),
                            t_median=float(np.median(tp)), t_p90=float(np.percentile(tp, 90))))
            tps.append(tp)
        d = pd.DataFrame(per)
        e = ref[ds]
        inc = 100 * sum(v["frac_defined"] * v["n"] for v in e["defined"].values()) / sum(v["n"] for v in e["defined"].values())
        r42 = d[d.seed == 42].iloc[0]
        diffs = [r42.t_median - e["median"], r42.t_p90 - e["p90"], r42.wrong - 100 * e["wrong_exit_frac"], r42.inclusion - inc]
        if max(abs(x) for x in diffs) > 0.01:
            sys.exit(f"GUARD FAILED (exit {ds}): seed-42 prefix vs evidence_arrival.json differences {diffs}")
        tp = np.concatenate(tps)
        admitted = int((~sel[sel.dataset == ds].early_exit_disabled).sum())
        if admitted != int(fl.loc[ds, "admitted"]):
            sys.exit(f"GUARD FAILED (exit {ds}): admitted {admitted} vs tab_m4_flags {fl.loc[ds, 'admitted']}")
        return dict(dataset=ds, admitted=admitted, wrong=d.wrong.mean(), wrong_min=d.wrong.min(), wrong_max=d.wrong.max(),
                    t_median=float(np.median(tp)), t_p90=float(np.percentile(tp, 90)), inclusion=d.inclusion.mean(),
                    flagged=float(fl.loc[ds, "unc"]))

    pairs = []
    for ds in IOT:
        r = stats(ds)
        pairs += [(f"JNExitWrong{DS[ds]}Five", f(r["wrong"])), (f"JNExitWrong{DS[ds]}Min", f(r["wrong_min"])),
                  (f"JNExitWrong{DS[ds]}Max", f(r["wrong_max"])), (f"JNExitInclusion{DS[ds]}Five", f(r["inclusion"])),
                  (f"JNExitTMedian{DS[ds]}Five", f(r["t_median"], 0)), (f"JNExitTPninety{DS[ds]}Five", f(r["t_p90"], 0)),
                  (f"JNExitFlag{DS[ds]}", f(r["flagged"]))]
    guard("exit-audit path (prefix outputs, Table IX) on IoT", pairs)
    GUARDS.append("exit audit: seed-42 prefix statistics reproduce evidence_arrival.json on all eleven datasets; "
                  "admitted counts agree between selected_test.csv and tab_m4_flags.csv")
    rows = [stats(ds) for ds in GP]
    pd.DataFrame(rows).to_csv(DATA / "tab_exit_general.csv", index=False)
    body = [f"{NAME[r['dataset']]} & {r['admitted']} & {r['wrong']:.2f} ({r['wrong_min']:.2f}--{r['wrong_max']:.2f}) & "
            f"{r['t_median']:.0f} & {r['t_p90']:.0f} & {r['inclusion']:.2f} & {r['flagged']:.2f} \\\\" for r in rows]
    tex_table("tab_exit_general",
              "Early-exit audit on the seven general-purpose benchmarks (as Table IX). Admitted: seeds (of five) on "
              "which the validation gate admits early exit. Wrong at 0.90: share of all test samples that cross the "
              "fixed diagnostic threshold $\\theta = 0.90$ with a wrong label, mean over five seeds (range). $t'$: "
              "first step whose running argmax is correct, among full-window-correct samples, pooled over five seeds. "
              "Inclusion: share of test samples on which $t'$ is defined. Flagged: share below $\\theta_{unc} = 0.5$, "
              "mean over seeds.",
              "tab:exit_general", "lrrrrrr",
              ["& & Wrong at 0.90 & \\multicolumn{2}{c}{$t'$} & & \\\\", "\\cmidrule(lr){4-5}",
               "Dataset & Admitted & \\% (range) & median & p90 & Inclusion \\% & Flagged \\% \\\\"],
              body, star=False, tag=tag)


# ------------------------------------------------------------------ README
def readme(tag, extras):
    items = [
        ("pareto_iot.pdf", "fig:pareto", "main", "numbers.tex (JNEnergy*, JNFone*); deployment_measurements.csv via deployment_energy2.row_energy (SRAM 20 pJ, DRAM 1300 pJ, reuse-aware, deployed, seed 42)"),
        ("exit_anatomy_ciciot.pdf", "fig:exit_anatomy", "main", "prefix outputs, CICIoT2023 seed 42 (results/prefix_outputs, not tracked)"),
        ("tree-to-spike.pdf", "Fig. 1", "main", "REDRAWN from the method description; the original source file is not in the repository"),
        ("cd_diagram.pdf", "fig:cd_diagram", "appendix", "numbers.tex (JNPooledRank*, JNPooledCD)"),
        ("memory_band.pdf", "fig:memory_band", "appendix", "deployment_measurements.csv via deployment_energy2.table"),
        ("margin_sweep.pdf", "fig:margin_sweep", "appendix", "prefix outputs, CICIoT2023 seed 42"),
        ("tab_results_general.tex", "tab:results_general", "appendix", "run records via make_results_table.cell; per_seed_all_cells.csv"),
        ("tab_encoder_ablation_general.tex", "tab:encoder_ablation_general", "appendix", "run records (results/encoder_ablation, results/Tree-to-Spike)"),
        ("tab_envelope.tex", "tab:envelope", "appendix", "results/deployment/gbdt_compression_repriced.csv"),
        ("tab_exit_general.tex", "tab:exit_general", "appendix", "prefix outputs; tab_m4_flags.csv; selected_test.csv"),
    ]
    lines = ["# Figures and tables for manuscript_tnnls.tex", "",
             f"Generated by `scripts/make_manuscript_figures.py` from tag `{TAG}` ({tag}). Regenerate with",
             "`python scripts/make_manuscript_figures.py`. Tables are typed values (no macros), as the manuscript requires.",
             "The plotted values are in `data/`.", "", "| File | Label | Where | Source |", "|---|---|---|---|"]
    lines += [f"| `{a}` | `{b}` | {c} | {d} |" for a, b, c, d in items]
    lines += ["", "## Guards (all passed; the script stops on any failure)", ""] + [f"- {g}" for g in GUARDS]
    lines += ["", "## Notes", ""] + [f"- {x}" for x in extras]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("[wrote] Figures/README.md")


def main():
    global PREVIEW, M
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=Path)
    a = ap.parse_args()
    if a.preview:
        PREVIEW = a.preview
        PREVIEW.mkdir(parents=True, exist_ok=True)
    tag = check_tag()
    M = macros()
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    pareto()
    share = exit_anatomy()
    schematic()
    joined = cd_diagram()
    memory_band()
    margin_sweep()
    results_general(tag)
    ablation_general(tag)
    envelope(tag)
    exit_general(tag)
    readme(tag, [
        "Exit anatomy: the two vertical lines are the mean exit step (2.25) and mean recovery step (11.16) of the "
        "wrong exits at 0.90 whose argmax later becomes correct (scripts/exit_margin_test.py). The bars are a "
        "different population: t' of all full-window-correct samples. Label them separately in the caption.",
        f"Exit anatomy: DDoS accounts for {share:.1f}% of the wrong exits at 0.90 (CICIoT2023, seed 42).",
        "Margin sweep: the y-axis is the macro-F1 gap, because the one-point admission bound is defined on macro-F1 "
        "(scripts/early_exit_sweep.py); the accuracy version is in scripts/exit_margin_test.py's log.",
        f"CD diagram cliques (mean-rank gap below CD): {joined}.",
        "Envelope: 'best' is the highest-scoring configuration within 128 KB, the main-text budget; counts use the "
        "same repriced bytes as \\JNEnvFitsIot.",
        "Pareto (a): TASNN's arithmetic energy (92-260 uJ) is off scale and drawn at the right axis limit with an arrow.",
        "Dataset display names are set in NAME at the top of the script.",
    ])


if __name__ == "__main__":
    main()
