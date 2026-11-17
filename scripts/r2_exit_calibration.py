#!/usr/bin/env python3
"""Early-exit calibration and the repeated wrong-exit audit (revision 2, step 4;
results/preregistration/2026-11-17_r2_steps4_6.md, rule R6).

Offline, from the cached prefix outputs of the reported Tree-to-Spike models (IoT four,
seeds 42-46). Calibrators are fitted on validation only:
  C1  per-step temperature tau_t (validation NLL); sensitivity: one global temperature
  C2  per-class exit thresholds on the uncalibrated confidence
Operating points on test: U (theta 0.90, the published audit), U-gate (the published
validation gate, recomputed here and checked against results/early_exit/selected_test.csv),
C1 (theta 0.90), C1-gate (the same gate on calibrated confidence), C1g (global temperature,
theta 0.90), C2.

    python scripts/r2_exit_calibration.py
Output: results/journal_r2/exit_calibration/{per_seed.csv, summary.csv, calibration.csv,
        thresholds.json, SUMMARY.md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import early_exit_sweep as EE  # noqa: E402

OUT = REPO / "results/journal_r2/exit_calibration"
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
NAME = {"ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024", "edgeiiot": "Edge-IIoTset", "toniot": "TON-IoT"}
SEEDS = range(42, 47)
AUDIT_TH = 0.90
CAL_STEPS = [1, 5, 10, 20]


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def nll(z, y):
    return float(-log_softmax(z)[np.arange(len(y)), y].mean())


def fit_tau(z, y):
    r = minimize_scalar(lambda lt: nll(z / np.exp(lt), y), bounds=(-3, 3), method="bounded",
                        options=dict(xatol=1e-4))
    return float(np.exp(r.x))


def rule(prob, theta_of_pred):
    """First crossing of a (possibly class-dependent) threshold; else the full window.
    Returns exit index (0-based), crossed flag, prediction."""
    conf, pred = prob.max(2), prob.argmax(2)
    fired = conf >= theta_of_pred(pred)
    ever = fired.any(1)
    T = prob.shape[1]
    first = np.where(ever, fired.argmax(1), T - 1)
    return first, ever, pred[np.arange(len(first)), first]


def audit(y, prob, pref, first, ever, p):
    T = prob.shape[1]
    full = prob[:, -1].argmax(1)
    early = ever & (first < T - 1)
    s = EE.score(y, p); sf = EE.score(y, full)
    sops = pref["cum_sops"][np.arange(len(first)), first]
    return dict(wrong_exit=100 * float((ever & (p != y)).mean()),
                harmful=100 * float((early & (p != y) & (full == y)).mean()),
                exit_share=100 * float(early.mean()), mean_exit=float((first + 1).mean()),
                median_exit=float(np.median(first + 1)),
                sop_reduction=100 * float(1 - sops.mean() / pref["cum_sops"][:, -1].mean()),
                macro_f1=s["macro_f1"], far=s["FAR"], d_macro_f1=s["macro_f1"] - sf["macro_f1"],
                full_macro_f1=sf["macro_f1"], full_far=sf["FAR"])


def gate(pv, yv, scale):
    """The validation gate of scripts/early_exit_sweep.py on logits z / scale."""
    pv2 = dict(pv, prefix_logits=pv["prefix_logits"].astype(np.float32) / scale)
    pred_f, _, sops_f, _ = EE.full_window(pv2)
    f_full = EE.score(yv, pred_f)["macro_f1"]
    cand = []
    for th in EE.THETAS:
        p, ex, ever, sops, _ = EE.replay(pv2, th)
        s = EE.score(yv, p)
        cand.append(dict(theta=th, macro_f1=s["macro_f1"], FAR=s["FAR"], mean_sops=sops.mean(),
                         eligible=bool(s["macro_f1"] >= f_full - EE.MARGIN)))
    elig = [c for c in cand if c["eligible"]]
    if not elig:
        return None
    return sorted(elig, key=lambda c: (c["mean_sops"], -c["macro_f1"], c["FAR"], -c["theta"]))[0]["theta"]


def per_class_thresholds(prob, y):
    K = prob.shape[2]
    conf, pred = prob.max(2), prob.argmax(2)
    full = pred[:, -1]
    th = {}
    for c in range(K):
        sel_full = full == c
        prec_full = float((y[sel_full] == c).mean()) if sel_full.any() else 1.0
        th[c] = None
        for t in EE.THETAS:
            f = (pred == c) & (conf >= t)
            ex = f.any(1)
            if ex.any() and (y[ex] == c).mean() >= prec_full:
                th[c] = t
                break
    return th


def calib_rows(ds, seed, prob_u, prob_c, y):
    rows = []
    for t in CAL_STEPS:
        for name, pr in (("uncalibrated", prob_u), ("C1", prob_c)):
            q = pr[:, t - 1]
            conf, pred = q.max(1), q.argmax(1)
            onehot = np.eye(q.shape[1])[y]
            rows.append(dict(dataset=ds, seed=seed, t=t, variant=name,
                             ece=EE.ece(conf, (pred == y).astype(float)),
                             brier=float(((q - onehot) ** 2).sum(1).mean()),
                             nll=float(-np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean())))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pub = pd.read_csv(REPO / "results/early_exit/selected_test.csv")
    aud = pd.read_csv(REPO / "results/tables/tab_exit_audit_iot.csv")
    rows, cal, thr = [], [], {}
    for ds in IOT:
        for seed in SEEDS:
            pv, pt = EE.load(ds, seed, "val"), EE.load(ds, seed, "test")
            yv, yt = pv["labels"], pt["labels"]
            zv, zt = pv["prefix_logits"].astype(np.float32), pt["prefix_logits"].astype(np.float32)
            T = zt.shape[1]
            rec = json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/results.json").read_text())
            f_rec = 100 * rec["test_plus"]["metrics"]["f1_macro"]
            f_pre = EE.score(yt, zt[:, -1].argmax(1))["macro_f1"]
            if abs(f_pre - f_rec) > 1e-6:
                raise SystemExit(f"{ds}/{seed}: prefix full window {f_pre} is not the record {f_rec}")
            # C1: per-step temperatures; C1g: one global temperature
            tau = np.array([fit_tau(zv[:, t], yv) for t in range(T)])
            tau_g = fit_tau(zv.reshape(-1, zv.shape[2]), np.repeat(yv, T))
            pu, pc = EE.softmax(zt, axis=2), EE.softmax(zt / tau[None, :, None], axis=2)
            pg = EE.softmax(zt / tau_g, axis=2)
            ops = {}
            ops["U"] = rule(pu, lambda p: AUDIT_TH)
            th_u = gate(pv, yv, 1.0)
            prow = pub[(pub.dataset == ds) & (pub.seed == seed)].iloc[0]
            pth = None if prow.early_exit_disabled else float(prow.selected_threshold)
            if th_u != pth:
                raise SystemExit(f"{ds}/{seed}: recomputed gate {th_u} is not the published {pth}")
            ops["U-gate"] = rule(pu, lambda p: th_u if th_u is not None else np.inf)
            ops["C1"] = rule(pc, lambda p: AUDIT_TH)
            th_c = gate(pv, yv, tau[None, :, None])
            ops["C1-gate"] = rule(pc, lambda p: th_c if th_c is not None else np.inf)
            ops["C1g"] = rule(pg, lambda p: AUDIT_TH)
            pcls = per_class_thresholds(EE.softmax(zv, axis=2), yv)
            vec = np.array([np.inf if pcls[c] is None else pcls[c] for c in range(zt.shape[2])])
            ops["C2"] = rule(pu, lambda p: vec[p])
            thr[f"{ds}/{seed}"] = dict(tau=tau.round(4).tolist(), tau_global=tau_g, gate_uncal=th_u,
                                       gate_c1=th_c, per_class={str(k): v for k, v in pcls.items()})
            for name, (first, ever, p) in ops.items():
                rows.append(dict(dataset=ds, seed=seed, op=name, **audit(yt, pu if name.startswith("U") or name == "C2"
                                                                         else (pc if name != "C1g" else pg),
                                                                         pt, first, ever, p)))
            u = [r for r in rows if r["dataset"] == ds and r["seed"] == seed and r["op"] == "U"][0]
            a = aud[(aud.dataset == ds) & (aud.seed == seed)].iloc[0]
            if abs(u["wrong_exit"] - a.wrong_exit) > 1e-6:
                raise SystemExit(f"{ds}/{seed}: audit wrong-exit {u['wrong_exit']} is not the published {a.wrong_exit}")
            cal += calib_rows(ds, seed, pu, pc, yt)
            print(f"{ds} s{seed}: tau_1 {tau[0]:.2f} tau_T {tau[-1]:.2f} global {tau_g:.2f} | gate U {th_u} C1 {th_c} | "
                  + " ".join(f"{r['op']} {r['wrong_exit']:.2f}" for r in rows[-len(ops):]), flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "per_seed.csv", index=False)
    pd.DataFrame(cal).to_csv(OUT / "calibration.csv", index=False)
    (OUT / "thresholds.json").write_text(json.dumps(thr, indent=1))
    s = d.groupby(["op", "dataset"]).mean(numeric_only=True).drop(columns="seed")
    s.to_csv(OUT / "summary.csv")
    report(d, s, pd.DataFrame(cal))


def report(d, s, cal):
    ops = ["U", "U-gate", "C1", "C1-gate", "C1g", "C2"]
    iot = {op: s.loc[op].loc[IOT].mean() for op in ops}
    L = ["# Early-exit calibration and the repeated wrong-exit audit (revision 2, step 4)", "",
         "Pre-registration: `results/preregistration/2026-11-17_r2_steps4_6.md` (R6). Tree-to-Spike, IoT four, "
         "seeds 42-46, test split, five-seed means. Wrong-exit = share of all test samples that cross the threshold "
         "with a wrong label (the published audit). Harmful = exits before T that are wrong where the full window is right.", "",
         "Operating points: U = uncalibrated, theta 0.90 (published audit); U-gate = published validation gate; "
         "C1 = per-step temperature, theta 0.90; C1-gate = validation gate on C1 confidence; C1g = global temperature, "
         "theta 0.90; C2 = per-class thresholds.", ""]
    for ds in IOT + ["IoT mean"]:
        L += [f"## {NAME.get(ds, ds)}", "", "| Operating point | Wrong-exit | Harmful | Exit before T | Mean exit step | SOP reduction | Macro-F1 | Change vs full | FAR |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for op in ops:
            r = iot[op] if ds == "IoT mean" else s.loc[op].loc[ds]
            L.append(f"| {op} | {r.wrong_exit:.2f}% | {r.harmful:.2f}% | {r.exit_share:.1f}% | {r.mean_exit:.2f} | "
                     f"{r.sop_reduction:.1f}% | {r.macro_f1:.2f} | {r.d_macro_f1:+.2f} | {r.far:.2f} |")
        L.append("")
    c = cal.groupby(["variant", "t"]).mean(numeric_only=True)[["ece", "brier", "nll"]]
    L += ["## Calibration of the prefix posterior (test, IoT mean over datasets and seeds)", "",
          "| t | ECE before | ECE after C1 | Brier before | Brier after | NLL before | NLL after |", "|--:|--:|--:|--:|--:|--:|--:|"]
    for t in CAL_STEPS:
        b, a = c.loc[("uncalibrated", t)], c.loc[("C1", t)]
        L.append(f"| {t} | {b.ece:.2f} | {a.ece:.2f} | {b.brier:.4f} | {a.brier:.4f} | {b.nll:.4f} | {a.nll:.4f} |")
    L += ["", "## R6", ""]
    u = s.loc["U"].loc[IOT]
    passing = []
    for op in ["C1", "C1-gate", "C1g", "C2"]:
        x = s.loc[op].loc[IOT]
        lower = int((x.wrong_exit < u.wrong_exit).sum())
        mean_lower = iot[op].wrong_exit < iot["U"].wrong_exit
        savings = iot[op].sop_reduction >= 10.0
        bound = bool((x.d_macro_f1 >= -1.0).all())
        ok = mean_lower and lower >= 3 and savings and bound
        if ok:
            passing.append(op)
        L.append(f"- **{op}**: wrong-exit lower than U on {lower}/4 datasets, IoT mean {iot[op].wrong_exit:.2f}% vs "
                 f"{iot['U'].wrong_exit:.2f}%; SOP reduction {iot[op].sop_reduction:.1f}% (>= 10% required); worst "
                 f"macro-F1 change {x.d_macro_f1.min():+.2f} (>= -1.0 required). **{'Passes' if ok else 'Fails'}.**")
    if passing:
        best = min(passing, key=lambda op: iot[op].wrong_exit)
        L += ["", f"**Decision: early exit is kept, with {best}** (lowest IoT-mean wrong-exit among passing calibrators)."]
    else:
        L += ["", "**Decision: no calibrator passes. Early exit is dropped from the deployment claims; the paper reports "
              "full-window results, with this audit as the reason.**"]
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
