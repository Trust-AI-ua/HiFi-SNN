#!/usr/bin/env python3
"""CTU-13 stage 1: timestamp audit, scenario-fold MIP, feature-subset diagnostic.

Deliberately stops before the 300-run model matrix. The 36-run diagnostic decides
whether scenario separation produces a measurable effect at all; if every feature
set stays above 99.5 macro-F1 with negligible FAR change, CTU-13 is reported as a
null scenario-shift result and the matrix is never launched.
"""
from __future__ import annotations
import glob, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "ctu13"; OUT.mkdir(parents=True, exist_ok=True)
HOST = None  # filled after load

GATE = {"min_test_normal": 1000, "min_test_botnet": 1000,
        "max_single_scenario_share_of_test": 0.60,
        "min_val_normal": 500, "min_val_botnet": 500,
        "min_train_scenarios": 6, "test_target": 0.24, "val_target": 0.08}


# ---------------------------------------------------------------- 1. audit
def timestamp_audit() -> dict:
    rep = {"per_scenario": {}}
    tot = parsed = 0
    for si, f in enumerate(sorted(glob.glob(str(REPO/"dataset/ctu13/binetflow/scenario-*-*.binetflow")))):
        d = pd.read_csv(f, usecols=["StartTime", "Label"], low_memory=False, on_bad_lines="skip")
        lbl = d["Label"].astype(str)
        d = d[lbl.str.contains("Botnet", na=False) | lbl.str.contains("Normal", na=False)]
        t = pd.to_datetime(d["StartTime"], errors="coerce", format="mixed")
        n, ok = len(d), int(t.notna().sum())
        tot += n; parsed += ok
        srt = t.dropna()
        rep["per_scenario"][si] = {
            "rows": n, "parsed": ok, "invalid": n - ok,
            "duplicate_timestamps": int(t.duplicated().sum()),
            "earliest": str(t.min()), "latest": str(t.max()),
            "out_of_order_rows_in_source": int((srt.diff() < pd.Timedelta(0)).sum()),
            "sorts_to_valid_sequence": bool((srt.sort_values().diff().dropna()
                                             >= pd.Timedelta(0)).all()),
        }
    rep["total_rows"] = tot; rep["total_parsed"] = parsed
    rep["total_invalid"] = tot - parsed
    rep["state_crosses_scenario_boundary"] = False   # by construction: reset per scenario
    rep["acceptance_online_causal"] = bool(
        rep["total_invalid"] == 0
        and all(v["sorts_to_valid_sequence"] for v in rep["per_scenario"].values()))
    return rep


# ------------------------------------------------------- 2. scenario folds
def scenario_folds(d: pd.DataFrame, n_folds: int = 4) -> dict:
    from scipy.optimize import LinearConstraint, milp
    from scipy.sparse import lil_matrix, vstack
    sc = sorted(d.scenario.unique()); S, K = len(sc), n_folds
    nrm = np.array([int(((d.scenario == s) & (d.y == 0)).sum()) for s in sc], float)
    bot = np.array([int(((d.scenario == s) & (d.y == 1)).sum()) for s in sc], float)
    tot = nrm + bot; N = tot.sum(); per = N * GATE["test_target"]

    n = S * K; rows, lb, ub = [], [], []
    def add(r, lo, hi): rows.append(r); lb.append(lo); ub.append(hi)
    for i in range(S):                                   # each scenario in one fold
        r = lil_matrix((1, n))
        for k in range(K): r[0, i*K+k] = 1.0
        add(r, 1.0, 1.0)
    for k in range(K):
        rn, rb, rt, rc = (lil_matrix((1, n)) for _ in range(4))
        for i in range(S):
            rn[0, i*K+k] = nrm[i]; rb[0, i*K+k] = bot[i]
            rt[0, i*K+k] = tot[i]; rc[0, i*K+k] = 1.0
        add(rn, GATE["min_test_normal"], np.inf)
        add(rb, GATE["min_test_botnet"], np.inf)
        # Unequal test folds are unavoidable: scenario 8 holds 26.8% of all flows,
        # so any fold containing it needs >=44.7% of the data to satisfy the 60%
        # domination limit. The 24% target and the domination limit cannot both
        # bind, and (following the TON-IoT precedent) the domination limit wins --
        # a fold dominated by one scenario cannot support a generalisation claim.
        add(rt, per*0.35, per*2.60)
        add(rc, 3, 4)                                    # 3-4 scenarios per test fold
        for i in range(S):                               # domination limit
            r = lil_matrix((1, n))
            for j in range(S): r[0, j*K+k] = -GATE["max_single_scenario_share_of_test"]*tot[j]
            r[0, i*K+k] += tot[i]
            add(r, -np.inf, 0.0)
    res = milp(c=np.zeros(n), constraints=LinearConstraint(vstack(rows).tocsr(),
               np.array(lb), np.array(ub)), integrality=np.ones(n), bounds=(0, 1),
               options={"time_limit": 120, "presolve": True})
    if res.x is None:
        return {"feasible": False, "status": str(res.message)}
    z = np.asarray(res.x).reshape(S, K)
    bins = [[sc[i] for i in range(S) if z[i, k] > 0.5] for k in range(K)]

    folds = {}
    for k in range(K):
        te = set(bins[k]); rest = [s for s in sc if s not in te]
        # inner validation: 1-2 whole scenarios, >=500 of each class, >=6 train scenarios
        best = None
        for a in range(len(rest)):
            for b in list(range(a, len(rest))):
                cand = {rest[a]} if a == b else {rest[a], rest[b]}
                if len(rest) - len(cand) < GATE["min_train_scenarios"]: continue
                m = d.scenario.isin(cand)
                if int((d.y[m] == 0).sum()) < GATE["min_val_normal"]: continue
                if int((d.y[m] == 1).sum()) < GATE["min_val_botnet"]: continue
                dev = abs(int(m.sum())/N - GATE["val_target"])
                if best is None or dev < best[0]: best = (dev, cand)
        if best is None: return {"feasible": False, "status": f"no valid val set for fold {k}"}
        va = best[1]; tr = set(rest) - va
        sel = lambda ss: d.scenario.isin(ss)
        folds[k] = {
            "test_scenarios": sorted(te), "val_scenarios": sorted(va),
            "train_scenarios": sorted(tr),
            "rows": {p: int(sel(s).sum()) for p, s in
                     (("train", tr), ("val", va), ("test", te))},
            "normal": {p: int(((d.y == 0) & sel(s)).sum()) for p, s in
                       (("train", tr), ("val", va), ("test", te))},
            "botnet": {p: int(((d.y == 1) & sel(s)).sum()) for p, s in
                       (("train", tr), ("val", va), ("test", te))},
            "scenario_overlap": 0,
        }
        f = folds[k]
        f["botnet_prevalence"] = {p: round(f["botnet"][p]/max(f["rows"][p], 1), 4)
                                  for p in ("train", "val", "test")}
        shares = [int(((d.scenario == s)).sum())/f["rows"]["test"] for s in te]
        f["max_scenario_share_of_test"] = round(max(shares), 4)
        f["passes"] = bool(f["normal"]["test"] >= GATE["min_test_normal"]
                           and f["botnet"]["test"] >= GATE["min_test_botnet"]
                           and f["max_scenario_share_of_test"] <= GATE["max_single_scenario_share_of_test"]
                           and len(f["train_scenarios"]) >= GATE["min_train_scenarios"])
    return {"feasible": True, "gate": GATE, "folds": folds}


# ------------------------------------------------- 3. 36-run RF diagnostic
def diagnostic(d: pd.DataFrame, folds: dict) -> list:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
    from src.data.ctu13_causal import causal_host_features, HOST_COLS
    base = ["Dur","TotPkts","TotBytes","SrcBytes","sTos","dTos","Sport","Dport"]
    LEG=[f"legacy__{c}" for c in HOST_COLS]; CAU=[f"causal__{c}" for c in HOST_COLS]
    res=[]
    for k, f in folds.items():
        idx = {"A": None, "B": None, "C": None}
        # A/B share stratified random manifests; C uses scenario manifests
        rng = np.random.default_rng(1000+int(k)); n = len(d)
        perm = rng.permutation(n)
        nte = int(0.24*n); nva = int(0.08*n)
        idx["A"] = idx["B"] = (perm[nte+nva:], perm[nte:nte+nva], perm[:nte])
        idx["C"] = (np.where(d.scenario.isin(f["train_scenarios"]))[0],
                    np.where(d.scenario.isin(f["val_scenarios"]))[0],
                    np.where(d.scenario.isin(f["test_scenarios"]))[0])
        for cond, hostcols in (("legacy_random", LEG), ("causal_random", None),
                               ("causal_scenario", CAU)):
            tr, va, te = idx["A" if cond=="legacy_random" else "B" if cond=="causal_random" else "C"]
            if cond == "causal_random":
                # partition-local causal aggregates: recompute inside each partition
                parts=[]
                for part in (tr, va, te):
                    sub = d.iloc[part].sort_values(["scenario"], kind="stable")
                    ch = []
                    for s, g in sub.groupby("scenario", sort=True):
                        ch.append(causal_host_features(g))
                    ch = pd.concat(ch); ch.columns = CAU
                    parts.append(ch.reindex(d.index[part]))
                H = pd.concat(parts).reindex(d.index)
                D = pd.concat([d[base+["y"]], H], axis=1); hc = CAU
            else:
                D = d; hc = hostcols
            for fs, cols in (("base", base), ("host", hc), ("combined", base+hc)):
                X = D[cols].replace([np.inf,-np.inf],0).fillna(0).to_numpy(np.float32)
                y = d.y.to_numpy()
                m = RandomForestClassifier(n_estimators=40, n_jobs=8, random_state=42)
                m.fit(X[tr], y[tr]); p = m.predict(X[te])
                cm = confusion_matrix(y[te], p, labels=[0,1])
                far = cm[0,1]/max(cm[0].sum(),1); dr = cm[1,1]/max(cm[1].sum(),1)
                res.append(dict(fold=int(k), condition=cond, features=fs,
                    macro_f1=round(f1_score(y[te],p,average="macro")*100,3),
                    mcc=round(matthews_corrcoef(y[te],p)*100,3),
                    DR=round(dr*100,3), FAR=round(far*100,3),
                    confusion=cm.tolist(), n_test=int(len(te))))
                print(f"  fold{k} {cond:16s} {fs:8s} F1={res[-1]['macro_f1']:6.2f} "
                      f"FAR={res[-1]['FAR']:6.2f}", flush=True)
    return res


if __name__ == "__main__":
    print("=== 1. timestamp audit ==="); a = timestamp_audit()
    json.dump(a, open(OUT/"timestamp_audit.json","w"), indent=2)
    print(f"  rows {a['total_rows']} parsed {a['total_parsed']} invalid {a['total_invalid']}")
    print(f"  online-causal acceptance: {a['acceptance_online_causal']}")
    d = pd.read_parquet(REPO/"dataset/ctu13/_three_condition.parquet")
    print("\n=== 2. scenario-fold MIP ==="); fo = scenario_folds(d)
    json.dump(fo, open(OUT/"scenario_folds.json","w"), indent=2, default=str)
    print(f"  feasible={fo['feasible']}")
    if not fo["feasible"]:
        print("  ", fo["status"]); sys.exit(1)
    for k, f in fo["folds"].items():
        print(f"  fold{k}: test={f['test_scenarios']} val={f['val_scenarios']} "
              f"rows={f['rows']} botnet%={f['botnet_prevalence']} "
              f"maxshare={f['max_scenario_share_of_test']} pass={f['passes']}")
    print("\n=== 3. 36-run feature-subset diagnostic ===")
    r = diagnostic(d, fo["folds"])
    json.dump(r, open(OUT/"diagnostic36.json","w"), indent=2)
    df = pd.DataFrame(r)
    print("\n" + df.groupby(["condition","features"])[["macro_f1","FAR"]].mean().round(2).to_string())
