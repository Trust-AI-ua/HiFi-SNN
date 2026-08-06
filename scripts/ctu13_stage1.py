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

# results/ctu13/SNN_LeakyParallel/ is a leftover phase-1 SNN ablation artifact from
# 2026-06-25. It is NOT output of this leakage experiment and must never be counted
# by skip detection: it has the shape <arm>/seed_<n>/results.json, so a naive
# results/ctu13/*/*/seed_*/results.json glob would read its 5 runs as completed
# cells of the 300-run matrix and silently skip real work.
# Skip detection must key on a results.json inside a directory this script owns,
# never on directory existence under results/ctu13/.
STALE_DIRS = {"SNN_LeakyParallel"}

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
            # int(), not numpy int64: these round-trip through JSON, and a str
            # scenario id makes d.scenario.isin(...) match zero rows.
            "test_scenarios": sorted(int(s) for s in te),
            "val_scenarios": sorted(int(s) for s in va),
            "train_scenarios": sorted(int(s) for s in tr),
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
def _partition_local_causal(d: pd.DataFrame, partitions, cau_cols: list) -> pd.DataFrame:
    """Recompute causal host features independently inside each partition.

    Condition B asks what the aggregates look like when a partition may only see
    its own rows, so the features cannot come from the precomputed causal__*
    columns (those were built over the whole scenario) -- they must be recomputed,
    which is why the raw identifier and time columns have to survive into the
    persisted artifact.

    Results are scattered back positionally through _row_pos. Label-based
    reindex-after-groupby is correct while the frame carries a unique contiguous
    RangeIndex, but nothing enforces that invariant, and a silent misalignment
    here would surface as a plausible metric rather than an error.
    """
    from src.data.ctu13_causal import (HOST_COLS, causal_host_features,
                                       require_raw_columns)
    require_raw_columns(d)
    n = len(d)
    pos_all = d["_row_pos"].to_numpy()
    if not np.array_equal(pos_all, np.arange(n, dtype=pos_all.dtype)):
        raise ValueError("_row_pos must be 0..n-1 over the full table")

    H = np.full((n, len(HOST_COLS)), np.nan, dtype=np.float64)
    seen = np.zeros(n, dtype=np.int64)
    for part in partitions:
        # sort by _t, not by index: causal_host_features assumes chronological
        # arrival, and a random partition destroys the artifact's row order.
        sub = d.iloc[part].sort_values(["scenario", "_t"], kind="stable")
        for _, g in sub.groupby("scenario", sort=True):
            f = causal_host_features(g)
            pos = g["_row_pos"].to_numpy()
            H[pos] = f.to_numpy()
            seen[pos] += 1
    assert seen.min() == 1 and seen.max() == 1, "every row written exactly once"
    assert not np.isnan(H).any(), "no unwritten positions"
    return pd.DataFrame(H, columns=cau_cols, index=d.index)


def diagnostic(d: pd.DataFrame, folds: dict, manifest_sha: str = None) -> list:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
    from src.data.ctu13_causal import HOST_COLS, METADATA_COLS
    base = ["Dur","TotPkts","TotBytes","SrcBytes","sTos","dTos","Sport","Dport"]
    LEG=[f"legacy__{c}" for c in HOST_COLS]; CAU=[f"causal__{c}" for c in HOST_COLS]
    meta = set(METADATA_COLS)
    assert meta.issubset(d.columns), \
        f"artifact missing raw columns: {sorted(meta - set(d.columns))}"
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
        # Catches a scenario-id dtype mismatch: the frozen manifest stores ids that
        # have been through JSON, and isin(["2","4","8"]) against an int64 column
        # matches zero rows -- an empty partition, not an error.
        got = dict(zip(("train", "val", "test"), (len(x) for x in idx["C"])))
        if "rows" in f:
            assert got == {p: int(f["rows"][p]) for p in got}, (
                f"fold {k} condition-C sizes {got} != frozen manifest {f['rows']}; "
                "check scenario-id dtype")
        assert all(got.values()), f"fold {k} has an empty condition-C partition: {got}"
        for cond, hostcols in (("legacy_random", LEG), ("causal_random", None),
                               ("causal_scenario", CAU)):
            tr, va, te = idx["A" if cond=="legacy_random" else "B" if cond=="causal_random" else "C"]
            if cond == "causal_random":
                # partition-local causal aggregates: recompute inside each partition
                H = _partition_local_causal(d, (tr, va, te), CAU)
                D = pd.concat([d[base+["y"]], H], axis=1); hc = CAU
            else:
                D = d; hc = hostcols
            for fs, cols in (("base", base), ("host", hc), ("combined", base+hc)):
                # Host identity must not reach the classifier: the CTU-13 label is a
                # property of the host, so SrcAddr as a feature would manufacture the
                # shortcut this experiment is measuring.
                assert meta.isdisjoint(cols), f"metadata leaked into {fs} features"
                X = D[cols].replace([np.inf,-np.inf],0).fillna(0).to_numpy(np.float32)
                y = d.y.to_numpy()
                m = RandomForestClassifier(n_estimators=40, n_jobs=8, random_state=42)
                m.fit(X[tr], y[tr]); p = m.predict(X[te])
                cm = confusion_matrix(y[te], p, labels=[0,1])
                far = cm[0,1]/max(cm[0].sum(),1); dr = cm[1,1]/max(cm[1].sum(),1)
                res.append(dict(fold=int(k), condition=cond, features=fs,
                    manifest_sha256=manifest_sha,
                    macro_f1=round(f1_score(y[te],p,average="macro")*100,3),
                    mcc=round(matthews_corrcoef(y[te],p)*100,3),
                    DR=round(dr*100,3), FAR=round(far*100,3),
                    confusion=cm.tolist(), n_test=int(len(te))))
                print(f"  fold{k} {cond:16s} {fs:8s} F1={res[-1]['macro_f1']:6.2f} "
                      f"FAR={res[-1]['FAR']:6.2f}", flush=True)
    return res


def _load_frozen_folds(path: Path, frame: pd.DataFrame = None) -> dict:
    """Thin alias; the canonical loader lives in src.data.ctu13_causal so that every
    consumer (this script, the fold-1 audit, the model matrix) shares one contract."""
    from src.data.ctu13_causal import load_frozen_folds
    return load_frozen_folds(path, frame)


if __name__ == "__main__":
    AUDIT = OUT / "timestamp_audit.json"
    FOLDS = OUT / "scenario_folds.json"
    # Both artifacts are FROZEN and results already exist against them. The MIP has
    # a 120 s time limit and no unique optimum, so re-solving can return a
    # different-but-feasible assignment and silently invalidate prior results.
    # Reuse when present; delete the file deliberately to regenerate.
    print("=== 1. timestamp audit ===")
    if AUDIT.exists():
        a = json.load(open(AUDIT)); print(f"  reusing frozen {AUDIT.name}")
    else:
        a = timestamp_audit(); json.dump(a, open(AUDIT, "w"), indent=2)
    print(f"  rows {a['total_rows']} parsed {a['total_parsed']} invalid {a['total_invalid']}")
    print(f"  online-causal acceptance: {a['acceptance_online_causal']}")
    d = pd.read_parquet(REPO/"dataset/ctu13/_three_condition.parquet")
    print("\n=== 2. scenario-fold MIP ===")
    if FOLDS.exists():
        fo = _load_frozen_folds(FOLDS, d)
        print(f"  reusing frozen {FOLDS.name} sha={fo['manifest_sha256'][:16]}... "
              f"(partition sizes verified)")
    else:
        fo = scenario_folds(d)
        json.dump(fo, open(FOLDS, "w"), indent=2, default=str)
    print(f"  feasible={fo['feasible']}")
    if not fo["feasible"]:
        print("  ", fo["status"]); sys.exit(1)
    for k, f in fo["folds"].items():
        print(f"  fold{k}: test={f['test_scenarios']} val={f['val_scenarios']} "
              f"rows={f['rows']} botnet%={f['botnet_prevalence']} "
              f"maxshare={f['max_scenario_share_of_test']} pass={f['passes']}")
    print("\n=== 3. 36-run feature-subset diagnostic ===")
    r = diagnostic(d, fo["folds"], fo.get("manifest_sha256"))
    json.dump(r, open(OUT/"diagnostic36.json","w"), indent=2)
    df = pd.DataFrame(r)
    print("\n" + df.groupby(["condition","features"])[["macro_f1","FAR"]].mean().round(2).to_string())
