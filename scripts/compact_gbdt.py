#!/usr/bin/env python3
"""Compact GBDT baseline by validation-only tree-prefix selection.

The tree count is never hand-picked from test. For each dataset and seed the full
teacher is fitted once, boosting-round prefixes are scored on VALIDATION, and the
smallest prefix within 1.0 absolute macro-F1 point of the full teacher is frozen.
Test is evaluated exactly once, afterwards.

Search is a doubling grid (1,2,4,...,N) followed by a binary search between the
last failing and first passing prefix, so the reported tree count is the true
smallest qualifying one rather than whichever power of two happened to pass.

If no prefix qualifies, the full teacher is retained -- a valid outcome, not a
failure.

    python scripts/compact_gbdt.py --datasets nslkdd --seeds 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASETS = ["nslkdd", "kddcup99", "unswnb15", "cicids2017", "cse-cic-ids2018",
            "iscxids2012", "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
SEEDS = [42, 43, 44, 45, 46]
MARGIN = 1.0        # absolute macro-F1 percentage points
OUT = REPO / "results" / "deployment"


def build(dataset, seed):
    """Fit the same teacher the Tree-to-Spike run used. Deterministic given seed."""
    from src.data import get_dataset, DATASET_LOADERS
    from src.training_treespike import (_fit_teacher, _arrays_from_bundle,
                                        _resolve, _set_seed)
    import yaml, inspect
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{dataset}/config.yaml"))
    rec_p = REPO / f"results/Tree-to-Spike/{dataset}/seed_{seed}/results.json"
    if rec_p.exists():
        cfg.update(json.load(open(rec_p))["config"])
    cfg["dataset"] = dataset; cfg["seed"] = seed
    cfg = _resolve(cfg)
    _set_seed(seed)
    params = inspect.signature(DATASET_LOADERS[dataset]).parameters
    kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=seed)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in params and cfg[opt] is not None:
            kw[opt] = cfg[opt]
    b = get_dataset(dataset, **kw)
    Xtr, ytr, Xva, yva, Xte, yte = _arrays_from_bundle(b)
    teacher, _, present = _fit_teacher(Xtr, ytr, b.num_classes, cfg)
    return teacher, present, b, (Xva, yva, Xte, yte), cfg


def f1_at(teacher, present, X, y, n):
    from sklearn.metrics import f1_score
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    p = present[teacher.predict(df, iteration_range=(0, n))]
    return 100 * f1_score(y, p, average="macro", zero_division=0)


def full_metrics(teacher, present, X, y, names, n):
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    p = present[teacher.predict(df, iteration_range=(0, n))]
    m = compute_metrics(y, p, y_score=None, class_names=names)
    b = compute_binary_ids_metrics(y, p, normal_class_index=0)
    return (100 * m["f1_macro"], 100 * m["matthews_corrcoef"],
            100 * b["false_alarm_rate"], 100 * b["detection_rate"])


def tree_stats(teacher, n_rounds):
    """Serialized bytes and node counts for the first n_rounds boosting rounds."""
    import tempfile, os
    df = teacher.get_booster().trees_to_dataframe()
    n_cls = max(int(df["Tree"].max()) + 1, 1) // max(
        int(teacher.get_params().get("n_estimators", 1)), 1) or 1
    keep = df[df["Tree"] < n_rounds * max(n_cls, 1)]
    nodes = int(len(keep))
    leaves = int((keep["Feature"] == "Leaf").sum())
    internal = nodes - leaves
    # actual bytes: serialize a truncated copy
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        bst = teacher.get_booster()
        bst.save_model(tmp)
        full_bytes = os.path.getsize(tmp)
    finally:
        os.unlink(tmp)
    total_rounds = max(int(teacher.get_params().get("n_estimators", 1)), 1)
    approx = int(full_bytes * min(n_rounds / total_rounds, 1.0))
    return dict(nodes=nodes, internal_nodes=internal, leaves=leaves,
                full_model_bytes=full_bytes, prefix_bytes_estimate=approx)


def select_rounds(teacher, present, Xva, yva, N):
    """Smallest boosting-round prefix within MARGIN macro-F1 of the full teacher.

    Takes VALIDATION arrays only; test data cannot reach the selection. Doubling
    grid (1, 2, 4, ..., N), then bisection between the last failing and first
    passing prefix. Returns (rounds, qualified, full-teacher validation macro-F1);
    if no prefix qualifies, the full teacher is kept (qualified = False).
    """
    f_full_val = f1_at(teacher, present, Xva, yva, N)
    need = f_full_val - MARGIN
    grid, n = [], 1
    while n < N:
        grid.append(n); n *= 2
    grid.append(N)
    first_ok, last_bad = None, 0
    for n in grid:
        if f1_at(teacher, present, Xva, yva, n) >= need:
            first_ok = n; break
        last_bad = n
    if first_ok is None:
        return N, False, f_full_val
    lo, hi = last_bad + 1, first_ok
    while lo < hi:
        mid = (lo + hi) // 2
        if f1_at(teacher, present, Xva, yva, mid) >= need:
            hi = mid
        else:
            lo = mid + 1
    return lo, True, f_full_val


def write_merged(df):
    """Merge on (dataset, seed): a run for some datasets must not delete the rest.

    The previous version overwrote the file, so rerunning two datasets would have
    erased the other nine. Rows carry a run ID and the newest row per key wins.
    """
    import datetime, subprocess
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    df = df.copy()
    df["run_id"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + sha
    f = OUT / "compact_gbdt.csv"
    if f.exists():
        df = pd.concat([pd.read_csv(f), df], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "seed"], keep="last")
    df.to_csv(f, index=False)
    return df


def guards(ds, seed, teacher, present, b, arrays):
    """Three guards, each fatal: same splits as the stored runs, validation-only
    selection, and the same teacher. Returns a record of what was checked."""
    import hashlib, inspect as _insp
    from src.metrics import compute_metrics
    from src.training_treespike import _arrays_from_bundle
    Xva, yva, Xte, yte = arrays
    Xtr, ytr = _arrays_from_bundle(b)[:2]
    fail, rec = [], {}
    # 1. splits: sizes and per-class train counts against BOTH stored student runs,
    #    exact label sequences against the prefix artifacts regenerated from them
    for model in ("Tree-to-Spike", "DNN-Tree"):
        r = json.loads((REPO / f"results/{model}/{ds}/seed_{seed}/results.json").read_text())
        # Tree-to-Spike records all three split sizes; DNN-Tree records only the test
        # size (test_plus.n_samples) and, below, the per-class training counts.
        sp = r["data"].get("splits") or {"test_plus": r["test_plus"].get("n_samples")}
        mine = dict(train=len(ytr), val=len(yva), test_plus=len(yte))
        if any(k in mine and v is not None and mine[k] != v for k, v in sp.items()):
            fail.append(f"split sizes differ from stored {model}: {mine} vs {sp}")
        names = list(b.class_names)
        cnt = {names[c]: int((ytr == c).sum()) for c in range(len(names)) if (ytr == c).any()}
        if cnt != {k: int(v) for k, v in r["data"]["class_counts_train"].items() if int(v) > 0}:
            fail.append(f"per-class train counts differ from stored {model}")
    for split, y in (("test", yte), ("val", yva)):
        z = np.load(REPO / f"results/prefix_outputs/{ds}/treespike/seed_{seed}/{split}_prefix.npz",
                    allow_pickle=True)
        if not np.array_equal(z["labels"], y):
            fail.append(f"{split} label sequence differs from the prefix artifact")
    rec["test_sha256"] = hashlib.sha256(np.ascontiguousarray(Xte).tobytes()).hexdigest()[:16]
    rec["val_sha256"] = hashlib.sha256(np.ascontiguousarray(Xva).tobytes()).hexdigest()[:16]
    # 2. validation only: the selector cannot receive test arrays
    sig = list(_insp.signature(select_rounds).parameters)
    if sig != ["teacher", "present", "Xva", "yva", "N"]:
        fail.append(f"select_rounds signature changed: {sig}")
    # 3. same teacher: its test macro-F1 must equal the stored run's teacher_upper_bound,
    #    which the five-seed teacher refit also reproduces
    df = pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])])
    got = 100 * compute_metrics(yte, present[teacher.predict(df)], y_score=None,
                                class_names=b.class_names)["f1_macro"]
    want = 100 * json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/results.json")
                            .read_text())["test_plus"]["teacher_upper_bound"]["f1_macro"]
    rec["teacher_test_f1"], rec["stored_teacher_f1"] = got, want
    if abs(got - want) > 1e-9:
        fail.append(f"teacher differs: test macro-F1 {got:.6f} vs stored {want:.6f}")
    return fail, rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--guard", action="store_true",
                    help="run the three guards on every cell and require --reproduce datasets to "
                         "match their stored round counts; write NOTHING unless every cell passes")
    ap.add_argument("--reproduce", nargs="*", default=[],
                    help="datasets whose stored selected_rounds must be reproduced exactly")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stored = pd.read_csv(OUT / "compact_gbdt.csv") if (OUT / "compact_gbdt.csv").exists() else None
    guard_fail, guard_log = [], []

    rows, t0 = [], time.time()
    for ds in args.datasets:
        for seed in args.seeds:
            try:
                teacher, present, b, (Xva, yva, Xte, yte), cfg = build(ds, seed)
            except Exception as e:
                print(f"[FAIL] {ds}/{seed}: {type(e).__name__}: {e}", flush=True)
                continue
            N = int(teacher.get_params().get("n_estimators", 200))
            sel, qualified, f_full_val = select_rounds(teacher, present, Xva, yva, N)
            if args.guard:
                fl, rec = guards(ds, seed, teacher, present, b, (Xva, yva, Xte, yte))
                if ds in args.reproduce:
                    want = int(stored[(stored.dataset == ds) & (stored.seed == seed)].selected_rounds.iloc[0])
                    rec["stored_rounds"], rec["reselected_rounds"] = want, sel
                    if sel != want:
                        fl.append(f"reproduction failed: selected {sel} rounds, stored {want}")
                guard_log.append(dict(dataset=ds, seed=seed, passed=not fl, failures="; ".join(fl), **rec))
                print(f"  [guard] {ds}/s{seed}: {'PASS' if not fl else 'FAIL: ' + '; '.join(fl)}", flush=True)
                guard_fail += [f"{ds}/s{seed}: {x}" for x in fl]
            f_sel_val = f1_at(teacher, present, Xva, yva, sel)
            t_full = full_metrics(teacher, present, Xte, yte, b.class_names, N)
            t_sel = full_metrics(teacher, present, Xte, yte, b.class_names, sel)
            st_full, st_sel = tree_stats(teacher, N), tree_stats(teacher, sel)

            rows.append(dict(
                dataset=ds, seed=seed, full_rounds=N, selected_rounds=sel,
                qualified=qualified, compression=N / max(sel, 1),
                val_full_macro_f1=f_full_val, val_sel_macro_f1=f_sel_val,
                test_full_macro_f1=t_full[0], test_sel_macro_f1=t_sel[0],
                d_test_macro_f1=t_sel[0] - t_full[0],
                test_full_mcc=t_full[1], test_sel_mcc=t_sel[1],
                test_full_FAR=t_full[2], test_sel_FAR=t_sel[2],
                d_test_FAR=t_sel[2] - t_full[2],
                test_full_DR=t_full[3], test_sel_DR=t_sel[3],
                full_nodes=st_full["nodes"], sel_nodes=st_sel["nodes"],
                full_internal=st_full["internal_nodes"], sel_internal=st_sel["internal_nodes"],
                full_bytes=st_full["full_model_bytes"],
                sel_bytes_estimate=st_sel["prefix_bytes_estimate"]))
            print(f"[{len(rows)}] {ds}/s{seed} rounds {N}->{sel} "
                  f"({'ok' if qualified else 'NO PREFIX QUALIFIED'}) "
                  f"val {f_full_val:.2f}->{f_sel_val:.2f} | test {t_full[0]:.2f}->"
                  f"{t_sel[0]:.2f} ({t_sel[0]-t_full[0]:+.2f}) "
                  f"nodes {st_full['nodes']}->{st_sel['nodes']} "
                  f"| {(time.time()-t0)/60:.1f}m", flush=True)

    df = pd.DataFrame(rows)
    if args.guard:
        pd.DataFrame(guard_log).to_csv(OUT / "compact_gbdt_guards.csv", index=False)
        if guard_fail:
            print(f"\nGUARDS FAILED on {len(guard_fail)} check(s); NOTHING WRITTEN:")
            for x in guard_fail:
                print("  " + x)
            sys.exit(1)
        print(f"\nall guards passed on {len(guard_log)} cells")
        df = df[~df.dataset.isin(args.reproduce)]      # reproduction cells only verify
    if len(df):
        write_merged(df)
    if len(df):
        print("\n=== per dataset (mean over seeds) ===")
        print(df.groupby("dataset")[["full_rounds", "selected_rounds", "compression",
                                     "test_full_macro_f1", "test_sel_macro_f1",
                                     "d_test_macro_f1", "d_test_FAR",
                                     "full_nodes", "sel_nodes"]].mean().round(3).to_string())
        print(f"\ncells where no prefix qualified: {int((~df.qualified).sum())} of {len(df)}")
        print(f"mean round compression: {df.compression.mean():.1f}x")
        print(f"cells losing >1 pp test macro-F1: {int((df.d_test_macro_f1 < -1).sum())}")
    print(f"\nwrote {OUT}/compact_gbdt.csv")
