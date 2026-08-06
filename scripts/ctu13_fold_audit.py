#!/usr/bin/env python3
"""CTU-13 fold audit: is fold 1's 27.40 combined macro-F1 a real scenario-shift outcome?

Fold 1 (test = scenarios 7, 9, 11) collapsed to 27.40 combined macro-F1 while fold 0
stayed at 99.50. That gap decides how CTU-13 gets reported, so it is explained before
any further aggregate is produced. Five sections:

  A  integrity      -- assert the frozen manifest, features and labels are intact,
                       so the collapse cannot be blamed on a defect
  B  decomposition  -- evaluate scenarios 7 / 9 / 11 separately, not just the fold
  C  seed variance  -- 3 feature sets x 2 protocols x 5 seeds = 30 runs, fold 1 only
  D  host identity  -- source-host overlap, purity, and error attribution
  E  feature shift  -- distance between fold-1 train and each test scenario

The folds are FROZEN. Nothing here regenerates them.
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results" / "ctu13"
PQ = REPO / "dataset/ctu13/_three_condition.parquet"
OLD_PQ = Path("/tmp/claude-1002/-home-student3-HiSNN/"
              "81f000bb-8428-4c08-be6f-f6eab257dd8b/scratchpad/_three_condition.OLD.parquet")

from src.data.ctu13_causal import HOST_COLS, METADATA_COLS          # noqa: E402
from ctu13_stage1 import _load_frozen_folds, _partition_local_causal  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
BASE = ["Dur", "TotPkts", "TotBytes", "SrcBytes", "sTos", "dTos", "Sport", "Dport"]
LEG = [f"legacy__{c}" for c in HOST_COLS]
CAU = [f"causal__{c}" for c in HOST_COLS]
FEATURE_SETS = {"base": BASE, "host": CAU, "combined": BASE + CAU}

# Set per fold by main() from the frozen manifest; the audit never hardcodes folds.
EXPECT_TEST: set = set()


def sha(obj) -> str:
    return hashlib.sha256(obj if isinstance(obj, bytes)
                          else json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def build_X(frame: pd.DataFrame, cols: list) -> np.ndarray:
    """Feature matrix. No fitted preprocessing: RF needs no scaling, and the only
    transforms (inf->0, NaN->0) are stateless, so nothing can leak from test."""
    assert set(METADATA_COLS).isdisjoint(cols), f"metadata in features: {cols}"
    return frame[cols].replace([np.inf, -np.inf], 0).fillna(0).to_numpy(np.float32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    both = len(np.unique(y_true)) > 1
    return {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro") * 100, 3),
        "mcc": round(matthews_corrcoef(y_true, y_pred) * 100, 3) if both else None,
        "DR": round(tp / max(tp + fn, 1) * 100, 3),
        "FAR": round(fp / max(fp + tn, 1) * 100, 3),
        "confusion": cm.tolist(),
        "support_normal": int(tn + fp), "support_botnet": int(tp + fn),
        "pred_normal": int(tn + fn), "pred_botnet": int(fp + tp),
        "pred_botnet_frac": round((fp + tp) / max(len(y_pred), 1), 4),
    }


# =============================================================== A. integrity
def section_a(d: pd.DataFrame, f: dict) -> dict:
    print("\n=== A. integrity ===")
    a: dict = {}
    tr_s, va_s, te_s = (set(f["train_scenarios"]), set(f["val_scenarios"]),
                        set(f["test_scenarios"]))

    assert te_s == EXPECT_TEST, f"test scenarios {te_s} != {EXPECT_TEST}"
    a["test_scenarios"] = sorted(te_s)
    assert tr_s.isdisjoint(va_s) and tr_s.isdisjoint(te_s) and va_s.isdisjoint(te_s)
    assert all(isinstance(s, int) for s in tr_s | va_s | te_s), "scenario ids not int"
    print(f"  scenarios: test={sorted(te_s)} val={sorted(va_s)} train={sorted(tr_s)}")
    print("  partitions disjoint, ids are int: PASS")

    tr = np.where(d.scenario.isin(sorted(tr_s)))[0]
    va = np.where(d.scenario.isin(sorted(va_s)))[0]
    te = np.where(d.scenario.isin(sorted(te_s)))[0]
    got = {"train": len(tr), "val": len(va), "test": len(te)}
    assert got == {p: int(f["rows"][p]) for p in got}, f"{got} != {f['rows']}"
    print(f"  row counts match frozen manifest: PASS {got}")

    test_frame = d.iloc[te]
    assert test_frame["scenario"].isin(sorted(EXPECT_TEST)).all()
    for p, idx in (("train", tr), ("val", va), ("test", te)):
        n0 = int((d.y.to_numpy()[idx] == 0).sum()); n1 = int((d.y.to_numpy()[idx] == 1).sum())
        assert n0 == int(f["normal"][p]) and n1 == int(f["botnet"][p]), \
            f"{p} class counts {n0}/{n1} != manifest {f['normal'][p]}/{f['botnet'][p]}"
    print("  class counts match manifest in all 3 partitions: PASS")

    # NaN / inf BEFORE the stateless cleanup, so masking cannot hide a defect.
    raw = d[BASE + CAU + LEG]
    n_nan = int(raw.isna().to_numpy().sum())
    n_inf = int(np.isinf(raw.to_numpy(np.float64)).sum())
    a["nan_cells_before_cleanup"] = n_nan
    a["inf_cells_before_cleanup"] = n_inf
    print(f"  NaN cells={n_nan} inf cells={n_inf} (checked pre-cleanup, unmasked)")
    assert n_nan == 0 and n_inf == 0, "NaN/inf present -- collapse may be an artifact"

    # Labels identical across conditions: every condition reads the same d.y.
    a["label_source"] = "single d.y column shared by all conditions"
    a["label_values"] = sorted(int(v) for v in d.y.unique())
    assert a["label_values"] == [0, 1]
    print(f"  label mapping identical across conditions: PASS {a['label_values']}")

    assert set(METADATA_COLS).isdisjoint(BASE + CAU + LEG)
    print("  metadata disjoint from every model feature list: PASS")

    # Exact duplicate feature rows crossing train->test (hash-based upper bound).
    h = pd.util.hash_pandas_object(d[BASE + CAU], index=False).to_numpy()
    n_dup = int(pd.Index(h[te]).isin(pd.Index(h[tr])).sum())
    a["test_rows_duplicated_in_train"] = n_dup
    a["test_rows_duplicated_in_train_pct"] = round(100 * n_dup / len(te), 4)
    print(f"  test rows with an exact feature duplicate in train: {n_dup} "
          f"({100*n_dup/len(te):.4f}%)  [hash upper bound]")

    a["preprocessing"] = ("none fitted: RandomForest needs no scaling and no categorical "
                          "encoding is used (base is numeric only); inf->0 and NaN->0 are "
                          "stateless, so no statistic crosses from test into train")
    print("  train-only preprocessing: N/A, nothing is fitted (see report)")

    a["manifest_sha256"] = sha((OUT / "scenario_folds.json").read_bytes())
    print(f"  frozen manifest sha256: {a['manifest_sha256'][:16]}...")

    # Rebuild changed schema only: compare feature blocks against the pre-fix artifact.
    if OLD_PQ.exists():
        old = pd.read_parquet(OLD_PQ, columns=LEG + CAU + ["y", "scenario"])
        same = bool(np.array_equal(old[LEG + CAU].to_numpy(), d[LEG + CAU].to_numpy())
                    and np.array_equal(old["y"].to_numpy(), d["y"].to_numpy())
                    and np.array_equal(old["scenario"].to_numpy(), d["scenario"].to_numpy()))
        a["features_identical_to_prefix_artifact"] = same
        a["causal_block_sha256"] = sha(np.ascontiguousarray(d[CAU].to_numpy()).tobytes())
        print(f"  legacy/causal/y/scenario bit-identical to pre-fix parquet: "
              f"{'PASS' if same else 'FAIL'}")
        assert same, "rebuild altered feature values, not just schema"
        del old
    else:
        a["features_identical_to_prefix_artifact"] = "pre-fix artifact unavailable"
        a["causal_block_sha256"] = sha(np.ascontiguousarray(d[CAU].to_numpy()).tobytes())
        print("  pre-fix parquet unavailable; recorded causal-block hash for future runs")
    print(f"  causal block sha256: {a['causal_block_sha256'][:16]}...")

    a["partition_index"] = {"train": len(tr), "val": len(va), "test": len(te)}
    return a, (tr, va, te)


# ================================================ B + C. decomposition, seeds
def section_bc(d: pd.DataFrame, f: dict, parts) -> tuple:
    from sklearn.ensemble import RandomForestClassifier
    print("\n=== B+C. per-scenario decomposition over 5 seeds "
          "(3 feature sets x 2 protocols x 5 seeds = 30 runs) ===")
    tr_c, va_c, te_c = parts
    y = d.y.to_numpy()
    n = len(d)
    runs, per_scenario = [], []
    preds_for_d: dict = {}

    for seed in SEEDS:
        # --- protocol 1: causal scenario-disjoint (frozen partition, precomputed
        # causal__* -- legitimate here, since a whole test scenario's history lies
        # entirely inside test, so nothing crosses the partition boundary).
        for fs, cols in FEATURE_SETS.items():
            X = build_X(d, cols)
            m = RandomForestClassifier(n_estimators=40, n_jobs=8, random_state=seed)
            m.fit(X[tr_c], y[tr_c])
            p_te = m.predict(X[te_c])
            runs.append(dict(seed=seed, protocol="causal_scenario", features=fs,
                             scope="fold", **metrics(y[te_c], p_te)))
            print(f"  seed{seed} causal_scenario {fs:8s} fold     "
                  f"F1={runs[-1]['macro_f1']:6.2f} FAR={runs[-1]['FAR']:6.2f}")
            # B: same fitted model, evaluated per test scenario
            sc_te = d.scenario.to_numpy()[te_c]
            for s in sorted(EXPECT_TEST):
                sm = sc_te == s
                per_scenario.append(dict(seed=seed, protocol="causal_scenario",
                                         features=fs, scope=f"scenario_{s}",
                                         **metrics(y[te_c][sm], p_te[sm])))
                r = per_scenario[-1]
                print(f"    scenario {s:>2}: F1={r['macro_f1']:6.2f} DR={r['DR']:6.2f} "
                      f"FAR={r['FAR']:6.2f} pred_botnet={r['pred_botnet_frac']:.3f} "
                      f"(N={r['support_normal']}, B={r['support_botnet']})")
            if fs == "combined" and seed == SEEDS[0]:
                preds_for_d = {"idx": te_c, "pred": p_te}

        # --- protocol 2: causal random (fresh permutation per seed, partition-local
        # recompute -- the random split cuts across scenarios).
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        nte, nva = int(0.24 * n), int(0.08 * n)
        tr_r, va_r, te_r = perm[nte + nva:], perm[nte:nte + nva], perm[:nte]
        H = _partition_local_causal(d, (tr_r, va_r, te_r), CAU)
        Dr = pd.concat([d[BASE + ["y"]], H], axis=1)
        for fs, cols in FEATURE_SETS.items():
            X = build_X(Dr, cols)
            m = RandomForestClassifier(n_estimators=40, n_jobs=8, random_state=seed)
            m.fit(X[tr_r], y[tr_r])
            p = m.predict(X[te_r])
            runs.append(dict(seed=seed, protocol="causal_random", features=fs,
                             scope="fold", **metrics(y[te_r], p)))
            print(f"  seed{seed} causal_random   {fs:8s} fold     "
                  f"F1={runs[-1]['macro_f1']:6.2f} FAR={runs[-1]['FAR']:6.2f}")
        del H, Dr

    return runs, per_scenario, preds_for_d


# ============================================================ D. host identity
def section_d(d: pd.DataFrame, parts, preds: dict) -> dict:
    print("\n=== D. host-identity audit ===")
    tr_c, _, te_c = parts
    src = d["SrcAddr"].to_numpy()
    y = d.y.to_numpy()
    train_hosts = set(src[tr_c].tolist())
    out: dict = {"scenarios": {}}

    for s in sorted(EXPECT_TEST):
        m = d.scenario.to_numpy() == s
        hs = src[m]
        uniq = set(hs.tolist())
        overlap = uniq & train_hosts
        counts = pd.Series(hs).value_counts()
        pur = pd.DataFrame({"host": hs, "y": y[m]}).groupby("host")["y"].agg(
            ["count", "mean"])
        purity = np.maximum(pur["mean"], 1 - pur["mean"])
        e = {
            "unique_hosts": len(uniq),
            "hosts_also_in_fold1_train": len(overlap),
            "hosts_unique_to_test": len(uniq - train_hosts),
            "host_overlap_pct": round(100 * len(overlap) / max(len(uniq), 1), 2),
            "flows_per_host_median": float(counts.median()),
            "flows_per_host_max": int(counts.max()),
            "flows_from_overlapping_hosts_pct": round(
                100 * float(counts.reindex(list(overlap)).fillna(0).sum()) / int(m.sum()), 2),
            "hosts_ge_90pct_single_class": int((purity >= 0.90).sum()),
            "hosts_ge_95pct_single_class": int((purity >= 0.95).sum()),
            "hosts_ge_99pct_single_class": int((purity >= 0.99).sum()),
            "mean_host_purity": round(float(purity.mean()), 4),
        }
        out["scenarios"][s] = e
        print(f"  scenario {s:>2}: hosts={e['unique_hosts']:>6} "
              f"overlap_with_train={e['hosts_also_in_fold1_train']:>5} "
              f"({e['host_overlap_pct']:>5.2f}%) flows_from_overlap="
              f"{e['flows_from_overlapping_hosts_pct']:>5.2f}% "
              f">=99% pure={e['hosts_ge_99pct_single_class']:>6}")

    all_test_hosts = set(src[te_c].tolist())
    ov = all_test_hosts & train_hosts
    out["fold_test_hosts"] = len(all_test_hosts)
    out["fold_host_overlap"] = len(ov)
    out["fold_host_overlap_pct"] = round(100 * len(ov) / max(len(all_test_hosts), 1), 2)
    out["host_disjoint"] = len(ov) == 0
    print(f"  fold test hosts={len(all_test_hosts)} overlapping train={len(ov)} "
          f"({out['fold_host_overlap_pct']}%)")
    print(f"  scenario protocol is host-disjoint: {out['host_disjoint']} "
          f"-- {'' if out['host_disjoint'] else 'do NOT call it host-disjoint'}")

    # Error attribution using the seed-42 combined scenario-disjoint model.
    if preds:
        idx, pred = preds["idx"], preds["pred"]
        err = pred != y[idx]
        df = pd.DataFrame({"host": src[idx], "scenario": d.scenario.to_numpy()[idx],
                           "err": err,
                           "fp": (pred == 1) & (y[idx] == 0),
                           "fn": (pred == 0) & (y[idx] == 1)})
        g = df.groupby("host")[["err", "fp", "fn"]].sum().sort_values("err",
                                                                     ascending=False)
        tot = int(err.sum())
        top5 = g.head(5)
        out["errors_total"] = tot
        out["top5_hosts_error_share_pct"] = round(100 * float(top5["err"].sum()) / max(tot, 1), 2)
        out["top5_hosts"] = [
            {"host": str(h), "errors": int(r["err"]), "fp": int(r["fp"]), "fn": int(r["fn"])}
            for h, r in top5.iterrows()]
        print(f"  seed-42 combined errors={tot}; top-5 hosts account for "
              f"{out['top5_hosts_error_share_pct']}%")
        for t in out["top5_hosts"]:
            print(f"    {t['host']:>18}  err={t['errors']:>7} fp={t['fp']:>7} fn={t['fn']:>7}")
        out["errors_by_scenario"] = {
            int(s): {"errors": int(v["err"]), "fp": int(v["fp"]), "fn": int(v["fn"])}
            for s, v in df.groupby("scenario")[["err", "fp", "fn"]].sum().iterrows()}
        print(f"  errors by scenario: {out['errors_by_scenario']}")
    return out


# ============================================================= E. feature shift
def section_e(d: pd.DataFrame, parts) -> dict:
    from scipy.stats import wasserstein_distance
    print("\n=== E. feature shift, fold-1 train vs each test scenario ===")
    tr_c, _, _ = parts
    out: dict = {"note": ("Wasserstein computed on train-standardised values so it is "
                          "comparable across features; SMD uses pooled train sd."),
                 "scenarios": {}}
    tr_df = d.iloc[tr_c]

    for s in sorted(EXPECT_TEST):
        te_df = d[d.scenario == s]
        rows = []
        for c in CAU + BASE:
            a = tr_df[c].to_numpy(np.float64); b = te_df[c].to_numpy(np.float64)
            sd = a.std()
            sd_safe = sd if sd > 1e-12 else 1.0
            lo, hi = a.min(), a.max()
            rows.append({
                "feature": c,
                "train_median": round(float(np.median(a)), 4),
                "test_median": round(float(np.median(b)), 4),
                "train_iqr": round(float(np.subtract(*np.percentile(a, [75, 25]))), 4),
                "test_iqr": round(float(np.subtract(*np.percentile(b, [75, 25]))), 4),
                "smd": round(float((b.mean() - a.mean()) / sd_safe), 4),
                "wasserstein_std": round(float(wasserstein_distance(a / sd_safe,
                                                                    b / sd_safe)), 4),
                "pct_outside_train_range": round(
                    float(100 * ((b < lo) | (b > hi)).mean()), 3),
            })
        df = pd.DataFrame(rows).sort_values("wasserstein_std", ascending=False)
        out["scenarios"][s] = df.to_dict("records")
        print(f"\n  -- scenario {s}: 6 most-shifted features "
              f"(of {len(CAU)} host + {len(BASE)} base)")
        print(df.head(6)[["feature", "smd", "wasserstein_std",
                          "pct_outside_train_range"]].to_string(index=False))
        host_w = df[df.feature.isin(CAU)]["wasserstein_std"].mean()
        base_w = df[df.feature.isin(BASE)]["wasserstein_std"].mean()
        print(f"  mean Wasserstein: host block={host_w:.3f}  base block={base_w:.3f}")
        out.setdefault("block_means", {})[s] = {
            "host_mean_wasserstein": round(float(host_w), 4),
            "base_mean_wasserstein": round(float(base_w), 4)}
    return out


# ======================================================================= main
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[1])
    args = ap.parse_args()

    fo = _load_frozen_folds(OUT / "scenario_folds.json")
    d = pd.read_parquet(PQ)
    print(f"loaded {len(d)} rows x {len(d.columns)} cols "
          f"| manifest sha={fo['manifest_sha256'][:16]}...")

    for fold in args.folds:
        f = fo["folds"][fold]
        EXPECT_TEST = set(f["test_scenarios"])
        print(f"\n{'='*70}\n=== CTU-13 fold-{fold} audit (test scenarios "
              f"{sorted(EXPECT_TEST)}) ===\n{'='*70}")
        rep = {"fold": fold, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "manifest_sha256": fo["manifest_sha256"]}

        rep["A_integrity"], parts = section_a(d, f)
        runs, per_scenario, preds = section_bc(d, f, parts)
        rep["C_runs"] = runs
        rep["B_per_scenario"] = per_scenario
        rep["D_host_identity"] = section_d(d, parts, preds)
        rep["E_feature_shift"] = section_e(d, parts)

        dest = OUT / f"fold{fold}_audit.json"
        json.dump(rep, open(dest, "w"), indent=2, default=str)

        print(f"\n=== fold {fold} SUMMARY: fold-level, mean/sd over 5 seeds ===")
        r = pd.DataFrame(runs)
        print(r.groupby(["protocol", "features"])["macro_f1"]
               .agg(["mean", "std", "min", "max"]).round(2).to_string())
        print(f"\n=== fold {fold} SUMMARY: per-scenario, causal_scenario, "
              f"mean over 5 seeds ===")
        ps = pd.DataFrame(per_scenario)
        print(ps.pivot_table(index="scope", columns="features",
                             values=["macro_f1", "FAR"]).round(2).to_string())
        print(f"\nwrote {dest}")
