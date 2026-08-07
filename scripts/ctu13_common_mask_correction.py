#!/usr/bin/env python3
"""Matched-mask duplicate correction for the CTU-13 host-context interaction.

The raw Delta_host compares base against combined on *different* retained row
sets: the base representation has 15-46% of its test rows exactly duplicated in
training (8 coarse features collide), while combined has ~0%. Correcting each
side on its own mask is a bound, not a comparison.

This evaluates both representations on ONE common mask per (condition, fold):

    M_common = NOT (dup_base OR dup_combined)

INFERENCE ONLY -- no SNN is retrained. Student weights come from the saved
checkpoints. The XGBoost teacher is refit because it was never serialised, but
it is deterministic given the seed and costs seconds; every reconstruction is
verified against the stored full-test macro-F1 before its masked metrics are
used, so a reconstruction that is not bit-faithful is reported, not silently
consumed.

    python scripts/ctu13_common_mask_correction.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.data.ctu13_causal import load_frozen_folds                    # noqa: E402
from src.data.ctu13_conditions import (BASE_COLS, CAU, PQ, FOLDS_JSON,  # noqa: E402
                                       _random_partition)
from src.data.leakage import encode_train_only                          # noqa: E402
from src.metrics import clear_eval_context                              # noqa: E402
from run_ctu13_matrix import build_cfg, result_dir                      # noqa: E402

MODELS = ["gbdt", "treespike", "dnn"]
CONDITIONS = ["causal_random", "causal_scenario"]
FEATURE_SETS = ["base", "combined"]
FOLDS = [0, 1, 2, 3]
SEEDS = [42, 43, 44, 45, 46]
OUT = REPO / "results" / "ctu13"
TOL = 0.05          # pp; reconstruction must match stored macro-F1 this closely


def block(y_true, y_pred) -> dict:
    from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {"n": int(len(y_true)),
            "macro_f1": 100 * f1_score(y_true, y_pred, average="macro", zero_division=0),
            "mcc": 100 * matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else None,
            "DR": 100 * tp / max(tp + fn, 1), "FAR": 100 * fp / max(fp + tn, 1)}


# --------------------------------------------------------------- data per cell
_cache: dict = {}


def cell_data(condition: str, fold: int, feature_set: str):
    """(Xtr, ytr, Xte, yte, te_idx) plus the raw feature frame for hashing.
    Cached: identical for every model and seed at a given (condition, fold, fs)
    because the partition is fold-keyed, not seed-keyed."""
    key = (condition, fold, feature_set)
    if key in _cache:
        return _cache[key]
    d = _cache.get("_pq")
    if d is None:
        d = pd.read_parquet(PQ)
        _cache["_pq"] = d
    fo = load_frozen_folds(FOLDS_JSON, d)
    f = fo["folds"][fold]
    y = d["y"].to_numpy(np.int64)

    if condition == "causal_scenario":
        tr = np.where(d.scenario.isin(f["train_scenarios"]))[0]
        va = np.where(d.scenario.isin(f["val_scenarios"]))[0]
        te = np.where(d.scenario.isin(f["test_scenarios"]))[0]
        host = d[CAU]
    else:
        tr, va, te = _random_partition(len(d), fold)
        from ctu13_stage1 import _partition_local_causal
        host = _partition_local_causal(d, (tr, va, te), CAU)

    feat = (d[BASE_COLS] if feature_set == "base"
            else pd.concat([d[BASE_COLS], host.set_axis(CAU, axis=1)], axis=1))
    feat = feat.reset_index(drop=True)
    h = pd.util.hash_pandas_object(feat, index=False).to_numpy()
    dup = np.asarray(pd.Index(h[te]).isin(pd.Index(h[tr])), dtype=bool)

    Xtr, _, Xte, _ = encode_train_only(feat, tr, va, te)
    out = (Xtr, y[tr], Xte, y[te], dup)
    _cache[key] = out
    return out


# ------------------------------------------------------- prediction reproduction
def predict(model: str, condition: str, feature_set: str, fold: int, seed: int,
            Xtr, ytr, Xte):
    """Rebuild the trained model and return its test predictions."""
    from src.training_treespike import (_fit_teacher, _resolve as _res_ts,
                                        _set_seed, _encode_forward_in_batches)
    cfg = build_cfg(model, condition, feature_set, fold, seed)

    if model == "gbdt":
        teacher, _, present = _fit_teacher(Xtr, ytr, 2, cfg)
        Xte_df = pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])])
        return present[teacher.predict(Xte_df)]

    ckpt = result_dir(model, condition, feature_set, fold) / f"seed_{seed}" / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    if model == "treespike":
        from src.models import TreeGuidedSpikeEncoder, SNNStudent
        cfg = _res_ts(cfg)
        _set_seed(seed)
        dev = torch.device(cfg["device"])
        teacher, _, _ = _fit_teacher(Xtr, ytr, 2, cfg)
        enc = TreeGuidedSpikeEncoder(
            T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
            jitter=cfg["jitter"], encoder_mode=cfg["encoder_mode"]).fit(
                teacher, Xtr.shape[1], X=Xtr)
        st = SNNStudent(enc.out_dim, 2, hidden=tuple(cfg["hidden"]),
                        beta=cfg["beta"], leaky=cfg["leaky"]).to(dev)
        st.load_state_dict(torch.load(ckpt, map_location=dev))
        with torch.no_grad():
            pred, *_ = _encode_forward_in_batches(st, enc, Xte, cfg["batch_size"], dev)
        return pred

    from src.models import TreeGuidedSpikeEncoder, DNNStudent
    from src.training_dnn import _resolve as _res_dnn, _forward_batches
    cfg = _res_dnn(cfg)
    _set_seed(seed)
    dev = torch.device(cfg["device"])
    teacher, _, _ = _fit_teacher(Xtr, ytr, 2, cfg)
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
                                 prune_quantile=cfg["prune_quantile"],
                                 jitter=cfg["jitter"]).fit(teacher, Xtr.shape[1])
    st = DNNStudent(enc.out_dim, 2, hidden=tuple(cfg["hidden"]),
                    clip_relu=cfg["clip_relu"]).to(dev)
    st.load_state_dict(torch.load(ckpt, map_location=dev))
    with torch.no_grad():
        pred, _ = _forward_batches(st, enc, Xte, cfg["batch_size"], dev)
    return pred


def stored_macro_f1(model, condition, feature_set, fold, seed) -> float:
    p = result_dir(model, condition, feature_set, fold) / f"seed_{seed}" / "results.json"
    return 100 * json.load(open(p))["test_plus"]["metrics"]["f1_macro"]


if __name__ == "__main__":
    clear_eval_context()          # masked metrics are computed here, not in compute_metrics
    rows, t0 = [], time.time()
    n_tot = len(CONDITIONS) * len(FOLDS) * len(MODELS) * len(SEEDS) * len(FEATURE_SETS)
    done = 0

    for condition in CONDITIONS:
        for fold in FOLDS:
            # one common mask per (condition, fold): test rows are the same set and
            # order for base and combined, since the partition is feature-independent
            _, _, _, yte, dup_b = cell_data(condition, fold, "base")
            _, _, _, yte2, dup_c = cell_data(condition, fold, "combined")
            assert np.array_equal(yte, yte2), "test labels differ between feature sets"
            common = ~(dup_b | dup_c)
            print(f"\n=== {condition} fold{fold}: test={len(yte)} "
                  f"dup_base={dup_b.sum()} dup_combined={dup_c.sum()} "
                  f"common_kept={common.sum()} ({100*common.mean():.2f}%) ===", flush=True)

            for model in MODELS:
                for seed in SEEDS:
                    for fs in FEATURE_SETS:
                        Xtr, ytr, Xte, yt, _ = cell_data(condition, fold, fs)
                        pred = predict(model, condition, fs, fold, seed, Xtr, ytr, Xte)
                        full = block(yt, pred)
                        ref = stored_macro_f1(model, condition, fs, fold, seed)
                        drift = full["macro_f1"] - ref
                        masked = block(yt[common], pred[common])
                        rows.append(dict(model=model, condition=condition, features=fs,
                                         fold=fold, seed=seed,
                                         stored_macro_f1=ref,
                                         repro_macro_f1=full["macro_f1"],
                                         repro_drift_pp=drift,
                                         full_macro_f1=full["macro_f1"],
                                         full_FAR=full["FAR"],
                                         common_macro_f1=masked["macro_f1"],
                                         common_mcc=masked["mcc"],
                                         common_DR=masked["DR"], common_FAR=masked["FAR"],
                                         common_n=masked["n"],
                                         common_kept_pct=100 * common.mean()))
                        done += 1
                    print(f"  [{done}/{n_tot}] {model}/f{fold}/s{seed} "
                          f"drift={drift:+.3f}pp elapsed={(time.time()-t0)/60:.1f}m",
                          flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "matrix_common_mask.csv", index=False)

    print("\n" + "=" * 78)
    print("RECONSTRUCTION FIDELITY (masked metrics are only valid if this is ~0)")
    print("=" * 78)
    print(df.groupby("model")["repro_drift_pp"].agg(["mean", "min", "max"]).round(4).to_string())
    bad = df[df.repro_drift_pp.abs() > TOL]
    print(f"\ncells exceeding +/-{TOL} pp: {len(bad)} of {len(df)}")
    if len(bad):
        print(bad[["model", "condition", "features", "fold", "seed",
                   "stored_macro_f1", "repro_macro_f1", "repro_drift_pp"]]
              .head(12).round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("MATCHED Delta_host = (comb_scen - base_scen) - (comb_rand - base_rand)")
    print("=" * 78)
    k = ["model", "fold", "seed"]
    def g(fs, cond, col):
        return df[(df.features == fs) & (df.condition == cond)].set_index(k)[col]
    for col, lab in (("full_macro_f1", "raw (own-mask, for reference)"),
                     ("common_macro_f1", "MATCHED common mask")):
        idx = g("base", "causal_random", col).index
        hi = ((g("combined", "causal_scenario", col).reindex(idx)
               - g("base", "causal_scenario", col).reindex(idx))
              - (g("combined", "causal_random", col).reindex(idx)
                 - g("base", "causal_random", col).reindex(idx))).dropna()
        t = hi.rename("d").reset_index().groupby("model")["d"].agg(["mean", "std", "min", "max"])
        print(f"\n-- {lab}")
        print(t.round(2).to_string())
        byf = hi.rename("d").reset_index().pivot_table(index="model", columns="fold", values="d")
        print("   by fold:"); print(byf.round(2).to_string())

    print("\n-- components on the common mask (macro-F1, mean over folds+seeds)")
    print(df.pivot_table(index="model", columns=["condition", "features"],
                         values="common_macro_f1").round(2).to_string())
    print(f"\nwrote {OUT/'matrix_common_mask.csv'}")
