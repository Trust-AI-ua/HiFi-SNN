#!/usr/bin/env python3
"""CICIoMT2024 seed-aligned duplicate-free re-evaluation. Inference only.

The provider capture-file partition is NOT changed and nothing is retrained. For
each seed the study's sampling protocol is reproduced exactly
(`subsample_fraction=0.25`, `RandomState(seed)` drawing ktr then kte, then a
stratified 10% validation split), the duplicate mask is built against *that
seed's own training partition* using canonical raw feature records, and every
checkpoint is scored three ways:

    full        the test set as originally evaluated
    unique      duplicates removed
    duplicate   the removed rows alone

The third is the diagnostic that matters: if duplicated rows are dramatically
easier than unique ones, the provider partition inflates the headline.

Every reconstruction is verified against the stored macro-F1 before its numbers
are used, so a model that fails to reproduce is reported rather than trusted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "ciciomt2024"
SEEDS = [42, 43, 44, 45, 46]
SUBSAMPLE = 0.25
VAL_FRACTION = 0.1
TOL = 0.05      # pp

RESULT_DIR = {
    "gbdt":      "results/Tree-to-Spike/ciciomt2024",        # teacher_upper_bound block
    "treespike": "results/Tree-to-Spike/ciciomt2024",
    "dnn":       "results/DNN-Tree/ciciomt2024",
    "tasnn":     "results/TASNN/ciciomt2024",
    "vanilla":   "results/ciciomt2024/SNN_LeakyParallel/latency",
}
MODELS = ["gbdt", "vanilla", "treespike", "dnn", "tasnn"]
_teacher_cache: dict = {}


def record_path(model: str, seed: int) -> Path:
    return REPO / RESULT_DIR[model] / f"seed_{seed}" / "results.json"


def stored_f1(model: str, seed: int) -> float:
    r = json.load(open(record_path(model, seed)))["test_plus"]
    blk = r["teacher_upper_bound"] if model == "gbdt" else r
    m = blk["metrics"] if "metrics" in blk else blk
    return 100 * m["f1_macro"]


def metrics_block(y_true, y_pred, n_classes, class_names) -> dict:
    from sklearn.metrics import (f1_score, matthews_corrcoef, confusion_matrix,
                                 balanced_accuracy_score, recall_score)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # binary IDS view: class 0 is Benign
    tb, pb = (y_true != 0).astype(int), (y_pred != 0).astype(int)
    bcm = confusion_matrix(tb, pb, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in bcm.ravel())
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1c = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "macro_f1": 100 * f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": 100 * matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else None,
        "bacc": 100 * balanced_accuracy_score(y_true, y_pred),
        "DR": 100 * tp / max(tp + fn, 1),
        "FAR": 100 * fp / max(fp + tn, 1),
        "per_class_recall": {class_names[i]: 100 * float(rec[i]) for i in labels},
        "per_class_f1": {class_names[i]: 100 * float(f1c[i]) for i in labels},
        "confusion_matrix": cm.tolist(),
        "support": {class_names[i]: int((y_true == i).sum()) for i in labels},
    }


def get_teacher(seed, Xtr, ytr, n_classes, cfg):
    from src.training_treespike import _fit_teacher
    key = (seed, cfg["teacher_n_estimators"], cfg["teacher_max_depth"], cfg["teacher_lr"])
    if key not in _teacher_cache:
        t0 = time.time()
        _teacher_cache.clear()          # only ever need the current seed's teacher
        _teacher_cache[key] = _fit_teacher(Xtr, ytr, n_classes, cfg)
        print(f"    [teacher fit {time.time()-t0:.0f}s]", flush=True)
    return _teacher_cache[key]


def predict(model: str, seed: int, bundle, Xtr, ytr, Xte):
    """Rebuild the trained model from its checkpoint and return test predictions."""
    cfg = json.load(open(record_path(model, seed)))["config"]
    rec = json.load(open(record_path(model, seed)))
    nc = bundle.num_classes
    dev = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    ck_dir = REPO / RESULT_DIR[model] / f"seed_{seed}"

    if model == "gbdt":
        teacher, _, present = get_teacher(seed, Xtr, ytr, nc, cfg)
        Xte_df = pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])])
        return present[teacher.predict(Xte_df)]

    if model in ("treespike", "dnn"):
        from src.models import TreeGuidedSpikeEncoder
        teacher, _, _ = get_teacher(seed, Xtr, ytr, nc, cfg)
        if model == "treespike":
            from src.models import SNNStudent
            from src.training_treespike import _encode_forward_in_batches
            enc = TreeGuidedSpikeEncoder(
                T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")
            ).fit(teacher, Xtr.shape[1], X=Xtr)
            st = SNNStudent(enc.out_dim, nc, hidden=tuple(cfg["hidden"]),
                            beta=cfg["beta"], leaky=cfg["leaky"]).to(dev)
            st.load_state_dict(torch.load(ck_dir / "checkpoint.pt", map_location=dev))
            with torch.no_grad():
                pred, *_ = _encode_forward_in_batches(st, enc, Xte, cfg["batch_size"], dev)
            return pred
        from src.models import DNNStudent
        from src.training_dnn import _forward_batches
        enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
                                     prune_quantile=cfg["prune_quantile"],
                                     jitter=cfg["jitter"]).fit(teacher, Xtr.shape[1])
        st = DNNStudent(enc.out_dim, nc, hidden=tuple(cfg["hidden"]),
                        clip_relu=cfg["clip_relu"]).to(dev)
        st.load_state_dict(torch.load(ck_dir / "checkpoint.pt", map_location=dev))
        with torch.no_grad():
            pred, _ = _forward_batches(st, enc, Xte, cfg["batch_size"], dev)
        return pred

    if model == "vanilla":
        from src.models import SNN
        from src.encoding import get_encoder
        from src.training import _eval
        enc = get_encoder(rec.get("encoding", "latency"))
        m = SNN(num_features=bundle.num_features, hidden_size=cfg["hidden_size"],
                num_classes=nc, neuron_name=rec.get("neuron", "LeakyParallel"),
                num_steps=cfg["num_steps"]).to(dev)
        m.load_state_dict(torch.load(ck_dir / "checkpoint.pt", map_location=dev))
        return _eval(m, bundle.test_loader, enc, cfg["num_steps"], dev)["y_pred"]

    from src.models import TASNN
    from src.training_tasnn import _eval as _eval_ta
    m = TASNN(num_features=bundle.num_features, num_classes=nc,
              num_steps=cfg["num_steps"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
              n_layers=cfg["n_layers"], ff=cfg["ff"], dropout=cfg["dropout"],
              pop_M=cfg["pop_M"], stc_channels=cfg["stc_channels"],
              stc_kernel=cfg["stc_kernel"], lif_hidden=tuple(cfg["lif_hidden"]),
              lif_beta=cfg["lif_beta"], lif_v_th=cfg["lif_v_th"],
              lif_gamma=cfg["lif_gamma"], surrogate_beta=cfg["surrogate_beta"],
              target_rate=cfg["target_rate"]).to(dev)
    m.load_state_dict(torch.load(ck_dir / "checkpoint.pt", map_location=dev))
    return _eval_ta(m, bundle.test_loader, dev)["y_pred"]


if __name__ == "__main__":
    from src.data import get_dataset
    from src.data.ciciomt2024 import _build_full_arrays, _category, BENIGN_FINE
    from src.data._label_modes import apply_label_mode
    from src.metrics import clear_eval_context
    clear_eval_context()

    print("hashing full provider arrays once ...", flush=True)
    Xtr_f, ytrf, Xte_f, ytef, fine = _build_full_arrays(REPO / "dataset/cic-iomt-2024")
    ytr_f, class_names = apply_label_mode("category", ytrf, fine, _category, BENIGN_FINE)
    yte_f, _ = apply_label_mode("category", ytef, fine, _category, BENIGN_FINE)
    htr = pd.util.hash_pandas_object(pd.DataFrame(np.asarray(Xtr_f)), index=False).to_numpy()
    hte = pd.util.hash_pandas_object(pd.DataFrame(np.asarray(Xte_f)), index=False).to_numpy()
    n_tr_f, n_te_f = len(htr), len(hte)
    del Xtr_f, Xte_f
    print(f"  train {n_tr_f}  test {n_te_f}  classes {class_names}", flush=True)

    rows, per_class_rows, t0 = [], [], time.time()
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        ktr = rng.choice(n_tr_f, int(n_tr_f * SUBSAMPLE), replace=False)
        kte = rng.choice(n_te_f, int(n_te_f * SUBSAMPLE), replace=False)
        y_pool = ytr_f[ktr]
        strat = y_pool if np.bincount(y_pool).min() >= 2 else None
        idx_tr, _ = train_test_split(np.arange(len(ktr)), test_size=VAL_FRACTION,
                                     random_state=seed, stratify=strat)
        train_rows = ktr[idx_tr]

        dup = np.asarray(pd.Index(hte[kte]).isin(pd.Index(htr[train_rows])), dtype=bool)
        dup_pool = np.asarray(pd.Index(hte[kte]).isin(pd.Index(htr[ktr])), dtype=bool)
        yte_seed = yte_f[kte]
        print(f"\n=== seed {seed}: test {len(kte)} | dup vs TRAIN partition "
              f"{int(dup.sum())} ({100*dup.mean():.3f}%) | vs train+val pool "
              f"{int(dup_pool.sum())} ({100*dup_pool.mean():.3f}%) ===", flush=True)

        bundle = get_dataset("ciciomt2024", batch_size=128, val_fraction=VAL_FRACTION,
                             subsample_fraction=SUBSAMPLE, label_mode="category",
                             seed=seed)
        Xte = bundle.test_loader.dataset.tensors[0].numpy()
        yte_b = bundle.test_loader.dataset.tensors[1].numpy()
        Xtr_b = bundle.train_loader.dataset.tensors[0].numpy()
        ytr_b = bundle.train_loader.dataset.tensors[1].numpy()
        assert len(yte_b) == len(kte), f"{len(yte_b)} != {len(kte)}"
        assert np.array_equal(yte_b, yte_seed), "bundle test labels != reproduced kte order"
        print(f"  bundle verified: test order matches kte, train {len(ytr_b)}", flush=True)

        for model in MODELS:
            try:
                pred = predict(model, seed, bundle, Xtr_b, ytr_b, Xte)
            except Exception as e:
                print(f"  [{model}] FAILED: {type(e).__name__}: {e}", flush=True)
                rows.append(dict(model=model, seed=seed, error=f"{type(e).__name__}: {e}"))
                continue
            full = metrics_block(yte_b, pred, bundle.num_classes, class_names)
            uniq = metrics_block(yte_b[~dup], pred[~dup], bundle.num_classes, class_names)
            dupo = (metrics_block(yte_b[dup], pred[dup], bundle.num_classes, class_names)
                    if dup.any() else None)
            ref = stored_f1(model, seed)
            drift = full["macro_f1"] - ref
            rows.append(dict(
                model=model, seed=seed, stored_macro_f1=ref,
                repro_macro_f1=full["macro_f1"], repro_drift_pp=drift,
                n_full=full["n"], n_unique=uniq["n"], n_removed=int(dup.sum()),
                pct_removed=100 * float(dup.mean()),
                macro_f1_before=full["macro_f1"], macro_f1_after=uniq["macro_f1"],
                d_macro_f1=uniq["macro_f1"] - full["macro_f1"],
                mcc_before=full["mcc"], mcc_after=uniq["mcc"],
                d_mcc=(uniq["mcc"] - full["mcc"]) if full["mcc"] is not None else None,
                bacc_before=full["bacc"], bacc_after=uniq["bacc"],
                d_bacc=uniq["bacc"] - full["bacc"],
                DR_before=full["DR"], DR_after=uniq["DR"], d_DR=uniq["DR"] - full["DR"],
                FAR_before=full["FAR"], FAR_after=uniq["FAR"], d_FAR=uniq["FAR"] - full["FAR"],
                dup_macro_f1=dupo["macro_f1"] if dupo else None,
                dup_bacc=dupo["bacc"] if dupo else None,
                dup_n=dupo["n"] if dupo else 0,
                full_block=full, unique_block=uniq, dup_block=dupo))
            for c in class_names:
                per_class_rows.append(dict(
                    model=model, seed=seed, cls=c,
                    support_before=full["support"][c], support_after=uniq["support"][c],
                    removed=full["support"][c] - uniq["support"][c],
                    recall_before=full["per_class_recall"][c],
                    recall_after=uniq["per_class_recall"][c],
                    f1_before=full["per_class_f1"][c], f1_after=uniq["per_class_f1"][c]))
            dup_txt = f" | dup-only {dupo['macro_f1']:.3f}" if dupo else ""
            print(f"  [{model:9s}] drift={drift:+.4f} | macroF1 {full['macro_f1']:.3f} -> "
                  f"{uniq['macro_f1']:.3f} ({uniq['macro_f1']-full['macro_f1']:+.3f} pp)"
                  f"{dup_txt}", flush=True)
        del bundle, Xte, Xtr_b
        print(f"  elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    df = pd.DataFrame([{k: v for k, v in r.items()
                        if not k.endswith("_block")} for r in rows])
    df.to_csv(OUT / "dupfree_reeval.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUT / "dupfree_reeval_per_class.csv", index=False)
    json.dump(rows, open(OUT / "dupfree_reeval_full.json", "w"), indent=2, default=str)

    print("\n" + "=" * 78)
    print("RECONSTRUCTION FIDELITY")
    print("=" * 78)
    ok = df[df.get("repro_drift_pp").notna()] if "repro_drift_pp" in df else df
    print(ok.groupby("model")["repro_drift_pp"].agg(["mean", "min", "max"]).round(4).to_string())
    print(f"cells exceeding +/-{TOL} pp: {int((ok.repro_drift_pp.abs() > TOL).sum())} of {len(ok)}")

    print("\n" + "=" * 78)
    print("DUPLICATE-FREE SENSITIVITY (mean over 5 seeds)")
    print("=" * 78)
    cols = ["macro_f1_before", "macro_f1_after", "d_macro_f1", "d_mcc", "d_bacc",
            "d_DR", "d_FAR", "pct_removed"]
    print(ok.groupby("model")[cols].mean().round(3).to_string())
    print("\nlargest |d_macro_f1| per model:")
    print(ok.groupby("model")["d_macro_f1"].apply(lambda s: s.abs().max()).round(3).to_string())

    print("\n" + "=" * 78)
    print("DUPLICATE ROWS vs UNIQUE ROWS (is the overlap 'easy'?)")
    print("=" * 78)
    print(ok.groupby("model")[["dup_macro_f1", "macro_f1_after", "dup_bacc"]]
            .mean().round(3).to_string())

    print("\n" + "=" * 78)
    print("PER-CLASS (mean over seeds+models)")
    print("=" * 78)
    pc = pd.DataFrame(per_class_rows)
    print(pc.groupby("cls")[["support_before", "removed", "recall_before", "recall_after",
                             "f1_before", "f1_after"]].mean().round(3).to_string())
    print(f"\nwrote {OUT}/dupfree_reeval.csv, dupfree_reeval_per_class.csv")
