#!/usr/bin/env python3
"""Matched false-alarm check: validation-selected compact GBDT against Tree-to-Spike
(revision 2, step 3b; results/preregistration/2026-11-17_r2_matched_far.md).

The GBDT is the configuration selected in step 3 (results/journal_r2/gbdt_valsel), refit with
its leaves quantized as packed. Its benign threshold t (predict benign if p_benign >= t, else
the most probable attack class) is set on validation:
  (A) to Tree-to-Spike's validation FAR for the same seed,
  (B) to FAR <= 1%.
(C) Threshold-free: benign-versus-attack ROC AUC of both models on test, and the GBDT's test
    DR at Tree-to-Spike's test FAR (a readout on the test ROC curve).

    python scripts/r2_matched_far.py <dataset> <seed>
Output: results/journal_r2/matched_far/<dataset>/seed_<S>.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
warnings.filterwarnings("ignore")
OUT = REPO / "results/journal_r2/matched_far"
FAR_GRID = [0.001, 0.0025, 0.005, 0.01, 0.02]
CHUNK = 32768


def threshold_for(pb_benign: np.ndarray, f: float) -> float:
    """Largest t with mean(pb_benign < t) <= f."""
    v = np.sort(pb_benign)
    k = int(np.floor(f * len(v) + 1e-9))
    return float(v[k]) if k < len(v) else float(np.nextafter(v[-1], np.inf))


def thresholded(P: np.ndarray, t: float) -> np.ndarray:
    """Local-index predictions: benign (0) if p_b >= t, else the most probable attack class."""
    att = 1 + P[:, 1:].argmax(1)
    return np.where(P[:, 0] >= t, 0, att)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("seed", type=int)
    ap.add_argument("--sym", action="store_true", help="amendment 1: (D) and (E) as well")
    a = ap.parse_args()
    ds, seed = a.dataset, a.seed
    out = (OUT.parent / "matched_far_sym" if a.sym else OUT) / ds / f"seed_{seed}.json"
    if out.exists():
        print(f"[skip] {out}"); return
    if any(os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise SystemExit("thread variables are set; the teacher would not be the production teacher")
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from src.data import get_dataset, DATASET_LOADERS
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    from src.models.tree_to_spike import TreeGuidedSpikeEncoder, SNNStudent
    from src.training_treespike import _arrays_from_bundle, _fit_teacher
    import r2_gbdt_valsel as G
    t0 = time.time()
    rec_t = json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/results.json").read_text())
    rec_g = json.loads((REPO / f"results/journal_r2/gbdt_valsel/{ds}/seed_{seed}.json").read_text())
    sel = rec_g["selected"]
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    cfg.update(rec_t["config"]); cfg["dataset"], cfg["seed"] = ds, seed
    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=seed)
    for o in ("subsample_fraction", "test_fraction", "label_mode"):
        if o in cfg and o in lp and cfg[o] is not None:
            kw[o] = cfg[o]
    b = get_dataset(ds, **kw)
    Xtr, ytr, Xva, yva, Xte, yte = _arrays_from_bundle(b)
    K, F = b.num_classes, Xtr.shape[1]
    cols = [f"f{i}" for i in range(F)]
    score = lambda y, p: compute_metrics(y, p, class_names=b.class_names)

    def ids(y, p):
        m, bm = score(y, p), compute_binary_ids_metrics(y, p, normal_class_index=0)
        return dict(macro_f1=100 * m["f1_macro"], far=100 * bm["false_alarm_rate"], dr=100 * bm["detection_rate"])

    # --- Tree-to-Spike, float path (the reported model) ---
    teacher, _, present_t = _fit_teacher(Xtr, ytr, K, cfg)
    tf1 = score(yte, present_t[teacher.predict(pd.DataFrame(Xte, columns=cols))])["f1_macro"]
    if tf1 != rec_t["test_plus"]["teacher_upper_bound"]["f1_macro"]:
        raise SystemExit(f"{ds}/{seed}: refit teacher {tf1!r} is not the stored teacher")
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                 jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")).fit(teacher, F, X=Xtr)
    st = SNNStudent(enc.out_dim, K, hidden=tuple(cfg["hidden"]), beta=cfg["beta"], leaky=cfg["leaky"])
    st.load_state_dict(torch.load(REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/checkpoint.pt", map_location="cpu"))
    st.eval()

    def t2s(X):
        pred, pb = [], []
        for i in range(0, len(X), CHUNK):
            with torch.no_grad():
                z = st(enc.encode(X[i:i + CHUNK]))[0]
            pred.append(z.argmax(1).numpy()); pb.append(torch.softmax(z, 1).numpy())
        return np.concatenate(pred), np.concatenate(pb)
    pt_va, Pt_va = t2s(Xva)
    pt_te, Pt_te = t2s(Xte)
    pbt_te = Pt_te[:, 0]
    got = score(yte, pt_te)["f1_macro"]
    if got != rec_t["test_plus"]["metrics"]["f1_macro"]:
        raise SystemExit(f"{ds}/{seed}: Tree-to-Spike {got!r} does not reproduce the record")
    T_va, T_te = ids(yva, pt_va), ids(yte, pt_te)

    # --- GBDT, the configuration selected in step 3 ---
    present = np.unique(ytr)
    if present[0] != 0:
        raise SystemExit(f"{ds}/{seed}: benign class absent from training")
    remap = np.full(K, -1, np.int64); remap[present] = np.arange(len(present))
    cnt = np.bincount(remap[ytr], minlength=len(present)).astype(np.float32)
    w = (cnt.sum() / (cnt + 1e-6))[remap[ytr]]
    k_units = len(present) if len(present) > 2 else 1
    if k_units == 1:
        raise SystemExit(f"{ds}: binary label set; the benign-threshold rule assumes multiclass")
    m = G.fit(Xtr, remap[ytr], w, sel["rounds"], sel["max_depth"], seed)
    qb = G.quantized_booster(m, sel["rounds"], k_units, sel["leaf_bits"])
    prob = lambda X: qb.predict(xgb.DMatrix(pd.DataFrame(X, columns=cols)))
    P_va, P_te = prob(Xva), prob(Xte)
    g_arg = ids(yte, present[P_te.argmax(1)])
    if abs(g_arg["macro_f1"] - sel["test_macro_f1"]) > 1e-9:
        raise SystemExit(f"{ds}/{seed}: refit GBDT {g_arg['macro_f1']!r} is not the selected {sel['test_macro_f1']!r}")
    ben_va = yva == 0
    res = {}
    for name, f in (("A_matched", T_va["far"] / 100), ("B_far1", 0.01)):
        t = threshold_for(P_va[ben_va, 0], f)
        res[name] = dict(target_far=100 * f, threshold=t, val=ids(yva, present[thresholded(P_va, t)]),
                         test=ids(yte, present[thresholded(P_te, t)]))
    # (C) threshold-free readouts on test
    att_te = (yte != 0).astype(int)
    t_c = threshold_for(P_te[yte == 0, 0], T_te["far"] / 100)
    res["C_test_roc"] = dict(auc_gbdt=float(roc_auc_score(att_te, 1 - P_te[:, 0])),
                             auc_t2s=float(roc_auc_score(att_te, 1 - pbt_te)),
                             gbdt_at_t2s_test_far=ids(yte, present[thresholded(P_te, t_c)]))
    if a.sym:   # amendment 1: Tree-to-Spike under the same benign-threshold rule, and the FAR curve
        def tuned(Pv, Pt, f):
            t = threshold_for(Pv[ben_va, 0], f)
            return dict(target_far=100 * f, threshold=t, val=ids(yva, thresholded(Pv, t)), test=ids(yte, thresholded(Pt, t)))
        if not np.array_equal(present_t, np.arange(K)):
            raise SystemExit(f"{ds}/{seed}: Tree-to-Spike class set is not 0..K-1")
        res["D_t2s_far1"] = tuned(Pt_va, Pt_te, 0.01)
        res["E_curve"] = [dict(target_far=100 * f, gbdt=tuned(P_va, P_te, f)["test"], t2s=tuned(Pt_va, Pt_te, f)["test"])
                          for f in FAR_GRID]
        # thresholded() returns local indices; GBDT entries are remapped below
        for e, f in zip(res["E_curve"], FAR_GRID):
            t = threshold_for(P_va[ben_va, 0], f)
            e["gbdt"] = ids(yte, present[thresholded(P_te, t)])
    rec = dict(dataset=ds, seed=seed, gbdt_config=sel, n_val=int(len(yva)), n_val_benign=int(ben_va.sum()),
               n_test=int(len(yte)), t2s=dict(val=T_va, test=T_te), gbdt_argmax_test=g_arg, **res,
               preregistration="results/preregistration/2026-11-17_r2_matched_far.md",
               seconds=round(time.time() - t0, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    A, B = res["A_matched"]["test"], res["B_far1"]["test"]
    print(f"[done] {ds} s{seed}: T2S F1 {T_te['macro_f1']:.2f} FAR {T_te['far']:.2f} DR {T_te['dr']:.2f} | "
          f"GBDT argmax F1 {g_arg['macro_f1']:.2f} FAR {g_arg['far']:.2f} | matched F1 {A['macro_f1']:.2f} "
          f"FAR {A['far']:.2f} DR {A['dr']:.2f} | FAR<=1% F1 {B['macro_f1']:.2f} FAR {B['far']:.2f} DR {B['dr']:.2f} | "
          f"AUC {res['C_test_roc']['auc_gbdt']:.4f} vs {res['C_test_roc']['auc_t2s']:.4f} [{rec['seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
