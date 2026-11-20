#!/usr/bin/env python3
"""Amendment 8, part 4: encoder tables and multichannel collisions
(results/preregistration/2026-11-17_neuro_stream.md). Runs in .venv.

For one dataset and seed: refit the production teacher (thread variables unset, or OMP_NUM_THREADS equal
to --omp, the record's own cap), check it gives each record's stored teacher macro-F1, fit the encoder,
and check that the integer encoder reproduces the deployed spike times (results/neuro/inputs). Then, per
retained feature: ordered thresholds, importance, channels with offsets, and each channel's level ->
spike-time table, and the collisions of the emitted multichannel tuples.

Encoder rule (src/models/tree_to_spike.py, scripts/r2_integer_inference.py IntEncoder): level k = number
of thresholds <= x (0 .. |E_j|); channel c of feature j with offset o_c emits one spike at
t = round((1 - clip(k / |E_j| + o_c, 0, 1)) * (T - 1)) (float32 level, NumPy round: half to even);
offsets o_c = (c - (m_j - 1) / 2) * jitter, m_j = max(1, ceil(importance_j / max importance * pop_max)).
Two levels collide when every channel of the feature emits the same time for both.

    .venv/bin/python scripts/neuro_encoder_tables.py <dataset> <seed> [--omp N] [--skip-production] [<tag>=<record dir> ...]
Output: results/neuro/review/p3_encoders/<ds>_seed<S>_<tag>.json (tag production, or the variant tag)
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/neuro/review/p3_encoders"


def tables(enc, ienc, Xte, imp):
    k_te = ienc.index(Xte)
    feats = []
    for j in ienc.feats:
        chans = [c for c, (jj, _) in enumerate(ienc.channels) if jj == j]
        n = len(enc.edges_[j])
        tup = np.stack([ienc.spike_lut[c] for c in chans], 1)                   # [n + 1, m_j]
        _, inv, cnt = np.unique(tup, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        merged_level = cnt[inv] > 1                                              # level shares its tuple
        lv_count = np.bincount(k_te[j], minlength=n + 1)
        seen = lv_count > 0
        seen_per_tuple = np.bincount(inv, weights=seen, minlength=len(cnt))
        merged_seen = seen & (seen_per_tuple[inv] > 1)                           # shares it with another level seen in test
        feats.append(dict(
            feature=int(j), importance=float(imp[j]), n_thresholds=n, thresholds=[float(e) for e in enc.edges_[j]],
            channels=[dict(channel=int(c), offset=float(ienc.channels[c][1]), spike_time_by_level=ienc.spike_lut[c].tolist())
                      for c in chans],
            n_levels=n + 1, distinct_tuples=int(len(cnt)), levels_lost=int(n + 1 - len(cnt)),
            levels_lost_centre_channel_alone=int(n + 1 - len(np.unique(ienc.spike_lut[min(chans, key=lambda c: abs(ienc.channels[c][1]))]))),
            single_channel_bound_violated=bool(n >= enc.T), collides=bool(len(cnt) < n + 1),
            test_values_in_merged_tuple=float(lv_count[merged_level].sum() / len(Xte)),
            test_values_merged_with_seen_level=float(lv_count[merged_seen].sum() / len(Xte)),
            test_levels_seen=int(seen.sum())))
    return feats


def main():
    args = sys.argv[1:]
    omp = None
    if "--omp" in args:
        i = args.index("--omp"); omp = args[i + 1]; del args[i:i + 2]
    skip = "--skip-production" in args
    args = [a for a in args if a != "--skip-production"]
    if os.environ.get("OMP_NUM_THREADS") != omp or any(os.environ.get(v) for v in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise SystemExit("thread variables must be unset (production teacher), or OMP_NUM_THREADS must equal --omp")
    ds, seed = args[0], int(args[1])
    variants = dict(a.split("=", 1) for a in args[2:])
    import r2_integer_inference as RI
    from src.data import get_dataset, DATASET_LOADERS
    from src.metrics import compute_metrics
    from src.models.tree_to_spike import TreeGuidedSpikeEncoder
    from src.training_treespike import _arrays_from_bundle, _fit_teacher
    base = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    recs = {} if skip else {"production": (f"results/Tree-to-Spike/{ds}", f"{ds}_seed{seed}.npz")}
    for t, d in variants.items():
        recs[t] = (d, f"{ds}_{t}_seed{seed}.npz")
    cfgs = {}
    for t, (d, _) in recs.items():
        rec = json.loads((REPO / d / f"seed_{seed}/results.json").read_text())
        c = dict(base); c.update(rec["config"]); c["dataset"], c["seed"] = ds, seed
        cfgs[t] = (c, rec)
    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    c0 = next(iter(cfgs.values()))[0]
    kw = dict(batch_size=c0["batch_size"], val_fraction=c0["val_fraction"], seed=seed)
    for o in ("subsample_fraction", "test_fraction", "label_mode"):
        if o in c0 and o in lp and c0[o] is not None:
            kw[o] = c0[o]
    b = get_dataset(ds, **kw)
    Xtr, ytr, Xva, yva, Xte, yte = _arrays_from_bundle(b)
    K, F = b.num_classes, Xtr.shape[1]
    score = lambda y, p: compute_metrics(y, p, class_names=b.class_names)["f1_macro"]
    teacher, _, present = _fit_teacher(Xtr, ytr, K, c0)
    t_f1 = score(yte, present[teacher.predict(pd.DataFrame(Xte, columns=[f"f{i}" for i in range(F)]))])
    imp = np.asarray(teacher.feature_importances_, dtype=np.float64)
    names = getattr(b, "feature_names", None)
    OUT.mkdir(parents=True, exist_ok=True)
    for t, (cfg, rec) in cfgs.items():
        if t_f1 != rec["test_plus"]["teacher_upper_bound"]["f1_macro"]:
            raise SystemExit(f"{ds}/{seed}/{t}: refit teacher is not the stored teacher (OMP_NUM_THREADS={omp})")
        enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                     jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")).fit(teacher, F, X=Xtr)
        ienc = RI.IntEncoder(enc)
        z = np.load(REPO / "results/neuro/inputs" / recs[t][1])
        s = np.concatenate([ienc.spike_times(Xte[i:i + RI.CHUNK]) for i in range(0, len(Xte), RI.CHUNK)])
        if not np.array_equal(s.astype(np.int8), z["spike_times"]):
            raise SystemExit(f"{ds}/{seed}/{t}: encoder tables do not reproduce the deployed spike times")
        feats = tables(enc, ienc, Xte, imp)
        if names is not None:
            for f in feats:
                f["name"] = str(names[f["feature"]])
        n = len(feats)
        r = dict(dataset=ds, seed=seed, model=t, record=recs[t][0], T=int(enc.T), pop_max=cfg["pop_max"],
                 prune_quantile=cfg["prune_quantile"], jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree"),
                 teacher_omp_num_threads=omp, teacher_test_macro_f1=t_f1, n_features_input=int(F),
                 n_features_retained=n, n_channels=int(ienc.spike_lut.__len__()),
                 rule="level k = #thresholds <= x (IEEE-754 order); t_c = round((1 - clip(k/|E_j| + o_c, 0, 1)) * (T - 1)), "
                      "float32 level, NumPy round half to even; one spike per channel per flow",
                 summary=dict(share_single_channel_bound_violated=sum(f["single_channel_bound_violated"] for f in feats) / n,
                              share_features_colliding=sum(f["collides"] for f in feats) / n,
                              share_level_distinctions_lost=sum(f["levels_lost"] for f in feats) / sum(f["n_levels"] for f in feats),
                              share_level_distinctions_lost_centre_channel_alone=sum(f["levels_lost_centre_channel_alone"] for f in feats)
                              / sum(f["n_levels"] for f in feats),
                              mean_test_values_in_merged_tuple=float(np.mean([f["test_values_in_merged_tuple"] for f in feats])),
                              mean_test_values_merged_with_seen_level=float(np.mean([f["test_values_merged_with_seen_level"] for f in feats]))),
                 check="integer encoder from these tables == deployed spike times on the whole test split; teacher == stored",
                 features=feats)
        (OUT / f"{ds}_seed{seed}_{t}.json").write_text(json.dumps(r, indent=1))
        print(ds, seed, t, json.dumps({k: round(v, 4) for k, v in r["summary"].items()}), flush=True)


if __name__ == "__main__":
    main()
