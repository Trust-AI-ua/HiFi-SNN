#!/usr/bin/env python3
"""|E_j| and the T >= max_j|E_j|+1 bound, all 11 datasets, production vs compressed.

Two reporting modes, because they answer different questions:

  all-feature     fraction of ALL features violating the bound
  retained-only   fraction of features the encoder actually KEEPS that violate it

Lemma 2 concerns only retained features -- pruned ones never enter the encoder --
so the retained-only rate is the one the theory is about. The manuscript's current
claim is stated against the all-feature rate, which overstates the violation.

The compressed arm refits the teacher at max_depth=3 with fewer rounds. If a
depth-3 ensemble satisfies the bound where the depth-6 production teacher does
not, then the envelope-compressed teacher is also the better-conditioned one, and
that belongs beside the compression sweep rather than in an appendix.
"""
from __future__ import annotations
import inspect, json, sys
from pathlib import Path
import numpy as np, yaml

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
DATASETS = ["kddcup99","nslkdd","ctu13","iscxids2012","cicids2017","cse-cic-ids2018",
            "unswnb15","ciciot2023","ciciomt2024","edgeiiot","toniot"]

def stats(ds, depth=None, est=None):
    from src.data import get_dataset, DATASET_LOADERS
    from src.models import TreeGuidedSpikeEncoder
    from src.training_treespike import _fit_teacher, _arrays_from_bundle
    cfg = yaml.safe_load(open(REPO/f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    dk = dict(seed=42)
    for o in ("subsample_fraction","test_fraction","label_mode"):
        if o in cfg and o in lp: dk[o] = cfg[o]
    b = get_dataset(ds, **dk); Xtr, ytr, *_ = _arrays_from_bundle(b)
    c = dict(cfg); c["seed"] = 42
    if depth: c["teacher_max_depth"] = depth
    if est:   c["teacher_n_estimators"] = est
    teacher, _, _ = _fit_teacher(Xtr, ytr, b.num_classes, c)
    T = cfg["T"]
    enc = TreeGuidedSpikeEncoder(T=T, pop_max=cfg["pop_max"],
            prune_quantile=cfg["prune_quantile"], jitter=cfg["jitter"]).fit(
            teacher, Xtr.shape[1], X=Xtr)
    all_e = {j: len(v) for j, v in enc.edges_.items()}          # every split feature
    kept = sorted({j for j, _ in enc.channels_})                 # encoder-retained
    ka = np.array([all_e[j] for j in kept if j in all_e])
    aa = np.array(list(all_e.values()))
    viol = lambda a: float((a + 1 > T).mean()) if len(a) else float("nan")
    return dict(T=int(T), n_split_features=int(len(aa)), n_retained=int(len(ka)),
                max_all=int(aa.max()) if len(aa) else 0,
                max_retained=int(ka.max()) if len(ka) else 0,
                median_retained=float(np.median(ka)) if len(ka) else 0.0,
                violation_all=viol(aa), violation_retained=viol(ka))

if __name__ == "__main__":
    out = {}
    f = REPO/"results/edge_stats_all.json"
    if f.exists(): out = json.load(open(f))
    print(f"{'dataset':17s} {'arm':10s} {'T':>3} {'maxE':>6} {'medE':>6} "
          f"{'viol_all':>9} {'viol_kept':>10}")
    for ds in DATASETS:
        for arm, kw in (("production", {}), ("compressed", dict(depth=3, est=25))):
            k = f"{ds}|{arm}"
            if k in out: continue
            try: out[k] = stats(ds, **kw)
            except Exception as e:
                print(f"{ds:17s} {arm:10s} FAILED {str(e)[:35]}"); continue
            r = out[k]
            print(f"{ds:17s} {arm:10s} {r['T']:3d} {r['max_retained']:6d} "
                  f"{r['median_retained']:6.0f} {r['violation_all']*100:8.1f}% "
                  f"{r['violation_retained']*100:9.1f}%")
            json.dump(out, open(f, "w"), indent=2)
