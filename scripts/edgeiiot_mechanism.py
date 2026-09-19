#!/usr/bin/env python3
"""Is Edge-IIoTset a second mechanism, or the SAME attractor operating pairwise?

Dataset-level modal share (50.5%) says "diffuse", but that aggregate is computed
over ALL wrong exits. If each collapsing class has its OWN concentrated absorber,
the aggregate looks diffuse purely because it sums over pairs -- and the mechanism
is identical to CICIoT2023, just distributed. Test 1 resolves this by measuring
concentration PER SOURCE CLASS.

Test 2 asks whether the encoder is protocol-blind. Edge-IIoTset's DDoS variants
differ only in transport (TCP/ICMP/UDP/HTTP). If gain-based pruning discards the
protocol-distinguishing features in favour of volumetric ones, the prefix cannot
separate the siblings by construction -- which would make the encoder's own
importance budgeting the cause of the early-exit failure, tying this section back
to the encoder ablation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from scripts.early_exit_audit import replay

DS = "edgeiiot"; TH = 0.90

def feature_names():
    """Rebuild the feature-name list exactly as the loader constructs X."""
    from src.data.edgeiiot import CSV_REL, DROP_COLUMNS, LABEL_COLUMNS
    p = REPO/"dataset/edge-iiotset"/CSV_REL
    df = pd.read_parquet(REPO/"dataset/edge-iiotset/_leakage_raw.parquet") \
         if (REPO/"dataset/edge-iiotset/_leakage_raw.parquet").exists() \
         else pd.read_csv(p, nrows=200000, low_memory=False)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.dropna(axis=0, how="any").drop_duplicates()
    df = df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])
    obj = df.select_dtypes(include="object").columns.tolist()
    num = [c for c in df.columns if c not in obj]
    names = list(num)
    if obj:
        names += list(pd.get_dummies(df[obj].astype(str), dummy_na=False).columns)
    return names

if __name__ == "__main__":
    conf, pred, y, names = replay(DS)
    T = conf.shape[1]; N = len(y)
    fired = conf >= TH; ex = fired.any(1)
    t_star = np.where(ex, fired.argmax(1), T-1)
    pe = pred[np.arange(N), t_star]

    print("="*78)
    print("TEST 1 -- per-class wrong-exit concentration (is 'diffuse' an artifact?)")
    print("="*78)
    print(f"{'source class':16s} {'n_wrong':>8} {'top target':>16} {'share':>7} "
          f"{'top2':>6} {'exit step sd':>13}")
    rows=[]
    for c in np.unique(y):
        m = (y == c) & ex & (pe != y)
        if m.sum() < 30: continue
        tgt, cnt = np.unique(pe[m], return_counts=True)
        o = np.argsort(-cnt); sh = cnt[o[0]]/cnt.sum()
        top2 = cnt[o[:2]].sum()/cnt.sum()
        sd = t_star[m].std()
        rows.append((names[c], int(m.sum()), names[int(tgt[o[0]])], sh, top2, sd))
        print(f"{names[c][:16]:16s} {m.sum():8d} {names[int(tgt[o[0]])][:16]:>16} "
              f"{sh*100:6.1f}% {top2*100:5.1f}% {sd:13.2f}")
    if rows:
        mean_sh = np.mean([r[3] for r in rows])
        print(f"\n  mean PER-CLASS top-target share: {mean_sh*100:.1f}%  "
              f"(dataset-level modal share was 50.5%)")
        print(f"  -> {'PAIRWISE ATTRACTORS (same mechanism, distributed)' if mean_sh>0.7 else 'GENUINELY DIFFUSE (second mechanism)'}")

    print("\n" + "="*78)
    print("TEST 2 -- is the encoder protocol-blind?")
    print("="*78)
    import yaml
    from src.data import get_dataset
    from src.models import TreeGuidedSpikeEncoder
    from src.training_treespike import _arrays_from_bundle, _fit_teacher
    cfg = yaml.safe_load(open(REPO/f"experiments/Tree-to-Spike/{DS}/config.yaml"))
    dk = dict(seed=42)
    for o in ("subsample_fraction","test_fraction","label_mode"):
        if o in cfg: dk[o] = cfg[o]
    b = get_dataset(DS, **dk); Xtr, ytr, *_ = _arrays_from_bundle(b)
    teacher,_,_ = _fit_teacher(Xtr, ytr, b.num_classes, {**cfg,"seed":42})
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
            prune_quantile=cfg["prune_quantile"], jitter=cfg["jitter"]).fit(
            teacher, Xtr.shape[1], X=Xtr)
    fn = feature_names()
    kept = sorted({j for j,_ in enc.channels_})
    print(f"  features total {Xtr.shape[1]} | names recovered {len(fn)} | kept {len(kept)}")
    if len(fn) >= Xtr.shape[1]:
        proto_kw = ("tcp","udp","icmp","http","arp","mqtt","dns")
        keptn = [fn[j] for j in kept if j < len(fn)]
        allp = [n for n in fn if any(k in n.lower() for k in proto_kw)]
        keptp = [n for n in keptn if any(k in n.lower() for k in proto_kw)]
        print(f"  protocol-related features: {len(allp)} present, {len(keptp)} retained "
              f"({len(keptp)/max(len(allp),1)*100:.0f}%)")
        for kw in proto_kw:
            a = [n for n in fn if kw in n.lower()]; k_ = [n for n in keptn if kw in n.lower()]
            if a: print(f"    {kw:5s}: {len(k_):3d}/{len(a):3d} retained")
    else:
        print("  (name reconstruction incomplete; skipping protocol breakdown)")
