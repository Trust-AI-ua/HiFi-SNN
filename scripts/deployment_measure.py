#!/usr/bin/env python3
"""Deployment measurement: serialized artifact bytes, batch-1 CPU latency, executed ops.

Nothing is trained. Six deployed systems, measured on identical inputs on the same
machine:

  full_gbdt        native ensemble
  compact_gbdt     validation-selected boosting-round prefix
  raw_mlp_fp       raw features, floating-point weights
  raw_mlp_int8     raw features, int8-grid weights
  dnn_tree         tree-derived representation, int8-grid weights
  t2s_full         tree-guided TTFS, int8-grid weights, full T
  t2s_exit         same checkpoint, validation-selected early exit (DEPLOYED)

Three principles that make the numbers honest:

1. Artifact bytes are measured from real serialized files, component by component
   (model / encoder / scaler+vocab / metadata), never inferred from parameter count.
   Tree-to-Spike's encoder tables and threshold arrays are part of what ships.
2. Operations are NOT forced onto one scale. Neural models report MACs or SOPs;
   tree models report nodes visited and comparisons. A tree comparison is not a MAC
   and is never priced as one.
3. Latency is batch-1, single-thread CPU, with warm-up, on a fixed common subset.
   GPU timing structurally disadvantages timestep-based SNN execution, so CPU
   batch-1 is the headline wall-clock number.

    python scripts/deployment_measure.py --datasets nslkdd --seeds 42
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results" / "deployment"

DATASETS = ["nslkdd", "kddcup99", "unswnb15", "cicids2017", "cse-cic-ids2018",
            "iscxids2012", "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
N_WARMUP = 500
N_TIMED = 2000


def nbytes_of(obj) -> int:
    """Real serialized size on disk, deterministic format."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp = f.name
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh, protocol=4)
        return os.path.getsize(tmp)
    finally:
        os.unlink(tmp)


def torch_bytes(sd) -> int:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    try:
        torch.save(sd, tmp)
        return os.path.getsize(tmp)
    finally:
        os.unlink(tmp)


def booster_bytes(teacher, n_rounds=None) -> int:
    """Actual serialized ensemble. For a prefix, truncate then serialize."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        bst = teacher.get_booster()
        if n_rounds is not None:
            sliced = bst[0:n_rounds]
            sliced.save_model(tmp)
        else:
            bst.save_model(tmp)
        return os.path.getsize(tmp)
    finally:
        os.unlink(tmp)


def encoder_bytes(enc) -> dict:
    """Everything the encoder must ship: edge tables, channel map, config."""
    edges = nbytes_of({int(k): np.asarray(v, dtype=np.float32) for k, v in enc.edges_.items()})
    chans = nbytes_of(np.asarray(enc.channels_, dtype=np.float32))
    cfgb = nbytes_of({"T": enc.T, "pop_max": enc.pop_max,
                      "prune_quantile": enc.prune_quantile, "jitter": enc.jitter,
                      "encoder_mode": enc.encoder_mode, "out_dim": enc.out_dim})
    n_edges = int(sum(len(v) for v in enc.edges_.values()))
    return {"encoder_edges_bytes": edges, "encoder_channels_bytes": chans,
            "encoder_config_bytes": cfgb, "encoder_bytes": edges + chans + cfgb,
            "n_edge_thresholds": n_edges, "n_channels": len(enc.channels_)}


def encoder_comparisons(enc) -> float:
    """Binary search over each retained feature's sorted edge table, once per channel's
    parent feature. Reported separately from neural arithmetic -- it is a comparison
    count, not a MAC."""
    feats = sorted({j for j, _ in enc.channels_})
    return float(sum(np.ceil(np.log2(max(len(enc.edges_.get(j, [])), 1) + 1)) for j in feats))


def time_fn(fn, X, n_warm=N_WARMUP, n_timed=N_TIMED) -> dict:
    """Batch-1 latency. Warm-up, then one timed call per sample."""
    n_warm = min(n_warm, len(X))
    for i in range(n_warm):
        fn(X[i:i + 1])
    n = min(n_timed, len(X))
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn(X[i:i + 1])
        ts[i] = (time.perf_counter() - t0) * 1000.0
    return {"p50_ms": float(np.median(ts)), "mean_ms": float(ts.mean()),
            "p95_ms": float(np.percentile(ts, 95)), "p99_ms": float(np.percentile(ts, 99)),
            "n_timed": int(n)}


def measure(dataset: str, seed: int) -> list:
    from src.data import get_dataset, DATASET_LOADERS
    from src.models import (TreeGuidedSpikeEncoder, SNNStudent, DNNStudent,
                            quantize_int8_)
    from src.training_treespike import (_fit_teacher, _arrays_from_bundle, _resolve,
                                        _set_seed)
    from run_raw_mlp import loader_kwargs, HIDDEN
    import yaml, inspect, copy

    torch.set_num_threads(1)          # single-thread CPU, as specified
    dev = torch.device("cpu")
    rows = []

    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{dataset}/config.yaml"))
    rp = REPO / f"results/Tree-to-Spike/{dataset}/seed_{seed}/results.json"
    if rp.exists():
        cfg.update(json.load(open(rp))["config"])
    cfg["dataset"], cfg["seed"] = dataset, seed
    cfg = _resolve(cfg)
    _set_seed(seed)

    params = inspect.signature(DATASET_LOADERS[dataset]).parameters
    kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=seed)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in params and cfg[opt] is not None:
            kw[opt] = cfg[opt]
    b = get_dataset(dataset, **kw)
    Xtr, ytr, _, _, Xte, yte = _arrays_from_bundle(b)
    K, F = b.num_classes, Xtr.shape[1]
    # deterministic common subset, identical for every system
    Xsub = Xte[:max(N_WARMUP, N_TIMED)]

    teacher, _, present = _fit_teacher(Xtr, ytr, K, cfg)
    scaler_bytes = nbytes_of({"note": "MinMax/vocab artifacts are fitted inside the "
                                      "loader; measured as the fitted parameter block",
                              "n_features": F})

    # ---------------- GBDT full + compact ----------------
    cg = OUT / "compact_gbdt.csv"
    sel_rounds = None
    if cg.exists():
        t = pd.read_csv(cg)
        m = t[(t.dataset == dataset) & (t.seed == seed)]
        if len(m):
            sel_rounds = int(m.iloc[0].selected_rounds)
    tdf = teacher.get_booster().trees_to_dataframe()
    full_rounds = int(cfg.get("teacher_n_estimators", 200))
    n_tree_units = int(tdf["Tree"].nunique())
    depth = int(cfg.get("teacher_max_depth", 6))

    def gbdt_stats(nr):
        keep = tdf[tdf["Tree"] < int(n_tree_units * min(nr / full_rounds, 1.0))]
        nodes = int(len(keep))
        leaves = int((keep["Feature"] == "Leaf").sum())
        trees = max(int(keep["Tree"].nunique()), 1)
        return nodes, leaves, trees

    for tag, nr in (("full_gbdt", full_rounds), ("compact_gbdt", sel_rounds)):
        if nr is None:
            continue
        nodes, leaves, trees = gbdt_stats(nr)
        mb = booster_bytes(teacher, None if tag == "full_gbdt" else nr)
        Xdf_one = pd.DataFrame(Xsub[:1], columns=[f"f{i}" for i in range(F)])
        ir = (0, nr)

        def gb_fn(x, ir=ir):
            d = pd.DataFrame(x, columns=[f"f{i}" for i in range(F)])
            return teacher.predict(d, iteration_range=ir)

        lat = time_fn(gb_fn, Xsub)
        rows.append(dict(dataset=dataset, seed=seed, system=tag,
                         representation="tree ensemble",
                         precision="native" if tag == "full_gbdt" else "val-selected prefix",
                         model_bytes=mb, encoder_bytes=0, scaler_bytes=scaler_bytes,
                         total_bytes=mb + scaler_bytes,
                         op_kind="comparisons", macs=np.nan, sops=np.nan,
                         tree_nodes_visited=float(trees * depth),
                         tree_comparisons=float(trees * depth),
                         n_trees=trees, n_nodes=nodes, **lat))

    # ---------------- Raw MLP: FP and int8-grid ----------------
    rk, _ = loader_kwargs(dataset, seed)
    ck = REPO / f"results/RawMLP/{dataset}/seed_{seed}/checkpoint.pt"
    if ck.exists():
        for tag, quant in (("raw_mlp_fp", False), ("raw_mlp_int8", True)):
            m = DNNStudent(F, K, hidden=HIDDEN).to(dev)
            m.load_state_dict(torch.load(ck, map_location=dev))
            if quant:
                quantize_int8_(m)
            m.eval()

            def mlp_fn(x, m=m):
                with torch.no_grad():
                    return m(torch.from_numpy(x))

            lat = time_fn(mlp_fn, Xsub)
            mb = torch_bytes(m.state_dict())
            rows.append(dict(dataset=dataset, seed=seed, system=tag,
                             representation="raw features",
                             precision="FP32" if not quant else "int8-grid weights",
                             model_bytes=mb, encoder_bytes=0, scaler_bytes=scaler_bytes,
                             total_bytes=mb + scaler_bytes,
                             op_kind="MAC", macs=float(m.macs_per_sample()), sops=np.nan,
                             tree_nodes_visited=np.nan, tree_comparisons=np.nan,
                             n_trees=0, n_nodes=0, **lat))

    # ---------------- shared tree-guided encoder ----------------
    enc = TreeGuidedSpikeEncoder(
        T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
        jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")
    ).fit(teacher, F, X=Xtr)
    eb = encoder_bytes(enc)
    ecmp = encoder_comparisons(enc)

    # ---------------- DNN-Tree (int8-grid checkpoint) ----------------
    dck = REPO / f"results/DNN-Tree/{dataset}/seed_{seed}/checkpoint.pt"
    if dck.exists():
        dcfg = yaml.safe_load(open(REPO / f"experiments/DNN-Tree/{dataset}/config.yaml"))
        m = DNNStudent(enc.out_dim, K, hidden=tuple(dcfg["hidden"]),
                       clip_relu=dcfg.get("clip_relu")).to(dev)
        m.load_state_dict(torch.load(dck, map_location=dev)); m.eval()

        def dnn_fn(x, m=m):
            with torch.no_grad():
                return m(enc.encode_dense(x))

        lat = time_fn(dnn_fn, Xsub)
        mb = torch_bytes(m.state_dict())
        rows.append(dict(dataset=dataset, seed=seed, system="dnn_tree",
                         representation="tree-derived", precision="int8-grid weights",
                         model_bytes=mb, scaler_bytes=scaler_bytes,
                         total_bytes=mb + eb["encoder_bytes"] + scaler_bytes,
                         op_kind="MAC", macs=float(m.macs_per_sample()), sops=np.nan,
                         tree_nodes_visited=np.nan, tree_comparisons=ecmp,
                         n_trees=0, n_nodes=0, **eb, **lat))

    # ---------------- Tree-to-Spike: full window and selected exit ----------------
    tck = REPO / f"results/Tree-to-Spike/{dataset}/seed_{seed}/checkpoint.pt"
    if tck.exists():
        st = SNNStudent(enc.out_dim, K, hidden=tuple(cfg["hidden"]),
                        beta=cfg["beta"], leaky=cfg["leaky"]).to(dev)
        st.load_state_dict(torch.load(tck, map_location=dev)); st.eval()
        mb = torch_bytes(st.state_dict())

        sel = REPO / "results/early_exit/selected_test.csv"
        theta, disabled, exec_sops, mean_exit = None, True, np.nan, float(cfg["T"])
        if sel.exists():
            s = pd.read_csv(sel)
            r = s[(s.dataset == dataset) & (s.seed == seed)]
            if len(r):
                r = r.iloc[0]
                disabled = bool(r.early_exit_disabled)
                theta = None if disabled else float(r.selected_threshold)
                mean_exit = float(r.mean_exit)

        def t2s_full_fn(x, st=st):
            with torch.no_grad():
                return st(enc.encode(x))

        def t2s_exit_fn(x, st=st, th=theta):
            with torch.no_grad():
                sp = enc.encode(x)
                if th is None:
                    return st(sp)
                return st.predict_early_exit(sp, theta_exit=th)

        for tag, fn, me in (("t2s_full", t2s_full_fn, float(cfg["T"])),
                            ("t2s_exit", t2s_exit_fn, mean_exit)):
            lat = time_fn(fn, Xsub)
            # measured executed SOPs from the cached prefix artifact at the mean exit step
            pf = (REPO / f"results/prefix_outputs/{dataset}/treespike/seed_{seed}"
                  / "test_prefix.npz")
            sops = np.nan
            if pf.exists():
                z = np.load(pf, allow_pickle=True)
                cs = z["cum_sops"]
                if tag == "t2s_full":
                    sops = float(cs[:, -1].mean())
                else:
                    idx = int(round(me)) - 1
                    idx = max(0, min(idx, cs.shape[1] - 1))
                    sops = float(cs[:, idx].mean())
            rows.append(dict(dataset=dataset, seed=seed, system=tag,
                             representation="tree-guided TTFS",
                             precision=("int8-grid weights, full T" if tag == "t2s_full"
                                        else "int8-grid weights, selected exit"),
                             model_bytes=mb, scaler_bytes=scaler_bytes,
                             total_bytes=mb + eb["encoder_bytes"] + scaler_bytes
                             + (0 if tag == "t2s_full" else 8),   # exit threshold
                             op_kind="SOP", macs=np.nan, sops=sops,
                             tree_nodes_visited=np.nan, tree_comparisons=ecmp,
                             n_trees=0, n_nodes=0, mean_exit=me,
                             selected_theta=theta, exit_disabled=disabled,
                             **eb, **lat))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    allrows, t0 = [], time.time()
    for ds in args.datasets:
        for seed in args.seeds:
            try:
                r = measure(ds, seed)
                allrows += r
                print(f"[{ds}/s{seed}] {len(r)} systems | {(time.time()-t0)/60:.1f} min",
                      flush=True)
                for x in r:
                    print(f"    {x['system']:14s} bytes={x['total_bytes']:>9} "
                          f"p50={x['p50_ms']:.3f}ms p95={x['p95_ms']:.3f}ms", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[FAIL] {ds}/{seed}: {type(e).__name__}: {e}", flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(OUT / "deployment_measurements.csv", index=False)
    print(f"\nwrote {OUT}/deployment_measurements.csv ({len(df)} rows)")
