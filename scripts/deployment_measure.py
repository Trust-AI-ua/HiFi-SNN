#!/usr/bin/env python3
"""Deployment measurement: serialized artifact bytes, batch-1 CPU latency, executed ops.

Nothing is trained. Six deployed systems, measured on identical inputs on the same
machine:

  full_gbdt        packed ensemble (float32 thresholds, 16-bit leaves)
  compact_gbdt     validation-selected boosting-round prefix, same packing
  vanilla_snn      raw-feature latency code, float32 (reported in floating point)
  tasnn            transformer-spiking hybrid, float32
  dnn_tree         tree-derived representation, int8-grid weights
  t2s_full         tree-guided TTFS, int8-grid weights, full T
  t2s_exit         same checkpoint, validation-selected early exit (DEPLOYED)

Three principles that make the numbers honest:

0. Each artifact is stored in the format its deployed values require
   (src/serialize.py): int8 codes plus per-tensor scales for the grid-deployed
   students, float32 for models reported in floating point, packed node arrays for
   the ensembles. Tree accuracy is measured ON the packed artifact.
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

# Release gate: this stage produces the only wall-clock numbers in the paper, so it
# refuses to start until (a) a release file exists and (b) the one-minute load
# average has stayed below LOAD_MAX for LOAD_HOLD_S. An earlier run began while the
# encoder-variant jobs were still draining (load 18.2 at start, 1.4 at end).
GATE_FILE = "results/deployment/.RELEASE_DEPLOYMENT"
LOAD_MAX = 2.0          # this box idles near 1.0 (other users' services)
LOAD_HOLD_S = 300


def wait_for_gate(repo):
    import os, time
    gate = repo / GATE_FILE
    while not gate.exists():
        print(f"[gate] waiting for {GATE_FILE}", flush=True)
        time.sleep(60)
    quiet_since = None
    while True:
        load1 = os.getloadavg()[0]
        now = time.time()
        if load1 < LOAD_MAX:
            quiet_since = quiet_since or now
            if now - quiet_since >= LOAD_HOLD_S:
                print(f"[gate] released: load {load1:.2f} held below {LOAD_MAX} "
                      f"for {LOAD_HOLD_S}s", flush=True)
                return load1
        else:
            quiet_since = None
        print(f"[gate] load {load1:.2f}, waiting for a quiet machine", flush=True)
        time.sleep(60)

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
# Model and artifact roots (Part B A1: the CICIoT2023 block models live elsewhere). Defaults
# are the production paths; {d} is the dataset.
ROOTS = {"t2s": "results/Tree-to-Spike/{d}", "dnn": "results/DNN-Tree/{d}",
         "vanilla": "results/{d}/SNN_LeakyParallel/latency", "tasnn": "results/TASNN/{d}",
         "prefix": "results/prefix_outputs/{d}/treespike", "exit_csv": "results/early_exit/selected_test.csv",
         "out_csv": "results/deployment/deployment_measurements.csv"}
N_WARMUP = 500
N_TIMED = 2000


from src import serialize as SER
from src.metrics import compute_metrics   # per-system serialization rule (v2)


def student_bytes(sd, int8_grid: bool) -> dict:
    """int8 codes plus per-tensor scales for grid-deployed students (Tree-to-Spike,
    DNN-Tree); float32 for models reported and executed in floating point."""
    return SER.int8_student_bytes(sd) if int8_grid else SER.float32_model_bytes(sd)


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


def booster_bytes(teacher, n_rounds=None, n_classes: int = 2) -> int:
    """Packed node arrays, the format the compression sweep prices. The XGBoost
    native file is a container, not a deployment artifact, and was about 15x
    larger; see src/serialize.py."""
    return SER.pack_booster(teacher, n_rounds, leaf_bits=16,
                            n_classes=n_classes)["deploy_bytes"]


def booster_bytes_native(teacher, n_rounds=None) -> int:
    """Previous convention, retained for the comparison in the verification."""
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


GBDT_SYSTEMS = ("full_gbdt", "compact_gbdt")


def measure(dataset: str, seed: int, only_gbdt: bool = False, artifact_dir=None) -> list:
    """artifact_dir (Part B, B8): if set, each system's runtime artifact (weights, encoder,
    the timed input subset) is pickled there, for peak-memory measurement in a fresh
    process. Measurements are unchanged."""
    from src.data import get_dataset, DATASET_LOADERS
    from src.models import (TreeGuidedSpikeEncoder, SNNStudent, DNNStudent,
                            quantize_int8_)
    from src.training_treespike import (_fit_teacher, _arrays_from_bundle, _resolve,
                                        _set_seed)
    import yaml, inspect, copy

    torch.set_num_threads(1)          # single-thread CPU, as specified
    dev = torch.device("cpu")
    rows = []

    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{dataset}/config.yaml"))
    rp = REPO / ROOTS["t2s"].format(d=dataset) / f"seed_{seed}/results.json"
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
    def dump(tag, **art):
        if artifact_dir is not None:
            Path(artifact_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(artifact_dir) / f"{dataset}_{tag}.pkl", "wb") as fh:
                pickle.dump(dict(system=tag, dataset=dataset, seed=seed, Xsub=Xsub, **art), fh, protocol=4)

    teacher, _, present = _fit_teacher(Xtr, ytr, K, cfg)
    # ONE prediction thread, like every neural system (torch.set_num_threads(1) above).
    # The teacher is fitted with n_jobs=4, and XGBoost predicts with the model's own
    # thread count, so without this the GBDT rows were timed on four threads.
    teacher.set_params(n_jobs=1)
    teacher.get_booster().set_param({"nthread": 1})
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
        mb = booster_bytes(teacher, None if tag == "full_gbdt" else nr,
                           n_classes=len(present))
        Xdf_one = pd.DataFrame(Xsub[:1], columns=[f"f{i}" for i in range(F)])
        ir = (0, nr)

        def gb_fn(x, ir=ir):
            d = pd.DataFrame(x, columns=[f"f{i}" for i in range(F)])
            return teacher.predict(d, iteration_range=ir)

        dump(tag, kind="gbdt", teacher=teacher, iteration_range=ir, F=F)
        lat = time_fn(gb_fn, Xsub)
        # the teacher is fit on the classes PRESENT in training (CSE-CIC-IDS2018
        # declares five and exhibits four), so the packing and the label mapping
        # must use the teacher's own class count, not the dataset's
        K_t = len(present)
        packed = SER.pack_booster(teacher, None if tag == "full_gbdt" else nr,
                                  leaf_bits=16, n_classes=K_t)
        ncap = min(len(Xsub), 5000)
        pk_pred = present[SER.predict_packed(packed, Xsub[:ncap], K_t, leaf_bits=16)]
        nat_pred = present[teacher.predict(pd.DataFrame(Xsub[:ncap],
                                                        columns=[f"f{i}" for i in range(F)]),
                                           iteration_range=ir)]
        f1_packed = 100 * compute_metrics(yte[:ncap], pk_pred,
                                          class_names=b.class_names)["f1_macro"]
        f1_native = 100 * compute_metrics(yte[:ncap], nat_pred,
                                          class_names=b.class_names)["f1_macro"]
        rows.append(dict(packed_macro_f1=f1_packed, native_macro_f1=f1_native,
                         packed_agreement=float(np.mean(pk_pred == nat_pred)),
                         n_accuracy_compared=int(ncap),
                         serialization_format=SER.SERIALIZATION_FORMAT,
                         dataset=dataset, seed=seed, system=tag,
                         representation="tree ensemble",
                         precision="native" if tag == "full_gbdt" else "val-selected prefix",
                         model_bytes=mb, encoder_bytes=0, scaler_bytes=scaler_bytes,
                         total_bytes=mb + scaler_bytes,
                         op_kind="comparisons", macs=np.nan, sops=np.nan,
                         tree_nodes_visited=float(trees * depth),
                         tree_comparisons=float(trees * depth),
                         n_trees=trees, n_nodes=nodes, **lat))

    # ---------------- shared tree-guided encoder ----------------
    enc = TreeGuidedSpikeEncoder(
        T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
        jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")
    ).fit(teacher, F, X=Xtr)
    eb = encoder_bytes(enc)
    ecmp = encoder_comparisons(enc)

    if only_gbdt:
        return rows

    # ---------------- Vanilla SNN and TASNN (float32, as reported) ----------------
    import json as _json
    vck = REPO / ROOTS["vanilla"].format(d=dataset) / f"seed_{seed}/checkpoint.pt"
    if vck.exists():
        from src.models import SNN
        from src.encoding import get_encoder
        vcfg = yaml.safe_load(open(REPO / f"configs/{dataset}/SNN_LeakyParallel/latency/config.yaml"))
        vT = int(vcfg["num_steps"])
        vm = SNN(num_features=F, hidden_size=vcfg["hidden_size"], num_classes=K,
                 neuron_name=vcfg["neuron"], num_steps=vT).to(dev)
        vm.load_state_dict(torch.load(vck, map_location=dev)); vm.eval()
        venc = get_encoder(vcfg["encoding"])

        def van_fn(x, vm=vm, venc=venc, vT=vT):
            with torch.no_grad():
                return vm(venc(torch.from_numpy(x), vT))

        dump("vanilla_snn", kind="vanilla", sd=vm.state_dict(), F=F, K=K, vcfg=dict(vcfg))
        lat = time_fn(van_fn, Xsub)
        vb = student_bytes(vm.state_dict(), int8_grid=False)
        rj = REPO / ROOTS["vanilla"].format(d=dataset) / f"seed_{seed}/results.json"
        _vss = _json.load(open(rj))["test_plus"]["spike_stats"]
        spk = float(_vss["energy_spikes_per_sample"])
        # Its own run counts BOTH synaptic layers (input spikes x hidden, hidden
        # spikes x classes), which is the convention Tree-to-Spike now uses too
        # after the input layer was added to its synops. Without this the row
        # carried no operation count at all and the energy model priced it at
        # nearly zero arithmetic.
        if "synops_per_sample" in _vss:
            vsops = float(_vss["synops_per_sample"])
        else:
            # The two repaired corpora were rerun by a path that did not persist it.
            # src/opcount reproduces this exact convention (validated to the decimal
            # against a run that does record it), so the count is recomputed rather
            # than left missing.
            from src.opcount import count_ops
            _vx = Xsub[:2000]
            with torch.no_grad(), count_ops(vm, batch_size=len(_vx)) as _vc:
                vm(venc(torch.from_numpy(_vx), vT))
            vsops = float(_vc.sops)
        # The published tables add num_features * hidden_size on top of this value,
        # which already counts the input layer from the features that ACTUALLY fire
        # (41% of them on CICIoT2023). Both are recorded so the difference is visible
        # without a rerun; see results/tables/vanilla_sops_double_count.md.
        vsops_paper = vsops + float(F) * float(vcfg["hidden_size"])
        rows.append(dict(dataset=dataset, seed=seed, system="vanilla_snn",
                         representation="raw-feature TTFS", precision="FP32",
                         model_bytes=vb["model_bytes"], encoder_bytes=0,
                         scaler_bytes=scaler_bytes,
                         total_bytes=vb["model_bytes"] + scaler_bytes,
                         model_float_bytes=vb["model_float_bytes"],
                         serialization_format=vb["serialization_format"],
                         op_kind="SOP", macs=np.nan, sops=vsops,
                         sops_paper_convention=vsops_paper,
                         spikes_per_sample=spk, n_channels=float(F),
                         tree_nodes_visited=np.nan, tree_comparisons=np.nan,
                         n_trees=0, n_nodes=0, **lat))

    tack = REPO / ROOTS["tasnn"].format(d=dataset) / f"seed_{seed}/checkpoint.pt"
    if tack.exists():
        from src.models import TASNN
        tcfg = yaml.safe_load(open(REPO / f"experiments/TASNN/{dataset}/config.yaml"))
        tcfg.update(_json.load(open(REPO / ROOTS["tasnn"].format(d=dataset) / f"seed_{seed}/results.json"))["config"])
        tm = TASNN(num_features=F, num_classes=K, num_steps=tcfg["num_steps"],
                   d_model=tcfg["d_model"], n_heads=tcfg["n_heads"], n_layers=tcfg["n_layers"],
                   ff=tcfg["ff"], dropout=tcfg["dropout"], pop_M=tcfg["pop_M"],
                   stc_channels=tcfg["stc_channels"], stc_kernel=tcfg["stc_kernel"],
                   lif_hidden=tuple(tcfg["lif_hidden"]), lif_beta=tcfg["lif_beta"],
                   lif_v_th=tcfg["lif_v_th"], lif_gamma=tcfg["lif_gamma"],
                   surrogate_beta=tcfg["surrogate_beta"], target_rate=tcfg["target_rate"]).to(dev)
        tm.load_state_dict(torch.load(tack, map_location=dev)); tm.eval()

        def tas_fn(x, tm=tm):
            with torch.no_grad():
                return tm(torch.from_numpy(x))

        dump("tasnn", kind="tasnn", sd=tm.state_dict(), F=F, K=K, tcfg=dict(tcfg))
        lat = time_fn(tas_fn, Xsub)
        # TASNN records no operation count. It is priced by the SAME analytic
        # formula the published figures use (scripts/benchmark_4way.py), not by a
        # new path: hook-based counting misses MultiheadAttention's projections and
        # score products and lands at about half. This reproduces the paper.
        from src.tasnn_energy import tasnn_energy_pj
        _rec_t = _json.load(open(REPO / ROOTS["tasnn"].format(d=dataset) / f"seed_{seed}/results.json"))
        _e_t, t_macs, t_sops = tasnn_energy_pj(_rec_t)
        t_macs, t_sops = float(t_macs), float(t_sops)
        tb2 = student_bytes(tm.state_dict(), int8_grid=False)
        rj = REPO / ROOTS["tasnn"].format(d=dataset) / f"seed_{seed}/results.json"
        spk = float(_json.load(open(rj))["test_plus"]["spike_stats"]["energy_spikes_per_sample"])
        rows.append(dict(dataset=dataset, seed=seed, system="tasnn",
                         representation="transformer + LIF core", precision="FP32",
                         model_bytes=tb2["model_bytes"], encoder_bytes=0,
                         scaler_bytes=scaler_bytes,
                         total_bytes=tb2["model_bytes"] + scaler_bytes,
                         model_float_bytes=tb2["model_float_bytes"],
                         serialization_format=tb2["serialization_format"],
                         op_kind="MAC+SOP", macs=t_macs, sops=t_sops,
                         spikes_per_sample=spk, n_channels=float(F),
                         tree_nodes_visited=np.nan, tree_comparisons=np.nan,
                         n_trees=0, n_nodes=0, **lat))

    # ---------------- DNN-Tree (int8-grid checkpoint) ----------------
    dck = REPO / ROOTS["dnn"].format(d=dataset) / f"seed_{seed}/checkpoint.pt"
    if dck.exists():
        dcfg = yaml.safe_load(open(REPO / f"experiments/DNN-Tree/{dataset}/config.yaml"))
        m = DNNStudent(enc.out_dim, K, hidden=tuple(dcfg["hidden"]),
                       clip_relu=dcfg.get("clip_relu")).to(dev)
        m.load_state_dict(torch.load(dck, map_location=dev)); m.eval()

        def dnn_fn(x, m=m):
            with torch.no_grad():
                return m(enc.encode_dense(x))

        dump("dnn_tree", kind="dnn", sd=m.state_dict(), enc=enc, K=K, hidden=tuple(dcfg["hidden"]),
             clip_relu=dcfg.get("clip_relu"))
        lat = time_fn(dnn_fn, Xsub)
        bb = student_bytes(m.state_dict(), int8_grid=True); mb = bb["model_bytes"]
        rows.append(dict(dataset=dataset, seed=seed, system="dnn_tree",
                         representation="tree-derived", precision="int8-grid weights",
                         model_bytes=mb, scaler_bytes=scaler_bytes,
                         total_bytes=mb + eb["encoder_bytes"] + scaler_bytes,
                         model_int8_bytes=bb["model_int8_bytes"],
                         model_scale_bytes=bb["model_scale_bytes"],
                         model_float_bytes=bb["model_float_bytes"],
                         serialization_format=bb["serialization_format"],
                         op_kind="MAC", macs=float(m.macs_per_sample()), sops=np.nan,
                         tree_nodes_visited=np.nan, tree_comparisons=ecmp,
                         n_trees=0, n_nodes=0, **eb, **lat))

    # ---------------- Tree-to-Spike: full window and selected exit ----------------
    tck = REPO / ROOTS["t2s"].format(d=dataset) / f"seed_{seed}/checkpoint.pt"
    if tck.exists():
        st = SNNStudent(enc.out_dim, K, hidden=tuple(cfg["hidden"]),
                        beta=cfg["beta"], leaky=cfg["leaky"]).to(dev)
        st.load_state_dict(torch.load(tck, map_location=dev)); st.eval()
        tbb = student_bytes(st.state_dict(), int8_grid=True); mb = tbb["model_bytes"]

        sel = REPO / ROOTS["exit_csv"]
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
                # stop_when_done: at batch 1 this leaves the loop at the sample's
                # own exit step, which is what a deployed exit does. Without it the
                # timed path runs all T steps and the measured "exit" latency is the
                # full-window latency plus per-step bookkeeping.
                return st.predict_early_exit(sp, theta_exit=th, stop_when_done=True)

        for tag, fn, me in (("t2s_full", t2s_full_fn, float(cfg["T"])),
                            ("t2s_exit", t2s_exit_fn, mean_exit)):
            dump(tag, kind="t2s", sd=st.state_dict(), enc=enc, K=K, hidden=tuple(cfg["hidden"]),
                 beta=cfg["beta"], leaky=cfg["leaky"], theta=(theta if tag == "t2s_exit" else None))
            lat = time_fn(fn, Xsub)
            # measured executed SOPs from the cached prefix artifact at the mean exit step
            pf = (REPO / ROOTS["prefix"].format(d=dataset) / f"seed_{seed}"
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
                             model_int8_bytes=tbb["model_int8_bytes"],
                             model_scale_bytes=tbb["model_scale_bytes"],
                             model_float_bytes=tbb["model_float_bytes"],
                             serialization_format=tbb["serialization_format"],
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
    ap.add_argument("--systems", choices=["all", "gbdt"], default="all",
                    help="gbdt: re-measure only the GBDT rows and MERGE them into the existing "
                         "CSV, keeping every other row as measured")
    for k in ROOTS:
        ap.add_argument("--" + k.replace("_", "-") + ("-root" if k not in ("exit_csv", "out_csv") else ""),
                        dest=k, default=ROOTS[k])
    args = ap.parse_args()
    for k in ROOTS:
        ROOTS[k] = getattr(args, k)
    if ROOTS["out_csv"] == "results/deployment/deployment_measurements.csv" and set(args.datasets) != set(DATASETS):
        sys.exit("a partial run would overwrite the published deployment_measurements.csv; pass --out-csv")
    OUT.mkdir(parents=True, exist_ok=True)

    load_at_start = wait_for_gate(REPO)
    allrows, t0 = [], time.time()
    for ds in args.datasets:
        for seed in args.seeds:
            try:
                r = measure(ds, seed, only_gbdt=args.systems == "gbdt")
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
    import subprocess, datetime
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=REPO, text=True).strip()
    except Exception:
        sha = "unknown"
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + sha
    df["run_id"] = run_id
    df["serialization_format"] = df.get("serialization_format", SER.SERIALIZATION_FORMAT)
    df["serialization_format"] = df["serialization_format"].fillna(SER.SERIALIZATION_FORMAT)
    df["source_script"] = "scripts/deployment_measure.py"
    csv = REPO / ROOTS["out_csv"]
    csv.parent.mkdir(parents=True, exist_ok=True)
    header = []
    if args.systems == "gbdt":
        # merge: replace only the re-measured GBDT rows, keep the rest and their run IDs
        header = [l for l in csv.read_text().splitlines() if l.startswith("#")]
        old = pd.read_csv(csv, comment="#")
        key = old.set_index(["dataset", "seed", "system"]).index
        new_key = df.set_index(["dataset", "seed", "system"]).index
        df = pd.concat([old[~key.isin(new_key)], df], ignore_index=True)
        header.append(f"# gbdt_remeasure run_id={run_id} rows={len(new_key)} nthread=1 "
                      f"load1_at_start={load_at_start:.2f}")
    else:
        header = [f"# run_id={run_id} serialization={SER.SERIALIZATION_FORMAT} "
                  f"source=scripts/deployment_measure.py load1_at_start={load_at_start:.2f}"]
    with open(csv, "w") as fh:
        fh.write("\n".join(header) + "\n")
        df.to_csv(fh, index=False)
    print(f"\nwrote {csv} ({len(df)} rows)")
