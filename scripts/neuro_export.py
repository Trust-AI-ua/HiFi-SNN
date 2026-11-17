#!/usr/bin/env python3
"""Export what track A needs from the training environment (results/preregistration/2026-11-17_neuro_stream.md).

Runs in .venv (the training environment), so the teacher is the production teacher. For one
dataset and seed it writes results/neuro/inputs/<ds>_seed<S>.npz with:
  spike_times   [N, C] int8   test-split spike time of every encoder channel (each channel
                              spikes exactly once; the integer encoder of step 2)
  labels        [N]           test labels
  float_logits  [N, K]        snnTorch float readout (mean output membrane)
  int_pred      [N]           integer NumPy predictions (Q = 8, 8-bit leak; step 2 headline)
  sample        [2000]        the stratified A2/A3 sample (indices into the test split)
  int_counts    [2000, 3]     integer NumPy spikes per layer over the window, on the sample
  W1..W3 int8, b1..b3 (float bias), s1..s3 (weight scales), theta1..3 and bq1..3 (Q = 8)
Guards: refit teacher equals the stored one; float and integer test macro-F1 equal the record
and the step-2 record; the integer encoder equals the float encoder.

    .venv/bin/python scripts/neuro_export.py <dataset> [seed]
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/neuro/inputs"
N_SAMPLE, MIN_PER_CLASS = 2000, 50


def stratified_sample(y: np.ndarray, n: int = N_SAMPLE, m: int = MIN_PER_CLASS, seed: int = 0) -> np.ndarray:
    """At least m per class (all if fewer), the rest in proportion to class size."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    take = np.minimum(counts, m)
    rest = n - take.sum()
    room = counts - take
    if rest > 0 and room.sum() > 0:
        extra = np.floor(rest * room / room.sum()).astype(int)
        extra = np.minimum(extra, room)
        short = rest - extra.sum()
        for i in np.argsort(-(rest * room / room.sum() - extra)):   # largest remainders
            if short <= 0:
                break
            if extra[i] < room[i]:
                extra[i] += 1; short -= 1
        take = take + extra
    idx = [rng.choice(np.flatnonzero(y == c), size=k, replace=False) for c, k in zip(classes, take)]
    return np.sort(np.concatenate(idx))


def int_counts(layers, st, T, mul=230, sh=8, Q=8):
    """Integer NumPy run (same arithmetic as r2_integer_inference.int_snn) with per-layer spike counts."""
    B = st.shape[0]
    v = [np.zeros((B, L["W"].shape[1]), np.int64) for L in layers]
    acc = np.zeros((B, layers[2]["W"].shape[1]), np.int64)
    cnt = np.zeros((B, 3), np.int64)
    for t in range(T):
        spk = (st == t)
        for li, L in enumerate(layers):
            cur = (spk.astype(np.int64) @ L["W"]) * (2 ** Q) + L["b"]
            r = v[li] > L["theta"]
            v[li] = ((v[li] * mul) >> sh) + cur - r * L["theta"]
            spk = v[li] > L["theta"]
            cnt[:, li] += spk.sum(1)
        acc = acc + v[2]
    return acc.argmax(1), cnt


def main():
    ds = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    out = OUT / f"{ds}_seed{seed}.npz"
    if any(os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise SystemExit("thread variables are set; the teacher would not be the production teacher")
    import r2_integer_inference as RI
    from src.data import get_dataset, DATASET_LOADERS
    from src.metrics import compute_metrics
    from src.models.tree_to_spike import TreeGuidedSpikeEncoder, SNNStudent
    from src.training_treespike import _arrays_from_bundle, _fit_teacher
    rec = json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/results.json").read_text())
    rint = json.loads((REPO / f"results/journal_r2/integer/{ds}/seed_{seed}.json").read_text())
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    cfg.update(rec["config"]); cfg["dataset"], cfg["seed"] = ds, seed
    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=seed)
    for o in ("subsample_fraction", "test_fraction", "label_mode"):
        if o in cfg and o in lp and cfg[o] is not None:
            kw[o] = cfg[o]
    b = get_dataset(ds, **kw)
    Xtr, ytr, Xva, yva, Xte, yte = _arrays_from_bundle(b)
    K, F = b.num_classes, Xtr.shape[1]
    score = lambda y, p: compute_metrics(y, p, class_names=b.class_names)["f1_macro"]
    teacher, _, present = _fit_teacher(Xtr, ytr, K, cfg)
    if score(yte, present[teacher.predict(pd.DataFrame(Xte, columns=[f"f{i}" for i in range(F)]))]) \
            != rec["test_plus"]["teacher_upper_bound"]["f1_macro"]:
        raise SystemExit(f"{ds}/{seed}: refit teacher is not the stored teacher")
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                 jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")).fit(teacher, F, X=Xtr)
    ienc, T = RI.IntEncoder(enc), enc.T
    st = SNNStudent(enc.out_dim, K, hidden=tuple(cfg["hidden"]), beta=cfg["beta"], leaky=cfg["leaky"])
    st.load_state_dict(torch.load(REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/checkpoint.pt", map_location="cpu"))
    st.eval()
    run, _, layers = RI.int_snn(st, 8, "8bit")
    stimes, logits, ipred = [], [], []
    for i in range(0, len(Xte), RI.CHUNK):
        x = Xte[i:i + RI.CHUNK]
        sp = enc.encode(x)
        s = ienc.spike_times(x)
        if not torch.equal(sp, (torch.arange(T).view(T, 1, 1) == torch.from_numpy(s).unsqueeze(0)).float()):
            raise SystemExit(f"{ds}/{seed}: integer encoder spikes differ from the float encoder")
        with torch.no_grad():
            logits.append(st(sp)[0].numpy())
        stimes.append(s.astype(np.int8)); ipred.append(run(s, T))
    stimes, logits, ipred = np.concatenate(stimes), np.concatenate(logits), np.concatenate(ipred)
    if score(yte, logits.argmax(1)) != rec["test_plus"]["metrics"]["f1_macro"]:
        raise SystemExit(f"{ds}/{seed}: float student does not reproduce the record")
    if abs(100 * score(yte, ipred) - rint["integer"]["tts_q8_leak8"]["macro_f1"]) > 1e-9:
        raise SystemExit(f"{ds}/{seed}: integer path does not reproduce step 2")
    samp = stratified_sample(yte)
    p2, cnt = int_counts(layers, stimes[samp].astype(np.int64), T)
    if not np.array_equal(p2, ipred[samp]):
        raise SystemExit("counting integer run disagrees with step 2's integer run")
    sd = {k: v.detach().numpy() for k, v in st.state_dict().items()}
    extra = {}
    for i, (n, L) in enumerate(zip(("fc1", "fc2", "fc3"), layers), 1):
        extra.update({f"W{i}": L["W"].T.astype(np.int8), f"b{i}": sd[f"{n}.bias"], f"s{i}": np.float64(L["s"]),
                      f"theta{i}": np.int64(L["theta"]), f"bq{i}": L["b"]})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, spike_times=stimes, labels=yte.astype(np.int64), float_logits=logits,
                        int_pred=ipred, sample=samp, int_counts=cnt, class_names=np.array(b.class_names),
                        T=np.int64(T), beta=np.float64(cfg["beta"]), **extra)
    print(f"[done] {ds} s{seed}: N={len(yte)} C={stimes.shape[1]} K={K} float F1 {100 * score(yte, logits.argmax(1)):.2f} "
          f"int F1 {100 * score(yte, ipred):.2f} sample {len(samp)} "
          f"(per class min {np.bincount(yte[samp]).min()})", flush=True)


if __name__ == "__main__":
    main()
