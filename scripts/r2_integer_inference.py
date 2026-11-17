#!/usr/bin/env python3
"""Integer-only inference for Tree-to-Spike and DNN-Tree (revision 2, step 2).

Pre-registered in results/preregistration/2026-11-16_r2_integer_gbdt.md. For one dataset and
seed: rebuild the production teacher and encoder, load the deployed checkpoints, and run the
test split through a NumPy reimplementation that uses only integer arithmetic: int8 weights,
int32 accumulators (checked), integer comparisons, shifts and lookup tables.

Encoder (both models). A feature's level index is the number of thresholds at or below it.
It is found by comparing IEEE-754 bit patterns as integers, mapped to a monotone key so that
negative values order correctly; this is exact. A per-channel table maps the index to the
spike time (Tree-to-Spike) or to the 8-bit input (DNN-Tree), so no floating-point arithmetic
runs at inference.

Tree-to-Spike. Spikes are binary, so no activation needs quantizing. Each layer's membrane is
a fixed-point integer with Q fractional bits in units of that layer's weight scale s:
  current   I = 2^Q * sum(int8 weights of the inputs that spiked) + round(2^Q b / s)
  update    v = leak(v) + I - r * theta,  r = [v_prev > theta] (snnTorch's delayed
            subtract-reset), theta = round(2^Q / s)
  spike     [v > theta]
  leak      (v * 230) >> 8   (beta = 0.8984; the model's beta is 0.9)
  readout   argmax of the summed output membrane (the float model's mean, times T)
DNN-Tree. TFLite-style: uint8 input (level x 255), int8 weights, int32 accumulator plus
integer bias, ReLU, requantization to uint8 by a fixed-point multiplier (m >> shift), with
each layer's activation range calibrated as the maximum over the validation split. Argmax of
the last accumulator.

Guards, before any integer result is kept:
  1. the refit teacher equals the stored one (test macro-F1 of the record);
  2. the checkpoints' weights lie on the per-tensor int8 grid;
  3. the float path reproduces each record's test macro-F1 exactly;
  4. the integer encoder emits exactly the float encoder's spikes and levels.

Variants (Tree-to-Spike): Q = 8 with the 8-bit leak (headline), Q = 8 with a 16-bit leak
((v * 7373) >> 13, beta = 0.90002), and Q = 0 with the 8-bit leak.

    python scripts/r2_integer_inference.py <dataset> <seed>

Output: results/journal_r2/integer/<dataset>/seed_<S>.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/journal_r2/integer"
INT32 = 2 ** 31 - 1
CHUNK = 32768
LEAKS = {"8bit": (230, 8), "16bit": (7373, 13)}


# ---------------------------------------------------------------- integer encoder
def sort_key(x: np.ndarray) -> np.ndarray:
    """Monotone int64 key of float32 values: order of keys == order of floats."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(b >= 0, b, -(b & 0x7FFFFFFF))


class IntEncoder:
    def __init__(self, enc, dense_scale: int = 255):
        self.T = enc.T
        self.feats = sorted({j for j, _ in enc.channels_})
        for j in self.feats:
            if j not in enc.edges_ or len(enc.edges_[j]) == 0:
                raise SystemExit(f"feature {j} has no thresholds; the encoder's sigmoid fallback is not integer")
        self.keys = {j: sort_key(np.asarray(enc.edges_[j], dtype=np.float32)) for j in self.feats}
        self.channels = enc.channels_
        self.spike_lut, self.dense_lut = [], []
        for j, o in enc.channels_:
            n = len(enc.edges_[j])
            # the encoder's own arithmetic, evaluated once per possible index (offline)
            lv = np.empty(n + 1, dtype=np.float32)
            lv[:] = np.clip(np.arange(n + 1) / float(n) + o, 0.0, 1.0)
            self.spike_lut.append(np.round((1.0 - lv) * (enc.T - 1)).astype(np.int64))
            self.dense_lut.append(np.round(lv.astype(np.float64) * dense_scale).astype(np.int64))

    def index(self, X):
        k = {j: np.searchsorted(self.keys[j], sort_key(X[:, j]), side="right") for j in self.feats}
        return k

    def spike_times(self, X):
        k = self.index(X)
        return np.stack([self.spike_lut[c][k[j]] for c, (j, _) in enumerate(self.channels)], 1)

    def dense_u8(self, X):
        k = self.index(X)
        return np.stack([self.dense_lut[c][k[j]] for c, (j, _) in enumerate(self.channels)], 1)


# ---------------------------------------------------------------- integer models
def int8_grid(W: np.ndarray):
    s = float(np.abs(W).max()) / 127.0
    q = W / s
    qi = np.round(q)
    err = float(np.abs(q - qi).max())
    if err > 1e-3 or np.abs(qi).max() > 127:
        raise SystemExit(f"weights not on the int8 grid (max deviation {err:.2e})")
    return qi.astype(np.int64), s


class Tracker:
    def __init__(self):
        self.max_abs = 0

    def __call__(self, a):
        m = int(np.abs(a).max()) if a.size else 0
        self.max_abs = max(self.max_abs, m)
        if m > INT32:
            raise SystemExit(f"int32 overflow: {m}")
        return a


def int_snn(student, Q=8, leak="8bit"):
    sd = {k: v.detach().cpu().numpy() for k, v in student.state_dict().items()}
    layers = []
    for n in ("fc1", "fc2", "fc3"):
        Wq, s = int8_grid(sd[f"{n}.weight"])
        layers.append(dict(W=Wq.T.copy(), b=np.round(sd[f"{n}.bias"] / s * 2 ** Q).astype(np.int64),
                           theta=int(round(1.0 / s * 2 ** Q)), s=s))
    mul, sh = LEAKS[leak]
    track = Tracker()

    def run(st, T):
        """st: [B, C] spike times -> predictions [B]."""
        B = st.shape[0]
        v = [np.zeros((B, L["W"].shape[1]), np.int64) for L in layers]
        acc = np.zeros((B, layers[2]["W"].shape[1]), np.int64)
        for t in range(T):
            spk = (st == t)
            for li, L in enumerate(layers):
                cur = track((spk.astype(np.int64) @ L["W"]) * (2 ** Q) + L["b"])
                r = v[li] > L["theta"]
                v[li] = track(((v[li] * mul) >> sh) + cur - r * L["theta"])
                spk = v[li] > L["theta"]
            acc = track(acc + v[2])
        return acc.argmax(1)
    return run, track, layers


def requant_multiplier(M: float):
    """M = m * 2^-shift with m in [2^30, 2^31)."""
    if M <= 0:
        raise SystemExit("non-positive requantization multiplier")
    shift = 0
    while M < 0.5:
        M *= 2; shift += 1
    while M >= 1.0:
        M /= 2; shift -= 1
    m = int(round(M * 2 ** 31))
    if m == 2 ** 31:
        m //= 2; shift -= 1
    return m, shift + 31


def int_dnn(model, Xva_u8, dense_scale=255):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    if model.clip_relu:
        raise SystemExit("clip_relu is not implemented in the integer path")
    q = [int8_grid(sd[f"fc{i}.weight"]) for i in (1, 2, 3)]
    b = [sd[f"fc{i}.bias"] for i in (1, 2, 3)]
    # activation ranges from the float model on the validation split
    with torch.no_grad():
        xv = torch.from_numpy(Xva_u8.astype(np.float32) / dense_scale)
        h1 = torch.relu(model.fc1(xv)); h2 = torch.relu(model.fc2(h1))
    s_in = [1.0 / dense_scale, float(h1.max()) / 255.0, float(h2.max()) / 255.0]
    layers = []
    for i in range(3):
        Wq, sw = q[i]
        s_acc = sw * s_in[i]
        L = dict(W=Wq.T.copy(), b=np.round(b[i] / s_acc).astype(np.int64))
        if i < 2:
            L["m"], L["shift"] = requant_multiplier(s_acc / s_in[i + 1])
        layers.append(L)
    track = Tracker()

    def run(xu8):
        h = xu8.astype(np.int64)
        for i, L in enumerate(layers):
            a = track(h @ L["W"] + L["b"])
            if i == 2:
                return a.argmax(1)
            a = np.maximum(a, 0)
            h = np.minimum((a * L["m"] + (1 << (L["shift"] - 1))) >> L["shift"], 255)
    return run, track, dict(act_max_val=s_in[1:], calibration="max over the validation split")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("seed", type=int)
    a = ap.parse_args()
    ds, seed = a.dataset, a.seed
    out = OUT / ds / f"seed_{seed}.json"
    if out.exists():
        print(f"[skip] {out}"); return
    if any(os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise SystemExit("thread variables are set; the teacher would not be the production teacher")
    from src.data import get_dataset, DATASET_LOADERS
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    from src.models.tree_to_spike import TreeGuidedSpikeEncoder, SNNStudent
    from src.models.dnn_student import DNNStudent
    from src.training_treespike import _arrays_from_bundle, _fit_teacher
    t0 = time.time()
    rec_t = json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/results.json").read_text())
    rec_d = json.loads((REPO / f"results/DNN-Tree/{ds}/seed_{seed}/results.json").read_text())
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
    teacher, _, present = _fit_teacher(Xtr, ytr, K, cfg)                     # guard 1
    tf1 = score(yte, present[teacher.predict(pd.DataFrame(Xte, columns=cols))])["f1_macro"]
    if tf1 != rec_t["test_plus"]["teacher_upper_bound"]["f1_macro"]:
        raise SystemExit(f"{ds}/{seed}: refit teacher {tf1!r} is not the stored teacher")
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                 jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")).fit(teacher, F, X=Xtr)
    ienc = IntEncoder(enc)
    T = enc.T

    st = SNNStudent(enc.out_dim, K, hidden=tuple(cfg["hidden"]), beta=cfg["beta"], leaky=cfg["leaky"])
    st.load_state_dict(torch.load(REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/checkpoint.pt", map_location="cpu"))
    st.eval()
    dcfg = rec_d["config"]
    dn = DNNStudent(enc.out_dim, K, hidden=tuple(dcfg["hidden"]), clip_relu=dcfg.get("clip_relu"))
    dn.load_state_dict(torch.load(REPO / f"results/DNN-Tree/{ds}/seed_{seed}/checkpoint.pt", map_location="cpu"))
    dn.eval()
    runs = {f"tts_q8_leak8": int_snn(st, 8, "8bit"), "tts_q8_leak16": int_snn(st, 8, "16bit"),
            "tts_q0_leak8": int_snn(st, 0, "8bit")}                           # guard 2 inside
    dn_run, dn_track, dn_info = int_dnn(dn, ienc.dense_u8(Xva))

    pf_t, pf_d = [], []
    pi = {k: [] for k in runs}; pi["dnn"] = []
    for i in range(0, len(Xte), CHUNK):
        x = Xte[i:i + CHUNK]
        sp = enc.encode(x)                                                   # float encoder
        stimes = ienc.spike_times(x)                                         # guard 4
        if not torch.equal(sp, (torch.arange(T).view(T, 1, 1) == torch.from_numpy(stimes).unsqueeze(0)).float()):
            raise SystemExit(f"{ds}/{seed}: integer encoder spikes differ from the float encoder")
        lv = enc.encode_dense(x).numpy()
        if not np.array_equal(ienc.index(x)[ienc.feats[0]],
                              np.searchsorted(np.asarray(enc.edges_[ienc.feats[0]]), x[:, ienc.feats[0]], side="right")):
            raise SystemExit(f"{ds}/{seed}: integer level index differs")
        with torch.no_grad():
            pf_t.append(st(sp)[0].argmax(1).numpy())
            pf_d.append(dn(torch.from_numpy(lv)).argmax(1).numpy())
        for k, (run, _, _) in runs.items():
            pi[k].append(run(stimes, T))
        pi["dnn"].append(dn_run(ienc.dense_u8(x)))
    pf_t, pf_d = np.concatenate(pf_t), np.concatenate(pf_d)
    for name, pf, rec in (("Tree-to-Spike", pf_t, rec_t), ("DNN-Tree", pf_d, rec_d)):   # guard 3
        got = score(yte, pf)["f1_macro"]
        if got != rec["test_plus"]["metrics"]["f1_macro"]:
            raise SystemExit(f"{ds}/{seed}: float {name} {got!r} does not reproduce the record "
                             f"{rec['test_plus']['metrics']['f1_macro']!r}")

    def summary(pred, ref):
        m = score(yte, pred); bm = compute_binary_ids_metrics(yte, pred, normal_class_index=0)
        return dict(macro_f1=100 * m["f1_macro"], accuracy=100 * m["accuracy"], mcc=100 * m["matthews_corrcoef"],
                    far=100 * bm["false_alarm_rate"], dr=100 * bm["detection_rate"],
                    agreement_with_float=float(np.mean(pred == ref)))
    res = dict(dataset=ds, seed=seed, n_test=int(len(yte)), T=T, channels=int(enc.out_dim),
               float=dict(tts=summary(pf_t, pf_t), dnn=summary(pf_d, pf_d)), integer={},
               dnn_calibration=dn_info, preregistration="results/preregistration/2026-11-16_r2_integer_gbdt.md")
    for k, (run, track, layers) in runs.items():
        res["integer"][k] = dict(**summary(np.concatenate(pi[k]), pf_t), max_abs_int=track.max_abs,
                                 thresholds=[L["theta"] for L in layers])
    res["integer"]["dnn_u8"] = dict(**summary(np.concatenate(pi["dnn"]), pf_d), max_abs_int=dn_track.max_abs)
    res["guards"] = "teacher, int8 grid, float reproduction of both records, integer encoder: all passed"
    res["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    I = res["integer"]
    print(f"[done] {ds} s{seed}: T2S float {res['float']['tts']['macro_f1']:.2f} -> int {I['tts_q8_leak8']['macro_f1']:.2f} "
          f"(agree {100 * I['tts_q8_leak8']['agreement_with_float']:.2f}%; leak16 {I['tts_q8_leak16']['macro_f1']:.2f}, "
          f"Q0 {I['tts_q0_leak8']['macro_f1']:.2f}) | DNN float {res['float']['dnn']['macro_f1']:.2f} -> int "
          f"{I['dnn_u8']['macro_f1']:.2f} (agree {100 * I['dnn_u8']['agreement_with_float']:.2f}%) [{res['seconds']:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main()
