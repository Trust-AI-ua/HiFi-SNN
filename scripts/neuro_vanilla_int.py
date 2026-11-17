#!/usr/bin/env python3
"""Amendment 4 (results/preregistration/2026-11-17_neuro_stream.md): the Vanilla SNN in fixed point,
by the same rules as the Tree-to-Spike integer path (scripts/r2_integer_inference.py, Q = 8, 8-bit leak).

Runs in .venv after scripts/neuro_export_vanilla.py. Rebuilds the seed-42 test split as that
script does (the latency encoder on the CPU, checked there to equal the GPU encoder) and runs:
  hidden (LeakyParallel = nn.RNN with ReLU, diagonal recurrence beta, threshold, no reset):
      v_t = max(0, ((v_{t-1} * mul) >> 8) + (x_t @ Wq_ih) * 2^Q + bq_h),   s_t = [v_t > theta_h]
  output (snnTorch Leaky, delayed subtractive reset):
      v_t = ((v_{t-1} * mul) >> 8) + (s_t @ Wq_out) * 2^Q + bq_out - theta_o [v_{t-1} > theta_o]
  readout: output spike count over the window, argmax (as the float model's forward()).
The int8_grid rule (s = max|W| / 127, per tensor, round) on W_ih and W_out; mul = round(256 beta) (218 for 0.85);
bq = round(b / s 2^Q); theta = round(thr / s 2^Q). Writes results/neuro/inputs/<ds>_vanilla_int_seed42.npz
(full-test integer predictions, sample predictions and spike counts, the integer parameters) and
results/neuro/a2_lava/vanilla_int_<ds>.json (full-test macro-F1, float record and integer).

    .venv/bin/python scripts/neuro_vanilla_int.py <dataset>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
INP = REPO / "results/neuro/inputs"
Q, SH = 8, 8
INT32 = 2 ** 31 - 1


def int_vanilla(P, st, T):
    """P: integer parameters; st: [B, F] spike times (-1 = none). Returns out counts [B, K], hidden counts [B], max |v|."""
    B = st.shape[0]
    vh = np.zeros((B, P["Wh"].shape[0]), np.int64)
    vo = np.zeros((B, P["Wo"].shape[0]), np.int64)
    oc = np.zeros_like(vo); hc = np.zeros(B, np.int64); vmax = 0
    for t in range(T):
        x = (st == t).astype(np.int64)
        vh = np.maximum(0, ((vh * P["mul"]) >> SH) + (x @ P["Wh"].T) * 2 ** Q + P["bh"])
        sh = (vh > P["th_h"]).astype(np.int64)
        r = vo > P["th_o"]
        vo = ((vo * P["mul_o"]) >> SH) + (sh @ P["Wo"].T) * 2 ** Q + P["bo"] - r * P["th_o"]
        so = vo > P["th_o"]
        hc += sh.sum(1); oc += so
        vmax = max(vmax, int(np.abs(vh).max()), int(np.abs(vo).max()))
    if vmax > INT32:
        raise SystemExit(f"int32 overflow: {vmax}")
    return oc, hc, vmax


def float_vanilla(v, st, T):
    """Float64 NumPy replay of the float model (largest |v|, for the SANA-FE float scaling)."""
    B = st.shape[0]
    vh = np.zeros((B, v["W_ih"].shape[0])); vo = np.zeros((B, v["W_out"].shape[0])); m = 0.0
    for t in range(T):
        x = (st == t).astype(np.float64)
        vh = np.maximum(0, v["beta_h"] * vh + x @ v["W_ih"].T + v["b_h"])
        sh = (vh > v["thr_h"]).astype(np.float64)
        r = vo > v["thr_out"]
        vo = v["beta_out"] * vo + sh @ v["W_out"].T + v["b_out"] - r * v["thr_out"]
        m = max(m, float(np.abs(vh).max()), float(np.abs(vo).max()))
    return m


def main():
    ds = sys.argv[1]
    import torch
    import backfill_snn_metrics as BF
    from src.data import get_dataset
    from src.encoding import get_encoder
    from src.metrics import compute_metrics
    rd = REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_42"
    rec = json.loads((rd / "results.json").read_text())
    cfg = rec["config"]
    v = dict(np.load(INP / f"{ds}_vanilla_seed42.npz"))
    T = int(v["T"])
    b = get_dataset(ds, **BF._dataset_kwargs_for(ds, "LeakyParallel", "latency", cfg))
    X = torch.cat([x for x, _ in b.test_loader]).float()
    y = torch.cat([t for _, t in b.test_loader]).numpy()
    xs = X.clamp(0, 1)
    st = torch.floor((1.0 - xs) * (T - 1)).long().clamp(0, T - 1)
    st = torch.where(xs > 0.01, st, torch.full_like(st, -1)).numpy().astype(np.int64)
    enc = get_encoder("latency")
    chk = enc(X[:4096], T)
    if not torch.equal(chk, (torch.arange(T).view(T, 1, 1) == torch.from_numpy(st[:4096]).unsqueeze(0)).float()):
        raise SystemExit("spike times do not reproduce the latency encoder")
    t2s = np.load(INP / f"{ds}_seed42.npz")
    same = len(t2s["labels"]) == len(y) and np.array_equal(t2s["labels"], y)
    if bool(same) != bool(v["same_split_as_t2s"]):
        raise SystemExit("split check differs from the Vanilla export")
    from neuro_export import stratified_sample
    samp = t2s["sample"] if same else stratified_sample(y)
    if not np.array_equal(st[samp].astype(np.int8), v["spike_times"]) or not np.array_equal(y[samp], v["labels"]):
        raise SystemExit("sample differs from the Vanilla export")
    s_h = float(np.abs(v["W_ih"]).max()) / 127.0; s_o = float(np.abs(v["W_out"]).max()) / 127.0
    Wh = np.round(v["W_ih"] / s_h).astype(np.int64); Wo = np.round(v["W_out"] / s_o).astype(np.int64)
    beta_h = float(np.unique(v["beta_h"])[0]) if len(np.unique(v["beta_h"])) == 1 else None
    if beta_h is None:
        raise SystemExit("per-neuron beta: not covered by the 8-bit scalar leak rule")
    P = dict(Wh=Wh, Wo=Wo, mul=int(round(256 * beta_h)), mul_o=int(round(256 * float(v["beta_out"]))),
             bh=np.round(v["b_h"] / s_h * 2 ** Q).astype(np.int64), bo=np.round(v["b_out"] / s_o * 2 ** Q).astype(np.int64),
             th_h=int(round(float(v["thr_h"]) / s_h * 2 ** Q)), th_o=int(round(float(v["thr_out"]) / s_o * 2 ** Q)))
    preds, vmax = [], 0
    for i in range(0, len(st), 8192):
        oc, _, m = int_vanilla(P, st[i:i + 8192], T)
        preds.append(oc.argmax(1)); vmax = max(vmax, m)
    ipred = np.concatenate(preds)
    soc, shc, _ = int_vanilla(P, st[samp], T)
    f1 = lambda p: compute_metrics(y, p, class_names=b.class_names)["f1_macro"]
    f_rec = rec["test_plus"]["metrics"]["f1_macro"]
    fmax = float_vanilla(v, st[samp], T)
    np.savez_compressed(INP / f"{ds}_vanilla_int_seed42.npz", int_pred_full=ipred, labels_full=y, sample=samp,
                        spike_times_full=st.astype(np.int8),
                        int_pred=soc.argmax(1), int_out_counts=soc, int_hidden_counts=shc, s_h=s_h, s_o=s_o,
                        float_max_abs_v_sample=fmax, **{f"P_{k}": np.asarray(val) for k, val in P.items()})
    r = dict(dataset=ds, seed=42, T=T, Q=Q, leak_mul=P["mul"], leak_mul_out=P["mul_o"], theta_h=P["th_h"],
             theta_o=P["th_o"], max_abs_v_int=vmax, float_max_abs_v_sample=fmax,
             macro_f1_float_record=100 * f_rec, macro_f1_int_full_test=100 * f1(ipred),
             delta_f1_int_minus_float=100 * (f1(ipred) - f_rec),
             agreement_int_float_sample=float(np.mean(soc.argmax(1) == v["pred"])),
             hidden_spikes_per_flow_sample=dict(int=float(shc.mean()), float=float(v["hidden_counts"].mean())),
             out_spikes_per_flow_sample=dict(int=float(soc.sum(1).mean()), float=float(v["out_counts"].mean())))
    out = REPO / "results/neuro/a2_lava"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"vanilla_int_{ds}.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
