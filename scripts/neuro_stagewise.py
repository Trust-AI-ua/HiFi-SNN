#!/usr/bin/env python3
"""Stagewise predictions along the execution path (amendment 8, results/preregistration/2026-11-17_neuro_stream.md).

Runs in .venv (no Lava import: the two Lava emulators are copied from src/lava_sublif/sublif.py and
checked against the guarded confirmation runs on the test split). For one dataset, one seed and one
split (val or test), every flow of the split through each stage, with the predicted class and the
per-class score the stage's readout takes the argmax of.

Vanilla SNN (SNN_LeakyParallel, latency; readout = output spike count over T = 25 steps):
  float_gpu         the record's path: the checkpoint on CUDA, the bundle's loader (test: equals the record)
  float64           NumPy float64 replay of the same equations
  proj_w            float64, both weight matrices projected to the per-tensor int8 grid (s = max|W| / 127)
  proj_w_leak12     proj_w with Lava's 12-bit leak, beta' = (4096 - round(4096 (1 - beta))) / 4096
  integer_q8        amendment 4: Q = 8 fixed point, 8-bit leak (test: equals inputs/<ds>_vanilla_int_seed<S>.npz)
  lava              the Lava fixed-point network, bit-exact emulation (test: equals the confirmation's full-test F1
                    and Lava's predictions on the 2,000-flow sample)
  lava_nosat        diagnostic: lava with the 24-bit membrane saturation removed
Tree-to-Spike (production student, and the T = 10 student where exported; test split only; readout = summed
output membrane): float (QAT checkpoint, weights already on the int8 grid, so float = projected weights),
integer_q8 (revision 2, step 2), lava (bit-exact emulation; checked as above).

    .venv/bin/python scripts/neuro_stagewise.py <dataset> <seed> <val|test>
Output: results/neuro/review/p1_stagewise/<ds>_seed<S>_<split>.npz (pred_<model>_<stage>, score_<model>_<stage>,
        labels, flow_id = row index in the split) and <ds>_seed<S>_<split>_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/review/p1_stagewise"
CH = 20000


def _decay(x, d):
    p = x.astype(np.int64) * (4096 - d)
    return (np.sign(p) * (np.abs(p) >> 12)).astype(np.int64)


def lava_vanilla(layers, st, T, du=4095, dv=614, M=2 ** 23):
    """Copy of src.lava_sublif.sublif.emulate_vanilla; M = 2^23 is Lava's 24-bit membrane. Also returns,
    per flow, whether the hidden membrane reached the saturation bound."""
    B = st.shape[0]
    u = [np.zeros((B, L["W"].shape[0]), np.int64) for L in layers]
    v = [np.zeros_like(x) for x in u]
    r = np.zeros_like(v[1]); oc = np.zeros_like(v[1]); cnt = np.zeros((B, 2), np.int64)
    sat = np.zeros(B, bool)
    for t in range(T):
        s = (st == t).astype(np.int64)
        for i, L in enumerate(layers):
            a = s @ L["W"].T.astype(np.int64)
            uu = _decay(u[i], du + 1) + (a << 6)
            uu = np.where(uu > M, uu - 2 * M, uu); uu = np.where(uu <= -M, uu + 2 * M, uu)
            u[i] = uu
            th = L["vth"] << 6
            vv = _decay(v[i], dv) + u[i] + L["bias_eff"]
            if i == 0:
                sat |= (vv >= M - 1).any(1)
                vv = np.clip(vv, 0, M - 1)
            else:
                sat |= (np.abs(vv - r * th) >= M - 1).any(1)
                vv = np.clip(vv - r * th, -M + 1, M - 1)
            s = (vv > th).astype(np.int64)
            if i == 1:
                r = s; oc += s
            v[i] = vv
            cnt[:, i] += s.sum(1)
    return oc, cnt, sat


def lava_t2s(layers, st, T, du=4095, dv=410):
    """Copy of src.lava_sublif.sublif.emulate (reset = subtract). Returns the summed output voltage and per-layer counts."""
    B, M = st.shape[0], 2 ** 23
    u = [np.zeros((B, L["W"].shape[0]), np.int64) for L in layers]
    v = [np.zeros_like(x) for x in u]; r = [np.zeros_like(x) for x in u]
    acc = np.zeros_like(v[-1]); cnt = np.zeros((B, len(layers)), np.int64)
    for t in range(T):
        s = (st == t).astype(np.int64)
        for i, L in enumerate(layers):
            a = s @ L["W"].T.astype(np.int64)
            uu = _decay(u[i], du + 1) + (a << 6)
            uu = np.where(uu > M, uu - 2 * M, uu); uu = np.where(uu <= -M, uu + 2 * M, uu)
            u[i] = uu
            th = L["vth"] << 6
            vv = np.clip(_decay(v[i], dv) + u[i] + L["bias_eff"] - r[i] * th, -M + 1, M - 1)
            s = (vv > th).astype(np.int64)
            v[i], r[i] = vv, s
            cnt[:, i] += s.sum(1)
        acc = acc + v[-1]
    return acc, cnt


def float_vanilla(st, W_ih, b_h, beta_h, thr_h, W_out, b_out, beta_out, thr_out, T):
    """Float64 replay: hidden nn.RNN (ReLU, diagonal recurrence) with threshold and no reset; output snnTorch
    Leaky with delayed subtractive reset; returns output spike counts."""
    B = st.shape[0]
    vh = np.zeros((B, W_ih.shape[0])); vo = np.zeros((B, W_out.shape[0])); oc = np.zeros((B, W_out.shape[0]), np.int64)
    for t in range(T):
        x = (st == t).astype(np.float64)
        vh = np.maximum(0, beta_h * vh + x @ W_ih.T + b_h)
        sh = (vh > thr_h).astype(np.float64)
        r = vo > thr_out
        vo = beta_out * vo + sh @ W_out.T + b_out - r * thr_out
        oc += vo > thr_out
    return oc


def batched(fn, st, *a):
    outs = [fn(st[i:i + CH], *a) for i in range(0, len(st), CH)]
    if isinstance(outs[0], tuple):
        return tuple(np.concatenate([o[k] for o in outs]) for k in range(len(outs[0])))
    return np.concatenate(outs)


def vanilla_split(ds, seed, split):
    """GPU replay of the Vanilla record on the split: spike times, labels, float_gpu predictions and counts."""
    import torch
    import backfill_snn_metrics as BF
    from src.data import get_dataset
    from src.encoding import get_encoder
    from src.metrics import compute_metrics
    from src.models import SNN
    rd = REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_{seed}"
    rec = json.loads((rd / "results.json").read_text())
    cfg = rec["config"]
    b = get_dataset(ds, **BF._dataset_kwargs_for(ds, "LeakyParallel", "latency", cfg))
    loader = b.test_loader if split == "test" else b.val_loader
    dev = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.05, dev)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    T = cfg["num_steps"]
    m = SNN(num_features=b.num_features, hidden_size=cfg["hidden_size"], num_classes=b.num_classes,
            neuron_name="LeakyParallel", num_steps=T).to(dev)
    m.load_state_dict(torch.load(rd / "checkpoint.pt", map_location=dev))
    m.eval()
    enc = get_encoder("latency")
    X, y, oc = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            sp = enc(xb.to(dev), T)
            stack, _, _, _ = m._output_spike_train(sp)
            if not torch.equal(m(sp)[0].argmax(1), stack.sum(0).argmax(1)):
                raise SystemExit("replay readout differs from forward()")
            oc.append(stack.sum(0).cpu().numpy()); X.append(xb.float()); y.append(yb.numpy())
    X, y, oc = torch.cat(X), np.concatenate(y), np.concatenate(oc).astype(np.int64)
    if split == "test":
        got = compute_metrics(y, oc.argmax(1), class_names=b.class_names)["f1_macro"]
        if abs(got - rec["test_plus"]["metrics"]["f1_macro"]) > 1e-12:
            raise SystemExit(f"{ds}/{seed}: GPU replay {got} is not the record")
    xs = X.clamp(0, 1)
    st = torch.floor((1.0 - xs) * (T - 1)).long().clamp(0, T - 1)
    st = torch.where(xs > 0.01, st, torch.full_like(st, -1)).numpy().astype(np.int64)
    chk = enc(X[:4096], T)
    if not torch.equal(chk, (torch.arange(T).view(T, 1, 1) == torch.from_numpy(st[:4096]).unsqueeze(0)).float()):
        raise SystemExit("spike times do not reproduce the latency encoder")
    return st, y, oc, T, list(b.class_names)


def vanilla_params(v, wclip=(100.0, 100.0)):
    """Integer weights and scales for the Vanilla network; wclip = |W| percentile per tensor that maps to 127
    (100 = max |W|, the amendment-4 rule; below 100 the weights beyond it saturate at +-127)."""
    out = {}
    for k, W, p in (("h", v["W_ih"], wclip[0]), ("o", v["W_out"], wclip[1])):
        s = float(np.percentile(np.abs(W), p)) / 127.0 if p < 100 else float(np.abs(W).max()) / 127.0
        out[f"s_{k}"] = s
        out[f"W{k}"] = np.clip(np.round(W / s), -127, 127).astype(np.int64)
    return out


def main():
    ds, seed, split = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    from sklearn.metrics import f1_score
    from neuro_a2_vanilla import vanilla_lava_params
    from neuro_a2_lava import lava_params
    from neuro_vanilla_int import int_vanilla, Q
    f1 = lambda y, p: 100 * f1_score(y, p, average="macro")
    v = dict(np.load(INP / f"{ds}_vanilla_seed{seed}.npz"))
    st, y, oc_gpu, T, names = vanilla_split(ds, seed, split)
    res = {"float_gpu": oc_gpu}
    beta_h = float(np.unique(v["beta_h"])[0])
    bh, bo = v["b_h"].astype(np.float64), v["b_out"].astype(np.float64)
    thr = (float(v["thr_h"]), float(v["thr_out"]))
    fl = lambda Wh, Wo, lh, lo: batched(float_vanilla, st, Wh, bh, lh, thr[0], Wo, bo, lo, thr[1], T)
    res["float64"] = fl(v["W_ih"].astype(np.float64), v["W_out"].astype(np.float64), beta_h, float(v["beta_out"]))
    P = vanilla_params(v)
    Wh_p, Wo_p = P["Wh"] * P["s_h"], P["Wo"] * P["s_o"]
    res["proj_w"] = fl(Wh_p, Wo_p, beta_h, float(v["beta_out"]))
    b12 = (4096 - int(round(4096 * (1 - beta_h)))) / 4096
    res["proj_w_leak12"] = fl(Wh_p, Wo_p, b12, b12)
    IP = dict(Wh=P["Wh"], Wo=P["Wo"], mul=int(round(256 * beta_h)), mul_o=int(round(256 * float(v["beta_out"]))),
              bh=np.round(v["b_h"] / P["s_h"] * 2 ** Q).astype(np.int64), bo=np.round(v["b_out"] / P["s_o"] * 2 ** Q).astype(np.int64),
              th_h=int(round(float(v["thr_h"]) / P["s_h"] * 2 ** Q)), th_o=int(round(float(v["thr_out"]) / P["s_o"] * 2 ** Q)))
    res["integer_q8"] = batched(lambda x: int_vanilla(IP, x, T)[0], st)
    vi = dict(P_Wh=P["Wh"], P_Wo=P["Wo"], s_h=P["s_h"], s_o=P["s_o"])
    vl, _, dv2 = vanilla_lava_params(v, vi)
    lv_oc, lv_cnt, lv_sat = batched(lambda x: lava_vanilla(vl, x, T, dv=dv2), st)
    res["lava"] = lv_oc
    res["lava_nosat"] = batched(lambda x: lava_vanilla(vl, x, T, dv=dv2, M=2 ** 61)[0], st)
    pred = {f"vanilla_{k}": o.argmax(1) for k, o in res.items()}
    score = {f"vanilla_{k}": o.astype(np.int16) for k, o in res.items()}
    checks = {}
    if split == "test":
        vint = np.load(INP / f"{ds}_vanilla_int_seed{seed}.npz")
        if not np.array_equal(vint["labels_full"], y) or not np.array_equal(vint["spike_times_full"].astype(np.int64), st):
            raise SystemExit("test split differs from the amendment-4 export")
        if not np.array_equal(pred["vanilla_integer_q8"], vint["int_pred_full"]):
            raise SystemExit("integer_q8 differs from the amendment-4 integer predictions")
        cj = json.loads((REPO / f"results/neuro/confirm/{ds}_seed{seed}_vanilla_fixed.json").read_text())
        if f1(y, pred["vanilla_lava"]) != cj["simulated_macro_f1_full_test"]:
            raise SystemExit("lava emulation differs from the confirmation's full-test macro-F1")
        import pandas as pd
        pf = pd.read_csv(REPO / f"results/neuro/confirm/per_flow_{ds}_seed{seed}_vanilla.csv")
        if not np.array_equal(pred["vanilla_lava"][vint["sample"]], pf.pred.values):
            raise SystemExit("lava emulation differs from Lava/SANA-FE on the 2,000-flow sample")
        checks["vanilla"] = "integer_q8 == amendment-4 export; lava == confirmation (full-test F1, sample predictions)"
        for tag, fn in (("t2s", f"{ds}_seed{seed}.npz"), ("t2s_T10", f"{ds}_T10_seed{seed}.npz")):
            if not (INP / fn).exists():
                continue
            z = np.load(INP / fn)
            zt = int(z["T"]); sts, ys = z["spike_times"].astype(np.int64), z["labels"]
            lp, _ = lava_params(z)
            acc, _ = batched(lambda x: lava_t2s(lp, x, zt), sts)
            row = "t2s_full" if tag == "t2s" else "T10"
            cj = json.loads((REPO / f"results/neuro/confirm/{ds}_seed{seed}_{row}.json").read_text())
            if f1(ys, acc.argmax(1)) != cj["simulated_macro_f1_full_test"]:
                raise SystemExit(f"{tag}: lava emulation differs from the confirmation's full-test macro-F1")
            pf = pd.read_csv(REPO / f"results/neuro/confirm/per_flow_{ds}_seed{seed}_{'t2s' if tag == 't2s' else 'T10'}_full.csv")
            if not np.array_equal(acc.argmax(1)[z["sample"]], pf.pred.values):
                raise SystemExit(f"{tag}: lava emulation differs from Lava/SANA-FE on the sample")
            same = np.array_equal(ys, y)
            for k, p, sc in (("float", z["float_logits"].argmax(1), z["float_logits"].astype(np.float32)),
                             ("integer_q8", z["int_pred"], None), ("lava", acc.argmax(1), acc.astype(np.float32))):
                pred[f"{tag}_{k}"] = p
                if sc is not None:
                    score[f"{tag}_{k}"] = sc
            checks[tag] = f"lava == confirmation; same test split and order as Vanilla: {same}"
            if not same:
                raise SystemExit(f"{tag}: test split differs from the Vanilla test split")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{ds}_seed{seed}_{split}.npz", flow_id=np.arange(len(y)), labels=y,
                        class_names=np.array(names), lava_hidden_saturated=lv_sat,
                        **{f"pred_{k}": p.astype(np.int8) for k, p in pred.items()},
                        **{f"score_{k}": s for k, s in score.items()})
    ref = pred["vanilla_float_gpu"]
    summ = dict(dataset=ds, seed=seed, split=split, n=int(len(y)), T_vanilla=T, checks=checks,
                weight_scale=dict(s_h=P["s_h"], s_o=P["s_o"], max_abs_W_ih=float(np.abs(v["W_ih"]).max()),
                                  max_abs_W_out=float(np.abs(v["W_out"]).max())),
                lava_hidden_saturated_frac=float(lv_sat.mean()),
                float_gpu_ties_frac=float(((oc_gpu == oc_gpu.max(1, keepdims=True)).sum(1) > 1).mean()),
                stages={k: dict(macro_f1=f1(y, p), agreement_with_vanilla_float_gpu=float((p == ref).mean()) if k.startswith("vanilla") else None,
                                per_class_f1=dict(zip(names, (100 * f1_score(y, p, average=None, labels=range(len(names)))).round(4).tolist())))
                        for k, p in pred.items()})
    (OUT / f"{ds}_seed{seed}_{split}_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: round(s["macro_f1"], 2) for k, s in summ["stages"].items()}),
          "sat", round(summ["lava_hidden_saturated_frac"], 4), "ties", round(summ["float_gpu_ties_frac"], 4), flush=True)


if __name__ == "__main__":
    main()
