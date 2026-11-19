#!/usr/bin/env python3
"""A3 input for the Vanilla SNN (results/preregistration/2026-11-17_neuro_stream.md).

Runs in .venv. Rebuilds the reported seed-42 Vanilla SNN (SNN_LeakyParallel, latency) and its
test split exactly as scripts/backfill_snn_metrics.py replays it, checks that the replay
reproduces the record's test macro-F1, and writes results/neuro/inputs/<ds>_vanilla_seed42.npz:
  spike_times [n, F] int8 for the A2/A3 sample (-1 = the feature never spikes), labels, pred,
  hidden_counts [n], out_counts [n] (spikes over the window), and the weights:
  W_ih [H, F], b_h [H] (b_ih + b_hh), beta_h [H] (diagonal of weight_hh), thr_h, W_out [K, H],
  b_out [K], beta_out, thr_out, T.
The sample: the Tree-to-Spike sample indices if the two test splits are identical (same labels
in the same order), otherwise the same stratified rule (scripts/neuro_export.py) on this split.

    .venv/bin/python scripts/neuro_export_vanilla.py <dataset> [seed]   (default 42; amendment 6 adds 43-46)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))


def main():
    ds = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    import backfill_snn_metrics as BF
    from neuro_export import stratified_sample
    from src.data import get_dataset
    from src.encoding import get_encoder
    from src.metrics import compute_metrics
    from src.models import SNN
    rd = REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_{seed}"
    rec = json.loads((rd / "results.json").read_text())
    cfg = rec["config"]
    b = get_dataset(ds, **BF._dataset_kwargs_for(ds, "LeakyParallel", "latency", cfg))
    # The record was evaluated on CUDA with the bundle's test loader (batch 128); LeakyParallel's
    # nn.RNN runs through cuDNN there, and the CPU replay does not reproduce it (Edge-IIoTset:
    # 77.55 vs 77.04). Replay exactly as src/training._eval ran, on the (shared) GPU.
    dev = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(0.05, dev)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    X = torch.cat([x for x, _ in b.test_loader]).float()
    y = torch.cat([t for _, t in b.test_loader]).numpy()
    T = cfg["num_steps"]
    m = SNN(num_features=b.num_features, hidden_size=cfg["hidden_size"], num_classes=b.num_classes,
            neuron_name="LeakyParallel", num_steps=T).to(dev)
    m.load_state_dict(torch.load(rd / "checkpoint.pt", map_location=dev))
    m.eval()
    enc = get_encoder("latency")
    preds, hid, outc = [], [], []
    with torch.no_grad():
        for xb, _ in b.test_loader:
            sp = enc(xb.to(dev), T)
            logits = m(sp)[0]
            stack, _, _, _ = m._output_spike_train(sp)
            if not torch.equal(logits.argmax(1), stack.sum(0).argmax(1)):
                raise SystemExit("replay readout differs from forward()")
            preds.append(logits.argmax(1).cpu().numpy())
            hid.append(m.lp(sp).sum((0, 2)).cpu().numpy()); outc.append(stack.sum((0, 2)).cpu().numpy())
    pred = np.concatenate(preds)
    got = compute_metrics(y, pred, class_names=b.class_names)["f1_macro"]
    if abs(got - rec["test_plus"]["metrics"]["f1_macro"]) > 1e-12:
        raise SystemExit(f"{ds}: Vanilla replay {got!r} is not the record {rec['test_plus']['metrics']['f1_macro']!r}")
    t2s = np.load(REPO / f"results/neuro/inputs/{ds}_seed{seed}.npz")
    same = len(t2s["labels"]) == len(y) and np.array_equal(t2s["labels"], y)
    samp = t2s["sample"] if same else stratified_sample(y)
    xs = X[samp].clamp(0, 1)
    times = torch.floor((1.0 - xs) * (T - 1)).long().clamp(0, T - 1)
    times = torch.where(xs > 0.01, times, torch.full_like(times, -1)).numpy().astype(np.int8)
    sp = enc(X[samp], T)                      # encoder on CPU: the same float32 floor as on the GPU (checked below)
    chk = (torch.arange(T).view(T, 1, 1) == torch.from_numpy(times.astype(np.int64)).unsqueeze(0)).float()
    if not torch.equal(chk, sp):
        raise SystemExit("spike-time export does not reproduce the latency encoder")
    if not torch.equal(enc(X[samp].to(dev), T).cpu(), sp):
        raise SystemExit("latency encoder differs between CPU and GPU")
    sd = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    whh = sd["lp.rnn.weight_hh_l0"]
    if np.abs(whh - np.diag(np.diag(whh))).max() > 0:
        raise SystemExit("weight_hh is not diagonal")
    np.savez_compressed(REPO / f"results/neuro/inputs/{ds}_vanilla_seed{seed}.npz", spike_times=times, labels=y[samp],
                        pred=pred[samp], hidden_counts=np.concatenate(hid)[samp], out_counts=np.concatenate(outc)[samp],
                        W_ih=sd["lp.rnn.weight_ih_l0"], b_h=sd["lp.rnn.bias_ih_l0"] + sd["lp.rnn.bias_hh_l0"],
                        beta_h=np.diag(whh), thr_h=np.float64(sd["lp.threshold"]), W_out=sd["fc.weight"],
                        b_out=sd["fc.bias"], beta_out=np.float64(sd["lif_out.beta"]),
                        thr_out=np.float64(sd["lif_out.threshold"]), T=np.int64(T), same_split_as_t2s=same)
    print(f"[done] {ds}: Vanilla F1 {100 * got:.2f} reproduced; same split as T2S: {same}; "
          f"T={T}, F={X.shape[1]}, H={cfg['hidden_size']}; sample hidden spikes/flow {np.concatenate(hid)[samp].mean():.1f}",
          flush=True)


if __name__ == "__main__":
    main()
