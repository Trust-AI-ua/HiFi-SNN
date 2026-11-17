#!/usr/bin/env python3
"""B2, stage 2 (GPU): Tree-to-Spike-Stream (results/preregistration/2026-11-17_neuro_stream.md).

Encoder: the production TreeGuidedSpikeEncoder fitted on the per-packet teacher (stage 1)
gives the kept features and their level edges (importance pruning at the 0.20 quantile). At
each packet each kept feature's level index is computed; an "up" channel spikes when it rises,
a "down" channel when it falls, nothing when unchanged (the level before packet 1 is that of
an all-zero state). No input after the flow's last packet.
Student: SNNStudent (64, 64 LIF, beta 0.9, subtractive reset), one step per packet, 32 steps;
running readout after packet k = mean output membrane over packets 1..k. Loss: the production
distillation loss (src/training_treespike.distillation_loss, production weights) on the running
readout at every packet up to the flow's length, against the teacher's posterior on the same
prefix. 10 epochs, batch 128, Adam lr 0.002 (the Edge-IIoTset production settings). Epoch
selection on validation macro-F1 at decision (src/stream_decision: per-packet temperatures and
gate refitted on validation each epoch). Deployed int8 (production quantize_int8_).

Shares the GPU with the revision-2 training queue (PI decision, 2026-11-17), so it keeps off the
CPU: every tensor (prefix features, teacher posteriors, spike trains) is moved to the GPU once
and batches are indexed there (no DataLoader, no workers); torch uses 2 CPU threads; GPU memory
is capped with set_per_process_memory_fraction. Deterministic: production _set_seed (cuDNN
deterministic, no benchmark) plus torch.use_deterministic_algorithms. Sharing the device does
not change either job's results; B2 counts operations and does not time them. The device
state at start is recorded in the report.

    python scripts/stream_b2_student.py <seed> [--mem-fraction 0.12] [--smoke]
Output: results/neuro/b2/seed_<S>/{student.pt, student_report.json}; outside git,
        dataset/edge-iiotset/_stream/b2/seed_<S>_student.npz (val/test running logits [n, 32, K],
        cumulative SOPs [n, 32] (input-inclusive), input spikes per packet [n, 32])
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # required by deterministic cuBLAS

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.edgeiiot_stream import load_stream  # noqa: E402
from src import stream_decision as SD  # noqa: E402

OUT = REPO / "results/neuro/b2"
BIG = REPO / "dataset/edge-iiotset/_stream/b2"
P = 32


class StreamEncoder:
    """Level edges from the teacher; spike trains computed on the GPU (torch.searchsorted)."""

    def __init__(self, teacher, n_features, device):
        from src.models.tree_to_spike import TreeGuidedSpikeEncoder
        cfg = yaml.safe_load(open(REPO / "experiments/Tree-to-Spike/edgeiiot/config.yaml"))
        enc = TreeGuidedSpikeEncoder(T=P, pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                     jitter=cfg["jitter"], encoder_mode="tree").fit(teacher, n_features)
        self.feats = sorted({j for j, _ in enc.channels_})
        self.edges = [np.asarray(enc.edges_[j], np.float32) for j in self.feats]
        self.edges_t = [torch.from_numpy(e).to(device) for e in self.edges]
        self.l0 = torch.tensor([int(np.searchsorted(e, np.float32(0), side="right")) for e in self.edges], device=device)
        self.out_dim = 2 * len(self.feats)

    def spikes(self, X, L):
        """X [n, P, F] float32 (GPU), L [n] (GPU) -> [n, P, 2J] bool (GPU)."""
        lv = torch.stack([torch.searchsorted(e, X[:, :, j].contiguous(), right=True)
                          for j, e in zip(self.feats, self.edges_t)], -1)
        prev = torch.cat([self.l0.view(1, 1, -1).expand(len(X), 1, -1), lv[:, :-1]], 1)
        valid = (torch.arange(P, device=X.device)[None, :] < L[:, None])[:, :, None]
        return torch.cat([(lv > prev) & valid, (lv < prev) & valid], -1)

    def spikes_numpy(self, X, L):
        """CPU reference of spikes(), for the guard."""
        lv = np.stack([np.searchsorted(e, X[:, :, j], side="right") for j, e in zip(self.feats, self.edges)], -1)
        l0 = self.l0.cpu().numpy()
        prev = np.concatenate([np.broadcast_to(l0, (len(X), 1, len(self.feats))), lv[:, :-1]], 1)
        valid = (np.arange(P)[None, :] < L[:, None])[:, :, None]
        return np.concatenate([(lv > prev) & valid, (lv < prev) & valid], -1)


def unroll(st, spk):
    """spk [B, P, C] float. Running readout [B, P, K], hidden spikes per step [B, P] (l1, l2), SOPs [B, P]."""
    m1, m2, m3 = st.lif1.init_leaky(), st.lif2.init_leaky(), st.lif3.init_leaky()
    cum, zs, s1n, s2n = 0.0, [], [], []
    for t in range(spk.shape[1]):
        s1, m1 = st.lif1(st.fc1(spk[:, t]), m1)
        s2, m2 = st.lif2(st.fc2(s1), m2)
        _, m3 = st.lif3(st.fc3(s2), m3)
        cum = cum + m3
        zs.append(cum / (t + 1)); s1n.append(s1.sum(1)); s2n.append(s2.sum(1))
    s1n, s2n = torch.stack(s1n, 1), torch.stack(s2n, 1)
    sops = spk.sum(2) * st.fanouts[0] + s1n * st.fanouts[1] + s2n * st.fanouts[2]
    return torch.stack(zs, 1), s1n, s2n, sops


@torch.no_grad()
def infer(st, S, bs=8192):
    """S: GPU spike trains [n, P, C] bool. Returns numpy running logits, cumulative SOPs, input spikes per packet."""
    Z, C, I = [], [], []
    for i in range(0, len(S), bs):
        spk = S[i:i + bs].float()
        z, _, _, sops = unroll(st, spk)
        Z.append(z.cpu().numpy()); C.append(torch.cumsum(sops, 1).cpu().numpy()); I.append(spk.sum(2).cpu().numpy())
    return np.concatenate(Z), np.concatenate(C), np.concatenate(I)


def decision_f1(Z, y, L, cum):
    taus = SD.fit_taus(Z, y, L)
    prob = SD.calibrated(Z, taus)
    theta, _ = SD.gate(prob, y, L, cum)
    _, Pd = SD.decide(prob, L, theta)
    return SD.scores(y, Pd.argmax(1))["macro_f1"], theta


def device_state():
    q = "--query-compute-apps=pid,process_name,used_memory"
    try:
        apps = subprocess.run(["nvidia-smi", q, "--format=csv,noheader"], capture_output=True, text=True).stdout
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        apps = gpu = "nvidia-smi unavailable"
    return dict(gpu=gpu.strip(), other_compute_processes=[l.strip() for l in apps.splitlines() if l.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--mem-fraction", type=float, default=0.12)
    ap.add_argument("--smoke", action="store_true", help="5,000 training flows, 1 epoch, no outputs kept")
    a = ap.parse_args()
    torch.set_num_threads(2)
    from xgboost import XGBClassifier
    from src.models.tree_to_spike import SNNStudent, quantize_int8_
    from src.training_treespike import distillation_loss, _set_seed
    cfg = yaml.safe_load(open(REPO / "experiments/Tree-to-Spike/edgeiiot/config.yaml"))
    seed, dev = a.seed, torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction, dev)
    torch.use_deterministic_algorithms(True)
    shared = device_state()
    s = load_stream(seed)
    K = len(s.class_names)
    teacher = XGBClassifier(importance_type="gain")
    teacher.load_model(OUT / f"seed_{seed}/teacher.json")
    tz = np.load(BIG / f"seed_{seed}_teacher.npz")
    enc = StreamEncoder(teacher, s.X["train"].shape[2], dev)
    idx_tr = np.arange(len(s.y["train"]))
    if a.smoke:
        idx_tr = np.sort(np.random.default_rng(0).choice(len(idx_tr), 5000, replace=False))
    # everything onto the GPU once
    G = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    L = {k: G(s.length[k].astype(np.int64)) for k in ("train", "val", "test")}
    S = {}
    for k in ("train", "val", "test"):
        Xk = G(s.X[k])
        S[k] = enc.spikes(Xk, L[k])
        del Xk
    chk = np.random.default_rng(1).choice(len(s.y["val"]), 2000, replace=False)
    if not np.array_equal(S["val"][torch.from_numpy(chk).to(dev)].cpu().numpy(),
                          enc.spikes_numpy(s.X["val"][chk], s.length["val"][chk])):
        raise SystemExit("GPU spike trains differ from the NumPy encoder")
    S["train"], L["train"] = S["train"][G(idx_tr)], L["train"][G(idx_tr)]
    ytr = G(s.y["train"][idx_tr].astype(np.int64))
    Ttr = G(tz["train_proba"][idx_tr])
    del tz
    epochs = 1 if a.smoke else cfg["epochs"]
    _set_seed(seed)
    st = SNNStudent(enc.out_dim, K, hidden=tuple(cfg["hidden"]), beta=cfg["beta"], leaky=cfg["leaky"]).to(dev)
    opt = torch.optim.Adam(st.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs, N = cfg["batch_size"], len(idx_tr)
    ar = torch.arange(P, device=dev)[None, :]
    curve, best, best_state = [], -1.0, None
    t0 = time.time()
    for ep in range(epochs):
        st.train()
        perm = torch.randperm(N).to(dev)
        run, nb = 0.0, 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            z, s1n, s2n, _ = unroll(st, S["train"][idx].float())
            valid = ar < L["train"][idx][:, None]
            yv = ytr[idx][:, None].expand(-1, P)[valid]
            rate = (s1n + s2n)[valid].sum() / (valid.sum() * sum(cfg["hidden"]))
            loss, _ = distillation_loss(z[valid], yv, Ttr[idx][valid], rate, tau=cfg["tau"], a_kd=cfg["a_kd"],
                                        a_rank=cfg["a_rank"], a_sparse=cfg["a_sparse"], a_fp=cfg["a_fp"])
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss.item()); nb += 1
        st.eval()
        Zv, Cv, _ = infer(st, S["val"])
        f1, theta = decision_f1(Zv, s.y["val"], s.length["val"], Cv)
        curve.append(dict(epoch=ep + 1, loss=run / max(1, nb), val_macro_f1_at_decision=f1, theta=theta,
                          seconds=round(time.time() - t0, 1)))
        print(curve[-1], flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.detach().clone() for k, v in st.state_dict().items()}
    st.load_state_dict(best_state)
    quantize_int8_(st)
    st.eval()
    res = dict(seed=seed, channels=enc.out_dim, kept_features=[s.features[j] for j in enc.feats],
               edges_per_feature=[int(len(e)) for e in enc.edges], curve=curve,
               selected_epoch=int(max(curve, key=lambda c: c["val_macro_f1_at_decision"])["epoch"]),
               train_seconds=round(time.time() - t0, 1), smoke=a.smoke,
               execution=dict(shared_gpu_with="revision-2 training queue (steps 5 and 6), PI decision 2026-11-17",
                              device_at_start=shared, memory_fraction=a.mem_fraction, torch_threads=2,
                              data_loader_workers=0, deterministic_algorithms=True,
                              cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                              max_gpu_memory_allocated_GB=round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2),
                              note="sharing the device changes neither job's results; B2 counts operations, not time"))
    if a.smoke:
        print(json.dumps(res, indent=1)); return
    out = {}
    for split in ("val", "test"):
        Z, C, I = infer(st, S[split])
        out.update({f"{split}_logits": Z.astype(np.float32), f"{split}_cum_sops": C.astype(np.float32),
                    f"{split}_in_spikes": I.astype(np.float32)})
    np.savez(BIG / f"seed_{seed}_student.npz", **out)
    torch.save(st.state_dict(), OUT / f"seed_{seed}/student.pt")
    (OUT / f"seed_{seed}/student_report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
