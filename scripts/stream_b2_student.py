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

    python scripts/stream_b2_student.py <seed> [--device cuda] [--smoke]
Output: results/neuro/b2/seed_<S>/{student.pt, student_report.json}; outside git,
        dataset/edge-iiotset/_stream/b2/seed_<S>_student.npz (val/test running logits [n, 32, K],
        cumulative SOPs [n, 32] (input-inclusive), input spikes per packet [n, 32])
"""
from __future__ import annotations

import argparse
import json
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
    def __init__(self, teacher, n_features):
        from src.models.tree_to_spike import TreeGuidedSpikeEncoder
        cfg = yaml.safe_load(open(REPO / "experiments/Tree-to-Spike/edgeiiot/config.yaml"))
        enc = TreeGuidedSpikeEncoder(T=P, pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                     jitter=cfg["jitter"], encoder_mode="tree").fit(teacher, n_features)
        self.feats = sorted({j for j, _ in enc.channels_})
        self.edges = [np.asarray(enc.edges_[j], np.float32) for j in self.feats]
        self.l0 = np.array([np.searchsorted(e, np.float32(0), side="right") for e in self.edges])
        self.out_dim = 2 * len(self.feats)

    def spikes(self, X, L):
        lv = np.stack([np.searchsorted(e, X[:, :, j], side="right") for j, e in zip(self.feats, self.edges)], -1)
        prev = np.concatenate([np.broadcast_to(self.l0, (len(X), 1, len(self.feats))), lv[:, :-1]], 1)
        valid = (np.arange(P)[None, :] < L[:, None])[:, :, None]
        return np.concatenate([(lv > prev) & valid, (lv < prev) & valid], -1)        # [n, P, 2J] bool


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
def infer(st, enc, X, L, device, bs=4096):
    Z, S, I = [], [], []
    for i in range(0, len(X), bs):
        spk = torch.from_numpy(enc.spikes(X[i:i + bs], L[i:i + bs]).astype(np.float32)).to(device)
        z, _, _, sops = unroll(st, spk)
        Z.append(z.cpu().numpy()); S.append(sops.cpu().numpy()); I.append(spk.sum(2).cpu().numpy())
    return np.concatenate(Z), np.cumsum(np.concatenate(S), 1), np.concatenate(I)


def decision_f1(Z, y, L, cum):
    taus = SD.fit_taus(Z, y, L)
    prob = SD.calibrated(Z, taus)
    theta, _ = SD.gate(prob, y, L, cum)
    _, Pd = SD.decide(prob, L, theta)
    return SD.scores(y, Pd.argmax(1))["macro_f1"], theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="5,000 training flows, 1 epoch, no outputs kept")
    a = ap.parse_args()
    from xgboost import XGBClassifier
    from src.models.tree_to_spike import SNNStudent
    from src.training_treespike import distillation_loss, _set_seed
    from src.models.tree_to_spike import quantize_int8_
    cfg = yaml.safe_load(open(REPO / "experiments/Tree-to-Spike/edgeiiot/config.yaml"))
    seed, device = a.seed, torch.device(a.device)
    s = load_stream(seed)
    K = len(s.class_names)
    teacher = XGBClassifier(importance_type="gain")
    teacher.load_model(OUT / f"seed_{seed}/teacher.json")
    tz = np.load(BIG / f"seed_{seed}_teacher.npz")
    enc = StreamEncoder(teacher, s.X["train"].shape[2])
    Xtr, Ltr, ytr, Ttr = s.X["train"], s.length["train"], s.y["train"], tz["train_proba"]
    if a.smoke:
        sub = np.random.default_rng(0).choice(len(ytr), 5000, replace=False)
        Xtr, Ltr, ytr, Ttr = Xtr[sub], Ltr[sub], ytr[sub], Ttr[sub]
    epochs = 1 if a.smoke else cfg["epochs"]
    _set_seed(seed)
    st = SNNStudent(enc.out_dim, K, hidden=tuple(cfg["hidden"]), beta=cfg["beta"], leaky=cfg["leaky"]).to(device)
    opt = torch.optim.Adam(st.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs, N = cfg["batch_size"], len(ytr)
    curve, best, best_state = [], -1.0, None
    t0 = time.time()
    for ep in range(epochs):
        st.train()
        perm = torch.randperm(N).numpy()
        run = 0.0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            spk = torch.from_numpy(enc.spikes(Xtr[idx], Ltr[idx]).astype(np.float32)).to(device)
            z, s1n, s2n, _ = unroll(st, spk)
            valid = torch.from_numpy(np.arange(P)[None, :] < Ltr[idx][:, None]).to(device)
            yv = torch.from_numpy(ytr[idx]).to(device)[:, None].expand(-1, P)[valid]
            tp = torch.from_numpy(Ttr[idx]).to(device)[valid]
            rate = (s1n + s2n)[valid].sum() / (valid.sum() * sum(cfg["hidden"]))
            loss, parts = distillation_loss(z[valid], yv, tp, rate, tau=cfg["tau"], a_kd=cfg["a_kd"],
                                            a_rank=cfg["a_rank"], a_sparse=cfg["a_sparse"], a_fp=cfg["a_fp"])
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss.item())
        st.eval()
        Zv, Cv, _ = infer(st, enc, s.X["val"], s.length["val"], device)
        f1, theta = decision_f1(Zv, s.y["val"], s.length["val"], Cv)
        curve.append(dict(epoch=ep + 1, loss=run / max(1, (N + bs - 1) // bs), val_macro_f1_at_decision=f1, theta=theta))
        print(curve[-1], flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.detach().clone() for k, v in st.state_dict().items()}
    st.load_state_dict(best_state)
    quantize_int8_(st)
    st.eval()
    res = dict(seed=seed, channels=enc.out_dim, kept_features=[s.features[j] for j in enc.feats],
               edges_per_feature=[int(len(e)) for e in enc.edges], curve=curve,
               selected_epoch=int(max(curve, key=lambda c: c["val_macro_f1_at_decision"])["epoch"]),
               train_seconds=round(time.time() - t0, 1), smoke=a.smoke)
    if a.smoke:
        print(json.dumps(res, indent=1)); return
    out = {}
    for split in ("val", "test"):
        Z, C, I = infer(st, enc, s.X[split], s.length[split], device)
        out.update({f"{split}_logits": Z.astype(np.float32), f"{split}_cum_sops": C.astype(np.float32),
                    f"{split}_in_spikes": I.astype(np.float32)})
    np.savez(BIG / f"seed_{seed}_student.npz", **out)
    torch.save(st.state_dict(), OUT / f"seed_{seed}/student.pt")
    (OUT / f"seed_{seed}/student_report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
