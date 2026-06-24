"""TASNN — Transformer-Augmented Spiking Neural Network (competitor baseline).

Faithful reimplementation of the *learned core* of TASNN, *"Energy-efficient
intrusion detection with a protocol-aware transformer-spiking hybrid model"*
(Scientific Reports 2026, 16:7095), adapted as a **fair** competitor in this repo.

It ports the reference model's architecture (tasnn.py in the reference release):

    MASE spike encoding      TTFS (learnable latency alpha) + population coding (Eq.10-11)
    FlowToken Transformer    feature-token transformer + [CLS]               (Eq.13)
    Spike-Temporal Conv      causal 1-D temporal convolution                  (Eq.15)
    LIF core                 2 LIF layers w/ lateral inhibition + surrogate   (Eq.16-17)
    XMG cross-modal gating   g = sigmoid(W_g h_TR) gates the spike current    (Eq.19)
    Dual heads               transformer head + spiking readout head          (Eq.24-25)
    Loss suite               focal + KD + homeostasis + energy + gate-entropy (Eq.26-32)

FAIRNESS vs FAITHFULNESS (read this)
------------------------------------
The competitor is given its **real model and training objective** so it is not a
strawman — but it is fed the **same data footing as every other model here**:
shared loaders (one-hot + log1p + MinMax->[0,1]), **no resampling**, and the same
splits. The reference's *preprocessing* stages — PAAN, Port-Role embeddings (PRE),
CPSMOTE-Tomek rebalancing, and Pseudo-Flow Reconstruction (PFR) — are deliberately
omitted (CPSMOTE-Tomek in particular would inflate minority-class metrics). The few
*model* pieces that are inseparable from that preprocessing are therefore dropped,
each flagged below:
  * EDC (Eventified Delta Coding) needs PFR delta channels -> omitted.
  * Neighbour flow-mate tokens + protocol-positional embeddings need PFR micro-flows
    and PRE protocol ids -> the FlowToken transformer runs on per-feature tokens
    ([CLS] + one token per input feature) instead of [CLS, cur, K neighbours].
  * Rank-order coding of categorical embeddings needs PRE embeddings -> omitted;
    the spike path is driven by MASE on the (already one-hot+scaled) feature vector.
The class-rebalancing knob (focal `alpha`) is gated by ``class_weighted`` and OFF by
default, matching every other model here, so no model gets a class-balance advantage.

Exposes ``forward(x) -> dict`` plus ``predict_early_exit`` for the shared
metric/energy/early-exit machinery (the trainer reads the dict).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Surrogate-gradient spike (Eq. 17)
# =====================================================================
class _SurrogateSpike(torch.autograd.Function):
    """Heaviside forward; sigmoid-derivative surrogate backward (Eq. 17)."""

    @staticmethod
    def forward(ctx, v_minus_thr, beta):
        ctx.save_for_backward(v_minus_thr)
        ctx.beta = beta
        return (v_minus_thr >= 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        sig = torch.sigmoid(ctx.beta * v)
        return grad_out * ctx.beta * sig * (1.0 - sig), None


def spike_fn(v_minus_thr, beta):
    return _SurrogateSpike.apply(v_minus_thr, beta)


# =====================================================================
# MASE — Multi-Scale Adaptive Spike Encoding (Eq. 10-11)
# =====================================================================
class MASE(nn.Module):
    """TTFS (learnable latency) + population Poisson coding of the feature vector.

    Input z: [B, F] in [0,1] (our MinMax features) -> spikes [B, T, F*(1+pop_M)].
    Receptive-field centres span [0,1] to match the MinMax range (the reference
    used [-3,3] for PAAN-Gaussianised inputs). Straight-through estimators keep
    the learnable TTFS scale differentiable.
    """

    def __init__(self, num_features: int, T: int, pop_M: int = 3,
                 tau0: float = 10.0, lam0: float = 0.8,
                 rf_min: float = 0.0, rf_max: float = 1.0):
        super().__init__()
        self.F, self.T, self.M = num_features, T, pop_M
        self.tau0, self.lam0 = tau0, lam0
        self.log_alpha = nn.Parameter(torch.zeros(num_features))   # alpha>0 (Eq.10)
        mu = torch.linspace(rf_min, rf_max, pop_M)
        self.register_buffer("mu", mu)
        sigma = torch.full((pop_M,), (rf_max - rf_min) / max(pop_M, 1))
        self.register_buffer("sigma", sigma)

    def forward(self, z: torch.Tensor) -> torch.Tensor:   # z: [B, F]
        B, dev, T = z.shape[0], z.device, self.T
        # ---- TTFS (Eq. 10): later spike for smaller value ----
        alpha = F.softplus(self.log_alpha)                          # [F]
        t_star = (self.tau0 * torch.exp(-alpha[None, :] * z)).clamp(0, T - 1)
        t_idx = torch.arange(T, device=dev).float()
        logits = -(t_idx[None, None, :] - t_star[:, :, None]) ** 2 / 0.5
        soft = F.softmax(logits, dim=-1)                            # [B, F, T]
        hard = F.one_hot(soft.argmax(-1), T).float()
        ttfs = (hard + (soft - soft.detach())).permute(0, 2, 1)     # [B, T, F]
        # ---- population coding (Eq. 11): Poisson per receptive field ----
        r = torch.exp(-(z[:, :, None] - self.mu[None, None, :]) ** 2 /
                      (2 * self.sigma[None, None, :] ** 2))         # [B, F, M]
        rate = (self.lam0 * r).clamp(0, 1)
        rate_T = rate[:, None].expand(B, T, self.F, self.M)
        pop = rate_T + (torch.bernoulli(rate_T) - rate_T).detach()  # ST Bernoulli
        pop = pop.reshape(B, T, self.F * self.M)
        return torch.cat([ttfs, pop], dim=2)                        # [B,T,F*(1+M)]


# =====================================================================
# FlowToken Transformer Encoder (FTE) — Eq. 13 (per-feature token adaptation)
# =====================================================================
class FlowTokenEncoder(nn.Module):
    """Feature-tokeniser + [CLS] + multi-head self-attention; returns the [CLS]
    embedding h_TR. Without PFR micro-flows we tokenise the record's own features
    ([CLS] + one token per feature) rather than [CLS, cur, K neighbours]."""

    def __init__(self, num_features: int, d_model: int, n_heads: int,
                 n_layers: int, ff: int, dropout: float = 0.1):
        super().__init__()
        self.feat_emb = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        self.feat_bias = nn.Parameter(torch.zeros(num_features, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B, F]
        B = x.shape[0]
        tok = x.unsqueeze(-1) * self.feat_emb + self.feat_bias      # [B, F, d]
        seq = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)   # [B, F+1, d]
        return self.norm(self.encoder(seq)[:, 0])                   # h_TR [B, d]


# =====================================================================
# Spike-Temporal Conv (STC) — Eq. 15  (causal temporal conv)
# =====================================================================
class STC(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int = 3):
        super().__init__()
        self.pad = kernel - 1
        self.conv = nn.Conv1d(c_in, c_out, kernel)

    def forward(self, s: torch.Tensor) -> torch.Tensor:   # [B, T, c_in]
        x = F.pad(s.transpose(1, 2), (self.pad, 0))        # causal left pad
        return self.conv(x).transpose(1, 2)                # [B, T, c_out]


# =====================================================================
# LIF layer with lateral inhibition (Eq. 16-17)
# =====================================================================
class LIFLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, beta: float,
                 v_th: float, gamma: float, surrogate_beta: float):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.lateral = nn.Parameter(0.01 * torch.randn(out_features, out_features))
        self.register_buffer("no_self", 1.0 - torch.eye(out_features))
        self.out_features = out_features
        self.beta, self.v_th, self.gamma, self.sbeta = beta, v_th, gamma, surrogate_beta

    def forward(self, drive: torch.Tensor) -> torch.Tensor:   # [B, T, in]
        B, T, _ = drive.shape
        V = drive.new_zeros(B, self.out_features)
        s_prev = drive.new_zeros(B, self.out_features)
        lateral = self.lateral * self.no_self
        out = []
        for t in range(T):
            I_in = self.fc(drive[:, t])
            lat = s_prev @ lateral.t()
            V = self.beta * V + I_in - self.gamma * lat
            s = spike_fn(V - self.v_th, self.sbeta)
            V = V * (1.0 - s)                                  # reset on spike
            s_prev = s
            out.append(s)
        return torch.stack(out, dim=1)                         # [B, T, out]


# =====================================================================
# Full TASNN model
# =====================================================================
class TASNN(nn.Module):
    """forward(x: [B, F]) -> dict with y_tr, y_snn, gate, mean_rate,
    spikes_per_sample, total_spikes (scalar), spike_sites (int)."""

    def __init__(self, num_features: int, num_classes: int, num_steps: int = 16,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 ff: int = 256, pop_M: int = 3, stc_channels: int = 64,
                 stc_kernel: int = 3, lif_hidden=(128, 64), lif_beta: float = 0.9,
                 lif_v_th: float = 1.0, lif_gamma: float = 0.1,
                 surrogate_beta: float = 5.0, target_rate: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        self.T = num_steps
        self.target_rate = target_rate
        h1, h2 = lif_hidden
        # transformer path
        self.fte = FlowTokenEncoder(num_features, d_model, n_heads, n_layers, ff, dropout)
        # spike path
        self.mase = MASE(num_features, num_steps, pop_M)
        c_spike = num_features * (1 + pop_M)
        self.stc = STC(c_spike, stc_channels, stc_kernel)
        self.xmg = nn.Linear(d_model, stc_channels)                # gate (Eq.19)
        self.lif1 = LIFLayer(stc_channels, h1, lif_beta, lif_v_th, lif_gamma, surrogate_beta)
        self.lif2 = LIFLayer(h1, h2, lif_beta, lif_v_th, lif_gamma, surrogate_beta)
        # dual heads (Eq. 24-25)
        self.head_tr = nn.Linear(d_model, num_classes)
        self.head_snn = nn.Linear(h2, num_classes)

    def _core(self, x: torch.Tensor):
        """Shared forward up to the spike trains. Returns (h_tr, gate, sp1, sp2)."""
        h_tr = self.fte(x)                              # [B, d_model]
        S = self.mase(x)                                # [B, T, c_spike]
        I = self.stc(S)                                 # [B, T, stc_channels]
        g = torch.sigmoid(self.xmg(h_tr))               # [B, stc_channels] (Eq.19)
        sp1 = self.lif1(g[:, None, :] * I)              # [B, T, h1]
        sp2 = self.lif2(sp1)                            # [B, T, h2]
        return h_tr, g, sp1, sp2

    def forward(self, x: torch.Tensor) -> Dict:
        h_tr, g, sp1, sp2 = self._core(x)
        readout = sp2.sum(dim=1)                                   # r_i (Eq.25)
        total_neurons = sp1.shape[2] + sp2.shape[2]
        spikes_per_sample = sp1.sum(dim=(1, 2)) + sp2.sum(dim=(1, 2))   # [B]
        total_spikes = spikes_per_sample.sum()
        spike_sites = sp1.shape[0] * self.T * total_neurons
        return {
            "y_tr": self.head_tr(h_tr),
            "y_snn": self.head_snn(readout),
            "gate": g,
            "mean_rate": spikes_per_sample / (self.T * total_neurons),
            "spikes_per_sample": spikes_per_sample,
            "total_spikes": total_spikes,
            "spike_sites": int(spike_sites),
        }

    @staticmethod
    def ensemble_logits(out: Dict) -> torch.Tensor:
        """Inference combination of the two heads (Eq. 24-25): mean of the head
        softmaxes, returned in log-space so it behaves like logits for argmax/AUC."""
        p = 0.5 * (torch.softmax(out["y_tr"], 1) + torch.softmax(out["y_snn"], 1))
        return torch.log(p.clamp_min(1e-9))

    # ---------------- early-exit inference (on the spiking head) ----------------
    @torch.no_grad()
    def predict_early_exit(self, x: torch.Tensor, conf_threshold: float = 0.9):
        h_tr, g, sp1, sp2 = self._core(x)
        B, T, _ = sp2.shape
        total_neurons = sp1.shape[2] + sp2.shape[2]
        total_spikes = sp1.sum() + sp2.sum()
        spike_sites = int(B * T * total_neurons)
        running = torch.cumsum(sp2, dim=1)                         # [B,T,h2]
        y_snn_t = self.head_snn(running)                           # [B,T,C]
        y_tr = self.head_tr(h_tr)[:, None, :]                      # [B,1,C]
        p_t = 0.5 * (torch.softmax(y_snn_t, -1) + torch.softmax(y_tr, -1))  # [B,T,C]
        conf, pred = p_t.max(dim=-1)                               # [B,T]
        exit_step = torch.full((B,), T - 1, dtype=torch.long, device=x.device)
        decided = torch.zeros(B, dtype=torch.bool, device=x.device)
        for t in range(T):
            newly = (conf[:, t] >= conf_threshold) & (~decided)
            exit_step[newly] = t
            decided |= newly
        final_pred = pred[torch.arange(B, device=x.device), exit_step]
        return final_pred, exit_step, total_spikes, spike_sites


# =====================================================================
# Loss suite (Eq. 26-32)
# =====================================================================
def focal_loss(logits, target, gamma=2.0, alpha=None):
    logp = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, target[:, None]).squeeze(1)
    loss = -((1.0 - logp_t.exp()) ** gamma) * logp_t
    if alpha is not None:
        loss = alpha[target] * loss
    return loss.mean()


def kd_loss(logits_tr, logits_snn, tau=3.0):
    p_tr = F.softmax(logits_tr / tau, dim=1)
    logp_snn = F.log_softmax(logits_snn / tau, dim=1)
    return F.kl_div(logp_snn, p_tr, reduction="batchmean") * (tau ** 2)


def homeostasis_loss(mean_rate, target_rate):
    return ((mean_rate - target_rate) ** 2).mean()


def gate_entropy_loss(gate, eps=1e-7):
    g = gate.clamp(eps, 1.0 - eps)
    return -(g * torch.log(g) + (1 - g) * torch.log(1 - g)).mean()


def compute_total_loss(out: Dict, target, *, w_tr=1.0, w_snn=1.0, w_kd=0.5,
                       w_homeo=0.1, w_energy=1e-3, w_gate=1e-3,
                       gamma=2.0, alpha=None, tau=3.0, target_rate=0.1):
    """Weighted objective (Eq. 32). Saliency-alignment (Eq. 28) is omitted: it
    belongs to the SAIF feature-selection stage, which sits on the privileged
    preprocessing we exclude."""
    l_tr = focal_loss(out["y_tr"], target, gamma, alpha)
    l_snn = focal_loss(out["y_snn"], target, gamma, alpha)
    l_kd = kd_loss(out["y_tr"], out["y_snn"], tau)
    l_homeo = homeostasis_loss(out["mean_rate"], target_rate)
    l_energy = out["spikes_per_sample"].mean()
    l_gate = gate_entropy_loss(out["gate"])
    total = (w_tr * l_tr + w_snn * l_snn + w_kd * l_kd + w_homeo * l_homeo +
             w_energy * l_energy + w_gate * l_gate)
    return total
