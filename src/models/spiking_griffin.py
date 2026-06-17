"""Spiking-Griffin: a spiking adaptation of Griffin for network intrusion detection.

A *second* hybrid alongside S-xLSTM (``src/models/s_xlstm.py``). It deliberately
reuses the exact same front-end (latency / TTFS encoding) and the same
``LeakyParallel`` spiking readout, so the two hybrids differ only in the
backbone and can be compared head-to-head with everything else held constant.

Backbone (interleaved, repeated x N layers):

    [ gated linear recurrence (RG-LRU) -> LeakyParallel spike readout ]   (teal + coral)
    [ windowed spike-driven local attention -> LeakyParallel readout ]    (purple + coral)

The spike layer sits *after* each sub-block: the recurrence stays linear and the
attention stays windowed/local and spike-driven, so training is parallelisable
and inference stays sub-quadratic. This mirrors the design figure (Griffin-style
interleaving of a gated linear recurrence with windowed local attention, each
followed by a LeakyParallel readout).

The model exposes the same forward signature as ``src/models/s_xlstm.py`` and
``src/models/snn.py`` — ``(logits, total_spikes, spike_sites)`` — so the shared
metric/energy machinery applies unchanged. Spike accounting includes every
``LeakyParallel`` emission: the two sub-block readouts *and* the spike-form
Q/K/V inside the windowed attention.

Only ``LeakyParallel`` and latency encoding are used, per the requested
restriction.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

import snntorch as snn
from snntorch import spikegen, surrogate


def _spk(out):
    """LeakyParallel returns spikes (some snntorch versions return (spk, mem))."""
    return out[0] if isinstance(out, (tuple, list)) else out


class GatedLinearRecurrence(nn.Module):
    """Diagonal gated linear recurrence (simplified RG-LRU), the teal block.

        a_t = sigmoid(W_a x_t)                          # input-dependent decay (selective leak)
        i_t = sigmoid(W_i x_t)                          # input gate
        h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * W_x x_t)

    Linear in h_t (hence parallelisable in principle); run sequentially here for
    clarity over the short latency window T. This is a numerically stable
    simplification of Griffin's RG-LRU with the same input-dependent-leak role.
    """

    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.d = d
        self.Wx = nn.Linear(in_dim, d)
        self.Wi = nn.Linear(in_dim, d)
        self.Wa = nn.Linear(in_dim, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [T, B, in_dim]
        T, B, _ = x.shape
        h = x.new_zeros(B, self.d)
        outs = []
        for t in range(T):
            xt = x[t]
            a = torch.sigmoid(self.Wa(xt))            # [B, d]
            i = torch.sigmoid(self.Wi(xt))            # [B, d]
            u = i * self.Wx(xt)                       # gated input
            h = a * h + torch.sqrt(torch.clamp(1 - a * a, min=1e-6)) * u
            outs.append(h)
        return torch.stack(outs, dim=0)               # [T, B, d]


class WindowedSpikeAttention(nn.Module):
    """Local (windowed), softmax-free, spike-driven attention — the purple block.

    Q/K/V are emitted as spikes by ``LeakyParallel`` neurons; scores are spike
    coincidence counts restricted to a local band (|i - j| <= window), averaged
    over the window (no softmax) and used to aggregate spike-form V. Cost is
    O(T * window) with spike operands. Returns the projected output plus the
    spike count and spike-site count of the Q/K/V neurons (for energy accounting).
    """

    def __init__(self, d: int, window: int, beta: float, spike_grad):
        super().__init__()
        self.window = window
        self.q = snn.LeakyParallel(d, d, beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.k = snn.LeakyParallel(d, d, beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.v = snn.LeakyParallel(d, d, beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor):  # x: [T, B, d]
        T = x.shape[0]
        q_spk = _spk(self.q(x))                          # [T, B, d] spikes
        k_spk = _spk(self.k(x))
        v_spk = _spk(self.v(x))
        q = q_spk.permute(1, 0, 2)                       # [B, T, d]
        k = k_spk.permute(1, 0, 2)
        v = v_spk.permute(1, 0, 2)
        scores = torch.bmm(q, k.transpose(1, 2))         # [B, T, T] spike coincidence counts
        idx = torch.arange(T, device=x.device)
        band = (idx[None, :] - idx[:, None]).abs() <= self.window     # [T, T] local mask
        scores = scores.masked_fill(~band, 0.0)
        denom = band.float().sum(-1, keepdim=True).clamp(min=1.0)     # [T, 1] window size
        out = torch.bmm(scores, v) / denom               # [B, T, d] windowed mean, no softmax
        out = self.proj(out).permute(1, 0, 2)            # [T, B, d]
        qkv_spikes = q_spk.sum() + k_spk.sum() + v_spk.sum()
        qkv_sites = 3 * q_spk.shape[0] * q_spk.shape[1] * q_spk.shape[2]
        return out, qkv_spikes, qkv_sites


class RecurrenceSubBlock(nn.Module):
    """Gated linear recurrence -> LeakyParallel spike readout. Returns
    (spk, n_spikes, n_sites)."""

    def __init__(self, in_dim: int, d: int, beta: float, spike_grad):
        super().__init__()
        self.rec = GatedLinearRecurrence(in_dim, d)
        self.norm = nn.LayerNorm(d)
        self.readout = snn.LeakyParallel(d, d, beta=beta, spike_grad=spike_grad,
                                         learn_beta=True, learn_threshold=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        spk = _spk(self.readout(self.norm(self.rec(x))))     # [T, B, d]
        return spk, spk.sum(), spk.shape[0] * spk.shape[1] * spk.shape[2]


class AttentionSubBlock(nn.Module):
    """Windowed spike attention -> LeakyParallel spike readout. Returns
    (spk, n_spikes, n_sites), where n_spikes includes the attention Q/K/V
    spikes plus the readout spikes."""

    def __init__(self, d: int, window: int, beta: float, spike_grad):
        super().__init__()
        self.attn = WindowedSpikeAttention(d, window, beta, spike_grad)
        self.norm = nn.LayerNorm(d)
        self.readout = snn.LeakyParallel(d, d, beta=beta, spike_grad=spike_grad,
                                         learn_beta=True, learn_threshold=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        attn_out, qkv_spikes, qkv_sites = self.attn(x)
        spk = _spk(self.readout(self.norm(attn_out)))        # [T, B, d]
        n_spikes = qkv_spikes + spk.sum()
        n_sites = qkv_sites + spk.shape[0] * spk.shape[1] * spk.shape[2]
        return spk, n_spikes, n_sites


class SpikingGriffin(nn.Module):
    """Full Spiking-Griffin model for tabular NIDS features.

    forward(x: [B, F]) -> (logits [B, C], total_spikes scalar, spike_sites int)
    """

    def __init__(self, num_features: int, num_classes: int, num_steps: int = 25,
                 d_model: int = 64, num_layers: int = 1, window: int = 4,
                 beta: float = 0.9):
        super().__init__()
        self.num_steps = num_steps
        self.d_model = d_model
        self.num_layers = num_layers
        self.window = window
        sg = surrogate.atan()
        layers = []
        in_dim = num_features
        for _ in range(num_layers):
            layers.append(RecurrenceSubBlock(in_dim, d_model, beta, sg))   # teal + coral
            layers.append(AttentionSubBlock(d_model, window, beta, sg))    # purple + coral
            in_dim = d_model
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(d_model, num_classes)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # clamp first: spikegen.latency requires [0, 1] and MinMax fit_transform can
        # emit a tiny float32 overshoot (e.g. 1.0000001) on some datasets (matches
        # the shared SNN encoders in src/encoding/).
        x = x.clamp(0.0, 1.0)
        return spikegen.latency(x, num_steps=self.num_steps,
                                normalize=True, linear=True, clip=True)   # [T, B, F]

    def _backbone(self, x: torch.Tensor):
        """x: [B, F] in [0,1] -> (spk [T, B, d], total_spikes, spike_sites)."""
        spk = self._encode(x)
        total_spikes = x.new_zeros(())
        total_sites = 0
        for layer in self.layers:
            spk, blk_spikes, blk_sites = layer(spk)
            total_spikes = total_spikes + blk_spikes
            total_sites += blk_sites
        return spk, total_spikes, total_sites

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        spk, total_spikes, total_sites = self._backbone(x)
        rate = spk.mean(dim=0)                           # [B, d_model]
        logits = self.readout(rate)                      # [B, num_classes]
        return logits, total_spikes, total_sites

    @torch.no_grad()
    def predict_early_exit(self, x: torch.Tensor, conf_threshold: float = 0.9):
        """Read out at each time step from the running spike-mean and stop as
        soon as softmax confidence crosses the threshold. Returns
        (predictions [B], exit_step [B], total_spikes, spike_sites)."""
        spk, total_spikes, total_sites = self._backbone(x)   # [T, B, d]
        T, B, _ = spk.shape
        running_mean = torch.cumsum(spk, dim=0) / torch.arange(
            1, T + 1, device=spk.device).view(T, 1, 1)
        conf, pred = torch.softmax(self.readout(running_mean), dim=-1).max(dim=-1)  # [T, B]
        exit_step = torch.full((B,), T - 1, dtype=torch.long, device=spk.device)
        decided = torch.zeros(B, dtype=torch.bool, device=spk.device)
        for t in range(T):
            newly = (conf[t] >= conf_threshold) & (~decided)
            exit_step[newly] = t
            decided |= newly
        final_pred = pred[exit_step, torch.arange(B, device=spk.device)]
        return final_pred, exit_step, total_spikes, total_sites
