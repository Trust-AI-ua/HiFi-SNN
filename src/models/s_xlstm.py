"""S-xLSTM: a spiking xLSTM hybrid for network intrusion detection.

This is the hybrid model proposed on top of the Phase-1 SNN ablation. It pairs
the ablation's two winning ingredients — latency (TTFS) spike encoding and the
``LeakyParallel`` spiking neuron — with an xLSTM ``mLSTM`` selective linear
recurrence. The pipeline mirrors the design figure:

    feature vector x in [0,1]^d
        -> latency (TTFS) encoding                       [Stage 2]   spikegen.latency
        -> S-xLSTM backbone x N:                         [Stage 3]
               mLSTM selective linear recurrence (teal, input-dependent gates)
               -> LeakyParallel spiking readout (coral, reset-free)
        -> readout + early-exit (accumulate spikes)      [Stage 4]
        -> attack class / alert                          [Output]

Design rationale (consistent with the Phase-1 findings)
-------------------------------------------------------
* The spike layer sits *after* the linear recurrence: the recurrence stays
  linear (so it keeps selectivity and is parallelisable in principle), and
  ``LeakyParallel`` is a reset-free parallel LIF, so the spiking signal between
  blocks is sparse and addition-friendly.
* Per-modality preprocessing (categorical one-hot + continuous min-max) is done
  upstream in ``src/data`` so the encoded feature vector is already in [0,1].
  Because ``spikegen.latency`` is non-differentiable, the first learnable
  projection lives in the first mLSTM block, immediately after encoding.
* Only ``LeakyParallel`` and latency encoding are used, per the requested
  restriction; no other neuron model or encoding is instantiated here.

The model exposes the same ``(logits, total_spikes, spike_sites)`` forward
signature as ``src/models/snn.py`` so the shared metric/energy machinery in
``src/metrics.py`` applies unchanged and the hybrid is directly comparable to
the SNN ablation.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import spikegen, surrogate


class mLSTMCell(nn.Module):
    """xLSTM mLSTM (matrix-memory) cell with exponential gating and stabiliser.

    Recurrence (linear in the matrix cell state C, hence parallelisable in
    principle; run here as a sequential loop over the short latency window T):

        C_t = f_t * C_{t-1} + i_t * (v_t k_t^T)
        n_t = f_t * n_{t-1} + i_t * k_t
        h_t = o_t * (C_t q_t) / max(|n_t . q_t|, 1)

    The forget gate f_t is the input-dependent leak — the conceptual bridge to
    the leaky-integrate view shared with the spiking readout. A running
    log-space stabiliser m_t keeps the exponential gates numerically safe.
    """

    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.d = d_model
        self.Wq = nn.Linear(in_dim, d_model)
        self.Wk = nn.Linear(in_dim, d_model)
        self.Wv = nn.Linear(in_dim, d_model)
        self.Wo = nn.Linear(in_dim, d_model)
        self.wi = nn.Linear(in_dim, 1)   # input gate (log-space, pre-exp)
        self.wf = nn.Linear(in_dim, 1)   # forget gate (log-space via logsigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [T, B, in_dim]
        T, B, _ = x.shape
        scale = self.d ** 0.5
        C = x.new_zeros(B, self.d, self.d)
        n = x.new_zeros(B, self.d)
        m = x.new_zeros(B, 1)            # stabiliser state
        outputs = []
        for t in range(T):
            xt = x[t]
            q = self.Wq(xt)
            k = self.Wk(xt) / scale
            v = self.Wv(xt)
            o = torch.sigmoid(self.Wo(xt))

            i_log = self.wi(xt)                  # [B, 1]
            f_log = F.logsigmoid(self.wf(xt))    # [B, 1]

            m_new = torch.maximum(f_log + m, i_log)          # stabilised max
            i = torch.exp(i_log - m_new)                     # [B, 1]
            f = torch.exp(f_log + m - m_new)                 # [B, 1]

            vk = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))   # [B, d, d]
            C = f.unsqueeze(2) * C + i.unsqueeze(2) * vk
            n = f * n + i * k                                # [B, d]

            Cq = torch.bmm(C, q.unsqueeze(2)).squeeze(2)     # [B, d]
            denom = torch.maximum((n * q).sum(-1, keepdim=True).abs(),
                                  torch.ones_like(m_new))    # [B, 1]
            h = o * (Cq / denom)
            m = m_new
            outputs.append(h)
        return torch.stack(outputs, dim=0)                   # [T, B, d]


class SxLSTMBlock(nn.Module):
    """One backbone block: mLSTM selective recurrence -> LeakyParallel readout.

    Returns the spike train plus this block's spike count and spike-site count
    so the caller can aggregate an energy proxy (spikes/sample) and sparsity.
    """

    def __init__(self, in_dim: int, d_model: int, beta: float, spike_grad):
        super().__init__()
        self.mlstm = mLSTMCell(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        # LeakyParallel: reset-free parallel LIF. Input [T, B, in] -> spikes
        # [T, B, hidden]. Membrane decay beta is learned; threshold fixed.
        self.readout = snn.LeakyParallel(
            input_size=d_model,
            hidden_size=d_model,
            beta=beta,
            spike_grad=spike_grad,
            learn_beta=True,
            learn_threshold=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        h = self.norm(self.mlstm(x))         # [T, B, d_model]
        spk = self.readout(h)                # [T, B, d_model]
        if isinstance(spk, (tuple, list)):   # defensive across snntorch versions
            spk = spk[0]
        n_spikes = spk.sum()
        n_sites = spk.shape[0] * spk.shape[1] * spk.shape[2]
        return spk, n_spikes, n_sites


class SxLSTM(nn.Module):
    """Full S-xLSTM model for tabular NIDS features.

    forward(x: [B, F]) -> (logits [B, C], total_spikes scalar, spike_sites int)
    """

    def __init__(self, num_features: int, num_classes: int, num_steps: int = 25,
                 d_model: int = 64, num_blocks: int = 2, beta: float = 0.9):
        super().__init__()
        self.num_steps = num_steps
        self.d_model = d_model
        self.num_blocks = num_blocks
        spike_grad = surrogate.atan()
        dims = [num_features] + [d_model] * num_blocks
        self.blocks = nn.ModuleList([
            SxLSTMBlock(dims[i], dims[i + 1], beta, spike_grad)
            for i in range(num_blocks)
        ])
        self.readout = nn.Linear(d_model, num_classes)   # Stage 4 head

    # ---------------- encoding + backbone ----------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latency (TTFS) encoding -> [T, B, F]. Non-differentiable; the first
        learnable projection lives in the first mLSTM block, after this."""
        # spikegen.latency strictly requires inputs in [0, 1]; clamp first because
        # MinMax fit_transform can emit a tiny float32 overshoot (e.g. 1.0000001)
        # on some datasets (matches the shared SNN encoders in src/encoding/).
        x = x.clamp(0.0, 1.0)
        return spikegen.latency(
            x, num_steps=self.num_steps,
            normalize=True, linear=True, clip=True,
        )

    def _backbone(self, x: torch.Tensor):
        """x: [B, F] in [0,1] -> (spk [T, B, d_model], total_spikes, spike_sites)."""
        spk = self._encode(x)                       # [T, B, F]
        total_spikes = x.new_zeros(())
        total_sites = 0
        for block in self.blocks:                   # Stage 3
            spk, blk_spikes, blk_sites = block(spk)
            total_spikes = total_spikes + blk_spikes
            total_sites += blk_sites
        return spk, total_spikes, total_sites       # [T, B, d_model]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        spk, total_spikes, total_sites = self._backbone(x)
        rate = spk.mean(dim=0)                       # accumulate spikes -> [B, d_model]
        logits = self.readout(rate)                  # [B, num_classes]
        return logits, total_spikes, total_sites

    # ---------------- early-exit inference ----------------
    @torch.no_grad()
    def predict_early_exit(self, x: torch.Tensor, conf_threshold: float = 0.9):
        """Read out at each time step from the running spike-mean and stop as
        soon as softmax confidence crosses the threshold.

        Returns (predictions [B], exit_step [B], total_spikes scalar,
        spike_sites int). Spike accounting is identical to ``forward`` so the
        energy reported under early-exit is comparable.
        """
        spk, total_spikes, total_sites = self._backbone(x)   # [T, B, d]
        T, B, _ = spk.shape
        running_mean = torch.cumsum(spk, dim=0) / torch.arange(
            1, T + 1, device=spk.device).view(T, 1, 1)
        logits_t = self.readout(running_mean)                # [T, B, C]
        conf, pred = torch.softmax(logits_t, dim=-1).max(dim=-1)  # [T, B]

        exit_step = torch.full((B,), T - 1, dtype=torch.long, device=spk.device)
        decided = torch.zeros(B, dtype=torch.bool, device=spk.device)
        for t in range(T):
            newly = (conf[t] >= conf_threshold) & (~decided)
            exit_step[newly] = t
            decided |= newly
        final_pred = pred[exit_step, torch.arange(B, device=spk.device)]
        return final_pred, exit_step, total_spikes, total_sites
