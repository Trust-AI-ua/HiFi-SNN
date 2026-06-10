"""Spiking-SSM (Design A): neuromorphic event-driven spiking state-space NIDS.

A sequence model for flow-window sequences ``[B, L, F]``:

    spiking front-end (embed + LIF, per step)
      -> stack of selective spiking state-space blocks:
             input-dependent Delta/B/C (Mamba-style), diagonal scan,
             spike-gated (event-driven) state updates, LIF nonlinearity, O(L)
      -> supervised classifier head  (mean output membrane over time)
      -> self-supervised next-window predictor head
      -> confidence-triggered early exit

Adapted from the reference prototype to fit this framework: it exposes spike
accounting ``(total_spikes, spike_sites)`` so the shared energy/sparsity metric
machinery applies, and the early-exit / self-supervised heads are retained.

Unlike S-xLSTM / Spiking-Griffin (which use latency encoding + LeakyParallel),
Design A keeps its native design: a learned spiking front-end (no separate
latency/rate/delta encoder) and ``snn.Leaky`` neurons with a fast-sigmoid
surrogate. The selective scan is an explicit O(L) loop for clarity.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


class SpikingFrontEnd(nn.Module):
    """Per-step feature embedding -> LIF spikes."""

    def __init__(self, n_features: int, d_model: int, beta: float = 0.9):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25),
                             reset_mechanism="subtract")

    def init_state(self):
        return self.lif.init_leaky()

    def step(self, x_t, mem):
        spk, mem = self.lif(self.embed(x_t), mem)
        return spk, mem


class SpikingSSMBlock(nn.Module):
    """Selective spiking state-space block (diagonal SSM + LIF), one step at a time.

    Selective (input-dependent) Delta/B/C in the Mamba style; the input term is
    gated by the incoming spikes ``u_t`` so the state updates only on events.
    """

    def __init__(self, d_model: int, d_state: int = 8, beta: float = 0.9):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        # diagonal state matrix (stable: A = -exp(A_log) < 0) and skip D
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, d_state, d_state)
                                             .repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25),
                             reset_mechanism="subtract")

    def init_state(self, B, device):
        h = torch.zeros(B, self.d_model, self.d_state, device=device)
        return h, self.lif.init_leaky()

    def step(self, u_t, state):
        """u_t: [B, d_model] (spikes from previous block). Returns spk, y(analog), state."""
        h, mem = state
        dt = F.softplus(self.dt_proj(u_t))                      # [B, d_model]
        Bt = self.B_proj(u_t)                                   # [B, d_state]
        Ct = self.C_proj(u_t)                                   # [B, d_state]
        A = -torch.exp(self.A_log)                              # [d_model, d_state]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))       # [B, d_model, d_state]
        dB = dt.unsqueeze(-1) * Bt.unsqueeze(1)                 # [B, d_model, d_state]
        h = dA * h + dB * u_t.unsqueeze(-1)                     # event-driven update
        y = (h * Ct.unsqueeze(1)).sum(-1) + self.D * u_t        # [B, d_model]
        spk, mem = self.lif(y + u_t, mem)                       # residual current -> spikes
        return spk, y, (h, mem)


class SpikingSSM(nn.Module):
    """Full Spiking-SSM model.

    forward(x: [B, L, F]) -> (logits [B, C], next_step_preds (list of L-1 [B, F]),
                              total_spikes scalar, spike_sites int)
    """

    def __init__(self, n_features: int, n_classes: int = 2, d_model: int = 32,
                 d_state: int = 8, n_blocks: int = 2, beta: float = 0.9):
        super().__init__()
        self.frontend = SpikingFrontEnd(n_features, d_model, beta=beta)
        self.blocks = nn.ModuleList(
            [SpikingSSMBlock(d_model, d_state, beta=beta) for _ in range(n_blocks)])
        self.classifier = nn.Linear(d_model, n_classes)     # supervised head
        self.predictor = nn.Linear(d_model, n_features)     # self-supervised head
        self.d_model = d_model
        self.n_blocks = n_blocks

    def _init_states(self, B, device):
        return self.frontend.init_state(), [b.init_state(B, device) for b in self.blocks]

    def forward(self, x: torch.Tensor):
        """x: [B, L, F] -> (logits, next_step_preds, total_spikes, spike_sites)."""
        B, L, _ = x.shape
        fe_mem, states = self._init_states(B, x.device)
        z_sum = torch.zeros(B, self.d_model, device=x.device)
        preds: List[torch.Tensor] = []
        total_spikes = x.new_zeros(())
        for t in range(L):
            s_t, fe_mem = self.frontend.step(x[:, t], fe_mem)
            total_spikes = total_spikes + s_t.sum()
            u, y_last = s_t, None
            for k, blk in enumerate(self.blocks):
                u, y_last, states[k] = blk.step(u, states[k])
                total_spikes = total_spikes + u.sum()
            z_sum = z_sum + u                                   # pool last-block spikes
            if t < L - 1:
                preds.append(self.predictor(y_last))            # predict next window
        logits = self.classifier(z_sum / L)
        # spike sites = (frontend + n_blocks) layers x L steps x B x d_model
        spike_sites = (1 + self.n_blocks) * L * B * self.d_model
        return logits, preds, total_spikes, spike_sites

    @torch.no_grad()
    def predict_early_exit(self, x: torch.Tensor, theta_exit: float = 0.9,
                           theta_uncertain: float = 0.5):
        """Confidence-triggered early exit along the sequence.

        Returns (preds [B], exit_step [B], uncertain_mask [B], total_spikes,
        spike_sites). Spike accounting matches ``forward``.
        """
        B, L, _ = x.shape
        fe_mem, states = self._init_states(B, x.device)
        z_sum = torch.zeros(B, self.d_model, device=x.device)
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        exit_step = torch.full((B,), L, dtype=torch.long, device=x.device)
        preds = torch.zeros(B, dtype=torch.long, device=x.device)
        total_spikes = x.new_zeros(())
        for t in range(L):
            s_t, fe_mem = self.frontend.step(x[:, t], fe_mem)
            total_spikes = total_spikes + s_t.sum()
            u = s_t
            for k, blk in enumerate(self.blocks):
                u, _, states[k] = blk.step(u, states[k])
                total_spikes = total_spikes + u.sum()
            z_sum = z_sum + u
            prob = F.softmax(self.classifier(z_sum / (t + 1)), dim=-1)
            conf, pred = prob.max(dim=-1)
            fire = (~done) & (conf >= theta_exit)
            preds[fire] = pred[fire]; exit_step[fire] = t + 1; done = done | fire
        # samples that never crossed theta_exit: final argmax, flag low-confidence
        prob = F.softmax(self.classifier(z_sum / L), dim=-1)
        conf, pred = prob.max(dim=-1)
        preds[~done] = pred[~done]
        uncertain = (~done) & (conf < theta_uncertain)
        spike_sites = (1 + self.n_blocks) * L * B * self.d_model
        return preds, exit_step, uncertain, total_spikes, spike_sites

    @torch.no_grad()
    def self_supervised_score(self, x: torch.Tensor) -> torch.Tensor:
        """Unlabeled anomaly score = mean next-window prediction error per sample [B]."""
        _, preds, _, _ = self.forward(x)
        errs = [F.mse_loss(preds[t], x[:, t + 1], reduction="none").mean(-1)
                for t in range(len(preds))]
        return torch.stack(errs, 0).mean(0)
