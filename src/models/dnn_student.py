"""Iso-architecture DNN twin of the Tree-to-Spike SNN student.

For a *fair* SNN-vs-DNN comparison (the iso-architecture principle), this is a
structural twin of ``SNNStudent`` in ``tree_to_spike.py``:

  * **Identical topology / parameters** — the same three ``nn.Linear`` layers
    (``in_dim -> hidden[0] -> hidden[1] -> n_classes``). The SNN's LIF neurons add
    no learnable weights, so ``DNNStudent`` has the *exact same* trainable
    parameter count as ``SNNStudent`` for a given ``in_dim`` / ``hidden`` /
    ``n_classes``.
  * **ReLU activation** mirrors the integrate-and-fire nonlinearity (clipped-ReLU
    is available via ``clip_relu`` to mirror a saturating IF response).
  * **One-shot (T=1) forward** — a non-spiking ANN processes the input once, with
    no temporal/membrane state, so its cost is dense Multiply-Accumulate (MAC)
    operations rather than the SNN's spike-triggered Accumulate (AC/SOP) ops.

It consumes the *same* tree-guided representation as the SNN student
(``TreeGuidedSpikeEncoder.encode_dense`` gives the [B, out_dim] bin levels), so
the only thing that differs between the two models is the computational paradigm.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNNStudent(nn.Module):
    """Dense MLP twin of ``SNNStudent`` (same Linear topology, ReLU, one-shot).

    forward(x [B, in_dim]) -> logits [B, n_classes].
    """

    def __init__(self, in_dim: int, n_classes: int, hidden=(64, 64),
                 clip_relu: float | None = None):
        super().__init__()
        hidden = tuple(hidden)
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], n_classes)
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden = hidden
        self.clip_relu = clip_relu

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        return torch.clamp(x, max=self.clip_relu) if self.clip_relu else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self._act(self.fc2(self._act(self.fc1(x)))))

    def macs_per_sample(self) -> int:
        """Dense Multiply-Accumulate ops for one one-shot forward pass.

        N_MAC = in*h0 + h0*h1 + h1*C  (biases excluded, matching the iso-ANN MAC
        convention used against the SNN's SynOps count).
        """
        h0, h1 = self.hidden
        return int(self.in_dim * h0 + h0 * h1 + h1 * self.n_classes)
