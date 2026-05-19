"""Generic SNN architecture parameterised by neuron type.

One module that covers all nine snntorch neuron families. Wiring differs by
neuron family:

  * SIMPLE  (Leaky, Lapicque, Synaptic, Alpha):
        Linear → Neuron → Linear → Neuron_out, looped over T timesteps.
  * RECURRENT (RLeaky, RSynaptic):
        Linear → RNeuron(hidden→hidden recurrent) → Linear → Leaky_out, looped.
  * SLSTM:
        SLSTM(F→H) consumed T-step by T-step → Linear → Leaky_out.
  * SConv2dLSTM:
        Reshape F→1×H×W, SConv2dLSTM → flatten → Linear → Leaky_out (T-step loop).
  * LeakyParallel:
        Vectorised across T (no python loop inside), → Linear → Leaky_out.

Output: spike-count tensor of shape [B, num_classes] — used as logits for CE.
Also exposes `total_spikes` for the energy proxy metric.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import snntorch as snn_lib

from src.neurons import (
    SIMPLE_NEURONS, RECURRENT_NEURONS, SEQUENCE_NEURONS,
    get_neuron_factory, DEFAULT_BETA,
)


class SNN(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, num_classes: int,
                 neuron_name: str, num_steps: int, conv_out_channels: int = 8):
        super().__init__()
        self.neuron_name = neuron_name
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_features = num_features
        self.conv_out_channels = conv_out_channels
        self._build()

    # ------------- construction -------------
    def _build(self):
        n = self.neuron_name
        if n in SIMPLE_NEURONS:
            self.fc1 = nn.Linear(self.num_features, self.hidden_size)
            self.lif1 = get_neuron_factory(n)()
            self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
            self.lif2 = get_neuron_factory(n)()
        elif n in RECURRENT_NEURONS:
            self.fc1 = nn.Linear(self.num_features, self.hidden_size)
            self.rlif = get_neuron_factory(n)(self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
            self.lif_out = snn_lib.Leaky(beta=DEFAULT_BETA, init_hidden=False)
        elif n == "SLSTM":
            self.slstm = get_neuron_factory(n)(self.num_features, self.hidden_size)
            self.fc = nn.Linear(self.hidden_size, self.num_classes)
            self.lif_out = snn_lib.Leaky(beta=DEFAULT_BETA, init_hidden=False)
        elif n == "SConv2dLSTM":
            side = math.ceil(math.sqrt(self.num_features))
            self.spatial_side = side
            self.padded_features = side * side
            self.sconv = get_neuron_factory(n)(in_channels=1,
                                               out_channels=self.conv_out_channels,
                                               kernel_size=3)
            flat = self.conv_out_channels * side * side
            self.fc = nn.Linear(flat, self.num_classes)
            self.lif_out = snn_lib.Leaky(beta=DEFAULT_BETA, init_hidden=False)
        elif n == "LeakyParallel":
            self.lp = get_neuron_factory(n)(self.num_features, self.hidden_size)
            self.fc = nn.Linear(self.hidden_size, self.num_classes)
            self.lif_out = snn_lib.Leaky(beta=DEFAULT_BETA, init_hidden=False)
        else:
            raise ValueError(f"Unsupported neuron {n}")

    # ------------- introspection -------------
    def _spike_sites(self, T: int, B: int) -> int:
        """Number of binary spike outputs the forward pass produces — denominator
        for the sparsity metric. Output-layer + hidden-layer counts only."""
        n = self.neuron_name
        if n == "SConv2dLSTM":
            hidden = self.conv_out_channels * self.spatial_side * self.spatial_side
        else:
            hidden = self.hidden_size
        return T * B * (hidden + self.num_classes)

    # ------------- forward -------------
    def forward(self, spike_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """spike_train: [T, B, F].

        Returns
        -------
        logits        : [B, num_classes] — spike-count at output layer.
        total_spikes  : scalar tensor    — sum of all hidden+output spikes (energy proxy).
        spike_sites   : int              — number of (T × neuron × sample) sites that
                                            *could* spike; used for sparsity.
        """
        T, B, F = spike_train.shape
        n = self.neuron_name
        device = spike_train.device
        spike_sites = self._spike_sites(T, B)

        if n in SIMPLE_NEURONS:
            # Initialise per-neuron state vectors.
            if n == "Leaky":
                mem1 = self.lif1.init_leaky()
                mem2 = self.lif2.init_leaky()
            elif n == "Lapicque":
                mem1 = self.lif1.init_lapicque()
                mem2 = self.lif2.init_lapicque()
            elif n == "Synaptic":
                syn1, mem1 = self.lif1.init_synaptic()
                syn2, mem2 = self.lif2.init_synaptic()
            elif n == "Alpha":
                syn_e1, syn_i1, mem1 = self.lif1.init_alpha()
                syn_e2, syn_i2, mem2 = self.lif2.init_alpha()
            spk_out_record = []
            spike_total = torch.zeros((), device=device)
            for t in range(T):
                cur1 = self.fc1(spike_train[t])
                if n in {"Leaky", "Lapicque"}:
                    spk1, mem1 = self.lif1(cur1, mem1)
                elif n == "Synaptic":
                    spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
                else:  # Alpha
                    spk1, syn_e1, syn_i1, mem1 = self.lif1(cur1, syn_e1, syn_i1, mem1)
                cur2 = self.fc2(spk1)
                if n in {"Leaky", "Lapicque"}:
                    spk2, mem2 = self.lif2(cur2, mem2)
                elif n == "Synaptic":
                    spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
                else:  # Alpha
                    spk2, syn_e2, syn_i2, mem2 = self.lif2(cur2, syn_e2, syn_i2, mem2)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk1.sum() + spk2.sum()
            spikes_out = torch.stack(spk_out_record, dim=0)  # [T, B, C]
            return spikes_out.sum(dim=0), spike_total, spike_sites

        if n in RECURRENT_NEURONS:
            init = self.rlif.init_rleaky() if n == "RLeaky" else self.rlif.init_rsynaptic()
            if n == "RLeaky":
                spk1, mem1 = init
                syn1 = None
            else:
                spk1, syn1, mem1 = init
            spk1 = spk1.to(device); mem1 = mem1.to(device)
            if syn1 is not None:
                syn1 = syn1.to(device)
            # Make sure shape matches batch
            if spk1.shape[0] != B:
                spk1 = torch.zeros(B, self.hidden_size, device=device)
                mem1 = torch.zeros(B, self.hidden_size, device=device)
                if syn1 is not None:
                    syn1 = torch.zeros(B, self.hidden_size, device=device)
            mem_out = self.lif_out.init_leaky()
            spk_out_record = []
            spike_total = torch.zeros((), device=device)
            for t in range(T):
                cur1 = self.fc1(spike_train[t])
                if n == "RLeaky":
                    spk1, mem1 = self.rlif(cur1, spk1, mem1)
                else:
                    spk1, syn1, mem1 = self.rlif(cur1, spk1, syn1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem_out = self.lif_out(cur2, mem_out)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk1.sum() + spk2.sum()
            return torch.stack(spk_out_record).sum(0), spike_total, spike_sites

        if n == "SLSTM":
            syn, mem = self.slstm.init_slstm()
            mem_out = self.lif_out.init_leaky()
            spk_out_record = []
            spike_total = torch.zeros((), device=device)
            for t in range(T):
                spk1, syn, mem = self.slstm(spike_train[t], syn, mem)
                cur = self.fc(spk1)
                spk2, mem_out = self.lif_out(cur, mem_out)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk1.sum() + spk2.sum()
            return torch.stack(spk_out_record).sum(0), spike_total, spike_sites

        if n == "SConv2dLSTM":
            syn, mem = self.sconv.init_sconv2dlstm()
            mem_out = self.lif_out.init_leaky()
            pad = self.padded_features - F
            spk_out_record = []
            spike_total = torch.zeros((), device=device)
            for t in range(T):
                x = spike_train[t]
                if pad > 0:
                    x = torch.cat([x, torch.zeros(B, pad, device=device, dtype=x.dtype)], dim=1)
                x = x.view(B, 1, self.spatial_side, self.spatial_side)
                spk1, syn, mem = self.sconv(x, syn, mem)
                flat = spk1.flatten(1)
                cur = self.fc(flat)
                spk2, mem_out = self.lif_out(cur, mem_out)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk1.sum() + spk2.sum()
            return torch.stack(spk_out_record).sum(0), spike_total, spike_sites

        if n == "LeakyParallel":
            spikes = self.lp(spike_train)  # [T, B, H]
            mem_out = self.lif_out.init_leaky()
            spk_out_record = []
            spike_total = spikes.sum()
            for t in range(T):
                cur = self.fc(spikes[t])
                spk2, mem_out = self.lif_out(cur, mem_out)
                spk_out_record.append(spk2)
                spike_total = spike_total + spk2.sum()
            return torch.stack(spk_out_record).sum(0), spike_total, spike_sites

        raise ValueError(f"Unsupported neuron {n}")  # pragma: no cover
