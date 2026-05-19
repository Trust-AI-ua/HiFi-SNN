"""Neuron factory.

Maps a string name (e.g. "Leaky") to a callable that returns a fresh snntorch
neuron instance with reasonable default hyperparameters. Models call this to
stay neuron-agnostic.
"""
from __future__ import annotations

from typing import Callable

import snntorch as snn

DEFAULT_BETA = 0.85
DEFAULT_ALPHA = 0.9  # snntorch.Alpha enforces alpha > beta


def _make_leaky():
    return snn.Leaky(beta=DEFAULT_BETA, init_hidden=False)


def _make_lapicque():
    return snn.Lapicque(beta=DEFAULT_BETA, init_hidden=False)


def _make_synaptic():
    return snn.Synaptic(alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, init_hidden=False)


def _make_alpha():
    return snn.Alpha(alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, init_hidden=False)


def _make_rleaky(hidden_features: int):
    return snn.RLeaky(beta=DEFAULT_BETA, linear_features=hidden_features, init_hidden=False)


def _make_rsynaptic(hidden_features: int):
    return snn.RSynaptic(alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
                         linear_features=hidden_features, init_hidden=False)


def _make_slstm(input_size: int, hidden_size: int):
    return snn.SLSTM(input_size=input_size, hidden_size=hidden_size)


def _make_sconv2dlstm(in_channels: int, out_channels: int, kernel_size: int = 3):
    return snn.SConv2dLSTM(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size)


def _make_leaky_parallel(input_size: int, hidden_size: int):
    return snn.LeakyParallel(input_size=input_size, hidden_size=hidden_size,
                             beta=DEFAULT_BETA)


# Categorisation — drives how the generic model wires each one up
SIMPLE_NEURONS = {"Leaky", "Lapicque", "Synaptic", "Alpha"}
RECURRENT_NEURONS = {"RLeaky", "RSynaptic"}
SEQUENCE_NEURONS = {"SLSTM", "SConv2dLSTM", "LeakyParallel"}

ALL_NEURONS = SIMPLE_NEURONS | RECURRENT_NEURONS | SEQUENCE_NEURONS


def get_neuron_factory(name: str) -> Callable:
    table = {
        "Leaky": _make_leaky,
        "Lapicque": _make_lapicque,
        "Synaptic": _make_synaptic,
        "Alpha": _make_alpha,
        "RLeaky": _make_rleaky,
        "RSynaptic": _make_rsynaptic,
        "SLSTM": _make_slstm,
        "SConv2dLSTM": _make_sconv2dlstm,
        "LeakyParallel": _make_leaky_parallel,
    }
    if name not in table:
        raise ValueError(f"Unknown neuron {name!r}. Available: {sorted(table)}")
    return table[name]
