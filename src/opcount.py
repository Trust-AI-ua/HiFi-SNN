"""Count executed arithmetic per sample, for models that do not report it.

Tree-to-Spike, DNN-Tree and the GBDT teacher each record their own operation
counts. The two competitor baselines (Vanilla SNN, TASNN) were added to the
deployment measurement for footprint and latency, and carried no ``macs`` or
``sops`` at all, so pricing them in the energy model produced a meaningless
number. This counts them by observation rather than by architecture-specific
arithmetic.

The rule, applied per linear or convolutional layer: if the layer's input is
BINARY (every value 0 or 1), the layer is driven by spikes and its cost is one
accumulate per incoming spike per output unit, which is a SOP. Otherwise it is a
dense product and each input-output pair is a MAC. A model that mixes the two,
as TASNN does, is therefore priced as the mixture it is, and neither class is
charged at the other's rate.

    with count_ops(model, batch_size=len(x)) as c:
        model(x)
    c.macs, c.sops       # synaptic ops per sample; c.leak_ops is separate

NOT VALID for models containing attention. ``nn.MultiheadAttention`` holds its
input projection in a raw ``in_proj_weight`` parameter and computes QK^T and AV
as bare ``matmul`` calls, none of which is a module a forward hook can see, so
TASNN comes out roughly half its true cost. TASNN is priced instead by the
analytic formula in scripts/benchmark_4way.py, which is the one the paper's
published figures use and which this module was checked against.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


class _Counter:
    def __init__(self):
        self.macs = 0.0
        self.sops = 0.0
        self.leak_ops = 0.0      # per-unit state updates, counted but NOT in sops
        self.n = 0
        self.layers = []


def _is_binary(t: torch.Tensor) -> bool:
    if not t.is_floating_point():
        return False
    return bool(((t == 0) | (t == 1)).all())


@contextmanager
def count_ops(model: nn.Module, batch_size: int | None = None):
    c = _Counter()
    handles = []

    def hook(mod, inp, out):
        x = inp[0]
        if x.dim() == 0:
            return
        b = x.shape[0] if batch_size is None else batch_size
        if isinstance(mod, nn.Linear):
            fan_out = mod.out_features
            if _is_binary(x):
                c.sops += float(x.sum()) * fan_out / b
                c.layers.append((type(mod).__name__, "SOP"))
            else:
                c.macs += float(x.numel()) * fan_out / b
                c.layers.append((type(mod).__name__, "MAC"))
        elif isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            k = 1
            for s in mod.kernel_size:
                k *= s
            per_out = k * mod.in_channels / mod.groups
            n_out = out.numel() / b
            if _is_binary(x):
                c.sops += per_out * n_out
                c.layers.append((type(mod).__name__, "SOP"))
            else:
                c.macs += per_out * n_out
                c.layers.append((type(mod).__name__, "MAC"))
        c.n += 1

    def rnn_hook(mod, inp, out):
        """nn.RNN is not a Linear, so its input matrix would go uncounted.

        The hidden-to-hidden matrix of a LeakyParallel cell is frozen at beta * I,
        a per-unit leak rather than a matmul, so it is priced as one accumulate
        per hidden unit per step. A genuine recurrent matrix is priced as a matmul.
        """
        x = inp[0]
        b = x.shape[0] if batch_size is None else batch_size
        H = mod.hidden_size
        steps = x.shape[1] if mod.batch_first else x.shape[0]
        if _is_binary(x):
            c.sops += float(x.sum()) * H / b
            c.layers.append(("RNN.input", "SOP"))
        else:
            c.macs += float(x.numel()) * H / b
            c.layers.append(("RNN.input", "MAC"))
        w = mod.weight_hh_l0.detach()
        d = torch.diag(w)
        if w.shape[0] == w.shape[1] and bool((d == d[0]).all()) and \
                bool((w - torch.diag(d) == 0).all()):
            # Counted separately: the SNN convention in this repo prices SYNAPTIC
            # operations only, for every system. Tree-to-Spike likewise excludes its
            # own membrane updates, so folding the leak into sops here would charge
            # one baseline for work the others are not charged for.
            c.leak_ops += H * steps
            c.layers.append(("RNN.leak", "leak"))
        else:
            c.macs += H * H * steps
            c.layers.append(("RNN.hidden", "MAC"))
        c.n += 1

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            handles.append(m.register_forward_hook(hook))
        elif isinstance(m, nn.RNN):
            handles.append(m.register_forward_hook(rnn_hook))
    try:
        yield c
    finally:
        for h in handles:
            h.remove()
