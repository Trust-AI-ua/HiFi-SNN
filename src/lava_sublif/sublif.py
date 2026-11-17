"""Lava process for reset rule R8, step 2 (results/preregistration/2026-11-17_neuro_stream.md).


SubLIFReset: Lava's fixed-point LIF arithmetic (24-bit u and v, 12-bit decays, activation and
threshold MSB-aligned by 6 bits, bias = mantissa << exponent), with two changes:
  * snnTorch's delayed subtractive reset instead of reset to zero:
        v_t = decay(v_{t-1}) + u_t + bias - vth_eff * r_{t-1},  r_t = [v_t > vth_eff]
  * the periodic state reset of LIFReset (every reset_interval steps at reset_offset), so
    that samples can run back to back; it also clears r.
Everything else (decay rounding, saturation, wrap of u) is copied from
lava.proc.lif.models.AbstractPyLifModelFixed. CPU fixed-point model only (select_tag
"fixed_pt"); no Loihi 2 microcode is written here.

ReluLIFReset (amendment 4): the Vanilla SNN's hidden neuron (no reset, floor at zero).

Also: emulate() and emulate_vanilla(), batched NumPy emulations of the same arithmetic (and of Lava's standard
LIFReset with zero reset), used to check the Lava runs and to attribute disagreement.
"""
from __future__ import annotations

import numpy as np
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.lif.models import AbstractPyLifModelFixed


class SubLIFReset(AbstractProcess):
    def __init__(self, *, shape, du=0, dv=0, bias_mant=0, bias_exp=0, vth=10, reset_interval=1, reset_offset=0,
                 name=None):
        super().__init__(shape=shape, name=name)
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.r = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.vth = Var(shape=(1,), init=vth)
        self.proc_params["reset_interval"] = reset_interval
        self.proc_params["reset_offset"] = reset_offset


@implements(proc=SubLIFReset, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PySubLIFResetModelFixed(AbstractPyLifModelFixed):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    r: np.ndarray = LavaPyType(np.ndarray, np.int32, precision=1)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.effective_vth = 0
        self.reset_interval = proc_params["reset_interval"]
        self.reset_offset = proc_params["reset_offset"] % self.reset_interval

    def scale_threshold(self):
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)
        self.isthrscaled = True

    def spiking_activation(self):
        return self.v > self.effective_vth

    def reset_voltage(self, spike_vector):
        pass                                      # subtractive reset is applied in run_spk

    def run_spk(self):
        a_in_data = self.a_in.recv()
        if (self.time_step % self.reset_interval) == self.reset_offset:
            self.u *= 0
            self.v *= 0
            self.r *= 0
        self.scale_bias()
        if not self.isthrscaled:
            self.scale_threshold()
        # the previous step's reset is subtracted inside the update, before the single clip
        # to the 24-bit range, as Lava clips its update
        self._subthr_with_reset(a_in_data)
        s = self.spiking_activation()
        self.r[:] = s.astype(np.int32)
        self.s_out.send(s)

    def _subthr_with_reset(self, activation_in):
        # current (Lava's code)
        decay_const_u = self.du + self.ds_offset
        decayed_curr = np.int64(self.u) * (self.decay_unity - decay_const_u)
        decayed_curr = np.int32(np.sign(decayed_curr) * np.right_shift(np.abs(decayed_curr), self.decay_shift))
        decayed_curr += np.left_shift(activation_in, self.act_shift)
        wrapped = np.where(decayed_curr > self.max_uv_val, decayed_curr - 2 * self.max_uv_val, decayed_curr)
        wrapped = np.where(wrapped <= -self.max_uv_val, decayed_curr + 2 * self.max_uv_val, wrapped)
        self.u[:] = wrapped
        # voltage (Lava's code plus the delayed subtractive reset)
        decay_const_v = self.dv + self.dm_offset
        lo, hi = -np.int32(self.max_uv_val) + 1, np.int32(self.max_uv_val) - 1
        dv_ = np.int64(self.v) * (self.decay_unity - decay_const_v)
        dv_ = np.int32(np.sign(dv_) * np.right_shift(np.abs(dv_), self.decay_shift))
        upd = np.int64(dv_) + self.u + self.effective_bias - np.int64(self.r) * self.effective_vth
        self.v[:] = np.clip(upd, lo, hi)


# ------------------------------------------------------------------ NumPy emulation
def _decay(x, d):
    p = x.astype(np.int64) * (4096 - d)
    return (np.sign(p) * (np.abs(p) >> 12)).astype(np.int64)


def emulate(layers, st, T, du=4095, dv=410, reset="subtract"):
    """Batched emulation of the Lava network in aligned time (the synaptic delays removed).

    layers: dicts with W [out, in] (the weights given to Dense, after truncation), bias_eff
    (mantissa << exponent), vth (pre-shift). st: [B, C] spike times. Returns the summed output
    voltage as Lava's monitor reads it (after the zero reset for "zero"), per-layer spike
    counts and the largest |v| reached.
    """
    B = st.shape[0]
    M = 2 ** 23
    u = [np.zeros((B, L["W"].shape[0]), np.int64) for L in layers]
    v = [np.zeros_like(x) for x in u]
    r = [np.zeros_like(x) for x in u]
    acc = np.zeros_like(v[-1])
    cnt = np.zeros((B, len(layers)), np.int64)
    vmax = 0
    for t in range(T):
        s = (st == t).astype(np.int64)
        for i, L in enumerate(layers):
            a = s @ L["W"].T.astype(np.int64)
            uu = _decay(u[i], du + 1) + (a << 6)
            uu = np.where(uu > M, uu - 2 * M, uu)
            uu = np.where(uu <= -M, uu + 2 * M, uu)
            u[i] = uu
            th = L["vth"] << 6
            vv = _decay(v[i], dv) + u[i] + L["bias_eff"]
            if reset == "subtract":
                vv = vv - r[i] * th
            vv = np.clip(vv, -M + 1, M - 1)
            vmax = max(vmax, int(np.abs(vv).max()))
            s = (vv > th).astype(np.int64)
            if reset == "zero":
                vv = np.where(s == 1, 0, vv)
            v[i], r[i] = vv, s
            cnt[:, i] += s.sum(1)
        acc = acc + v[-1]
    return acc, cnt, vmax


# ------------------------------------------------------------------ Vanilla SNN (amendment 4)
class ReluLIFReset(AbstractProcess):
    """LeakyParallel's hidden neuron in Lava's fixed-point arithmetic: no reset, potential floored
    at zero (the ReLU of nn.RNN), spike when v > vth_eff. Same u/v/bias/decay arithmetic as
    SubLIFReset and the same periodic state reset."""

    def __init__(self, *, shape, du=0, dv=0, bias_mant=0, bias_exp=0, vth=10, reset_interval=1, reset_offset=0,
                 name=None):
        super().__init__(shape=shape, name=name)
        self.a_in = InPort(shape=shape)
        self.s_out = OutPort(shape=shape)
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.du = Var(shape=(1,), init=du)
        self.dv = Var(shape=(1,), init=dv)
        self.bias_mant = Var(shape=shape, init=bias_mant)
        self.bias_exp = Var(shape=shape, init=bias_exp)
        self.vth = Var(shape=(1,), init=vth)
        self.proc_params["reset_interval"] = reset_interval
        self.proc_params["reset_offset"] = reset_offset


@implements(proc=ReluLIFReset, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyReluLIFResetModelFixed(AbstractPyLifModelFixed):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    vth: int = LavaPyType(int, np.int32, precision=17)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.effective_vth = 0
        self.reset_interval = proc_params["reset_interval"]
        self.reset_offset = proc_params["reset_offset"] % self.reset_interval

    def scale_threshold(self):
        self.effective_vth = np.left_shift(self.vth, self.vth_shift)
        self.isthrscaled = True

    def spiking_activation(self):
        return self.v > self.effective_vth

    def reset_voltage(self, spike_vector):
        pass                                      # no reset

    def run_spk(self):
        a_in_data = self.a_in.recv()
        if (self.time_step % self.reset_interval) == self.reset_offset:
            self.u *= 0
            self.v *= 0
        self.scale_bias()
        if not self.isthrscaled:
            self.scale_threshold()
        decay_const_u = self.du + self.ds_offset
        decayed_curr = np.int64(self.u) * (self.decay_unity - decay_const_u)
        decayed_curr = np.int32(np.sign(decayed_curr) * np.right_shift(np.abs(decayed_curr), self.decay_shift))
        decayed_curr += np.left_shift(a_in_data, self.act_shift)
        wrapped = np.where(decayed_curr > self.max_uv_val, decayed_curr - 2 * self.max_uv_val, decayed_curr)
        wrapped = np.where(wrapped <= -self.max_uv_val, decayed_curr + 2 * self.max_uv_val, wrapped)
        self.u[:] = wrapped
        decay_const_v = self.dv + self.dm_offset
        dv_ = np.int64(self.v) * (self.decay_unity - decay_const_v)
        dv_ = np.int32(np.sign(dv_) * np.right_shift(np.abs(dv_), self.decay_shift))
        upd = np.int64(dv_) + self.u + self.effective_bias
        self.v[:] = np.clip(upd, 0, np.int32(self.max_uv_val) - 1)       # the ReLU floor, then Lava's 24-bit ceiling
        s = self.spiking_activation()
        self.s_out.send(s)


def emulate_vanilla(layers, st, T, du=4095, dv=614):
    """Batched emulation of the Lava Vanilla network (ReluLIFReset hidden, SubLIFReset output) in
    aligned time. st: [B, F] spike times (-1 = none). Returns output spike counts [B, K], per-layer
    spike counts [B, 2] and the largest |v|."""
    B = st.shape[0]
    M = 2 ** 23
    u = [np.zeros((B, L["W"].shape[0]), np.int64) for L in layers]
    v = [np.zeros_like(x) for x in u]
    r = np.zeros_like(v[1])
    oc = np.zeros_like(v[1])
    cnt = np.zeros((B, 2), np.int64)
    vmax = 0
    for t in range(T):
        s = (st == t).astype(np.int64)
        for i, L in enumerate(layers):
            a = s @ L["W"].T.astype(np.int64)
            uu = _decay(u[i], du + 1) + (a << 6)
            uu = np.where(uu > M, uu - 2 * M, uu)
            uu = np.where(uu <= -M, uu + 2 * M, uu)
            u[i] = uu
            th = L["vth"] << 6
            vv = _decay(v[i], dv) + u[i] + L["bias_eff"]
            if i == 0:
                vv = np.clip(vv, 0, M - 1)
            else:
                vv = np.clip(vv - r * th, -M + 1, M - 1)
            vmax = max(vmax, int(np.abs(vv).max()))
            s = (vv > th).astype(np.int64)
            if i == 1:
                r = s
                oc += s
            v[i] = vv
            cnt[:, i] += s.sum(1)
    return oc, cnt, vmax
