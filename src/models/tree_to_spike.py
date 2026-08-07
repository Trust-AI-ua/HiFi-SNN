"""Tree-to-Spike (Design C): GBDT-guided spike encoder + compact SNN student.

Train-time: a cost-sensitive XGBoost teacher guides (a) a fixed spike encoder
whose latency bins come from the tree split thresholds and whose per-feature
channel budget comes from feature importance (with weak features pruned), and
(b) a compact LIF MLP student trained by distillation. Deploy-time: the
**encoder + SNN student only** run (no tree), int8-quantized, with
confidence-triggered early exit.

This module holds the two reusable pieces — ``TreeGuidedSpikeEncoder`` (the
contribution) and ``SNNStudent`` — plus the int8 fake-quant helper. The teacher
fit, distillation loss, and training loop live in
``src/training_treespike.py``.

Unlike the latency/LeakyParallel hybrids, Design C keeps its native design: a
tree-guided latency encoder (not snntorch's latency) and ``snn.Leaky`` neurons
with a fast-sigmoid surrogate (``leaky=False`` gives the non-leaky IF ablation).
The ``SNNStudent`` exposes spike accounting so the shared energy/sparsity
metrics apply, plus a SynOps proxy that the design emphasises.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


# ======================================================================================
# Tree-guided spike encoder (the contribution)
# ======================================================================================
class TreeGuidedSpikeEncoder:
    """Turns tabular flow features into spike trains using a trained GBDT teacher.

    - split thresholds  -> per-feature bins -> latency (time-to-first-spike) code
    - feature importance -> #channels per feature (population budget) + pruning

    The encoder is a *fixed function* after ``fit()``; the tree is not executed
    at runtime. ``encode(X)`` returns spikes ``[T, B, out_dim]``.
    """

    def __init__(self, T: int = 20, pop_max: int = 3, prune_quantile: float = 0.20,
                 jitter: float = 0.12, encoder_mode: str = "tree"):
        self.T = T
        self.pop_max = pop_max          # max channels for the most important feature
        self.prune_quantile = prune_quantile
        self.jitter = jitter
        # encoder_mode (ablation): "tree" = importance-pruned features + importance-weighted
        # population (the contribution); "uniform" = every feature-with-edges, 1 channel each
        # (no importance); "all" = every feature-with-edges, pop_max channels each (dense upper
        # bound). uniform/all keep the same tree-threshold latency code, only the channel
        # allocation differs - so those arms probe the channel budget.
        # "quantile" = the teacher-ruler control: IDENTICAL channels to "tree" (same pruning,
        # same importance budget, same jitter) and the same per-feature edge COUNT, but the
        # edges are equal-frequency quantiles of the training data instead of XGBoost split
        # thresholds. Only the edge placement changes, so tree-vs-quantile isolates whether
        # the teacher's split points ("the ruler") matter at matched budget. Requires X in fit().
        self.encoder_mode = encoder_mode
        self.edges_ = {}                # feat_idx -> sorted unique split thresholds
        self.channels_ = []             # list of (feat_idx, level_offset)
        self.out_dim = 0

    def fit(self, teacher, n_features: int, X=None):
        if self.encoder_mode == "raw":
            # Attribution control (no tree involvement anywhere): one channel per
            # feature, TTFS applied directly to the already-MinMax'd raw value.
            # Same T, same student, same schedule as "tree" - only the encoder
            # changes, which is what isolates the tree-derived representation.
            self.edges_ = {}
            self.channels_ = [(j, 0.0) for j in range(n_features)]
            self.out_dim = len(self.channels_)
            return self

        booster = teacher.get_booster()
        tdf = booster.trees_to_dataframe()
        # collect split thresholds per feature (feature names are 'f0','f1',...)
        for _, row in tdf.iterrows():
            feat, sp = row["Feature"], row["Split"]
            if feat == "Leaf" or sp is None or (isinstance(sp, float) and np.isnan(sp)):
                continue
            j = int(str(feat)[1:])  # 'f12' -> 12
            self.edges_.setdefault(j, []).append(float(sp))
        for j in list(self.edges_.keys()):
            self.edges_[j] = np.unique(np.asarray(self.edges_[j], dtype=np.float32))

        imp = np.asarray(teacher.feature_importances_, dtype=np.float32)  # gain, sums to 1
        keep_thr = np.quantile(imp[imp > 0], self.prune_quantile) if (imp > 0).any() else 0.0
        imp_max = imp.max() + 1e-9

        def _add_population(j, n_pop):
            offsets = (np.arange(n_pop) - (n_pop - 1) / 2.0) * self.jitter
            for o in offsets:
                self.channels_.append((j, float(o)))

        self.channels_ = []
        if self.encoder_mode == "all":          # ablation: every feature, full population
            for j in range(n_features):
                if j in self.edges_:
                    _add_population(j, self.pop_max)
        elif self.encoder_mode == "uniform":    # ablation: every feature, 1 channel (no importance)
            for j in range(n_features):
                if j in self.edges_:
                    self.channels_.append((j, 0.0))
        else:                                   # "tree"/"quantile": importance-pruned + weighted
            for j in range(n_features):
                if imp[j] <= 0 or imp[j] < keep_thr or j not in self.edges_:
                    continue  # pruned
                _add_population(j, max(1, int(np.ceil((imp[j] / imp_max) * self.pop_max))))
        if not self.channels_:  # safety: keep everything if pruning removed all
            self.channels_ = [(j, 0.0) for j in self.edges_]

        if self.encoder_mode == "quantile":
            # teacher-ruler control: swap each kept feature's edges for equal-frequency
            # quantiles of the TRAINING data with the SAME edge count. Channels (pruning,
            # budget, jitter) are already identical to "tree", so only placement changes.
            if X is None:
                raise ValueError("encoder_mode='quantile' requires the training X in fit()")
            X = np.asarray(X, dtype=np.float32)
            for j in sorted({j for j, _ in self.channels_}):
                k = len(self.edges_[j])
                qs = np.linspace(0, 1, k + 2)[1:-1]          # k interior quantiles
                e = np.unique(np.quantile(X[:, j], qs).astype(np.float32))
                if len(e) == 0:                               # constant feature: keep tree edges
                    continue
                self.edges_[j] = e
        self.out_dim = len(self.channels_)
        return self

    def _levels(self, X: np.ndarray):
        """Map each kept feature value to a bin level in [0,1] via tree thresholds."""
        feats = sorted({j for j, _ in self.channels_})
        if self.encoder_mode == "raw":
            # raw control: the loader already MinMax-scales to [0,1], so the value
            # *is* the level (no binning, no thresholds).
            return {j: np.clip(X[:, j], 0.0, 1.0) for j in feats}
        lvl = {}
        for j in feats:
            e = self.edges_.get(j, None)
            if e is None or len(e) == 0:
                lvl[j] = 1.0 / (1.0 + np.exp(-X[:, j]))  # fallback: squashed raw value
            else:
                lvl[j] = np.searchsorted(e, X[:, j], side="right") / float(len(e))
        return lvl

    def _level_channels(self, X) -> np.ndarray:
        """Map [B, F] -> [B, out_dim] continuous bin levels in [0, 1] (the dense,
        pre-spike representation shared by ``encode`` and ``encode_dense``)."""
        X = np.asarray(X, dtype=np.float32)
        B = X.shape[0]
        lvl = self._levels(X)
        level_ch = np.empty((B, self.out_dim), dtype=np.float32)
        for c, (j, o) in enumerate(self.channels_):
            level_ch[:, c] = np.clip(lvl[j] + o, 0.0, 1.0)
        return level_ch

    def encode_dense(self, X) -> torch.Tensor:
        """Dense one-shot analog of ``encode``: returns the [B, out_dim] bin-level
        matrix in [0, 1] with no temporal dimension. This is the input the
        iso-architecture DNN twin consumes (same channels as the SNN student sees,
        but processed in a single forward pass instead of as a T-step spike train)."""
        return torch.from_numpy(self._level_channels(X))

    def encode(self, X) -> torch.Tensor:
        """X: np.ndarray [B, F]  ->  spikes torch.FloatTensor [T, B, out_dim]."""
        level_ch = self._level_channels(X)
        spike_time = np.round((1.0 - level_ch) * (self.T - 1)).astype(np.int64)  # [B, C]
        st = torch.from_numpy(spike_time)
        grid = torch.arange(self.T).view(self.T, 1, 1)
        spikes = (grid == st.unsqueeze(0)).float()  # [T, B, C]
        return spikes


# ======================================================================================
# Compact SNN student (snntorch)
# ======================================================================================
class SNNStudent(nn.Module):
    """3-layer LIF MLP. ``leaky=False`` gives the non-leaky IF ablation (beta=1).

    forward(spikes [T, B, in]) -> (logits [B, C], total_spikes, spike_sites, synops)
      * logits      : mean output membrane over time
      * total_spikes: summed hidden-layer (L1+L2) spikes over the batch+timesteps
      * spike_sites : T * B * (hidden1 + hidden2)  (denominator for sparsity)
      * synops      : synaptic-operations proxy per sample (spikes * fan-out)
    """

    def __init__(self, in_dim: int, n_classes: int, hidden=(64, 64),
                 beta: float = 0.9, leaky: bool = True):
        super().__init__()
        b = beta if leaky else 1.0
        sg = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.lif1 = snn.Leaky(beta=b, spike_grad=sg, reset_mechanism="subtract")
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.lif2 = snn.Leaky(beta=b, spike_grad=sg, reset_mechanism="subtract")
        self.fc3 = nn.Linear(hidden[1], n_classes)
        self.lif3 = snn.Leaky(beta=b, spike_grad=sg, reset_mechanism="subtract")
        self.hidden = hidden
        self.fanouts = (hidden[0], hidden[1], n_classes)

    def forward(self, spikes: torch.Tensor):
        T, B, _ = spikes.shape
        m1, m2, m3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        mem_sum = 0.0
        n_spk1 = n_spk2 = 0.0
        for t in range(T):
            s1, m1 = self.lif1(self.fc1(spikes[t]), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            s3, m3 = self.lif3(self.fc3(s2), m3)
            mem_sum = mem_sum + m3
            n_spk1 = n_spk1 + s1.sum()
            n_spk2 = n_spk2 + s2.sum()
        logits = mem_sum / T                                   # mean output membrane
        total_spikes = n_spk1 + n_spk2                         # hidden-layer spikes
        spike_sites = T * B * (self.hidden[0] + self.hidden[1])
        synops = (n_spk1 * self.fanouts[1] + n_spk2 * self.fanouts[2]) / B   # per sample
        return logits, total_spikes, spike_sites, synops

    @torch.no_grad()
    def forward_prefix(self, spikes: torch.Tensor):
        """Per-timestep prefix state, so early-exit analysis becomes post-processing.

        One full-window pass yields everything needed to replay ANY threshold,
        recompute calibration, or re-derive class-conditional exit behaviour
        offline -- instead of one inference run per threshold.

        Returns a dict of CPU tensors:
          prefix_logits [B,T,K]  z^(t) = (1/t) sum_{s<=t} U3^(s), the same quantity
                                 forward() returns at t=T and predict_early_exit()
                                 thresholds on. prefix_logits[:, -1] == forward() logits.
          cum_spk1/cum_spk2 [B,T]  cumulative per-sample hidden spikes
          cum_sops      [B,T]    cumulative per-sample synaptic operations, using the
                                 same accounting as forward()'s `synops`
                                 (spk1*fanout1 + spk2*fanout2), so exiting at t has a
                                 real measured cost rather than a t/T approximation
          cum_in_spikes [B,T]    cumulative encoder input spikes per sample
        """
        T, B, _ = spikes.shape
        dev = spikes.device
        m1, m2, m3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        mem_sum = torch.zeros(B, self.fc3.out_features, device=dev)
        c1 = torch.zeros(B, device=dev); c2 = torch.zeros(B, device=dev)
        cin = torch.zeros(B, device=dev)
        pl = torch.empty(B, T, self.fc3.out_features, device=dev)
        k1 = torch.empty(B, T, device=dev); k2 = torch.empty(B, T, device=dev)
        kin = torch.empty(B, T, device=dev)
        for t in range(T):
            s1, m1 = self.lif1(self.fc1(spikes[t]), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            s3, m3 = self.lif3(self.fc3(s2), m3)
            mem_sum = mem_sum + m3
            pl[:, t, :] = mem_sum / (t + 1)
            c1 = c1 + s1.sum(dim=1); c2 = c2 + s2.sum(dim=1)
            cin = cin + spikes[t].sum(dim=1)
            k1[:, t] = c1; k2[:, t] = c2; kin[:, t] = cin
        return {"prefix_logits": pl.cpu(), "cum_spk1": k1.cpu(), "cum_spk2": k2.cpu(),
                "cum_sops": (k1 * self.fanouts[1] + k2 * self.fanouts[2]).cpu(),
                "cum_in_spikes": kin.cpu()}

    @torch.no_grad()
    def predict_early_exit(self, spikes: torch.Tensor, theta_exit: float = 0.9,
                           theta_uncertain: float = 0.5):
        """SNN-only inference with confidence-triggered early exit. Returns
        (preds [B], exit_step [B], uncertain [B], total_spikes, spike_sites, synops)."""
        T, B, _ = spikes.shape
        m1, m2, m3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        mem_sum = torch.zeros(B, self.fc3.out_features, device=spikes.device)
        done = torch.zeros(B, dtype=torch.bool, device=spikes.device)
        exit_step = torch.full((B,), T, dtype=torch.long, device=spikes.device)
        preds = torch.zeros(B, dtype=torch.long, device=spikes.device)
        n_spk1 = n_spk2 = 0.0
        for t in range(T):
            s1, m1 = self.lif1(self.fc1(spikes[t]), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            s3, m3 = self.lif3(self.fc3(s2), m3)
            mem_sum = mem_sum + m3
            n_spk1 = n_spk1 + s1.sum()
            n_spk2 = n_spk2 + s2.sum()
            prob = F.softmax(mem_sum / (t + 1), dim=-1)
            conf, pred = prob.max(dim=-1)
            fire = (~done) & (conf >= theta_exit)
            preds[fire] = pred[fire]
            exit_step[fire] = t + 1
            done = done | fire
        # samples that never crossed theta_exit: final argmax, flag low-confidence
        prob = F.softmax(mem_sum / T, dim=-1)
        conf, pred = prob.max(dim=-1)
        preds[~done] = pred[~done]
        uncertain = (~done) & (conf < theta_uncertain)
        total_spikes = n_spk1 + n_spk2
        spike_sites = T * B * (self.hidden[0] + self.hidden[1])
        synops = (n_spk1 * self.fanouts[1] + n_spk2 * self.fanouts[2]) / B
        return preds, exit_step, uncertain, total_spikes, spike_sites, synops


def quantize_int8_(model: nn.Module) -> nn.Module:
    """Post-training per-tensor int8 fake-quant of Linear weights (deploy-time
    stand-in for QAT). Returns the model, modified in place."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                w = m.weight.data
                scale = w.abs().max() / 127.0 + 1e-12
                m.weight.data = torch.round(w / scale).clamp(-127, 127) * scale
    return model
