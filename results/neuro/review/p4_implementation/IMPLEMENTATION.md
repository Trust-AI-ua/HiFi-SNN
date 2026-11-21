# Implementation details: Tree-to-Spike student and Vanilla SNN, float to Lava

Every equation below is the code's. File references are to the repository at the commit that carries this file.
Per-seed numeric values (scales, thresholds, bias ranges, Lava parameters) are in `deployment_params.csv`. One
flow traced step by step through both Lava networks is in `trace_edgeiiot_seed42.csv`.

## Tree-to-Spike student

- **Encoder** (`src/models/tree_to_spike.py`, `TreeGuidedSpikeEncoder`; integer form `scripts/r2_integer_inference.py`, `IntEncoder`).
  - Thresholds E_j are the teacher's unique split values for feature j.
  - Level k = number of thresholds at or below x. It is found by comparing IEEE-754 bit patterns as integers, which is exact.
  - Channel c of feature j has offset o_c = (c - (m_j - 1)/2) * 0.12. It emits one spike at
    t = round((1 - clip(k/|E_j| + o_c, 0, 1)) * (T - 1)), with a float32 level and NumPy rounding (half to even).
  - The channel count is m_j = max(1, ceil(importance_j / max importance * 3)).
  - Features with importance below the 20th percentile are pruned.
  - The full tables are in `../p3_encoders/`.
- **Network** (`SNNStudent`): 3 Linear layers (C -> 64 -> 64 -> K), each followed by snnTorch `Leaky`.
  - Leak: beta 0.9. Threshold: 1.0. Reset: subtract, delayed (r_t = [v_{t-1} > 1]).
  - Readout: mean output membrane over T, argmax.
  - Trained in float, then projected **after training** by `quantize_int8_` (per tensor: s = max|W|/127, W <- round(W/s) s).
    The deployed checkpoint is the projected one; `checkpoint_fp32.pt` is the unprojected student.
- **Integer Q = 8** (`r2_integer_inference.int_snn`):
  - Current I = 2^8 * sum of the int8 weights of inputs that spiked + round(2^8 b / s).
  - Update v = ((v * 230) >> 8) + I - r * theta, with theta = round(2^8 / s).
  - Spike [v > theta]; readout argmax of summed output v.
  - Accumulators are int64 and checked to stay within int32.
- **Lava fixed point** (`scripts/neuro_a2_lava.py` `lava_params`; process `src/lava_sublif/sublif.py` `SubLIFReset`, emulation `emulate`):
  - Weights 2 * int8 (Lava's 8-bit mixed-sign weights are even).
  - vth = round(2/s), compared as vth << 6.
  - Bias 2b/s * 64 as a 13-bit mantissa with exponent.
  - du = 4095 (the current lasts one step); dv = 410 (beta 0.8999).
  - Membrane: 24-bit saturating. Reset: delayed subtract (v -= r * (vth << 6)).
  - Readout: summed output v.

## Vanilla SNN (SNN_LeakyParallel, latency coding)

- **Encoder:** latency code on MinMax features. t = floor((1 - x) * (T - 1)), no spike if x <= 0.01. One channel per feature, T = 25.
- **Network** (`src/models/snn.py`, `SNN` with `LeakyParallel`):
  - Hidden layer: an `nn.RNN` (ReLU) with diagonal recurrence beta = 0.85, threshold 1.0 and **no reset**,
    v_t = max(0, beta v_{t-1} + W_ih x_t + b), s_t = [v_t > 1].
  - Output: snnTorch `Leaky`, beta 0.85, threshold 1.0, delayed subtractive reset.
  - Readout: output spike count over T, argmax; ties go to the lower class index.
  - Trained in float; no projection during training.
- **Projection** (`scripts/neuro_vanilla_int.py`): the same rule as the student, per tensor s = max|W|/127 on W_ih and W_out.
  - On Edge-IIoTset, max|W_ih| is 10.4 to 16.4 against a median of 0.10 to 0.11.
  - So s is 0.08 to 0.13, the median |q| is 1 and 23 to 36% of the input weights round to 0.
  - The threshold is 7.5 to 12 int8 steps.
- **Integer Q = 8** (`neuro_vanilla_int.int_vanilla`):
  - Hidden: v = max(0, ((v * 218) >> 8) + 2^8 W_ih x + bq), with no reset.
  - Output: v = ((v * 218) >> 8) + 2^8 W_out s + bq - r * theta_o.
  - theta = round(thr/s * 2^8); bq = round(b/s * 2^8).
- **Lava fixed point** (`scripts/neuro_a2_vanilla.py` `vanilla_lava_params`; process `ReluLIFReset` hidden, `SubLIFReset` output; emulation `emulate_vanilla`):
  - Weights 2 * int8; vth = round(2 thr/s) << 6; bias 2b/s * 64 as mantissa with exponent.
  - du = 4095; dv = 614 (beta 0.8501).
  - Hidden: clipped to [0, 2^23 - 1], no reset. Output: delayed subtract.
  - Readout: output spike count.
  - On Edge-IIoTset no flow reaches the saturation bound (`../p1_stagewise/*_summary.json`).

## Verification chain (each run stops on a failed check)

1. The float stage reproduces the record's test macro-F1, for the student and the Vanilla GPU replay.
2. The integer encoder emits exactly the float encoder's spikes.
3. Integer predictions equal the saved step-2 and amendment-4 predictions.
4. Lava equals its NumPy emulation bit for bit on the 2,000-flow sample, and the full-test emulation equals Lava on that sample.
5. SANA-FE's per-layer spikes and readout equal Lava's on every sampled flow before any energy is reported (amendment 2).

## Trace columns (`trace_edgeiiot_seed42.csv`)

- Each row is one step, one layer and one watched neuron: the most active neuron of that layer on this flow, so resets are visible.
- Columns:
  - `v_prev`: the membrane before the step.
  - `leak_term`: the decay, sign(v)(|v (4096 - dv)| >> 12).
  - `u`: this step's synaptic input, the sum of 2 * int8 weights of spiking inputs, << 6.
  - `bias_eff`: the bias.
  - `reset_subtracted`: r * (vth << 6) where the previous step spiked; 0 for the Vanilla hidden layer.
  - `v`: after the step.
  - `threshold`: vth << 6.
  - `spike`: the watched neuron's spike this step.
  - `layer_spikes`: the layer's total spikes this step.
- The trace's readout is checked against the batched emulation.
