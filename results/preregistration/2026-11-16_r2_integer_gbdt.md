# Pre-registration: revision-2 checks, steps 1 to 3 (integer execution, validation-selected compact GBDT, missing false-alarm rates)

Written 2026-11-16 on branch `journal-r2` (from `main` at db54f7f), before any of the runs
below. Published numbers: tag `tnnls-journal-r1`. Scope: the four IoT benchmarks
(CICIoT2023, CICIoMT2024, Edge-IIoTset, TON-IoT), seeds 42 to 46. Thread variables stay
unset; every refit teacher must equal the stored one.

After step 3 the work stops for a decision on the paper's framing (rules R2 and R3).

## Step 1. Missing false-alarm rates

Report macro-F1, FAR and DR (five-seed mean and SD) for the compact GBDT and the full GBDT, and
for PLE-MLP at fp32, per-tensor int8 and per-channel int8, beside Tree-to-Spike and DNN-Tree.
Sources: stored records and tables. The compact and full GBDT values are recomputed from the
refit teachers of step 2 as a cross-check; a mismatch is reported, not overwritten.

## Step 2. Integer-only execution

Reimplement inference in NumPy with integer arithmetic only, using int8 weights and int32
accumulators (checked: no value leaves the int32 range).

- **Encoder (both models).** The level index is the count of thresholds at or below the
  feature, found by comparing IEEE-754 bit patterns as integers (exact for non-negative
  floats). A per-channel table maps the index to the spike time, or for the dense twin to
  the 8-bit input. No floating-point arithmetic.
- **Tree-to-Spike.** Binary spikes need no activation quantization. Membrane in fixed point
  with Q = 8 fractional bits in units of each layer's weight scale; input current is the sum
  of int8 weights of the spiking inputs plus the integer bias; leak `(v * 230) >> 8`
  (beta = 0.8984 for 0.9); threshold `round(2^Q / s)`; subtract-reset delayed one step, as in
  snnTorch. Readout: argmax of the summed output membrane.
- **DNN-Tree.** TFLite-style: 8-bit input, int8 weights, int32 accumulators, integer bias,
  ReLU, requantization to uint8 by a fixed-point multiplier, with activation ranges
  calibrated on the validation split (max over validation). Argmax of the last accumulator.
- **Guards.** First, the floating-point path must reproduce each stored test macro-F1
  exactly from the checkpoint, and the checkpoint weights must lie on the int8 grid. Second,
  the integer encoder must emit exactly the spikes and levels of the float encoder.
- **Reported.** Integer macro-F1, FAR and agreement with the stored predictions, per dataset
  and seed. Sensitivity: a 16-bit leak (`(v * 7373) >> 13`, beta = 0.90002) and Q = 0.
- **R2 (decision).** Integer execution "hurts the spiking model" if the IoT-mean change in
  Tree-to-Spike macro-F1 (integer minus reported) is at most -0.5 points, or any dataset's
  five-seed mean change is at most -1.0. The Tree-to-Spike minus DNN-Tree difference under
  integer execution is reported beside the published +0.72.

## Step 3. Validation-selected compact GBDT

- **Grid.** The 48-configuration grid of `scripts/gbdt_compression.py` exactly as that code
  defines it (rounds 25, 50, 100, 200 by depth 3, 4, 6; the "gain_prune 0.9" arm; 16- and
  8-bit leaves), with the same hyperparameters, cost weights and `n_jobs=8`.
- **Note.** The "gain_prune 0.9" arm counts trees across classes and refits with that many
  boosting rounds, so it adds rounds rather than pruning. It is kept for comparability with
  the published envelope and reported as "more rounds".
- **Seeds and pricing.** Five seeds (42 to 46). Size uses the verified packed format (2 B
  feature, 4 B float32 threshold, 4 B children; leaves at the leaf bit width), taken from
  `src/serialize.pack_booster` on the actual model.
- **Selection.** Highest validation macro-F1 among configurations within 128 KiB; ties go to
  fewer bytes. The test split is used once, for the selected configuration.
- **Implementation.** Configurations that differ only in the number of rounds are prefixes of
  one fitted model (`iteration_range`). Guard: on seed 42 this reproduces the published
  test macro-F1 of the sweep for at least three configurations per dataset.
- **Reported.** Test macro-F1, FAR, size and configuration per dataset and seed, and the
  paired difference with Tree-to-Spike.
- **R3 (decision).** The tree "wins on more benchmarks" if the validation-selected GBDT within
  128 KiB has a higher five-seed mean macro-F1 than Tree-to-Spike on 2 or more of the 4 IoT
  datasets, or a higher IoT mean. Published reference: the test-selected seed-42 envelope
  beat Tree-to-Spike on 2 of 4 (CICIoT2023, TON-IoT).
