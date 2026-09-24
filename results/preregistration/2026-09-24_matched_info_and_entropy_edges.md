# Pre-registration: matched-information dense twin (E1) and supervised non-tree edges (E2)

Written 2026-09-24, before any E1 or E2 run. Committed with the code, before launch.
Scope: the four IoT benchmarks (CICIoT2023, CICIoMT2024, Edge-IIoTset, TON-IoT),
seeds 42-46, the production splits (each run records its loader_kwargs). Everything
not named below is unchanged from production. At N = 4 datasets all comparisons are
descriptive: means, per-dataset signs, five-seed SDs; no significance claims.

## E1. Matched-information dense twin ("DNN-TQ")

Question: is the dense twin's floating-point lead over Tree-to-Spike (+1.05 IoT mean,
M3 reruns) the extra level resolution it receives, or the cost of spiking computation?

Change: DNN-Tree is trained and evaluated on the levels exactly as the spike train
encodes them, l_hat = 1 - round((1 - l)(T - 1)) / (T - 1), T = 20, instead of the
continuous bin levels l. Same teacher, channels, loss (distillation minus sparsity),
optimizer, epochs, seed. Evaluated in fp32 (captured before projection) and at the
deployed per-tensor int8 grid. Output: results/DNN-TQ/<dataset>/seed_<S>/.

Primary statistic: gap_TQ = DNN-TQ fp32 - Tree-to-Spike fp32 (IoT mean, five seeds;
Tree-to-Spike fp32 from the M3 reruns, 84.72). Reference gap with level input: +1.05.

  A. gap_TQ <= +0.35 (at least two thirds of the gap closed): "given the same input
     information, spiking and dense computation match in floating point." Report the
     int8 comparison alongside.
  B. gap_TQ >= +0.70 (at most one third closed): "about one point is the cost of
     spiking computation itself; the resolution the spike train discards is not it."
  C. otherwise: "the input resolution explains part of the gap"; report the fraction
     closed, 1 - gap_TQ / 1.05.
Per-dataset gaps and the int8 results are reported whichever outcome occurs.

## E2. Supervised non-tree edges ("entropy")

Question: does the tree's threshold PLACEMENT matter, or would any class-supervised
univariate placement do as well?

Change: encoder_mode "entropy". Identical channels to "tree" (same pruning, importance
budget, jitter) and, per kept feature, the same edge COUNT |E_j|, like the quantile
control; the edges are class-supervised univariate cut points on the training data:
greedy best-first splitting that maximizes information gain (Fayyad-Irani boundary
points), stopped at |E_j| cuts. When no positive-gain split remains before |E_j|, the
largest remaining segment is split at its median so the count still matches; the
number of such filler cuts is recorded. Full production loss (with distillation),
deployed int8, like the other ablation arms. Output: results/encoder_ablation/<ds>/entropy/.

Statistic: d = tree - entropy macro-F1 (five-seed means), per dataset and IoT mean.

  A. Tree clearly wins: IoT mean d >= +1.0 AND tree ahead on >= 3 of 4 datasets.
     Claim: tree-derived threshold placement is the mechanism.
  B. Match: |IoT mean d| < 1.0. Claim becomes "supervised threshold placement", with
     the tree as the source that also supplies importances and the teacher.
  C. Entropy clearly wins: IoT mean d <= -1.0 AND entropy ahead on >= 3 of 4. Reported
     as found: the tree is not the best source of thresholds.
  D. Otherwise (|d| >= 1.0 but split 2/2): mixed; per-dataset reporting, no mechanism
     claim beyond supervised placement.
SOPs per sample (full window, input-inclusive) are reported for both arms.

## Guards (must pass before the corresponding runs start)

  G1. The modified encoder, in "tree" and "quantile" modes, is bitwise identical to the
      committed encoder (edges, channels, encode(), encode_dense()) on every IoT dataset
      at seed 42; "entropy" keeps the tree's channels exactly.
  G2. The modified DNN-Tree trainer with the production input reproduces the stored
      DNN-Tree TON-IoT seed-42 int8 macro-F1 exactly, and its fp32 capture reproduces
      the M3 rerun's recorded fp32 value exactly.
  G3. l_hat from encode_dense equals 1 - (first-spike step)/(T - 1) of encode() for
      every channel.
  G4. The entropy cutter recovers known cut points on synthetic data and is
      deterministic.
  After the runs: every new run's split (test size, per-class training counts) matches
  the stored Tree-to-Spike run for its dataset and seed.
