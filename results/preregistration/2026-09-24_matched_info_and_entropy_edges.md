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

## Amendment 1 (2026-09-24, before any training)

The first launch was stopped by the operator during G1, before any E1 or E2 run
trained (log: results/repaired_runs/e1_e2.log). No result of either experiment
existed when this amendment was written. The registered thresholds are unchanged.

E2, weighted criterion. The entropy cutter scores each candidate cut on the
teacher's cost-weighted class masses: every class's counts are multiplied by the
teacher's class weight w_c = N / (n_c + 1e-6), raised to teacher_cost_power (1.0 in
all four configs), computed in float32 exactly as _fit_teacher computes it. The
cutter and the teacher are then fitted to the same weighted objective, and the two
arms differ only in how cuts are chosen. The median fallback still splits on
unweighted sample counts (quantile semantics). The no-gain test for a filler is now
relative (gain <= 1e-12 x the segment's weighted n*entropy), since weighting
rescales the criterion.

E2, fallback share. Each run records, per kept feature, the tree's edge count, the
entropy cut count, the filler (median) cuts and any shortfall (results.json,
data.entropy_edges). The fallback share of a dataset is its filler cuts over its
entropy cuts, pooled over the five seeds, and it is reported per dataset. Reading
rule: where the share exceeds one half, that dataset's comparison is labelled
"largely against quantile placement", which Table V already covers, and is not
read as a test of supervised placement. The A-D outcome is still computed on all
four datasets as registered; the labels qualify its reading.

E2, scope. The write-up states in one clause that E2 isolates edge placement only:
channel selection, the per-feature edge budget and pruning stay tree-derived in
both arms.

E1, three precisions. Each E1 cell also scores per-channel int8 (one symmetric
scale per output row of every nn.Linear, s_i = max_j |W_ij| / 127, biases fp32),
projected from the same fp32 weights on a copy, before the deployed per-tensor
projection. References (tab_perchannel.csv, five-seed IoT means): Tree-to-Spike
fp32 84.72, per-tensor 84.64, per-channel 84.69; DNN-Tree with level input fp32
85.77, per-tensor 84.14, per-channel 85.33. The registered outcome is decided on
fp32 only; per-tensor and per-channel gaps are reported beside it.

E1, per-dataset reporting. Every E1 quantity is reported per dataset. For each
change from level input to spike-time input (chg = DNN-TQ - DNN-Tree level, at
fp32, per-tensor and per-channel), a dataset's share is its chg over the sum of
the four datasets' chg. When one dataset's share exceeds one half, the write-up
says that this dataset accounts for more than half of that change.

Added guards (must pass before any run starts):

  G2 (extended). The trainer's per-channel score for DNN-Tree TON-IoT seed 42 with
      the production input reproduces scripts/perchannel_check.py's recorded value
      exactly.
  G4 (extended). Unit class weights reproduce the unweighted cutter bitwise; the
      weighted cutter recovers separable boundaries; on imbalanced synthetic data
      the teacher's weights move the cut.
  G5. On every IoT dataset at seed 42, the class weights given to the cutter equal,
      bitwise, the per-sample weights the teacher is fitted with (captured at
      XGBClassifier.fit); the recorded diagnostics agree with the encoder's edges.
  G6. The per-channel projection in src equals perchannel_check.py's, bitwise, on
      the stored DNN-Tree and Tree-to-Spike TON-IoT seed-42 checkpoints.
  G1 compares against the encoder at 0af66b2, the last commit before E1/E2.

## Amendment 2 (2026-09-24, before any training)

The relaunch after amendment 1 stopped at G2, before any E1 or E2 run trained. G2
failed because the launcher capped OpenMP at two threads: XGBoost limits n_jobs to
the OpenMP thread count, and the TON-IoT seed-42 teacher fitted with two threads
scores 0.968515 test macro-F1 against the stored 0.969213; with the default
environment it reproduces 0.969213 exactly. The launcher no longer caps threads.
Every stored IoT run (Tree-to-Spike, DNN-Tree, both M3 reruns, the four ablation
arms) records the same teacher for its dataset and seed, so no reported comparison
mixes teachers.

The post-run check is extended: besides the split, each new cell's recorded teacher
test macro-F1 must equal the stored Tree-to-Spike run's exactly, so each arm is
compared under the same teacher. The analysis refuses to report otherwise. The
registered rules are unchanged.
