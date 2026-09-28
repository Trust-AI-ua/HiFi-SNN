# Pre-registration: Part B analyses and experiments (TNNLS journal revision)

Written 2026-09-28, before any Part B analysis or run. The plan
(https://claude.ai/artifact/BeSAJ3buj3SA47NSLM7VP6) was approved by the reviewer
with the decisions recorded below as D1-D9. This file is committed and pushed to the
remote before the first Part B analysis starts, so its timestamp precedes every
result. Later changes are dated amendments appended here before the affected runs.

Repository state: branch `leakage-repair` at 792f702; published numbers at tag
`tnnls-submission` (2adfc77, 749 macros). Scope: the four IoT benchmarks are primary
(CICIoT2023, CICIoMT2024, Edge-IIoTset, TON-IoT); general-purpose datasets where
stated. Seeds 42-46.

## Common protocol

- Outputs go under `results/journal_partB/<item>/`. Every record carries the commit,
  a run id, the script and the loader arguments. Aggregation only appends.
- Guards run before anything they protect and stop that item on failure:
  - a new run's split (test size, per-class training counts) and, where a teacher is
    fitted, its test macro-F1 must equal the stored Tree-to-Spike run's;
  - thread variables (OMP/MKL/OPENBLAS/TORCH_NUM_THREADS) stay unset, since the
    XGBoost teacher depends on the OpenMP thread count;
  - a new code path must reproduce stored values on unchanged cells before it
    produces new ones.
- The H200 is dedicated to this work; the GPU queue runs at full width.
- All wall-clock and memory measurements (B3 latency, B8 peak memory) run last, after
  the GPU queue has drained, under the existing gate of
  `scripts/deployment_measure.py`: the release file
  `results/deployment/.RELEASE_DEPLOYMENT` must exist and the one-minute load average
  must stay below 2.0 for 300 s.
- Each item gets its own section in `scripts/make_numbers.py`. One clean regeneration
  at the end, then a new annotated tag.

**D1 (tag).** `tnnls-submission` stays as the submitted state. A new annotated tag
`tnnls-journal-r1` is created at the end. This pre-registration commit is pushed
before the first run.

## B1. Empirical check of Proposition 1, and the encoder audit (no training)

**Scope (D2).** All eleven datasets, seeds 42-46 (55 cells). The IoT four are reported
in the main text, all eleven in the appendix.

**Procedure.**
1. Refit the teacher exactly as in production. Guard: its test macro-F1 equals the
   stored Tree-to-Spike run's, for all 55 cells.
2. Build the tree encoder with every split threshold, before pruning.
3. For each test flow, build a surrogate with the same level vector. For each split
   feature j, let k be the number of thresholds tau at or below x_j (the encoder's
   convention: a value equal to a threshold counts as above it).
   - k >= 1: x'_j = tau_k.
   - k = 0: x'_j = the largest float32 value below tau_1.
   - Features never used in a split: x'_j = 0 (stricter than leaving them).
4. Score original and surrogate through the production path (a DataFrame into
   XGBoost, float32 casting, missing-value routing). Assert that no input is NaN.
5. Report per dataset and seed: the share of identical predicted labels and the
   largest absolute difference in predicted probabilities.

**Export.** One row per dataset, seed and feature: retained after pruning, |E_j|,
allocated channels, T, unique unshifted levels on test, unique occupied time bins on
test, gain, dtype, comparison convention. From it: 352/767, per-dataset medians, the
share of retained features exceeding the bound, and the share of total gain removed
by pruning.

**Guard.** On the IoT four, the export reproduces 352 violating and 767 retained
features exactly (the current macros, from the M3 saved encoder tables).

**Rule.** Expected: 100% agreement. Any mismatch is reported with its cause (dtype,
boundary convention, NaN routing), then either fixed in code or written into the
proposition's scope.

## B2. Is the entropy control already the per-feature supervised tree? (no training unless triggered)

**Procedure.** For each IoT dataset at seed 42 and each kept feature j, fit
`sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=|E_j|+1)`
with no depth limit and the teacher's cost weights as sample weights
(w_c = N/(n_c + 1e-6), float32, as in `_fit_teacher`). Compare its thresholds with the
entropy cutter's cuts.

**Fillers (D3).** Only information-gain cuts are compared. The cutter's median filler
cuts are reported separately.

**Coincidence.** Tree cuts are matched one-to-one to the nearest entropy cut of the
same feature. A pair coincides when it splits the feature's distinct training values
the same way (x >= cut), or when the two cuts differ by less than the smallest gap
between adjacent entropy cuts of that feature (for single-cut features, only the
partition criterion applies). Reported: exact-match share, coincidence share, and
median absolute offset in scaled [0, 1] units, per dataset and pooled.

**Rule.** If at least 95% of cuts coincide (pooled over the IoT four), the two are
reported as the same construction (recursive information-gain splitting is a
per-feature entropy tree) and nothing further runs. Otherwise a fourth arm runs:
20 runs under exactly the E2 protocol and guards, reported as found.

## B3. Target-aware piecewise-linear encoding with a dense MLP

**Package.** `rtdl_num_embeddings==0.0.12` (Gorishniy et al.), installed with
`--no-deps` so PyTorch is not changed. The version is recorded in every run.

**Encoding.** `compute_bins(X_train, n_bins, y=y_train, regression=False,
tree_kwargs={"min_samples_leaf": 64, "min_impurity_decrease": 1e-4})` on training rows
only (the package's documented setting), then `PiecewiseLinearEncoding(bins)` with
within-bin coordinates intact. Columns with fewer than two distinct training values
pass through unencoded.

**Model and training (D4).** PLE, then Linear(d, W), ReLU, Linear(W, W), ReLU,
Linear(W, K). No dropout. Adam, weight decay 0, batch size 128. Unweighted
cross-entropy from labels, as in the reference recipe; no distillation. The same
repaired loaders, loader arguments, splits and seeds as the stored runs (split
guard).

**Search (D4, D5).** Seed 42, per dataset. Grid: bins {16, 64} x width {128, 256} x
learning rate {1e-3, 3e-4} x epochs {10, 30}: 16 configurations per dataset, 64
search runs. Validation macro-F1 is computed after every epoch and the best epoch's
checkpoint is kept (ties: the earliest epoch). The configuration with the highest
validation macro-F1 is selected (ties: fewer MACs).

**Final runs.** The selected configuration on seeds 42-46; seed 42 reuses its search
run (16 new runs).

**Precision (D5).** fp32 (the selected checkpoint), per-tensor int8
(`quantize_int8_`) and per-channel int8 (`quantize_int8_per_channel_`) on every
nn.Linear weight; bin edges stay float32. These are the three precisions of
Table VIII.

**Reported.** Macro-F1 +/- sample SD at each precision; FAR; parameters; encoder
storage (bin edges at 4 B); MACs, with the encoding counted as (encoded features x
bins) MACs; operation-priced energy at the same Horowitz prices (4.6 pJ per 32-bit
MAC, 0.23 pJ int8); CPU batch-1 latency with the protocol of
`scripts/deployment_measure.py`, seed 42, under the gate. Disclosed asymmetry: PLE-MLP
gets checkpoint selection and up to 30 epochs; the other models use a fixed 10-epoch
budget.

**Rule (the reviewer's 4.4, adopted as is).** If PLE-MLP is more accurate, it is
reported as a strong conventional-hardware alternative alongside the dense twin and
GBDT; the contribution stays the representation for a spiking substrate. No wording
that depends on the direction of the result is written before it exists.

## B4. One non-random split

**Feasibility first, no training (D6).** Option (a), downloading the full TON_IoT
release, is not pursued.
- (c) CICIoMT2024: establish from the dataset's documentation (the CIC dataset page
  and the dataset paper) whether the provider's train and test pcaps are separate
  captures or sessions. The evidence is quoted with its source. Supporting check: the
  exact-duplicate rate between the provider partitions (from B5).
- (b) TON-IoT host-disjoint: groups are source IPs. Per seed, a seeded permutation of
  hosts assigns hosts to test until the test set holds at least 20% of rows (the
  paper's test fraction); validation is 10% of the remaining rows, stratified, as in
  the loader. Per-class training support is counted for every seed.

**Runs.**
- If CICIoMT2024 is confirmed capture-disjoint: its provider split is reported as the
  paper's non-random evaluation. The existing Tree-to-Spike, DNN-Tree and entropy runs
  are used as they are, and only a validation-selected GBDT is added: cost-weighted
  XGBoost, rounds {50, 100, 200} x depth {3, 4, 6}, selected on validation macro-F1,
  seeds 42-46.
- If the TON-IoT host-disjoint split keeps training support for every class on every
  seed: the four arms (Tree-to-Spike, DNN-Tree, entropy-edge Tree-to-Spike, the
  validation-selected GBDT) run there on seeds 42-46, with teacher, encoder and gates
  fitted on training and validation data only. If not, the classes that lose training
  support are reported, per seed, and nothing runs on that split.

**Rule.** Report per-class support, macro-F1, FAR and paired differences. The question
is whether the tree-vs-entropy advantage and the parity with the dense twin persist,
not whether absolute scores drop. Results are reported whatever they are.

## B5. Data-handling checks (no training unless triggered)

**Train-only vocabulary.** Per dataset and seed: one-hot columns whose category never
occurs in that seed's training partition, and the share of test rows carrying such a
category. The TON-IoT and Edge-IIoTset loaders build one-hot columns with
`pandas.get_dummies` on the full table before the split; CICIoT2023 and CICIoMT2024
are numeric only. General-purpose datasets with categoricals are reported too.

**Trigger.** If affected test rows exceed 0.1% on any seed of a primary dataset, that
dataset is rerun for all four models (20 runs) with the vocabulary fitted on training
rows and unknown categories mapped to an all-zero one-hot (D7). The earlier cells are
kept, marked superseded.

**Cross-partition duplicates.** Exact feature-vector duplicates between train and
test on the preprocessed arrays, per dataset, class and seed. Reported, with no
rerun. A class is dominated when more than 50% of its test rows are duplicated in
training (D7); for such classes, per-class results are reported with and without
the duplicates.

**False-positive term.** `src/training_treespike.py` lines 157-159 return a constant
zero with no gradient when a batch has no benign samples. A unit test confirms a
finite loss and a zero false-positive gradient on such a batch, and a nonzero one on
a batch with benign samples. Sec. III-C gets one sentence. Retraining only if the
test shows NaN or a wrong gradient.

## B6. Convergence evidence (no training unless triggered)

**Curves.** Validation curves by epoch on the IoT four for tree, entropy, quantile,
no-KD, raw, DNN-Tree and Vanilla SNN. Macro-F1 where it is stored (Tree-to-Spike and
DNN-Tree, `results/diagnostics/val_curves/`); validation accuracy elsewhere, as the
declared proxy, labelled as such in the figure (D8).

**Trigger (D8).** For each arm and dataset, the mean over seeds of (value at epoch 10
minus value at epoch 8), excluding the three stalled entropy runs. If it exceeds 1.0
point for any non-stalled arm, the full affected block (all arms, all seeds) is rerun
with a 20-epoch budget and validation checkpoint selection. Otherwise the curves go
into the appendix and the fixed-budget results stand.

**Stalled runs.** CICIoT2023 seeds 44 and 45 and CICIoMT2024 seed 46 (entropy arm) are
rerun with an observer that records, per epoch, the loss terms, per-layer firing rates
and the output-class distribution. Guard: the rerun must reproduce the stored training
curve and test score exactly; otherwise it is reported as not reproducible.

## B7. Envelope selection on validation

The 768-row compression sweep stores test macro-F1 only. Reselection on validation
would need a refit of the whole sweep (about 27 h), which is not done. The envelope
table stays labelled exploratory. Nothing runs.

## B8. Peak memory

Package-energy counters are not exposed (KVM virtual machine, no
`/sys/class/powercap`), so the energy part is skipped under the item's own rule.
Peak resident memory per model over a fixed workload (the latency workload of
`scripts/deployment_measure.py`) is measured with the timing runs, under the same
gate (D9).

## Order of execution

1. Commit and push this file.
2. No-training items: B1, B2, the B5 checks, the B6 analysis, the B7 note, and the B4
   feasibility checks. Results and trigger outcomes are reported before any GPU run.
3. GPU queue at full width: the B6 diagnostic reruns, any triggered reruns (B2 arm,
   B5 vocabulary, B6 longer budget), the B3 search and final runs, the B4 arms.
4. After the queue drains: B3 latency and B8 memory under the gate.
5. Aggregation, macros, one clean regeneration of `numbers.tex`, tag
   `tnnls-journal-r1`.

## Amendment 1 (2026-09-28, after the no-training results, before any GPU run)

The no-training outcomes were committed and pushed at 5d9baff before this amendment.
The reviewer's decisions on them:

**A1. B6 fired: run as registered.** The reported CICIoT2023 values become the
20-epoch, validation-selected runs. The fixed-budget cells move to the appendix,
labelled "original fixed-budget protocol". "The full affected block" means every arm
that appears in a main-text CICIoT2023 cell, seeds 42-46:
Tree-to-Spike, DNN-Tree, Vanilla SNN, TASNN, quantile, uniform, all, no-KD, raw
(m8_raw), entropy, entropy-tree (the B2 arm, A2) and the spike-time-input dense twin
(DNN-TQ): 12 arms, 60 runs. Protocol: 20 epochs; after every epoch, macro-F1 of the
fp32 model on the loop's own validation pass; the best epoch's weights are kept (ties:
earliest); fp32 weights are captured before the deployed int8 projection, so Table
VIII's per-tensor and per-channel columns regenerate from the same runs. Regenerated
from the new CICIoT2023 runs: Tree-to-Spike's early-exit threshold selection, the
operation counts, the M8 decomposition and the exit audit. Every table caption labels
CICIoT2023 as trained under the 20-epoch budget. Guard: for each trainer (Tree-to-Spike
family, DNN-Tree/DNN-TQ, Vanilla SNN, TASNN) the new code path, run with selection off
and 10 epochs, reproduces a stored run exactly before any block run. The block starts
after the reviewer has seen its arm list and run count.

**A2. B2 fired: the fourth arm runs**, 20 runs (IoT four, seeds 42-46, 10 epochs) under
the E2 protocol and guards, plus its CICIoT2023 cells in the A1 block. Encoder mode
"entropy_tree": the tree's channels and per-feature edge count |E_j| as in "entropy";
the cuts are the thresholds of `DecisionTreeClassifier(criterion="entropy",
max_leaf_nodes=|E_j|+1)` fitted per feature with the teacher's class weights, each
converted to the encoder's convention as the smallest distinct training value above
the threshold (the same partition of the training values); when the tree stops early,
the cutter's median fillers complete the count. Guards: the existing encoder modes stay
bitwise identical to the encoder at 5d9baff; the sample weights equal the teacher's.

**A3. B4 CICIoMT2024 adopted as the non-random evaluation**, described as: "the
provider's pcap-file-level partition: test flows come from capture files never used in
training; whether these files are separate recording sessions is not documented." The
existing Tree-to-Spike, DNN-Tree and entropy runs are used as they are. Added: the
validation-selected GBDT (cost-weighted XGBoost, rounds {50, 100, 200} x depth
{3, 4, 6}, selected on validation macro-F1, seeds 42-46).

**A4. B4 TON-IoT host-disjoint split: recorded as not usable.** On every seed 1 to 5
classes have no test rows (results/journal_partB/B4), so macro-F1 would cover a
different class set on each seed. The decision rests on feasibility data only; no model
was run on this split.

**A5. M3 Edge-IIoTset reruns.** The four M3 cells that do not reproduce the stored runs
(Tree-to-Spike and DNN-Tree, seeds 45 and 46) are rerun in the default environment.
If they reproduce the stored runs, Table VIII is regenerated from them; if not, the
per-seed disagreement and its cause are reported in the appendix.

**A6. B5 dominated classes.** Per-class results with and without the cross-partition
duplicates, for CIC-IDS2017 probe, CSE-CIC-IDS2018 r2l and DoS, and KDDCup99 DoS, from
the stored models only (no training), in the appendix.
