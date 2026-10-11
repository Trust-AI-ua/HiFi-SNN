# Reproducing the paper

This guide maps every table, figure and pre-registered item of the Tree-to-Spike manuscript to
the commands that produce it. Run everything from the repository root with the environment
of the [README](../README.md#installation). Commands that train take a dataset and seeds; the
reported runs use all eleven datasets (or the four IoT ones where stated) and seeds 42 to 46.

The committed run records already contain every result, so steps 1 to 4 are needed only to
rerun experiments. Steps 5 and 6 regenerate the tables, numbers and figures from the records
in a few minutes.

## Ground rules

- **Leave thread variables unset.** `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and torch's thread setting change the XGBoost teacher, and with it
  every downstream model. Each record stores the teacher's test macro-F1
  (`test_plus.teacher_upper_bound`); a rerun is valid only if it equals the stored value.
- **Skip logic.** Runners skip a cell whose `results.json` exists; pass `--force` to rerun.
- **Split signature.** Each record stores the test size and per-class training counts; the
  analysis scripts refuse to compare runs whose signatures differ.
- **Hardware.** The reported runs used one NVIDIA H200. Most cells take minutes; a
  Tree-to-Spike run on CICIoT2023 with 20 epochs takes several hours.

## 1. Main comparison (Table III, Table IV, appendix tables)

```bash
D=toniot; S="42 43 44 45 46"
python scripts/run_treespike.py $D --seeds $S                                 # Tree-to-Spike -> results/Tree-to-Spike/
python scripts/run_dnn.py       $D --seeds $S                                 # DNN-Tree      -> results/DNN-Tree/
python scripts/run_tasnn.py     $D --seeds $S                                 # TASNN         -> results/TASNN/
python scripts/run_sweep.py     $D --only LeakyParallel --encodings latency --seeds $S   # Vanilla SNN -> results/<D>/SNN_LeakyParallel/latency/
python scripts/backfill_snn_metrics.py --only $D --neuron LeakyParallel --encoding latency   # Vanilla SNN SOPs and early exit
```

Configurations live in `experiments/<Model>/<dataset>/config.yaml` (Vanilla SNN:
`configs/<dataset>/SNN_LeakyParallel/latency/config.yaml`). All models train for 10 epochs;
CICIoT2023 is the exception below.

### CICIoT2023: 20 epochs with validation selection (amendment A1)

Pre-registered item B6 found the 10-epoch budget short on CICIoT2023, so all twelve arms that
appear in a CICIoT2023 cell were rerun for 20 epochs, keeping the epoch with the best
validation macro-F1.

```bash
python scripts/partB_block_guard.py          # the new path must reproduce stored 10-epoch runs first
python scripts/partB_b6_block.py <arm> <seed>   # arm: tree dnn vanilla tasnn dnn_tq raw quantile uniform all nokd entropy entropy_tree
python scripts/partB_swap_in.py              # puts block runs in the reported places (once; writes a manifest)
python scripts/partB_a1_followup.py          # prefixes, exit thresholds, exit audit for the block runs
```

The replaced 10-epoch runs are in `results/fixed_budget_ciciot2023/` (appendix values).

## 2. Mechanism

| Item | Commands | Output |
|---|---|---|
| Encoder ablation (Table V): quantile, uniform, all thresholds, no KD | `python scripts/run_encoder_variant.py <dataset> <arm> <seed>`, then `python scripts/encoder_ablation.py --plot` | `results/encoder_ablation/` |
| Raw-latency arm (M8 step 3) | `python scripts/run_m8_arm.py <dataset> <seed>`, `python scripts/m8_input_spikes.py`, `python scripts/m8_decomposition.py` | `results/encoder_ablation/<d>/m8_raw/`, `results/tables/tab_m8_decomposition*.csv` |
| E1: dense twin on spike-time input | `python scripts/run_dnn_tq.py --guard`, then `python scripts/run_dnn_tq.py <dataset> <seed>` | `results/DNN-TQ/` |
| E2: entropy edges | `python scripts/run_encoder_variant.py <dataset> entropy <seed>`; `python scripts/analyze_e1_e2.py` | `results/encoder_ablation/<d>/entropy/` |
| B2: entropy tree vs cutter, and the fourth arm | `python scripts/partB_b2_entropy_tree.py`; `python scripts/partB_b2arm_guard.py`; `python scripts/run_encoder_variant.py <dataset> entropy_tree <seed>` | `results/journal_partB/B2/`, `results/encoder_ablation/<d>/entropy_tree/` |
| Student vs teacher | `python scripts/teacher_vs_student.py` | `results/tables/tab_teacher_vs_student.csv` |
| Single-threshold features | `python scripts/single_edge_check.py` | `results/diagnostics/single_edge_partition.csv` |

## 3. Precision (Table VIII)

Reruns that capture fp32 weights before the int8 projection, then score per-tensor and
per-channel int8 on the same weights.

```bash
python scripts/val_curves.py Tree-to-Spike <dataset> --seed <seed>      # also DNN-Tree
python scripts/perchannel_check.py <Tree-to-Spike|DNN-Tree> <dataset> <seed> cuda
python scripts/perchannel_check.py --aggregate                          # results/tables/tab_perchannel.csv
python scripts/eval_m3_fp32_models.py                                   # Vanilla SNN and TASNN
python scripts/make_m3_table.py                                         # results/tables/tab_m3_quant.csv
```

## 4. Proposition 1, window length, early exit, data checks

| Item | Commands | Output |
|---|---|---|
| B1: Proposition 1 on every test flow, encoder audit | `python scripts/partB_b1_prop1.py` | `results/journal_partB/B1/` |
| Threshold counts per feature | `python scripts/m5_edges.py`, `python scripts/edge_stats_all.py` | `results/tables/tab_m5_edges.csv` |
| Window length T (ladder) | `python scripts/run_ladder.py --arms T10 T40 T80 --workers 4` | `results/ladder/<d>/T<T>/` |
| Early exit: prefixes, thresholds, audit | `python scripts/extract_prefix.py`, `python scripts/early_exit_sweep.py`, `python scripts/evidence_arrival.py`, `python scripts/m4_remainder.py`, `python scripts/exit_audit_iot.py`, `python scripts/exit_margin_test.py --dataset ciciot2023` | `results/early_exit/`, `results/tables/tab_m4_flags.csv`, `results/tables/tab_exit_audit_iot.csv` |
| B4: non-random split, validation-selected GBDT | `python scripts/partB_b4_feasibility.py`, `python scripts/partB_b4_gbdt.py` | `results/journal_partB/B4*/` |
| B5: unseen categories, duplicates | `python scripts/partB_b5_data_checks.py` | `results/journal_partB/B5/` |
| A6: dominated classes without duplicates | `python scripts/partB_a6_dominated.py` | `results/journal_partB/A6/` |
| B6: convergence trigger, stalled runs | `python scripts/partB_b6_convergence.py`, `python scripts/partB_b6_stalled.py <dataset> <seed>` | `results/journal_partB/B6/` |
| B3: PLE-MLP | `python scripts/partB_b3_ple.py search <dataset> <0-15>`, `... select`, `... final <dataset> <seed>` (needs `rtdl_num_embeddings`) | `results/journal_partB/B3/` |

## 5. Deployment and energy

Timing and memory are measured last, on a quiet machine: the measurement waits for the file
`results/deployment/.RELEASE_DEPLOYMENT` and a one-minute load average below 2.0 for 300 s.

```bash
python scripts/compact_gbdt.py                     # validation-selected compact teacher
python scripts/gbdt_compression.py                 # compression sweep (exploratory envelope)
python scripts/reprice_deployment_trees.py         # packed-tree pricing of the sweep
python scripts/partB_gated_measure.py              # deployment table (11 datasets), PLE-MLP latency, B8 peak memory
python scripts/deployment_energy2.py --iot         # total energy with memory, IoT four
python scripts/deployment_energy2.py
python scripts/iot_arithmetic.py                   # arithmetic energy, IoT four
```

## 6. Tables, numbers and figures

From the run records to every derived table:

```bash
python scripts/per_seed_cells.py
python scripts/make_results_table.py
python scripts/benchmark_4way.py
python scripts/make_arch_table.py
python scripts/m9_stats.py                          # pooled statistics, eleven datasets
python scripts/m9_iot.py                            # Friedman test, IoT four
python scripts/make_dep_systems_table.py
python scripts/make_table1_breakdown.py
python scripts/teacher_per_dataset.py --rejoin     # teacher table (quality columns kept, deployment join redone)
python scripts/make_teacher_table.py
python scripts/encoder_ablation.py --plot
```

Then the numbers and the figures:

```bash
python scripts/make_numbers.py               # results/tables/numbers_provenance.json, and numbers.tex locally
python scripts/make_manuscript_figures.py    # Figures/*.pdf, Figures/data/*.csv, Figures/README.md
```

`make_numbers.py` writes every value the manuscript quotes, with its source file, aggregation
and meaning. Each section refuses on a mismatched split or teacher. `make_manuscript_figures.py`
refuses unless its inputs equal the committed ones, then checks that its own code paths
reproduce the published values before drawing (the checks are listed in
[`Figures/README.md`](../Figures/README.md)).

On the committed records, `make_numbers.py` reproduces all 1,205 values of the tag
`tnnls-journal-r1` exactly.
