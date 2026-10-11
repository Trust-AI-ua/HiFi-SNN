# Scripts

Every script runs from the repository root (`python scripts/<name>.py`) and documents its
arguments in its module docstring. [`docs/REPRODUCING.md`](../docs/REPRODUCING.md) gives the order
in which they reproduce the paper. Thread variables (`OMP_NUM_THREADS` and similar) must stay unset.

## Training entry points

Each writes `results/.../seed_<S>/results.json` and skips cells that exist.

| Script | Purpose |
|---|---|
| [`run_treespike.py`](run_treespike.py) | Run Tree-to-Spike (Design C) for one dataset over one or more seeds. |
| [`run_dnn.py`](run_dnn.py) | Run the DNN-Tree (iso-architecture ANN twin of Tree-to-Spike) for one dataset over one or more seeds. Mirrors scripts/run_treespike.py: dataset is a required positional arg; each seed writes results/D |
| [`run_tasnn.py`](run_tasnn.py) | Run TASNN (Transformer-Augmented Spiking NN competitor) for one dataset over seeds. |
| [`run_sweep.py`](run_sweep.py) | Run the Phase-1 SNN sweep for one dataset. |
| [`run_dnn_tq.py`](run_dnn_tq.py) | E1: matched-information dense twin (pre-registered 2026-09-24). |
| [`run_encoder_variant.py`](run_encoder_variant.py) | Run ONE encoder-ablation cell (dataset, arm, seed), so the arms can run in a worker pool instead of encoder_ablation.py's sequential in-process loop. |
| [`run_m8_arm.py`](run_m8_arm.py) | M8 control arm: raw-feature latency encoding into the Tree-to-Spike student. |
| [`run_ladder.py`](run_ladder.py) | Attribution / component-isolation ladder for the TNNLS revision. |
| [`val_curves.py`](val_curves.py) | Per-epoch validation macro-F1 curves for DNN-Tree / Tree-to-Spike (diagnostic). |
| [`partB_b6_block.py`](partB_b6_block.py) | B6 CICIoT2023 block (amendment A1): one run per (arm, seed), 20 epochs, validation selection. |
| [`partB_b3_ple.py`](partB_b3_ple.py) | B3: target-aware piecewise-linear encoding with a dense MLP (pre-registered, Part B). |

## Record backfills

Add fields to existing records by replaying saved checkpoints; no retraining.

| Script | Purpose |
|---|---|
| [`backfill_snn_metrics.py`](backfill_snn_metrics.py) | Backfill early-exit + SynOps metrics for completed SNN ablation runs. |
| [`backfill_encoder_ttfs.py`](backfill_encoder_ttfs.py) | Backfill the encoder's mean time-to-first-spike (TTFS) for the encoder ablation. |
| [`backfill_vanilla_ttfs.py`](backfill_vanilla_ttfs.py) | Backfill the Vanilla SNN's mean encoder time-to-first-spike (TTFS). |
| [`backfill_dnn_activations.py`](backfill_dnn_activations.py) | Backfill DNN-Tree activated-neuron counts (the dense analog of hidden spikes). |

## Precision

| Script | Purpose |
|---|---|
| [`perchannel_check.py`](perchannel_check.py) | Would per-channel int8 scaling restore the dense twin? No training. |
| [`eval_m3_fp32_models.py`](eval_m3_fp32_models.py) | M3 for the models whose reported numbers are fp32: Vanilla SNN (SNN_LeakyParallel/latency) and TASNN. Inference only, nothing retrained. |
| [`make_m3_table.py`](make_m3_table.py) | M3 table: fp32 vs int8-grid test macro-F1 for all four models x 11 datasets. |
| [`layerwise_quant.py`](layerwise_quant.py) | Quantize one DNN-Tree layer at a time (per-tensor and per-output-channel) and score on test. |
| [`weight_scale_stats.py`](weight_scale_stats.py) | Per-layer weight-scale statistics for the int8 projection (M3 mechanism). |
| [`fc1_ratio.py`](fc1_ratio.py) | Per-layer weight-scale ratio before and after the dataset repair (M6/M3). |

## Early exit

| Script | Purpose |
|---|---|
| [`extract_prefix.py`](extract_prefix.py) | Extract per-timestep prefix artifacts for Tree-to-Spike. Inference only. |
| [`early_exit_sweep.py`](early_exit_sweep.py) | Validation-selected early exit, replayed offline from cached prefix artifacts. |
| [`early_exit_audit.py`](early_exit_audit.py) | Why does the validation gate disable early exit on 5 of 11 datasets? |
| [`evidence_arrival.py`](evidence_arrival.py) | Evidence-arrival step t': when does the correct class first become the argmax? |
| [`m4_remainder.py`](m4_remainder.py) | M4 remainder, from the regenerated (input-inclusive) prefix artifacts. |
| [`exit_audit_iot.py`](exit_audit_iot.py) | Early-exit audit statistics on the IoT four, all five seeds, from prefix artifacts. |
| [`exit_margin_test.py`](exit_margin_test.py) | Is max-posterior the wrong exit statistic? Two diagnostics plus a candidate remedy. |

## Encoder and mechanism

| Script | Purpose |
|---|---|
| [`encoder_ablation.py`](encoder_ablation.py) | Encoder ablation: what does the GBDT teacher actually contribute? |
| [`m5_edges.py`](m5_edges.py) | M5: how many retained features violate the TTFS resolution bound T >= \|E_j\| + 1. |
| [`edge_stats_all.py`](edge_stats_all.py) | \|E_j\| and the T >= max_j\|E_j\|+1 bound, all 11 datasets, production vs compressed. |
| [`m8_input_spikes.py`](m8_input_spikes.py) | Input-layer operations actually executed by the M8 raw-latency arm. |
| [`m8_decomposition.py`](m8_decomposition.py) | The four-step decomposition between the conventional SNN and Tree-to-Spike. |
| [`single_edge_check.py`](single_edge_check.py) | Do the tree and the entropy cutter pick the same cut on single-threshold features? |
| [`teacher_vs_student.py`](teacher_vs_student.py) | Teacher (full GBDT) versus Tree-to-Spike student, paired by seed, IoT subset. |
| [`analyze_e1_e2.py`](analyze_e1_e2.py) | Apply the pre-registered rules to E1 and E2 (results/preregistration/ 2026-09-24_matched_info_and_entropy_edges.md, with its amendment). Refuses to report while any cell is missing or any new run's sp |
| [`reproduce_crossed_2x2.py`](reproduce_crossed_2x2.py) | Reproduce the crossed 2x2 encoder x objective attribution from committed results. |
| [`reproduce_loss_ablations.py`](reproduce_loss_ablations.py) | Reproduce the loss-term ablation statistics: FAR effect, multiplicity, class weighting. |
| [`reproduce_noninferiority.py`](reproduce_noninferiority.py) | Reproduce the dense-twin one-point non-inferiority test from committed results. |
| [`sensitivity_analysis.py`](sensitivity_analysis.py) | Sensitivity sweep analysis: is Tree-to-Spike's default stable in its neighbourhood? |

## Statistics

| Script | Purpose |
|---|---|
| [`per_seed_cells.py`](per_seed_cells.py) | Rebuild the 44-cell per-seed table (11 datasets x 4 models x 5 seeds). |
| [`m9_iot.py`](m9_iot.py) | Statistics on the IoT subset (the paper's primary scope): descriptive results and the Friedman omnibus, for the main text. No post-hoc test and no CD diagram: at N = 4 blocks the Nemenyi critical diff |
| [`m9_stats.py`](m9_stats.py) | M9: Friedman / Iman-Davenport / Nemenyi over the four models, the Tree-to-Spike vs DNN-Tree equivalence analysis, and the CD diagram. |
| [`statistics_utils.py`](statistics_utils.py) | Canonical statistics for the paper's paired analyses. Single source of truth. |

## Deployment and energy

| Script | Purpose |
|---|---|
| [`deployment_measure.py`](deployment_measure.py) | Deployment measurement: serialized artifact bytes, batch-1 CPU latency, executed ops. |
| [`deployment_energy2.py`](deployment_energy2.py) | Memory-inclusive energy, per dataset, under two reuse assumptions. |
| [`iot_arithmetic.py`](iot_arithmetic.py) | Arithmetic energy on the IoT four, with every system counted the same way. |
| [`compact_gbdt.py`](compact_gbdt.py) | Compact GBDT baseline by validation-only tree-prefix selection. |
| [`gbdt_compression.py`](gbdt_compression.py) | Run 4: GBDT compression Pareto -- can any tree ensemble reach a 256 KB envelope? |
| [`reprice_deployment_trees.py`](reprice_deployment_trees.py) | Correct the leaf pricing of the GBDT rows in the deployment measurement. |
| [`envelope_recompute.py`](envelope_recompute.py) | Reprice the compression sweep under the verified packed format. |
| [`verify_serialization.py`](verify_serialization.py) | Verify the deployment serialization before it is used for measurement. |
| [`measure_encoder_storage.py`](measure_encoder_storage.py) | Measure the frozen tree-guided encoder's deployment storage per dataset. |
| [`latency_elp.py`](latency_elp.py) | Clean batch-1 latency + Energy-Latency-Product (ELP) re-measure from checkpoints, for Vanilla SNN (LeakyParallel/latency), Tree-to-Spike, and DNN-Tree. |
| [`fix_gbdt_dedup.py`](fix_gbdt_dedup.py) | One-off: drop rows the keyed merge failed to supersede in gbdt_compression.csv. |

## Data integrity and the leakage audit

| Script | Purpose |
|---|---|
| [`leakage_feature_audit.py`](leakage_feature_audit.py) | Feature-level leakage audit across all eleven benchmarks (read-only, CPU). |
| [`run_leakage.py`](run_leakage.py) | Leakage-resistant evaluation (round-2 reviewer item 1). |
| [`run_gbdt_leakage.py`](run_gbdt_leakage.py) | Cost-sensitive GBDT teacher evaluated under the frozen leakage manifests. |
| [`edgeiiot_layout_audit.py`](edgeiiot_layout_audit.py) | Edge-IIoTset row-layout audit on the published DNN-EdgeIIoT-dataset.csv. |
| [`edgeiiot_mechanism.py`](edgeiiot_mechanism.py) | Is Edge-IIoTset a second mechanism, or the SAME attractor operating pairwise? |
| [`layout_detector_control.py`](layout_detector_control.py) | Does the layout-only detector isolate the Edge-IIoTset artifact? |
| [`verify_edgeiiot_repair.py`](verify_edgeiiot_repair.py) | Verify the Edge-IIoTset row repair (src/data/repair.py) against the pcaps. |
| [`loader_provenance.py`](loader_provenance.py) | Loader provenance for runs made before trainers recorded their loader kwargs. |
| [`feature_names.py`](feature_names.py) | Recover feature names for the loaders' preprocessed arrays (audit helper). |
| [`ciciomt_dupfree_mask.py`](ciciomt_dupfree_mask.py) | CICIoMT2024 duplicate-free test mask + a model-free bound on its impact. |
| [`ciciomt_dupfree_reeval.py`](ciciomt_dupfree_reeval.py) | CICIoMT2024 seed-aligned duplicate-free re-evaluation. Inference only. |
| [`tag_summary_provenance.py`](tag_summary_provenance.py) | Flag every row of the summary CSVs with its provenance (never strip rows). |
| [`make_repaired_tables.py`](make_repaired_tables.py) | Tables that the repaired re-runs make final (no latency, no energy). |

## Pre-registered Part B items

Shared helpers in `partB_common.py`; the registration is `results/preregistration/2026-09-28_journal_partB.md`.

| Script | Purpose |
|---|---|
| [`partB_common.py`](partB_common.py) | Shared helpers for the Part B analyses (results/preregistration/2026-09-28_journal_partB.md). |
| [`partB_b1_prop1.py`](partB_b1_prop1.py) | B1: empirical check of Proposition 1 and the encoder audit (pre-registered, Part B; no training). |
| [`partB_b2_entropy_tree.py`](partB_b2_entropy_tree.py) | B2: is the entropy control already a per-feature supervised tree? (pre-registered, Part B) |
| [`partB_b2arm_guard.py`](partB_b2arm_guard.py) | Guards for the B2 fourth arm (encoder_mode "entropy_tree", amendment A2), before any run. |
| [`partB_b4_feasibility.py`](partB_b4_feasibility.py) | B4 feasibility (no training), pre-registered in Part B. |
| [`partB_b4_gbdt.py`](partB_b4_gbdt.py) | B4 (amendment A3): validation-selected GBDT on CICIoMT2024's provider split. |
| [`partB_b5_data_checks.py`](partB_b5_data_checks.py) | B5: train-only vocabulary and cross-partition duplicates (pre-registered, Part B). |
| [`partB_a6_dominated.py`](partB_a6_dominated.py) | A6 (Part B amendment 1): per-class results with and without cross-partition duplicates, for the classes B5 found dominated (> 50% of test rows duplicated in training), from the stored models only (no  |
| [`partB_b6_convergence.py`](partB_b6_convergence.py) | B6: convergence evidence from the stored per-epoch logs (pre-registered, Part B). |
| [`partB_b6_stalled.py`](partB_b6_stalled.py) | B6 diagnostic reruns of the three stalled entropy runs (pre-registered, Part B). |
| [`partB_block_guard.py`](partB_block_guard.py) | Reproduction guards for the B6 CICIoT2023 block code path (amendment A1), before any block run. |
| [`partB_swap_in.py`](partB_swap_in.py) | Put the Part B results in the reported places (amendments A1, A5; the 2026-10-07 deployment decision), moving what they replace aside. Nothing is deleted. |
| [`partB_a1_followup.py`](partB_a1_followup.py) | A1 follow-up for the CICIoT2023 block: early exit, flags and exit audit from the selected checkpoints of the block's Tree-to-Spike runs (pre-registration amendment A1). |
| [`partB_a5_recompute.py`](partB_a5_recompute.py) | A5 consequence: Table VIII and the E1 references with the four Edge-IIoTset M3 cells replaced. |
| [`partB_gated_measure.py`](partB_gated_measure.py) | Gated Part B measurements: CICIoT2023 seed-42 deployment re-measurement (A1), B3 PLE-MLP latency, B8 peak resident memory. Pre-registration 2026-09-28 (D9, A1). |
| [`partB_b8_memchild.py`](partB_b8_memchild.py) | B8 child: peak resident memory of one deployed system over the latency workload. |
| [`partB_teacher_thread_scope.py`](partB_teacher_thread_scope.py) | Scope of the deployment_measure.py thread defect (found 2026-10-07 by guard Gb). |
| [`partB_fill_vanilla_sops.py`](partB_fill_vanilla_sops.py) | Apply deployment_measure.py's Vanilla SNN operation-count rule to the CICIoT2023 row. |

## Guards

Checks that a new code path reproduces stored values before it is used.

| Script | Purpose |
|---|---|
| [`guard_e1_e2.py`](guard_e1_e2.py) | Guards G1, G3, G4, G5, G6 for the E1/E2 experiments (see results/preregistration/2026-09-24_matched_info_and_entropy_edges.md, with its amendment). Exits non-zero on any failure; the launch script run |
| [`guard_sops.py`](guard_sops.py) | Guard: every new operation-count path must reproduce the published one. |
| [`guard_vanilla_ceiling.py`](guard_vanilla_ceiling.py) | Proof that the published Vanilla SNN operation counts double-count the input layer. |

## Tables, numbers and figures

| Script | Purpose |
|---|---|
| [`make_results_table.py`](make_results_table.py) | Combined per-dataset results table (classification + binary IDS + efficiency) for all 4 models, 11 datasets -> results/Journal_results/tables/table_results_full.tex (+ .md). |
| [`benchmark_4way.py`](benchmark_4way.py) | 4-way comparison artifacts for the IEEE write-up. |
| [`make_arch_table.py`](make_arch_table.py) | 4-way architecture/size/spikes table -> results/Journal_results/tables/table_architecture.{tex,md}. |
| [`make_dep_systems_table.py`](make_dep_systems_table.py) | Assemble the deployed-configuration view of the deployment measurement. |
| [`make_table1_breakdown.py`](make_table1_breakdown.py) | Per-component artifact breakdown for the two int8 systems (Table I). |
| [`teacher_per_dataset.py`](teacher_per_dataset.py) | Per-dataset teacher table: quality, footprint, latency, energy. |
| [`make_teacher_table.py`](make_teacher_table.py) | Emit results/tab_teacher_per_dataset.tex from the computed CSV. |
| [`make_label_map_table.py`](make_label_map_table.py) | Appendix table: the exact native-label -> KDD 5-class mapping for CIC-IDS2017 and CSE-CIC-IDS2018, generated from the loaders' LABEL_TO_CLASS dicts (single source of truth: src/data/cicids2017.py, src |
| [`make_plots.py`](make_plots.py) | Generate the Phase-1 SNN-sweep result figures and tables. |
| [`make_numbers.py`](make_numbers.py) | Generate results/tables/numbers.tex: every number the manuscript quotes, as a macro. |
| [`check_numbers.py`](check_numbers.py) | Check a manuscript against results/tables/numbers.tex. |
| [`make_manuscript_figures.py`](make_manuscript_figures.py) | Manuscript figures and appendix tables -> Figures/. |
| [`make_precision_memory_figures.py`](make_precision_memory_figures.py) | The precision-profile and memory-band figures as standalone vector PDFs. |
