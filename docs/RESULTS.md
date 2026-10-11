# Results map

Everything under `results/` is committed output. Each training run writes one
`seed_<S>/results.json` record (configuration, split signature, training curve, test metrics,
spike and operation counts, early exit, teacher score); every other file is derived from those
records by a script in `scripts/`. The values the manuscript quotes are listed, each with the
file it was read from, in [`results/tables/numbers_provenance.json`](../results/tables/numbers_provenance.json).

Model weights (`*.pt`) and per-timestep early-exit logits (`results/prefix_outputs/`, about
13 GB) are not committed; the scripts in [`REPRODUCING.md`](REPRODUCING.md) regenerate them.

## Run records

| Path | What | Paper |
|---|---|---|
| `Tree-to-Spike/<dataset>/seed_<S>/` | Tree-to-Spike, 11 datasets x 5 seeds | main comparison |
| `DNN-Tree/<dataset>/seed_<S>/` | dense twin | main comparison |
| `<dataset>/SNN_LeakyParallel/latency/seed_<S>/` | Vanilla SNN | main comparison |
| `TASNN/<dataset>/seed_<S>/` | TASNN | main comparison |
| `encoder_ablation/<dataset>/<arm>/seed_<S>/` | encoder arms: `quantile`, `uniform`, `all`, `nokd`, `m8_raw`, `entropy` (E2), `entropy_tree` (B2) | encoder ablation, M8, E2, B2 |
| `DNN-TQ/<dataset>/seed_<S>/` | dense twin on spike-time input | E1 |
| `ladder/<dataset>/<arm>/seed_<S>/` | sensitivity ladder, 10-epoch runs; `T10`, `T40`, `T80` are the window sweep | window length T |
| `fixed_budget_ciciot2023/` | the 10-epoch CICIoT2023 runs that the 20-epoch block replaced (all twelve arms, with their per-channel files) | appendix, A1 |
| `summary*.csv` | one row per run as the trainers appended it; the CICIoT2023 rows are the 10-epoch runs (the block runs are only in their records) | |

CICIoT2023 records in the reported places are the 20-epoch, validation-selected block
(amendment A1); `journal_partB/swap_manifest.json` lists every file that was moved or copied.

## Derived tables

| Path | What |
|---|---|
| `tables/numbers_provenance.json` | every reported value, with source, aggregation and meaning |
| `diagnostics/per_seed_all_cells.csv` | macro-F1, accuracy, balanced accuracy and MCC per run |
| `Journal_results/tables/` | results, efficiency, architecture and ablation tables (Markdown) |
| `Journal_results/figures/` | overview figures for all eleven datasets (bars, heatmaps, mean ranks, critical-difference diagrams, energy) |
| `tables/tab_perchannel.csv`, `tables/tab_m3_quant.csv` | macro-F1 at fp32, per-tensor and per-channel int8 |
| `tables/tab_m8_decomposition*.csv` | step-by-step decomposition from Vanilla SNN to Tree-to-Spike |
| `tables/tab_iot_arithmetic.csv` | arithmetic energy per sample, IoT four |
| `tables/tab_teacher_per_dataset.csv`, `tables/tab_teacher_vs_student.csv` | GBDT teacher and student-minus-teacher |
| `tables/tab_dep_systems.csv`, `tables/tab_artifact_breakdown.csv` | deployed systems and artifact composition |
| `tables/tab_m5_edges.csv` | thresholds per feature and the window bound |
| `tables/tab_exit_audit_iot.csv`, `tables/tab_m4_flags.csv`, `tables/m4_accuracy_vs_ops.csv` | early-exit audit, uncertainty flags, accuracy versus operations |
| `early_exit/` | validation-selected exit threshold per run and its test effect |
| `deployment/deployment_measurements.csv` | seed-42 deployment table: artifact bytes, CPU batch-1 latency, operation counts (11 datasets) |
| `deployment/energy_crossover_iot.csv` | total energy with memory across SRAM prices |
| `deployment/compact_gbdt.csv`, `deployment/gbdt_compression*.csv` | compact teacher and compression sweep |
| `diagnostics/` | intermediate analyses (validation curves, per-channel files, leakage audit, guards) |
| `evidence_arrival.json`, `edge_stats_all.json` | evidence-arrival statistics, per-feature threshold counts |

## Pre-registered items

| Path | Item |
|---|---|
| `preregistration/` | the registrations, written before the runs (E1/E2 2026-09-24; Part B 2026-09-28, amendment 1) |
| `journal_partB/B1/` | Proposition 1 on every test flow; encoder audit |
| `journal_partB/B2/`, `journal_partB/B2arm_guard/` | entropy cutter vs per-feature entropy tree; guard of the fourth arm |
| `journal_partB/B3/` | PLE-MLP search, selection and final runs |
| `journal_partB/B4/`, `journal_partB/B4_gbdt/` | non-random split feasibility; validation-selected GBDT on CICIoMT2024 |
| `journal_partB/B5/` | unseen categories and cross-partition duplicates |
| `journal_partB/A6/` | dominated classes with and without duplicates |
| `journal_partB/B6/` | convergence trigger, the CICIoT2023 block, stalled-run diagnosis, follow-up |
| `journal_partB/B8/` | gated deployment re-measurement, PLE-MLP latency, peak memory |
| `journal_partB/A5_m3_edge/` | Edge-IIoTset M3 reruns in the default environment |
| `journal_partB/thread_scope/` | which teachers a thread cap changes |

## Audit trail (`archive/`)

| Path | What |
|---|---|
| `archive/pre_repair/` | runs on Edge-IIoTset and CSE-CIC-IDS2018 before the leakage repair |
| `archive/leakage/` | the leakage-audit runs (host-disjoint and flow-level protocols) |
| `archive/superseded/` | originals that later corrections replaced: thread-capped M3 cells (A5), the capped-teacher deployment table, the pre-repair Edge-IIoTset ladder, earlier deployment outputs, a September snapshot of the summary tables |
| `archive/repaired_runs/` | logs and driver output of the post-repair reruns |
| `archive/logs/` | run logs of the reported experiments |
| `archive/journal_partB_interim/`, `archive/notes/` | dated interim notes; their values are superseded by `numbers_provenance.json` |
| `archive/final_provenance.json` | the August provenance manifest, replaced by `numbers_provenance.json` |
