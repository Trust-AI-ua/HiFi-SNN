# Review evidence package (amendment 8)

Pre-registration: `results/preregistration/2026-11-17_neuro_stream.md`, amendment 8 and its dated correction.
Everything is simulation (Lava fixed point, SANA-FE); there are no physical-chip measurements.

| Item | Where | Produced by |
|---|---|---|
| 1. Paired execution-path predictions, Edge-IIoTset, seeds 42-46 | `p1_stagewise/`: `STAGEWISE.md`, `stage_f1.csv`, `per_class_f1.csv`, `predictions_{val,test}_seed<S>.csv.gz` (every flow: flow_id, true class, the predicted class at every stage), `edgeiiot_seed<S>_<split>.npz` (the same plus per-class scores: output spike counts for the Vanilla SNN, logits or summed membrane for the student) | `scripts/neuro_stagewise.py`, `scripts/neuro_review_stagewise_summary.py` |
| 2. Simulator and exit records, four benchmarks | `p2_simulator/`: `per_flow_records.csv.gz`, `class_counts.csv`, `energy_prevalence.csv`, `energy_by_class.csv`, `ENERGY.md` (gate location and charged costs at the end) | `scripts/neuro_review_energy.py` |
| 3. Encoder tables and multichannel collisions | `p3_encoders/`: one JSON per encoder (thresholds, importance, channels, offsets, level-to-spike-time tables, rule), `collisions.csv`, `COLLISIONS.md` | `scripts/neuro_encoder_tables.py`, `scripts/neuro_review_collisions.py` |
| 4. Implementation details | `p4_implementation/`: `IMPLEMENTATION.md` (equations with file references), `deployment_params.csv` (per seed and layer), `trace_edgeiiot_seed42.csv` (one flow, every step and layer) | `scripts/neuro_review_impl.py` |
| 5. Selection and summary records | `p5_selection/`: `gate_search_c1.csv` (every validation candidate), `selected_policy.csv` (validation and test, FAR separately), `gbdt_budget_candidates.csv`, `far_operating_points.csv` (validation and test FAR separately) | `scripts/neuro_review_gate_search.py`, `scripts/neuro_review_records.py` |
| Validation-tuned fixed-point Vanilla SNN | `a8_tuned_vanilla/`: `selection.json` (frozen before test), `TUNED.md`, `tuned_vanilla.csv`, `tuned_per_class_f1.csv`, per-seed JSON, per-flow CSV and full-test predictions | `scripts/neuro_a8_select.py`, `scripts/neuro_a8_tuned.py` (guarded), `scripts/neuro_a8_runner.sh`, `scripts/neuro_review_tuned_summary.py` |

Existing records also relevant to item 5, unchanged:
- `results/neuro/confirm/confirm_a7.csv`, `confirm.csv`, `confirm_five_seed.csv`;
- `results/journal_r2/integer_per_seed.csv`, `results/tables/m3_quant_per_seed.csv`, `results/deployment/serialization_verification.csv` (byte counts);
- `results/journal_r2/exit_calibration/` (`thresholds.json`, `per_seed.csv`), `results/journal_r2/gbdt_valsel_per_seed.csv`, `results/journal_r2/far_table.csv`;
- `scripts/make_numbers.py` writes `results/tables/numbers.tex`, the source of every typed number.

Notes:
- **Flow IDs.** `flow_id` is the row index in that seed's test split. The split is drawn per seed, so the same index is a different flow on another seed. Within a seed, Tree-to-Spike and Vanilla share the test split and the 2,000-flow sample (checked), so flows pair.
- **Large files.** The `.npz` and `.csv.gz` prediction files are not committed; they are on disk here.
- **Not done.** No capture- or device-disjoint split was run.
