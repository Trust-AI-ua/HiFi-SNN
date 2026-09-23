# Superseded outputs

Kept for provenance. Do not cite any number in this directory.

| File | Why superseded | Current source |
|---|---|---|
| `tables_deployment_audit.tex` | Pre-repair deployment audit: pooled accuracy (79.86 / 79.80 / 79.68), fp32 artifact sizes (53.4 KB), native-JSON teacher sizes (5.04 MB), teacher macro-F1 82.13, and the old memory band 0.94 to 2.06 | `results/tables/tab_dep_systems.tex`, `results/tables/tab_teacher_per_dataset.tex`, `results/deployment/energy_crossover_iot.csv` |
| `encoder_ablation.tex` | Unsplit pre-repair ablation table (2026-07-15) | `results/Journal_results/tables/encoder_ablation_{full,general,iot}.tex` |
| `encoder_ablation_observation.tex` | Hand-written observation quoting pre-repair ablation values (e.g. CSE-CIC-IDS2018 75.9) | the three ablation tables above, and `results/tables/numbers.tex` |
| `student_vs_teacher_observation.tex` | Pre-repair student-versus-teacher observation | `results/tables/tab_teacher_vs_student.{csv,md}` (five seeds, paired, IoT) |
| `limitations_footprint.tex` | Footprint limitation written against fp32 artifact sizes (2026-07-28) | `results/tables/tab_dep_systems.tex`, `results/tables/tab_artifact_breakdown.md` |
| `table_architecture.tex` | Unsplit pre-repair anatomy table (2026-07-28) | `results/Journal_results/tables/table_architecture_{full,general,iot}.tex` |
| `energy_memory_model.md` | Output of `scripts/energy_memory_model.py`, whose prose conclusions are hard-coded and predate the leakage repair, the input-layer accounting and the Vanilla SNN double-count correction | `scripts/deployment_energy2.py --iot` |

Moved here 2026-09-23 at the submission-number freeze (tag `tnnls-submission`). The five
`.tex` files above had never been committed: a blanket `*.tex` ignore rule hid them.
