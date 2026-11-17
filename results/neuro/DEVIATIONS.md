# Track A and B: process deviations (internal; not for the paper)

## D1. Lava model discovery executed repository scripts (2026-11-17)

**What happened.** Lava's compiler finds a custom process's models by importing every `.py`
file in the directory of the process's module. The first version of the subtractive-reset
process lived in `scripts/`. Two toy runs in `.venv-neuro` therefore imported the scripts in
`scripts/`; seven of them ran their work at import, having no `if __name__ == "__main__":`
guard (benchmark_4way, edgeiiot_layout_audit, fix_gbdt_dedup, layerwise_quant,
make_teacher_table, partB_teacher_thread_scope, and the import hung in
edgeiiot_layout_audit's CSV read).

**Effect.** 24 tracked files were rewritten (23 figure PDFs in results/Journal_results/figures
and results/deployment/gbdt_compression.csv, deduplicated by fix_gbdt_dedup) and 7 ignored
tables (results/Journal_results/tables/table_*.tex, results/tables/tab_teacher_per_dataset.tex).
No file was deleted or moved (none of the scripts that ran deletes or moves files).

**Repair and checks.** The 24 tracked files were restored from git (`git checkout`); the tree
had no tracked modifications at the start of the session. The 7 ignored tables: their tracked
`.md` twins were byte-identical after the accidental run, and the teacher table regenerated in
`.venv` is byte-identical to the accidental one, so their content is unchanged.

**Prevention.**
1. The Lava process lives alone in `src/lava_sublif/` (only an empty `__init__.py` beside it).
2. Every script in `scripts/` that ran work at import now has its body under
   `if __name__ == "__main__":` (the seven above, including my own neuro_a3_install_check;
   wrapped with the module's AST checked identical apart from the new `if`). Import-time
   side effects in three guarded scripts were moved under their guards (latency_elp's
   torch.set_num_threads(1), gbdt_compression's and encoder_ablation's mkdir). Checked:
   importing each of the ten has no effect on `git status --porcelain --ignored` or on any file;
   make_teacher_table and benchmark_4way reproduce their outputs (tables byte-identical; PDFs
   identical apart from /CreationDate). Three library modules (partB_common, statistics_utils,
   run_gbdt_leakage) run nothing at import and are left as they are.
3. Every neuro-environment run goes through `scripts/neuro_guard.py`: it refuses to start with
   tracked modifications and fails the run if tracked files change, if new paths appear outside
   the run's declared outputs, or if any file outside them changes (modification time, which
   also covers ignored files). A literally clean `git status --porcelain` is not possible on
   this machine (untracked files of the user and of the concurrent training queue), so new
   untracked paths are checked against the run's outputs instead.

## D2. Reset rule step 3 removed (2026-11-17, PI decision)

The zero-reset retrain of reset rule R8 (paper P6) is dropped and was never queued. See
amendment 2 of the pre-registration.
