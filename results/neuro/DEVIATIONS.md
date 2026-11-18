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

## D2. Reset rule step 3: removed, then kept as a contingency (2026-11-17, PI decisions)

The zero-reset retrain of reset rule R8 (paper P6) was dropped in amendment 2 and never queued.
Amendment 4 keeps it as a contingency: it runs only if Loihi 2 turns out not to support the
delayed subtractive reset (INRC or Loihi 2 neuron documentation).

## D3. Vanilla export: GPU and CPU numerics differ (2026-11-17)

The first Vanilla export replayed the seed-42 LeakyParallel model on the CPU and got 77.55
test macro-F1 on Edge-IIoTset against the record's 77.04. The record was evaluated on CUDA,
where nn.RNN runs through cuDNN; the CPU kernel gives different spikes near threshold. The
export now replays on the GPU with the bundle's own test loader (batch 128, cuDNN
deterministic; commit 98518528) and reproduces every record exactly. The latency encoder itself
is identical on CPU and GPU (checked in the export). Anything that replays a recurrent snnTorch
model must use the device it was evaluated on.

## D4. Guard events during A1 (2026-11-17)

- A1 stopped three times before computing any result, on interoperability problems between the
  pinned packages: `snn.utils` not imported; nir 1.0.8 passing gzip options to h5py 3.16 for
  0-d parameters; NIR's reader rejecting snnTorch 0.9.4's graph (batch dimension in the Input
  type, scalar LIF parameters). The workarounds are recorded in each A1 result
  (`export_workarounds`); none changes a parameter value.
- During the second run, another Claude session working in the same repository added the same
  one-line import to `scripts/neuro_a1_nir.py` and reverted it with `git checkout` (content equal
  to HEAD; reported by that session). The guard flagged the file on modification time, and also
  flagged two new scripts copied in under a second before the run started (the guard's
  one-second margin). No tracked content changed; the run had failed on its own error.
  The two sessions agreed that only this session edits, runs or commits track A/B.

## D5. Guard failures from this session's own commits (2026-11-17)

The guarded full-test scoring (`neuro_lava_fulltest.py`, with the early-exit extension) failed its
guard. That was not because of the run: during it, this session committed `.gitignore`,
`scripts/neuro_export_variants.py` and `scripts/neuro_frontier_runner.sh`, and the run rewrote its
own tracked output `results/neuro/a2_lava/lava_fulltest.csv`. The guard now accepts changes to tracked
files inside a run's declared outputs and lists them; the scoring is rerun under the guard after the
frontier. Rule kept: no commits while a guarded run is active.
