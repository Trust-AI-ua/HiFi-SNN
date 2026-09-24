#!/usr/bin/env bash
# After the Vanilla SNN CSE-CIC-IDS2018 rerun at 0.5: backfill its operation counts,
# GUARD that all four models now share one subsample per seed, then regenerate every
# output that reads that row. Stops at the first failure.
set -e
cd "$(dirname "$0")/.."
L=results/repaired_runs/vanilla_cse_downstream.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $L; }
until grep -q "all five done" results/repaired_runs/vanilla_cse_half.log; do sleep 60; done
n=$(ls results/cse-cic-ids2018/SNN_LeakyParallel/latency/seed_*/results.json 2>/dev/null | wc -l)
[ "$n" = 5 ] || { log "ABORT: $n of 5 runs produced results.json"; exit 1; }
log "START backfill"
.venv/bin/python -u scripts/backfill_snn_metrics.py --only cse-cic-ids2018 --neuron LeakyParallel --encoding latency >> $L 2>&1
log "START guard: shared subsample"
.venv/bin/python -u scripts/loader_provenance.py > results/repaired_runs/loader_provenance_after.log 2>&1
grep -q "cross-model disagreements: 0" results/repaired_runs/loader_provenance_after.log \
  || { log "ABORT: provenance disagreements remain (see loader_provenance_after.log)"; exit 1; }
log "ok guard: all four models share one subsample on every dataset and seed"
log "START regeneration"
.venv/bin/python -u scripts/per_seed_cells.py >> $L 2>&1
.venv/bin/python -u scripts/make_results_table.py >> $L 2>&1
.venv/bin/python -u scripts/benchmark_4way.py >> $L 2>&1
.venv/bin/python -u scripts/make_arch_table.py >> $L 2>&1
.venv/bin/python -u scripts/m9_stats.py >> $L 2>&1
.venv/bin/python -u scripts/eval_m3_fp32_models.py --models vanilla --datasets cse-cic-ids2018 --force >> $L 2>&1
.venv/bin/python -u scripts/make_m3_table.py >> $L 2>&1
.venv/bin/python -u scripts/make_numbers.py >> $L 2>&1
log "downstream done"
