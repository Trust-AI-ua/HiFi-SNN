#!/usr/bin/env bash
# Resume Part B after the 2026-09-30 shutdown (see results/journal_partB/RESUME.md).
# The machine came back on a new kernel (6.8.0-136 -> 6.8.0-142; driver, CUDA, cuDNN, torch,
# xgboost unchanged). The block reproduction guard is rerun first; the block starts only if
# this new guard run passes (exit code 0), so the 54 remaining runs rest on a guard made on
# the stack that trains them.
cd "$(dirname "$0")/.."
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS TORCH_NUM_THREADS
L=results/journal_partB/logs; Q=$L/queue.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $Q; }
log "RESUME after reboot (kernel $(uname -r)): rerunning the block reproduction guard before the block"
if .venv/bin/python -u scripts/partB_block_guard.py > $L/B6_block_guard_postreboot.log 2>&1; then
  log "post-reboot block guard PASSED; starting the block"
  exec bash scripts/run_partB_block.sh
else
  log "ABORT: post-reboot block guard FAILED (see B6_block_guard_postreboot.log); block not started"
  exit 1
fi
