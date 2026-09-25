#!/usr/bin/env bash
# Once the E1 pool's last cell lands: apply the registered E1 rules early (E1's
# analysis does not depend on E2), then add one E2 worker alongside the running
# 4-worker pool. The extra worker walks the E2 cell list from the END, so it takes
# the cells the main pool would start last; src/joblock.py and the results.json skip
# stop both pools from running the same cell. Own log names (e2x_*), so the main
# pool's later [skip]/[busy] logs do not overwrite them.
cd "$(dirname "$0")/.."
L=results/repaired_runs/e1_e2.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $L; }
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS TORCH_NUM_THREADS   # teacher reproducibility
until [ -f results/DNN-TQ/ciciot2023/seed_46/results.json ]; do sleep 30; done
log "E1 cells 20/20; START early E1 analysis"
.venv/bin/python -u scripts/analyze_e1_e2.py > results/repaired_runs/analyze_e1_early.log 2>&1 \
  || log "early E1 analysis refused (see analyze_e1_early.log)"
log "early E1 analysis done; START extra E2 worker (1, reverse order)"
for d in ciciot2023 ciciomt2024 edgeiiot toniot; do for s in 46 45 44 43 42; do echo "$d $s"; done; done |
  xargs -P 1 -L 1 bash -c '.venv/bin/python -u scripts/run_encoder_variant.py "$0" entropy "$1" > results/repaired_runs/e2x_$0_$1.log 2>&1 || echo "FAIL e2x $0 $1"' >> $L
log "extra E2 worker done"
