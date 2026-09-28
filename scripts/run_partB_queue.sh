#!/usr/bin/env bash
# Part B GPU queue (pre-registration 2026-09-28 with amendment 1). Dependencies, in order:
#   1. B2 fourth arm (20 runs, E2 protocol)       after scripts/partB_b2arm_guard.py passes
#   2. B3 selection and final runs (16 new)        after all 64 search runs exist
#   3. B6 CICIoT2023 block (60 runs)               after B1, B3, the B6 stalled reruns, A5 and A3
#                                                  have drained AND the block guard has passed
# Thread variables stay unset (the XGBoost teacher depends on them).
cd "$(dirname "$0")/.."
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS TORCH_NUM_THREADS
L=results/journal_partB/logs; Q=$L/queue.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $Q; }
guard_ok(){ ls $1 2>/dev/null | while read g; do grep -q '"ok": true' "$g" && echo yes; done | grep -q yes; }
running(){ pgrep -f "$1" > /dev/null; }

# 1. B2 fourth arm
until ! running partB_b2arm_guard.py; do sleep 60; done
if guard_ok "results/journal_partB/B2arm_guard/*/guard.json"; then
  log "B2 arm guard passed; START B2 fourth arm (20 runs, 4 workers)"
  ( for d in toniot edgeiiot ciciomt2024 ciciot2023; do for s in 42 43 44 45 46; do echo "$d $s"; done; done |
    xargs -P 4 -L 1 bash -c '.venv/bin/python -u scripts/run_encoder_variant.py "$0" entropy_tree "$1" > results/journal_partB/logs/B2arm_$0_$1.log 2>&1 || echo "FAIL B2arm $0 $1" >> results/journal_partB/logs/queue.log'
    log "B2 fourth arm pool done" ) &
else
  log "ABORT B2 fourth arm: guard did not pass (see B2arm_guard.log)"
fi

# 2. B3 selection and final runs
until [ "$(ls results/journal_partB/B3/search/*/cfg_*/results.json 2>/dev/null | wc -l)" -ge 64 ]; do sleep 120; done
until ! running "partB_b3_ple.py search"; do sleep 60; done
log "B3 search complete (64); START selection and final runs"
.venv/bin/python -u scripts/partB_b3_ple.py select > $L/B3_select.log 2>&1 || log "FAIL B3 select"
( for d in toniot edgeiiot ciciomt2024 ciciot2023; do for s in 42 43 44 45 46; do echo "$d $s"; done; done |
  xargs -P 4 -L 1 bash -c '.venv/bin/python -u scripts/partB_b3_ple.py final "$0" "$1" > results/journal_partB/logs/B3_final_$0_$1.log 2>&1 || echo "FAIL B3final $0 $1" >> results/journal_partB/logs/queue.log'
  log "B3 final runs done" ) &
sleep 30

# 3. B6 block: wait for the listed items to drain and the block guard to pass
until ! running partB_b1_prop1.py && ! running partB_b3_ple.py && ! running partB_b6_stalled.py \
      && ! running "val_curves.py" && ! running partB_b4_gbdt.py && ! running partB_block_guard.py; do sleep 120; done
if ! guard_ok "results/journal_partB/B6_block_guard/*/guard.json"; then
  log "ABORT B6 block: reproduction guard did not pass (see B6_block_guard.log)"; wait; exit 1
fi
log "queue drained and block guard passed; START B6 CICIoT2023 block (60 runs, 4 workers)"
for arm in tree entropy entropy_tree quantile nokd raw all uniform tasnn vanilla dnn_tq dnn; do
  for s in 42 43 44 45 46; do echo "$arm $s"; done
done | xargs -P 4 -L 1 bash -c '.venv/bin/python -u scripts/partB_b6_block.py "$0" "$1" > results/journal_partB/logs/B6block_$0_$1.log 2>&1 || echo "FAIL B6block $0 $1" >> results/journal_partB/logs/queue.log'
log "B6 block done"
wait
log "Part B GPU queue done"
