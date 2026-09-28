#!/usr/bin/env bash
# Part B GPU queue, slot-limited (replaces run_partB_queue.sh after load reached ~110 on 16
# cores). Jobs are fed in dependency order; each waits until fewer than MAX heavy Part B
# python processes are running. Concurrency changes speed only, never a result.
#   1. B3 search (remaining cells; completed cells are skipped by the script)
#   2. B2 fourth arm (20 runs)                  after scripts/partB_b2arm_guard.py passes
#   3. B3 selection and final runs               after all 64 search runs exist
#   4. B6 CICIoT2023 block (60 runs)             after B1, B3, the B6 stalled reruns, A5 and A3
#                                                 have drained AND the block guard has passed
cd "$(dirname "$0")/.."
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS TORCH_NUM_THREADS
MAX=${MAX:-6}
L=results/journal_partB/logs; Q=$L/queue.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $Q; }
heavy(){ pgrep -fc '^\.venv/bin/python -u scripts/(partB_|run_encoder_variant|val_curves)'; }
running(){ pgrep -f "^\.venv/bin/python -u scripts/$1" > /dev/null; }
guard_ok(){ for g in $1; do [ -f "$g" ] && grep -q '"ok": true' "$g" && return 0; done; return 1; }
slot(){ while [ "$(heavy)" -ge "$MAX" ]; do sleep 30; done; }
launch(){ slot; ( eval "$1" > "$2" 2>&1 || echo "[$(date -u +%FT%TZ)] FAIL $2" >> $Q ) & sleep 20; }

log "START slot-limited queue (MAX=$MAX heavy processes)"
# 1. B3 search
for d in toniot edgeiiot ciciomt2024 ciciot2023; do for i in $(seq 0 15); do
  [ -f results/journal_partB/B3/search/$d/cfg_$(printf %02d $i)/results.json ] && continue
  launch ".venv/bin/python -u scripts/partB_b3_ple.py search $d $i" $L/B3_search_${d}_$i.log
done; done
# 2. B2 fourth arm
while running partB_b2arm_guard.py; do sleep 60; done
if guard_ok "results/journal_partB/B2arm_guard/*/guard.json"; then
  log "B2 arm guard passed; feeding B2 fourth arm (20 runs)"
  for d in toniot edgeiiot ciciomt2024 ciciot2023; do for s in 42 43 44 45 46; do
    [ -f results/encoder_ablation/$d/entropy_tree/seed_$s/results.json ] && continue
    launch ".venv/bin/python -u scripts/run_encoder_variant.py $d entropy_tree $s" $L/B2arm_${d}_$s.log
  done; done
else
  log "ABORT B2 fourth arm: guard did not pass (see B2arm_guard.log)"
fi
# 3. B3 selection and finals
until [ "$(ls results/journal_partB/B3/search/*/cfg_*/results.json 2>/dev/null | wc -l)" -ge 64 ]; do sleep 60; done
while running "partB_b3_ple.py search"; do sleep 30; done
log "B3 search complete (64); selection and final runs"
.venv/bin/python -u scripts/partB_b3_ple.py select > $L/B3_select.log 2>&1 || log "FAIL B3 select"
for d in toniot edgeiiot ciciomt2024 ciciot2023; do for s in 42 43 44 45 46; do
  launch ".venv/bin/python -u scripts/partB_b3_ple.py final $d $s" $L/B3_final_${d}_$s.log
done; done
# 4. B6 block
until ! running partB_b1_prop1.py && ! running partB_b3_ple.py && ! running partB_b6_stalled.py \
      && ! running val_curves.py && ! running partB_b4_gbdt.py && ! running partB_block_guard.py \
      && ! pgrep -f "while pgrep" > /dev/null; do sleep 60; done
if ! guard_ok "results/journal_partB/B6_block_guard/*/guard.json"; then
  log "ABORT B6 block: reproduction guard did not pass (see B6_block_guard.log)"; wait; exit 1
fi
log "listed queue drained and block guard passed; feeding B6 CICIoT2023 block (60 runs)"
for arm in tree entropy entropy_tree quantile nokd raw all uniform tasnn vanilla dnn_tq dnn; do
  for s in 42 43 44 45 46; do
    [ -f results/journal_partB/B6/block/$arm/ciciot2023/seed_$s/results.json ] && continue
    launch ".venv/bin/python -u scripts/partB_b6_block.py $arm $s" $L/B6block_${arm}_$s.log
  done
done
wait
log "Part B GPU queue done"
