#!/usr/bin/env bash
# B6 CICIoT2023 block (amendment A1), slot-limited. Replaces step 4 of run_partB_queue2.sh,
# whose drain check `pgrep -f "while pgrep"` matched other users' processes on this shared
# machine and blocked the block after its conditions were met (2026-09-29 00:47Z to
# 2026-09-30). Every process check here is restricted to this user (pgrep -u).
cd "$(dirname "$0")/.."
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS TORCH_NUM_THREADS
MAX=${MAX:-6}
ME=$(id -un)
L=results/journal_partB/logs; Q=$L/queue.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $Q; }
heavy(){ pgrep -u "$ME" -fc '^\.venv/bin/python -u scripts/(partB_|run_encoder_variant|val_curves)'; }
running(){ pgrep -u "$ME" -f "^\.venv/bin/python -u scripts/$1" > /dev/null; }
guard_ok(){ for g in results/journal_partB/B6_block_guard/*/guard.json; do grep -q '"ok": true' "$g" && return 0; done; return 1; }
slot(){ while [ "$(heavy)" -ge "$MAX" ]; do sleep 30; done; }
launch(){ slot; ( eval "$1" > "$2" 2>&1 || echo "[$(date -u +%FT%TZ)] FAIL $2" >> $Q ) & sleep 20; }

for p in partB_b1_prop1.py partB_b3_ple.py partB_b6_stalled.py val_curves.py partB_b4_gbdt.py partB_block_guard.py run_encoder_variant.py; do
  if running $p; then log "ABORT B6 block: $p still running"; exit 1; fi
done
guard_ok || { log "ABORT B6 block: no passing reproduction guard"; exit 1; }
log "START B6 CICIoT2023 block scheduler (60 runs, MAX=$MAX; finished runs and runs held by a live worker are skipped)"
for arm in tree entropy entropy_tree quantile nokd raw all uniform tasnn vanilla dnn_tq dnn; do
  for s in 42 43 44 45 46; do
    [ -f results/journal_partB/B6/block/$arm/ciciot2023/seed_$s/results.json ] && continue
    lk=results/journal_partB/B6/block/.locks/${arm}_$s.lock      # held by a live worker: skip, and
    [ -f "$lk" ] && kill -0 "$(cat "$lk")" 2>/dev/null && continue   # never reopen its log
    launch ".venv/bin/python -u scripts/partB_b6_block.py $arm $s" $L/B6block_${arm}_$s.log
  done
done
wait
log "B6 block done"
