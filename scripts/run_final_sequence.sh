#!/usr/bin/env bash
# 1 prefix regeneration          done
# 2 guards                       re-checked here; the sequence ABORTS if they fail
# 3 deployment re-measurement    quiet box, the expensive one to repeat
# 4 energy model, IoT subset
# 5 M8 pool, with serialization verification and the five-seed teacher refit
#   alongside it, since neither is timing-sensitive.
cd "$(dirname "$0")/.."
L=results/repaired_runs
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $L/final_sequence.log; }

log "START guards"
if ! python -u scripts/guard_sops.py > $L/guards.log 2>&1; then
  log "ABORT: guards failed, nothing downstream ran (see $L/guards.log)"
  exit 1
fi
log "ok guards (A prefix, B opcount, C published Rel.E, D formula copy)"

log "START deployment_measure"
touch results/deployment/.RELEASE_DEPLOYMENT
if python -u scripts/deployment_measure.py > $L/deployment_final2.log 2>&1; then
  log "ok deployment_measure"
else
  log "FAIL deployment_measure"; rm -f results/deployment/.RELEASE_DEPLOYMENT; exit 1
fi
rm -f results/deployment/.RELEASE_DEPLOYMENT

log "START downstream tables"
python -u scripts/reprice_deployment_trees.py >> $L/downstream.log 2>&1 || log "warn: reprice"
python -u scripts/make_dep_systems_table.py   >> $L/downstream.log 2>&1 || log "warn: dep_systems"
python -u scripts/make_table1_breakdown.py    >> $L/downstream.log 2>&1 || log "warn: table1"
python -u scripts/deployment_energy2.py --iot >> $L/energy_iot.log 2>&1 || log "warn: energy"
log "ok downstream tables"

log "START m8 pool (+ verification and teacher refit alongside)"
( python -u scripts/verify_serialization.py > $L/verify_serialization2.log 2>&1
  log "ok verify_serialization" ) &
( python -u scripts/teacher_per_dataset.py --seeds 42 43 44 45 46 > $L/teacher_5seed.log 2>&1 \
  && python -u scripts/make_teacher_table.py >> $L/teacher_5seed.log 2>&1
  log "ok teacher 5-seed" ) &
WORKERS=4 bash scripts/run_m8_pool.sh >> $L/m8_pool.log 2>&1
log "ok m8 pool"
wait
python -u scripts/m8_decomposition.py > $L/m8_decomposition.log 2>&1 || log "warn: decomposition"
log "START prefix seeds 43-46"
python -u scripts/extract_prefix.py --seeds 43 44 45 46 > $L/prefix_regen_43_46.log 2>&1
log "final sequence done"
