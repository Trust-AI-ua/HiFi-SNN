#!/usr/bin/env bash
# Replaces run_post_repair_queue.sh, whose encoder-ablation stage ran its 40
# cells sequentially in one process (~53 GPU-h serial) and which would have
# timed the deployment measurement while the GPU and CPU were loaded.
#   A (parallel, now)   encoder variants, 3 workers
#   B (serial, now)     GBDT compression, early-exit chain, fc1 ratio -- no
#                       wall-clock measurements, so contention does not matter
#   C (quiet, last)     after A, B and the M3 capture (run_m3_repaired.sh):
#                       ablation aggregation, then deployment measurement and the
#                       teacher tables that read it, on an otherwise idle machine
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/repaired_runs; mkdir -p "$L"
ts () { date -u +%FT%TZ; }
stage () { local n=$1; shift; echo "[$(ts)] START $n"; "$@" > $L/queue_$n.log 2>&1 \
           && echo "[$(ts)] ok $n" || echo "[$(ts)] FAIL $n"; }

# ---- A: encoder variants in a pool
(
  for d in cse-cic-ids2018 edgeiiot; do
    for v in quantile nokd uniform all; do
      for s in 42 43 44 45 46; do echo "$d $v $s"; done
    done
  done | xargs -P "${WORKERS:-3}" -L 1 bash -c \
    '.venv/bin/python -u scripts/run_encoder_variant.py "$0" "$1" "$2" > results/repaired_runs/encvar_$0_$1_$2.log 2>&1 || echo "FAIL encvar $0 $1 $2"'
  echo "[$(ts)] encoder variants done"
) &
POOL=$!

# ---- B: contention-insensitive stages
stage gbdt_compression $PY -u scripts/gbdt_compression.py --datasets cse-cic-ids2018 edgeiiot
stage early_exit_sweep $PY -u scripts/early_exit_sweep.py
stage early_exit_audit $PY -u scripts/early_exit_audit.py --datasets ciciot2023 edgeiiot
stage evidence_arrival $PY -u scripts/evidence_arrival.py
stage exit_margin      $PY -u scripts/exit_margin_test.py
stage fc1_ratio        $PY -u scripts/fc1_ratio.py

# ---- C: wait for everything else, then measure on a quiet machine
wait $POOL
while pgrep -f "run_m3_repaired.sh" > /dev/null; do sleep 120; done
stage encoder_ablation_plot $PY -u scripts/encoder_ablation.py --plot
echo "[$(ts)] load before timing: $(cat /proc/loadavg)"
stage deployment_measure $PY -u scripts/deployment_measure.py
echo "[$(ts)] load after timing:  $(cat /proc/loadavg)"
stage teacher_table $PY -u scripts/teacher_per_dataset.py
stage teacher_tex   $PY -u scripts/make_teacher_table.py
echo "[$(ts)] post-repair queue done"
