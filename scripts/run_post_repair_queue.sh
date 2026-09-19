#!/usr/bin/env bash
# Everything downstream of the leakage repair, queued behind the headline re-runs.
# Each stage logs to results/repaired_runs/queue_<stage>.log and a failure is
# recorded but does not abort the queue.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/repaired_runs; mkdir -p "$L"
stage () { echo "[$(date -u +%FT%TZ)] START $1"; shift; "$@" && echo "[$(date -u +%FT%TZ)] ok" || echo "[$(date -u +%FT%TZ)] FAIL $*"; }

# wait for the 4-model x 5-seed x 2-dataset re-runs
while pgrep -f "run_repaired.sh" > /dev/null; do sleep 120; done
echo "[$(date -u +%FT%TZ)] headline re-runs finished"

# 1. encoder variants (quantile / nokd / uniform / all) on the two repaired datasets
mkdir -p results/pre_repair/encoder_ablation
for d in cse-cic-ids2018 edgeiiot; do
  for v in quantile nokd uniform all; do
    [ -d "results/encoder_ablation/$d/$v" ] && mkdir -p "results/pre_repair/encoder_ablation/$d" \
      && mv "results/encoder_ablation/$d/$v" "results/pre_repair/encoder_ablation/$d/"
  done
done
stage encoder_ablation $PY -u scripts/encoder_ablation.py > $L/queue_encoder_ablation.log 2>&1

# 2. GBDT teacher compression sweep, the two repaired datasets only (append-merged)
stage gbdt_compression $PY -u scripts/gbdt_compression.py --datasets cse-cic-ids2018 edgeiiot \
  > $L/queue_gbdt_compression.log 2>&1

# 3. deployment measurement (all 11: the script rewrites the CSV wholesale)
stage deployment_measure $PY -u scripts/deployment_measure.py > $L/queue_deployment_measure.log 2>&1

# 4. per-dataset teacher table (reads deployment_measurements.csv, so it follows)
stage teacher_table $PY -u scripts/teacher_per_dataset.py > $L/queue_teacher.log 2>&1
stage teacher_tex   $PY -u scripts/make_teacher_table.py  > $L/queue_teacher_tex.log 2>&1

# 5. early exit: threshold selection (all 11, wholesale rewrite), then the audits
stage early_exit_sweep $PY -u scripts/early_exit_sweep.py > $L/queue_exit_sweep.log 2>&1
stage early_exit_audit $PY -u scripts/early_exit_audit.py --datasets ciciot2023 edgeiiot \
  > $L/queue_exit_audit.log 2>&1
stage evidence_arrival $PY -u scripts/evidence_arrival.py > $L/queue_evidence.log 2>&1
stage exit_margin      $PY -u scripts/exit_margin_test.py > $L/queue_margin.log 2>&1

# 6. weight-scale statistics (M6 closure)
stage fc1_ratio $PY -u scripts/fc1_ratio.py > $L/queue_fc1_ratio.log 2>&1

echo "[$(date -u +%FT%TZ)] post-repair queue done"
