#!/usr/bin/env bash
# Deployment measurement (gated on an idle machine), then the two analyses that
# read it. M8 starts only after this finishes: training alongside it would
# reproduce the load contamination of the first attempt.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/repaired_runs
ts () { date -u +%FT%TZ; }
stage () { local n=$1; shift; echo "[$(ts)] START $n"; "$@" > $L/final_$n.log 2>&1 \
           && echo "[$(ts)] ok $n" || echo "[$(ts)] FAIL $n"; }
stage deployment_measure $PY -u scripts/deployment_measure.py
stage teacher_table      $PY -u scripts/teacher_per_dataset.py
stage teacher_tex        $PY -u scripts/make_teacher_table.py
stage envelope_recompute $PY -u scripts/envelope_recompute.py
echo "[$(ts)] deployment sequence done"
