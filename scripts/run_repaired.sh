#!/usr/bin/env bash
# Re-run the two repaired corpora (leakage audit): 4 models x 5 seeds x 2 datasets.
# Pre-repair results were moved to results/pre_repair/, so the runners' skip-guards
# see a clean slate; no --force needed.
cd "$(dirname "$0")/.."
L=results/repaired_runs; mkdir -p "$L"
for d in cse-cic-ids2018 edgeiiot; do
  for s in 42 43 44 45 46; do
    for m in Tree-to-Spike DNN-Tree TASNN _vanilla; do
      echo "$d $s $m"
    done
  done
done | xargs -P "${WORKERS:-4}" -L 1 bash -c '
  d=$0; s=$1; m=$2
  if [ "$m" = "_vanilla" ]; then run=experiments/$d/SNN_LeakyParallel/latency/run.py; tag=vanilla
  else run=experiments/$m/$d/run.py; tag=$m; fi
  HISNN_SEED=$s .venv/bin/python -u "$run" > results/repaired_runs/${tag}_${d}_${s}.log 2>&1 \
    || echo "FAIL $tag $d $s"'
echo "repaired neural runs done"
