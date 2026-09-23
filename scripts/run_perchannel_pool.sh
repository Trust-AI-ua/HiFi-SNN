#!/usr/bin/env bash
# Per-channel int8 check: 2 models x 4 IoT datasets x 5 seeds, no training.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
{ for m in DNN-Tree Tree-to-Spike; do for d in toniot ciciomt2024 edgeiiot ciciot2023; do
    for s in 42 43 44 45 46; do
      [ -f "results/diagnostics/perchannel/${m}_${d}_seed${s}.json" ] || echo "$m $d $s"
    done; done; done; } |
xargs -P 4 -L 1 bash -c 'python -u scripts/perchannel_check.py "$0" "$1" "$2" 2>&1 | grep -v -i warn' \
  >> results/repaired_runs/perchannel.log
echo "[$(date -u +%FT%TZ)] perchannel pool done" >> results/repaired_runs/perchannel.log
