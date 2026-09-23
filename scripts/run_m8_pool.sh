#!/usr/bin/env bash
# M8 control arm: 4 IoT benchmarks x seeds 42-46 = 20 cells.
# Single-threaded workers (these students cannot use torch's default pool, and
# one thread each keeps the per-worker cost near one core). src/joblock.py makes
# a second pool safe on the same list. Cheapest dataset first, so the first
# complete dataset arrives soonest.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1
mkdir -p results/repaired_runs
echo "[$(date -u +%FT%TZ)] m8 pool start, workers=${WORKERS:-5}"
{
  for d in edgeiiot toniot ciciomt2024 ciciot2023; do
    for s in 42 43 44 45 46; do echo "$d $s"; done
  done
} | xargs -P "${WORKERS:-5}" -L 1 bash -c '
  .venv/bin/python -u scripts/run_m8_arm.py "$0" "$1" \
    > results/repaired_runs/m8_$0_$1.log 2>&1 || echo "FAIL m8 $0 $1"'
echo "[$(date -u +%FT%TZ)] m8 pool done"
