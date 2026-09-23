#!/usr/bin/env bash
# Supplemental single-threaded workers alongside the pools already running.
# The box has 16 cores and the existing 7 jobs hold ~14.3 of them, at ~2.6 cores
# each: torch sizes its thread pool to the core count, but these small networks
# cannot use it. Workers started here pin torch/BLAS to one thread, so each adds
# ~1 core instead of ~2.6. src/joblock.py stops the two pools running the same
# cell. Cheapest-first (Edge-IIoTset), so breadth arrives soonest.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1
L=results/repaired_runs
{
  for m in DNN-Tree Tree-to-Spike; do
    for s in 46 45 44 43 42; do echo "M3 $m edgeiiot $s"; done
  done
  for v in all uniform nokd quantile; do
    for s in 46 45 44 43 42; do echo "VAR edgeiiot $v $s"; done
  done
} | xargs -P "${WORKERS:-3}" -L 1 bash -c '
  if [ "$0" = "M3" ]; then
    .venv/bin/python -u scripts/val_curves.py "$1" "$2" --seed "$3" \
      > results/repaired_runs/x_m3_$1_$2_$3.log 2>&1 || echo "FAIL x_m3 $1 $2 $3"
  else
    .venv/bin/python -u scripts/run_encoder_variant.py "$1" "$2" "$3" \
      > results/repaired_runs/x_encvar_$1_$2_$3.log 2>&1 || echo "FAIL x_encvar $1 $2 $3"
  fi'
echo "[$(date -u +%FT%TZ)] extra pool done"
