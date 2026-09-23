#!/usr/bin/env bash
# Second supplemental pool for the remaining CSE-CIC-IDS2018 encoder variants.
# The running pool uses 3 workers at ~4.9 cores each (92% of a 16-core box), but a
# cell costs ~32-37 core-hours regardless of thread count: 2.1 cores -> 15.4 h,
# 4.9 cores -> 7.5 h. More jobs at fewer threads therefore finishes more cells per
# hour. Workers here take 2 threads each; src/joblock.py stops both pools from
# claiming the same cell, and finished cells are skipped.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for v in all uniform nokd; do
  for s in 46 45 44 43 42; do echo "$v $s"; done
done | xargs -P "${WORKERS:-5}" -L 1 bash -c \
  '.venv/bin/python -u scripts/run_encoder_variant.py cse-cic-ids2018 "$0" "$1" \
     > results/repaired_runs/x2_encvar_$0_$1.log 2>&1 || echo "FAIL x2 $0 $1"'
echo "[$(date -u +%FT%TZ)] extra pool 2 done"
