#!/usr/bin/env bash
# M3 fp32-capture reruns for Tree-to-Spike (11 datasets x 5 seeds), same design
# as run_m3_dnn.sh. cse-cic-ids2018 first: its per-epoch val curves are also the
# SNN reference for the DNN-Tree quantization diagnosis. Diagnostic only: writes
# under results/diagnostics/val_curves/, never results/Tree-to-Spike/.
cd "$(dirname "$0")/.."
L=results/diagnostics/val_curves/logs; mkdir -p "$L"
for d in cse-cic-ids2018 kddcup99 nslkdd ctu13 iscxids2012 cicids2017 unswnb15 ciciot2023 ciciomt2024 edgeiiot toniot; do
  for s in 42 43 44 45 46; do
    [ -f "results/diagnostics/val_curves/Tree-to-Spike/$d/runs/seed_$s/test_fp32_int8.json" ] && continue
    echo "$d $s"
  done
done | xargs -P "${WORKERS:-5}" -L 1 bash -c '.venv/bin/python -u scripts/val_curves.py Tree-to-Spike "$0" --seed "$1" > results/diagnostics/val_curves/logs/m3_t2s_$0_$1.log 2>&1 || echo "FAIL $0 $1"'
echo "m3 t2s grid done"
