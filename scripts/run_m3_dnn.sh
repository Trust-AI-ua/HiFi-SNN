#!/usr/bin/env bash
# M3 fp32-capture reruns for DNN-Tree (all 11 datasets x 5 seeds). Diagnostic
# only: writes under results/diagnostics/val_curves/, never results/DNN-Tree/.
# Skip-guard: a cell with runs/seed_<S>/test_fp32_int8.json is not rerun.
cd "$(dirname "$0")/.."
L=results/diagnostics/val_curves/logs; mkdir -p "$L"
for d in kddcup99 nslkdd ctu13 iscxids2012 cicids2017 unswnb15 ciciot2023 ciciomt2024 edgeiiot toniot cse-cic-ids2018; do
  for s in 42 43 44 45 46; do
    [ -f "results/diagnostics/val_curves/DNN-Tree/$d/runs/seed_$s/test_fp32_int8.json" ] && continue
    echo "$d $s"
  done
done | xargs -P 4 -L 1 bash -c '.venv/bin/python -u scripts/val_curves.py DNN-Tree "$0" --seed "$1" > results/diagnostics/val_curves/logs/m3_dnn_$0_$1.log 2>&1 || echo "FAIL $0 $1"'
echo "m3 dnn grid done"
