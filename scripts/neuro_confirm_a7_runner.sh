#!/bin/bash
# Amendment 7: five-seed chip simulation on TON-IoT and CIC-IoT2023. Exports (.venv; GPU replay for Vanilla)
# for seeds 43-46, all finished before any guarded run starts; then the guarded confirmation runs (seed 42 is a
# rerun on the existing inputs) and the summary. Status: results/neuro/logs/confirm_a7.done
cd "$(dirname "$0")/.."
L=results/neuro/logs
D=$L/confirm_a7.done
: > $D
P="nice -n 10 env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python"
t10() {     # TON-IoT T = 10 export: uncapped teacher first, then the ladder's thread caps; the teacher guard decides
  s=$1; log=$L/confirm_export_toniot_$s.log
  $P scripts/neuro_export_variants.py --seed $s toniot T10=results/ladder/toniot/T10 >> $log 2>&1 && { echo "toniot $s T10 teacher uncapped" >> $D; return 0; }
  for n in 2 4 8; do
    nice -n 10 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$n .venv/bin/python scripts/neuro_export_variants.py --seed $s --omp $n \
      toniot T10=results/ladder/toniot/T10 >> $log 2>&1 && { echo "toniot $s T10 teacher OMP_NUM_THREADS=$n" >> $D; return 0; }
  done
  return 1
}
chain() {   # exports for one dataset and seed
  ds=$1; s=$2; log=$L/confirm_export_${ds}_$s.log
  : > $log
  if [ $s != 42 ]; then
    $P scripts/neuro_export.py $ds $s >> $log 2>&1 || { echo "$ds $s export failed" >> $D; return; }
    [ $ds = toniot ] && { t10 $s || { echo "$ds $s T10 export failed" >> $D; return; }; }
    $P scripts/neuro_export_vanilla.py $ds $s >> $log 2>&1 || { echo "$ds $s vanilla export failed" >> $D; return; }
    $P scripts/neuro_vanilla_int.py $ds $s >> $log 2>&1 || { echo "$ds $s vanilla int failed" >> $D; return; }
  fi
  echo "$ds $s exports ok" >> $D
}
for s in 42 43 44 45 46; do chain toniot $s & chain ciciot2023 $s & done; wait
sleep 5     # every export has finished, well outside the guard's one-second start margin
run() {
  ds=$1; s=$2; shift 2
  grep -q "^$ds $s exports ok" $D || return
  nice -n 10 python3 -I scripts/neuro_guard.py --allow results/neuro/confirm -- .venv-neuro/bin/python scripts/neuro_confirm.py $ds $s "$@" > $L/confirm_${ds}_$s.log 2>&1
  echo "$ds $s confirm exit $?" >> $D
}
for s in 42 43 44 45 46; do run toniot $s T10 vanilla t2s & sleep 3; run ciciot2023 $s t2s vanilla & sleep 3; done; wait
.venv/bin/python scripts/neuro_confirm_summary.py > $L/confirm_a7_summary.log 2>&1; echo "summary exit $?" >> $D
