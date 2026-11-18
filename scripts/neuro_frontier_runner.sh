#!/bin/bash
# Amendment 5/5b frontier. Exports (.venv, teacher refit per teacher group), then per dataset the guarded
# frontier (all frontier runs share their allowed output directory), then the guarded rerun of the
# full-test scoring (D5) and the summary.
cd "$(dirname "$0")/.."
L=results/neuro/logs
E="nice -n 10 .venv/bin/python scripts/neuro_export_variants.py"
ex() { PYTHONDONTWRITEBYTECODE=1 "$@"; }
run_ds() {
  ds=$1; shift
  while pgrep -f "neuro_export_variants.py.* $ds " >/dev/null; do sleep 60; done
  tags=""; for t in T10 uniform q40 M2 prune_only T40 T80; do
    if [ -f results/neuro/inputs/${ds}_${t}_seed42.npz ]; then tags="$tags $t"; else echo "$ds $t: no export" >> $L/frontier.done; fi; done
  nice -n 10 python3 -I scripts/neuro_guard.py --allow results/neuro/frontier -- .venv-neuro/bin/python scripts/neuro_frontier.py $ds $tags > $L/frontier_$ds.log 2>&1
  echo "$ds frontier exit $?" >> $L/frontier.done
}
# exports (default teacher, and OMP_NUM_THREADS=2 for the ladder runs whose stored teacher needs it)
ex $E toniot uniform=results/encoder_ablation/toniot/uniform prune_only=results/ladder/toniot/prune_only > $L/export_variants_toniot.log 2>&1 &
ex env OMP_NUM_THREADS=2 $E --omp 2 toniot T10=results/ladder/toniot/T10 T40=results/ladder/toniot/T40 T80=results/ladder/toniot/T80 q40=results/ladder/toniot/q40 M2=results/ladder/toniot/M2 > $L/export_variants_toniot_omp2.log 2>&1 &
ex $E edgeiiot T40=results/ladder/edgeiiot/T40 T80=results/ladder/edgeiiot/T80 > $L/export_variants_edgeiiot_2.log 2>&1 &
ex $E ciciomt2024 T40=results/ladder/ciciomt2024/T40 T80=results/ladder/ciciomt2024/T80 prune_only=results/ladder/ciciomt2024/prune_only > $L/export_variants_ciciomt2024_2.log 2>&1 &
ex env OMP_NUM_THREADS=2 $E --omp 2 ciciomt2024 q40=results/ladder/ciciomt2024/q40 M2=results/ladder/ciciomt2024/M2 > $L/export_variants_ciciomt2024_omp2.log 2>&1 &
ex $E ciciot2023 T40=results/ladder/ciciot2023/T40 T80=results/ladder/ciciot2023/T80 prune_only=results/ladder/ciciot2023/prune_only > $L/export_variants_ciciot2023_2.log 2>&1 &
ex env OMP_NUM_THREADS=2 $E --omp 2 ciciot2023 M2=results/ladder/ciciot2023/M2 > $L/export_variants_ciciot2023_omp2.log 2>&1 &
sleep 20
for ds in toniot edgeiiot ciciomt2024 ciciot2023; do run_ds $ds & sleep 5; done
wait
nice -n 10 python3 -I scripts/neuro_guard.py --allow results/neuro/a2_lava -- .venv-neuro/bin/python scripts/neuro_lava_fulltest.py > $L/lava_fulltest.log 2>&1
echo "lava_fulltest exit $?" >> $L/frontier.done
.venv/bin/python scripts/neuro_frontier_summary.py > $L/frontier_summary.log 2>&1
echo "summary exit $?" >> $L/frontier.done
