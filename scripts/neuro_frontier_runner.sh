#!/bin/bash
# Amendment 5 frontier: per dataset, once its variant export has finished and no a2_lava-writing guarded
# run is active, run the guarded frontier (all frontier runs share the allowed output directory).
cd "$(dirname "$0")/.."
L=results/neuro/logs
TAGS="T10 uniform q40 M2 prune_only"
run_ds() {
  ds=$1
  while pgrep -f "neuro_export_variants.py $ds " >/dev/null || pgrep -f "neuro_lava_fulltest.py" >/dev/null; do sleep 60; done
  for t in $TAGS; do [ -f results/neuro/inputs/${ds}_${t}_seed42.npz ] || { echo "$ds export missing $t" >> $L/frontier.done; return; }; done
  nice -n 10 python3 -I scripts/neuro_guard.py --allow results/neuro/frontier -- .venv-neuro/bin/python scripts/neuro_frontier.py $ds $TAGS > $L/frontier_$ds.log 2>&1
  echo "$ds frontier exit $?" >> $L/frontier.done
}
for ds in toniot edgeiiot ciciomt2024 ciciot2023; do run_ds $ds & sleep 5; done
wait
