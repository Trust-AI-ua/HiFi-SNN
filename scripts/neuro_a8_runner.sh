#!/bin/bash
# Amendment 8, part 2: the validation-selected fixed-point Vanilla SNN through the guarded chain
# (Edge-IIoTset primary, CICIoMT2024 secondary; seeds 42-46). Run after scripts/neuro_a8_select.py select + export,
# with nothing else writing to the repository. Status: results/neuro/logs/a8_tuned.done
cd "$(dirname "$0")/.."
L=results/neuro/logs
D=$L/a8_tuned.done
: > $D
run() {
  nice -n 10 python3 -I scripts/neuro_guard.py --allow results/neuro/review/a8_tuned_vanilla -- \
    .venv-neuro/bin/python scripts/neuro_a8_tuned.py $1 $2 > $L/a8_tuned_$1_$2.log 2>&1
  echo "$1 $2 tuned exit $?" >> $D
}
for s in 42 43 44 45 46; do run edgeiiot $s & sleep 3; run ciciomt2024 $s & sleep 3; done; wait
echo "all done" >> $D
