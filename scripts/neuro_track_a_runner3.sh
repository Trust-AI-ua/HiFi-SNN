#!/bin/bash
# Track A, third pass: A1 with the restore fix; full-test Lava-arithmetic macro-F1. Guarded, sequential.
cd "$(dirname "$0")/.."
L=results/neuro/logs
G="python3 -I scripts/neuro_guard.py"
N=.venv-neuro/bin/python
nice -n 10 $G --allow results/neuro/a1_nir -- $N scripts/neuro_a1_nir.py > $L/a1.log 2>&1; echo "a1 exit $?" >> $L/track_a3.done
nice -n 10 $G --allow results/neuro/a2_lava -- $N scripts/neuro_lava_fulltest.py > $L/lava_fulltest.log 2>&1; echo "lava_fulltest exit $?" >> $L/track_a3.done
