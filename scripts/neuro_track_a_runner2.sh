#!/bin/bash
# Track A, second pass (amendment 4): A1 (NIR, fixed export), Lava Vanilla, A3 with vanilla_fixed/vanilla_float.
# Each neuro-environment run guarded; sequential, so no guarded run sees another's outputs.
cd "$(dirname "$0")/.."
L=results/neuro/logs
G="python3 -I scripts/neuro_guard.py"
N=.venv-neuro/bin/python
nice -n 10 $G --allow results/neuro/a1_nir -- $N scripts/neuro_a1_nir.py > $L/a1.log 2>&1; echo "a1 exit $?" >> $L/track_a2.done
nice -n 10 $G --allow results/neuro/a2_lava -- $N scripts/neuro_a2_vanilla.py > $L/a2_vanilla.log 2>&1; echo "a2_vanilla exit $?" >> $L/track_a2.done
nice -n 10 $G --allow results/neuro/a3_sanafe -- $N scripts/neuro_a3_sanafe.py subtract > $L/a3.log 2>&1; echo "a3 exit $?" >> $L/track_a2.done
