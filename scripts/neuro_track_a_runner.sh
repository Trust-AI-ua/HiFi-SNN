#!/usr/bin/env bash
# Track A after the exports: per dataset the Vanilla export (training env, shared GPU), then A2 in
# the pre-registered order (Lava's own LIF first, then the subtractive process), then A1 on all
# four datasets, the Edge-IIoTset re-export that adds the encoder edge counts (its first export
# predates that field), and A3 with the P5-passing variant. Every .venv-neuro run goes through
# scripts/neuro_guard.py. Log: results/neuro/logs/track_a.done
set -u
cd "$(dirname "$0")/.."
L=results/neuro/logs
G="python3 -I scripts/neuro_guard.py"
N=.venv-neuro/bin/python
for d in toniot ciciomt2024 ciciot2023; do
  nice -n 10 .venv/bin/python scripts/neuro_export_vanilla.py $d > $L/export_vanilla_$d.log 2>&1; echo "$d vanilla_export exit $?" >> $L/track_a.done
  for v in zero subtract; do
    nice -n 10 $G --allow results/neuro/a2_lava -- $N scripts/neuro_a2_lava.py $v $d > $L/a2_${d}_$v.log 2>&1; echo "$d a2_$v exit $?" >> $L/track_a.done
  done
done
nice -n 10 $G --allow results/neuro/a1_nir -- $N scripts/neuro_a1_nir.py > $L/a1.log 2>&1; echo "a1 exit $?" >> $L/track_a.done
nice -n 10 .venv/bin/python scripts/neuro_export.py edgeiiot 42 > $L/export_edgeiiot_edges.log 2>&1; echo "edgeiiot re-export exit $?" >> $L/track_a.done
nice -n 10 $G --allow results/neuro/a3_sanafe -- $N scripts/neuro_a3_sanafe.py subtract > $L/a3.log 2>&1; echo "a3 exit $?" >> $L/track_a.done
