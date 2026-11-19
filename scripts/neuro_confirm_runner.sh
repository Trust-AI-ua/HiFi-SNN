#!/bin/bash
# Amendment 6. Waits for the revision-2 training queue (step 6), writes the steps 5-6 summary, then exports
# (.venv; GPU replay for Vanilla), the guarded confirmation runs (shared allowed output directory) and the summary.
cd "$(dirname "$0")/.."
L=results/neuro/logs
while [ "$(wc -l < results/journal_r2/logs/train_queue.done)" -lt "$(wc -l < results/journal_r2/logs/train_queue.txt)" ] || pgrep -f "r2_train_cell.py" >/dev/null; do sleep 120; done
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/r2_summarize_steps5_6.py > $L/steps5_6_summary.log 2>&1; echo "steps5_6 summary exit $?" >> $L/confirm.done
P="nice -n 10 env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python"
chain() {   # exports for one dataset and seed
  ds=$1; s=$2
  if [ $s != 42 ]; then
    $P scripts/neuro_export.py $ds $s > $L/confirm_export_${ds}_$s.log 2>&1 || { echo "$ds $s export failed" >> $L/confirm.done; return; }
    [ $ds = edgeiiot ] && { $P scripts/neuro_export_variants.py --seed $s edgeiiot T10=results/ladder/edgeiiot/T10 >> $L/confirm_export_${ds}_$s.log 2>&1 || { echo "$ds $s T10 export failed" >> $L/confirm.done; return; }; }
    $P scripts/neuro_export_vanilla.py $ds $s >> $L/confirm_export_${ds}_$s.log 2>&1 || { echo "$ds $s vanilla export failed" >> $L/confirm.done; return; }
    $P scripts/neuro_vanilla_int.py $ds $s >> $L/confirm_export_${ds}_$s.log 2>&1 || { echo "$ds $s vanilla int failed" >> $L/confirm.done; return; }
  fi
  echo "$ds $s exports ok" >> $L/confirm.done
}
for s in 42 43 44 45 46; do chain edgeiiot $s & chain ciciomt2024 $s & done; wait
run() {
  ds=$1; s=$2; shift 2
  grep -q "^$ds $s exports ok" $L/confirm.done || return
  nice -n 10 python3 -I scripts/neuro_guard.py --allow results/neuro/confirm -- .venv-neuro/bin/python scripts/neuro_confirm.py $ds $s "$@" > $L/confirm_${ds}_$s.log 2>&1
  echo "$ds $s confirm exit $?" >> $L/confirm.done
}
for s in 42 43 44 45 46; do run edgeiiot $s T10 vanilla t2s & sleep 3; run ciciomt2024 $s t2s vanilla & sleep 3; done; wait
.venv/bin/python scripts/neuro_confirm_summary.py > $L/confirm_summary.log 2>&1; echo "summary exit $?" >> $L/confirm.done
