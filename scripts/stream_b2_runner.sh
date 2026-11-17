#!/usr/bin/env bash
# B2 pipeline per seed (PI decisions 2026-11-17): each seed's student starts on the GPU, beside the
# revision-2 queue, as soon as its teacher has finished; then its evaluation (CPU, no compact GBDT)
# and an exploratory summary. After all five: the evaluations with the compact GBDT reference and
# the summary on which rule P7 is applied. Logs: results/neuro/logs/b2_*.log, b2_runner.done.
set -u
cd "$(dirname "$0")/.."
L=results/neuro/logs
PY=.venv/bin/python
wait_teacher() { until grep -q "^$1 exit" $L/b2_teacher.done 2>/dev/null; do sleep 60; done; grep -q "^$1 exit 0" $L/b2_teacher.done; }
for s in 42 43 44 45 46; do
  if ! wait_teacher $s; then echo "$s teacher failed" >> $L/b2_runner.done; continue; fi
  if [ $s = 42 ]; then
    nice -n 10 $PY scripts/stream_b2_student.py 42 --smoke > $L/b2_student_42_smoke.log 2>&1 || { echo "42 smoke failed" >> $L/b2_runner.done; exit 1; }
  fi
  nice -n 10 $PY scripts/stream_b2_student.py $s > $L/b2_student_$s.log 2>&1; echo "$s student exit $?" >> $L/b2_runner.done
  nice -n 10 $PY scripts/stream_b2_eval.py $s --no-compact > $L/b2_eval_$s.log 2>&1; echo "$s eval exit $?" >> $L/b2_runner.done
  nice -n 10 $PY scripts/stream_b2_summary.py > $L/b2_summary.log 2>&1
done
for s in 42 43 44 45 46; do
  nice -n 10 $PY scripts/stream_b2_eval.py $s > $L/b2_eval_full_$s.log 2>&1; echo "$s eval_full exit $?" >> $L/b2_runner.done
done
nice -n 10 $PY scripts/stream_b2_summary.py > $L/b2_summary.log 2>&1; echo "summary exit $?" >> $L/b2_runner.done
