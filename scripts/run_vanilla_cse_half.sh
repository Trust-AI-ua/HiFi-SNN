#!/usr/bin/env bash
# Rerun the Vanilla SNN on CSE-CIC-IDS2018 at subsample_fraction 0.5 (the shared
# protocol), five seeds, so all four models share one subsample and test set per seed.
# The full-corpus runs are in results/superseded/vanilla_cse_full_corpus_latency/.
cd "$(dirname "$0")/.."
for s in 42 43 44 45 46; do
  HISNN_SEED=$s nohup .venv/bin/python -u experiments/cse-cic-ids2018/SNN_LeakyParallel/latency/run.py \
    > results/repaired_runs/vanilla_cse_half_$s.log 2>&1 &
  echo "[$(date -u +%FT%TZ)] started seed $s pid $!" >> results/repaired_runs/vanilla_cse_half.log
  sleep 120
done
wait
echo "[$(date -u +%FT%TZ)] all five done" >> results/repaired_runs/vanilla_cse_half.log
