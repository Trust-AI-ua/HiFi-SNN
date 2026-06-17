#!/usr/bin/env bash
# Sequentially run every pending hybrid (model, dataset) sweep in ONE process,
# so the full matrix gets worked through without oversubscribing a shared GPU.
#
# Why a queue (not 17 parallel tmux sessions): the GPU is shared and often near
# 100% util, so launching every sweep at once just thrashes — each job crawls
# and some can OOM. Running them back-to-back finishes the whole set faster and
# is kinder to other users. The per-model orchestrators already skip-guard on an
# existing results.json, so this is safe to re-run and resumes where it left off.
#
# Scope: all four hybrids x {nslkdd, cicids2017, ctu13, iscxids2012,
# cse-cic-ids2018}. kddcup99 is intentionally omitted (already complete).
# S-xLSTM/{nslkdd,ctu13,iscxids2012} are omitted here because they are already
# done or running in their own tmux sessions (avoids a write race).
#
# Usage:
#   scripts/run_hybrids_queue.sh                       # seeds 42..46
#   scripts/run_hybrids_queue.sh --seeds 42 43 44
# Launch detached so it survives disconnects:
#   tmux new-session -d -s hisnn-hybrids-queue \
#     "scripts/run_hybrids_queue.sh 2>&1 | tee -a results/hybrids-queue.log"
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
PY=.venv/bin/python

SEEDS=(42 43 44 45 46)
if [[ "${1:-}" == "--seeds" ]]; then shift; SEEDS=("$@"); fi

declare -A SCRIPT=(
  [S-xLSTM]=scripts/run_sxlstm.py
  [Spiking-Griffin]=scripts/run_griffin.py
  [Spiking-SSM]=scripts/run_ssm.py
  [Tree-to-Spike]=scripts/run_treespike.py
)

# (model dataset) queue, cheapest datasets first; cse-cic last (heaviest + the
# vanilla SNN sweep is still using it). One pair per line.
QUEUE=(
  "Spiking-Griffin nslkdd"  "Spiking-SSM nslkdd"  "Tree-to-Spike nslkdd"
  "S-xLSTM cicids2017"  "Spiking-Griffin cicids2017"  "Spiking-SSM cicids2017"  "Tree-to-Spike cicids2017"
  "Spiking-Griffin ctu13"  "Spiking-SSM ctu13"  "Tree-to-Spike ctu13"
  "Spiking-Griffin iscxids2012"  "Spiking-SSM iscxids2012"  "Tree-to-Spike iscxids2012"
  "S-xLSTM cse-cic-ids2018"  "Spiking-Griffin cse-cic-ids2018"  "Spiking-SSM cse-cic-ids2018"  "Tree-to-Spike cse-cic-ids2018"
)

echo "=================================================================="
echo "hybrid queue start: $(date)"
echo "seeds: ${SEEDS[*]}   pairs: ${#QUEUE[@]}"
echo "=================================================================="
n=0; ok=0; fail=0; failed_pairs=()
for pair in "${QUEUE[@]}"; do
  read -r model ds <<< "$pair"
  n=$((n+1))
  echo ""
  echo "############### [$n/${#QUEUE[@]}] $model / $ds  ($(date +%H:%M:%S)) ###############"
  if "$PY" "${SCRIPT[$model]}" "$ds" --seeds "${SEEDS[@]}"; then
    ok=$((ok+1)); echo "--- [$n/${#QUEUE[@]}] $model/$ds OK ---"
  else
    fail=$((fail+1)); failed_pairs+=("$model/$ds")
    echo "--- [$n/${#QUEUE[@]}] $model/$ds FAILED (continuing) ---"
  fi
done

echo ""
echo "=================================================================="
echo "hybrid queue done: $(date)   ok=$ok fail=$fail"
[[ $fail -gt 0 ]] && printf '  FAILED: %s\n' "${failed_pairs[@]}"
echo "=================================================================="
