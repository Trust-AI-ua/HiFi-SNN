#!/usr/bin/env bash
# Launch a DNN-Tree (Design C) run for ONE dataset in a detached tmux session.
# Mirrors the other run_*_tmux.sh wrappers: session name and log file are suffixed
# by the dataset, so datasets can run in parallel.
#
# Usage:
#   scripts/run_dnn_tmux.sh kddcup99 --seeds 42 43 44 45 46
#   scripts/run_dnn_tmux.sh kddcup99 --seeds 44 45 46 --force
#
# Monitor:
#   tmux ls
#   tail -f results/sweep-dnn-kddcup99.log
#   tmux attach -t hisnn-dnn-kddcup99        # Ctrl+B then D to detach
#   tmux kill-session -t hisnn-dnn-kddcup99  # cancel

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <dataset> [run_dnn.py args...]"
    echo "example: $0 kddcup99 --seeds 42 43 44 45 46"
    exit 2
fi

DATASET=$1
SESSION="hisnn-dnn-${DATASET}"
LOG="results/sweep-dnn-${DATASET}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Choose one:"
    echo "  tmux attach -t $SESSION         # reconnect"
    echo "  tmux kill-session -t $SESSION   # cancel and try again"
    exit 1
fi

mkdir -p results
echo "Starting DNN-Tree run at $(date) — dataset=$DATASET args: $*" | tee "$LOG"

PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON=$(command -v python3 || command -v python)
fi

CMD="$PYTHON scripts/run_dnn.py $*"
CMD="$CMD && echo 'DNN-Tree $DATASET DONE at \$(date)' || echo 'DNN-Tree $DATASET FAILED at \$(date) — see log'"
CMD="$CMD; echo; echo 'Press Ctrl+B then D to detach, or Ctrl+D to close this pane.'; exec bash"

tmux new-session -d -s "$SESSION" "stdbuf -oL -eL bash -c '$CMD' 2>&1 | tee -a $LOG"

cat <<EOF

DNN-Tree run launched in tmux session: $SESSION
Log file:                                   $LOG

Useful commands:
  tail -f $LOG                       # follow output
  tmux attach -t $SESSION            # attach (Ctrl+B then D to detach)
  tmux ls                            # list ALL sessions
  tmux kill-session -t $SESSION      # cancel THIS dataset only
EOF
