#!/usr/bin/env bash
# Launch a Spiking-Griffin run for ONE dataset inside a detached tmux session.
# Mirrors scripts/run_sxlstm_tmux.sh and scripts/run_sweep_tmux.sh: the session
# name and log file are suffixed by the dataset, so datasets can run in parallel.
#
# Usage:
#   scripts/run_griffin_tmux.sh kddcup99 --seeds 42 43 44 45 46
#   scripts/run_griffin_tmux.sh kddcup99 --seeds 44 45 46 --force
#
# Monitor:
#   tmux ls
#   tail -f results/sweep-griffin-kddcup99.log
#   tmux attach -t hisnn-griffin-kddcup99        # Ctrl+B then D to detach
#   tmux kill-session -t hisnn-griffin-kddcup99  # cancel

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <dataset> [run_griffin.py args...]"
    echo "example: $0 kddcup99 --seeds 42 43 44 45 46"
    exit 2
fi

DATASET=$1
SESSION="hisnn-griffin-${DATASET}"
LOG="results/sweep-griffin-${DATASET}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Choose one:"
    echo "  tmux attach -t $SESSION         # reconnect"
    echo "  tmux kill-session -t $SESSION   # cancel and try again"
    exit 1
fi

mkdir -p results
echo "Starting Spiking-Griffin run at $(date) — dataset=$DATASET args: $*" | tee "$LOG"

PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON=$(command -v python3 || command -v python)
fi

CMD="$PYTHON scripts/run_griffin.py $*"
CMD="$CMD && echo 'Spiking-Griffin $DATASET DONE at \$(date)' || echo 'Spiking-Griffin $DATASET FAILED at \$(date) — see log'"
CMD="$CMD; echo; echo 'Press Ctrl+B then D to detach, or Ctrl+D to close this pane.'; exec bash"

tmux new-session -d -s "$SESSION" "stdbuf -oL -eL bash -c '$CMD' 2>&1 | tee -a $LOG"

cat <<EOF

Spiking-Griffin run launched in tmux session: $SESSION
Log file:                                     $LOG

Useful commands:
  tail -f $LOG                       # follow output
  tmux attach -t $SESSION            # attach (Ctrl+B then D to detach)
  tmux ls                            # list ALL sessions
  tmux kill-session -t $SESSION      # cancel THIS dataset only
EOF
