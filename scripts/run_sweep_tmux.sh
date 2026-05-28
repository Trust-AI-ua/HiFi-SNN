#!/usr/bin/env bash
# Launch the Phase-1 sweep inside a detached tmux session so you can
# disconnect from the shell and the job keeps running.
#
# The session name and log file are suffixed by the dataset, so you can run
# multiple datasets in parallel (e.g. nslkdd in one session, cicids2017 in
# another) without them clobbering each other.
#
# Usage examples:
#   scripts/run_sweep_tmux.sh nslkdd --seeds 43 44 45 46
#   scripts/run_sweep_tmux.sh kddcup99 --seeds 42 43 44
#   scripts/run_sweep_tmux.sh cicids2017 --seeds 42 --force
#
# Once launched:
#   tmux ls                                    # list all running sessions
#   tmux attach -t hisnn-sweep-nslkdd          # reconnect to a specific one
#   tail -f results/sweep-nslkdd.log           # follow output without attaching
#   tmux kill-session -t hisnn-sweep-nslkdd    # cancel that sweep

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <dataset> [run_sweep.py args...]"
    echo "example: $0 nslkdd --seeds 43 44 45 46"
    exit 2
fi

DATASET=$1
SESSION="hisnn-sweep-${DATASET}"
LOG="results/sweep-${DATASET}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Choose one:"
    echo "  tmux attach -t $SESSION         # reconnect"
    echo "  tmux kill-session -t $SESSION   # cancel and try again"
    exit 1
fi

mkdir -p results
echo "Starting sweep at $(date) — dataset=$DATASET args: $*" | tee "$LOG"

# Pick the venv Python explicitly so the tmux shell doesn't depend on user shell config.
PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON=$(command -v python3 || command -v python)
fi

# Chain the sweep with plot regeneration so you come back to finished figures.
CMD="$PYTHON scripts/run_sweep.py $*"
CMD="$CMD && echo '--- regenerating plots ---' && $PYTHON scripts/make_plots.py $DATASET"
CMD="$CMD && echo 'sweep + plots DONE at \$(date)' || echo 'sweep FAILED at \$(date) — see log'"
# Keep the pane alive after the command finishes so you can read the tail.
CMD="$CMD; echo; echo 'Press Ctrl+B then D to detach, or Ctrl+D to close this pane.'; exec bash"

tmux new-session -d -s "$SESSION" "stdbuf -oL -eL bash -c '$CMD' 2>&1 | tee -a $LOG"

cat <<EOF

Sweep launched in tmux session: $SESSION
Log file:                       $LOG

Useful commands while it runs:
  tail -f $LOG                       # follow output in this shell
  tmux attach -t $SESSION            # attach to the tmux pane (Ctrl+B then D to detach)
  tmux ls                            # list ALL sessions (incl. other datasets running in parallel)
  tmux kill-session -t $SESSION      # cancel THIS sweep only
  watch -n5 nvidia-smi               # watch GPU state

When the sweep finishes, plots will be regenerated automatically. You'll see
"sweep + plots DONE" near the bottom of the log.
EOF
