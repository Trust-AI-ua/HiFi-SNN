#!/usr/bin/env bash
# A1 follow-up chain, runnable without an operator or network: waits for the block prefix
# extraction to finish, requires all five cells to pass its exact-reconstruction gate, runs
# the validation-only early-exit selection on the block prefixes, then the follow-up script
# (which first reproduces the published exit audit and flag rows). Commits results locally
# at the end (no push; push when the network is back). Also waits for A6 and commits it.
cd "$(dirname "$0")/.."
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS TORCH_NUM_THREADS
L=results/journal_partB/logs; Q=$L/queue.log; FU=results/journal_partB/B6/followup
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $Q; }
alive(){ ps -u "$(id -un)" -o args | awk -v s="$1" '$1 ~ /python$/ && $3 ~ s' | grep -q .; }

while alive 'extract_prefix.py$'; do sleep 30; done
n_ok=$(grep -c "| OK" $L/A1_prefix.log)
if [ "$n_ok" -ne 5 ]; then log "ABORT A1 follow-up: prefix extraction passed $n_ok/5 cells (see A1_prefix.log)"; exit 1; fi
log "A1 prefix extraction: 5/5 cells reproduce their block records exactly; early-exit selection"
.venv/bin/python -u scripts/early_exit_sweep.py --prefix-root $FU/prefix_outputs --out-root $FU/early_exit \
  > $L/A1_exit_sweep.log 2>&1 || { log "FAIL A1 exit sweep"; exit 1; }
.venv/bin/python -u scripts/partB_a1_followup.py > $L/A1_followup.log 2>&1 || { log "FAIL A1 follow-up (see A1_followup.log)"; exit 1; }
log "A1 follow-up done (flags, exit audit, selection); committing locally"
git add scripts/extract_prefix.py scripts/early_exit_sweep.py scripts/partB_a1_followup.py \
        scripts/run_partB_followup_offline.sh $FU/early_exit $FU/flags.csv $FU/exit_audit.csv $FU/summary.json \
        $L/A1_*.log $L/queue.log $FU/prefix_outputs/extraction_report.json 2>/dev/null
git commit -q -m "A1 follow-up for the CICIoT2023 block: prefixes, early-exit selection, flags, exit audit

Prefix outputs of the block's tree runs reproduce each record exactly (5/5); the patched
extractor and exit sweep reproduce the published outputs bitwise on unchanged cells; the
follow-up script reproduces the published exit audit and IoT flag rows before computing
CICIoT2023. Committed offline; not yet pushed.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "A1 follow-up committed locally ($(git rev-parse --short HEAD))"

while alive 'partB_a6_dominated.py$'; do sleep 60; done
git add results/journal_partB/A6 $L/A6.log $L/queue.log 2>/dev/null
git commit -q -m "A6: dominated general-purpose classes with and without cross-partition duplicates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "A6 committed locally ($(git rev-parse --short HEAD))"
log "offline follow-up chain finished; push when the network is back"
