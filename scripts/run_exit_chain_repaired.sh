#!/usr/bin/env bash
# The early-exit chain replays cached per-timestep logits from
# results/prefix_outputs/; those were written 2026-08-07, i.e. before the leakage
# repair, so the 2026-09-20 chain run reported pre-repair numbers for the two
# re-run datasets (the other nine are unaffected). Regenerate the prefixes from
# the repaired checkpoints, then replay the whole chain.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/repaired_runs
ts () { date -u +%FT%TZ; }
stage () { local n=$1; shift; echo "[$(ts)] START $n"; "$@" > $L/queue_$n.log 2>&1 \
           && echo "[$(ts)] ok $n" || echo "[$(ts)] FAIL $n"; }
stage extract_prefix $PY -u scripts/extract_prefix.py --datasets cse-cic-ids2018 edgeiiot
stage early_exit_sweep2 $PY -u scripts/early_exit_sweep.py
stage early_exit_audit2 $PY -u scripts/early_exit_audit.py --datasets ciciot2023 edgeiiot
stage evidence_arrival2 $PY -u scripts/evidence_arrival.py
stage exit_margin2      $PY -u scripts/exit_margin_test.py
stage edgeiiot_mechanism2 $PY -u scripts/edgeiiot_mechanism.py
echo "[$(ts)] exit chain (repaired) done"
