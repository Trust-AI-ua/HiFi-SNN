#!/usr/bin/env bash
# One-off guard for the running M3 Tree-to-Spike grid (run_m3_t2s.sh): its
# Edge-IIoTset cells are obsolete (dataset contaminated, rerun pending), so stop
# them as they start. Exits on its own when the grid driver exits.
DRIVER_PID="$1"
while kill -0 "$DRIVER_PID" 2>/dev/null; do
  for pid in $(ps -eo pid=,args= | awk '/[v]al_curves.py Tree-to-Spike edgeiiot/ {print $1}'); do
    kill "$pid" && echo "$(date -u +%FT%TZ) stopped edgeiiot cell pid $pid"
  done
  sleep 30
done
echo "$(date -u +%FT%TZ) grid driver exited; guard done"
