#!/usr/bin/env bash
# M3 (fp32 vs int8) for the two repaired corpora. Runs after the headline re-runs,
# because val_curves.py checks its int8 side against their results.json; runs in
# parallel with run_post_repair_queue.sh. Pre-repair M3 outputs are archived through
# scripts/safe_mv.py, which refuses if any job could still depend on them.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=results/repaired_runs; mkdir -p "$L"
while pgrep -f "run_repaired.sh" > /dev/null; do sleep 120; done
echo "[$(date -u +%FT%TZ)] headline re-runs finished; starting M3 capture"

A=results/pre_repair/val_curves
for m in DNN-Tree Tree-to-Spike; do
  for d in cse-cic-ids2018 edgeiiot; do
    src=results/diagnostics/val_curves/$m/$d
    [ -e "$src" ] || continue
    $PY scripts/safe_mv.py "$src" "$A/$m/$d" || { echo "REFUSED $src; aborting"; exit 2; }
  done
done

# vanilla / TASNN: keep the pre-repair rows, then remove them so the skip-guard reruns them
$PY - <<'PYEOF'
import pandas as pd
p = "results/diagnostics/m3_fp32_models.csv"
d = pd.read_csv(p)
aff = d.dataset.isin(["cse-cic-ids2018", "edgeiiot"])
d[aff].to_csv("results/pre_repair/m3_fp32_models_prerepair_rows.csv", index=False)
d[~aff].to_csv(p, index=False)
print(f"archived {int(aff.sum())} pre-repair rows")
PYEOF

for d in cse-cic-ids2018 edgeiiot; do
  for s in 42 43 44 45 46; do
    for m in DNN-Tree Tree-to-Spike; do echo "$m $d $s"; done
  done
done | xargs -P "${WORKERS:-3}" -L 1 bash -c \
  '.venv/bin/python -u scripts/val_curves.py "$0" "$1" --seed "$2" > results/repaired_runs/m3_$0_$1_$2.log 2>&1 || echo "FAIL m3 $0 $1 $2"'

$PY -u scripts/eval_m3_fp32_models.py --datasets cse-cic-ids2018 edgeiiot > $L/m3_fp32_models.log 2>&1 \
  || echo "FAIL eval_m3_fp32_models"
$PY -u scripts/make_m3_table.py > $L/m3_table.log 2>&1 || echo "FAIL make_m3_table"
$PY -u scripts/fc1_ratio.py > $L/m3_fc1_ratio.log 2>&1 || echo "FAIL fc1_ratio"
echo "[$(date -u +%FT%TZ)] M3 repaired done"
