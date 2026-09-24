#!/usr/bin/env bash
# E1 (matched-information dense twin) and E2 (supervised non-tree edges), pre-registered in
# results/preregistration/2026-09-24_matched_info_and_entropy_edges.md.
# Guards first; no training starts unless every guard passes.
cd "$(dirname "$0")/.."
L=results/repaired_runs/e1_e2.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $L; }
REF=0af66b2          # last commit before the encoder/trainer changes
# No thread cap: XGBoost caps n_jobs at the OpenMP thread count, and the teacher fit
# depends on it (OMP_NUM_THREADS=2 changed the TON-IoT seed-42 teacher; G2 caught it).
# The production teachers reproduce with the default environment.
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS TORCH_NUM_THREADS
log "START guards G4 G6 G1 G3 G5 (vs $REF; amendment 1)"
.venv/bin/python -u scripts/guard_e1_e2.py --ref $REF > results/repaired_runs/guard_e1_e2.log 2>&1 \
  || { log "ABORT: G1/G3/G4/G5/G6 failed (see guard_e1_e2.log)"; exit 1; }
log "ok G4 G6 G1 G3 G5"
log "START guard G2 (DNN-Tree TON-IoT seed 42: int8, fp32, per-channel reproduction)"
.venv/bin/python -u scripts/run_dnn_tq.py --guard > results/repaired_runs/guard_g2.log 2>&1 \
  || { log "ABORT: G2 failed (see guard_g2.log)"; exit 1; }
log "ok G2"
cells(){ for d in toniot edgeiiot ciciomt2024 ciciot2023; do for s in 42 43 44 45 46; do echo "$d $s"; done; done; }
log "START E1 pool (20 DNN-TQ cells, 2 workers) and E2 pool (20 entropy cells, 4 workers)"
( cells | xargs -P 2 -L 1 bash -c '.venv/bin/python -u scripts/run_dnn_tq.py "$0" "$1" > results/repaired_runs/e1_$0_$1.log 2>&1 || echo "FAIL e1 $0 $1"' >> $L; log "E1 pool done" ) &
( cells | xargs -P 4 -L 1 bash -c '.venv/bin/python -u scripts/run_encoder_variant.py "$0" entropy "$1" > results/repaired_runs/e2_$0_$1.log 2>&1 || echo "FAIL e2 $0 $1"' >> $L; log "E2 pool done" ) &
wait
log "START analysis"
.venv/bin/python -u scripts/analyze_e1_e2.py > results/repaired_runs/analyze_e1_e2.log 2>&1 || log "analysis refused (see analyze_e1_e2.log)"
log "E1 E2 done"
