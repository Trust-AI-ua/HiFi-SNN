#!/usr/bin/env python3
"""One Tree-to-Spike training cell for revision-2 steps 5 and 6
(results/preregistration/2026-11-17_r2_steps4_6.md).

    python scripts/r2_train_cell.py variance  <dataset> <data_seed> <seed>
    python scripts/r2_train_cell.py factorial <dataset> <tree|raw> <kd|nokd> <T> <seed>

Configuration: the production one, as the stored records were made. That is
experiments/Tree-to-Spike/<ds>/config.yaml, and for CICIoT2023 the Part B block protocol
(20 epochs, validation macro-F1 epoch selection, all precisions recorded; scripts/partB_b6_block.py).
The cell's factors are applied on top. `data_seed` seeds only the loader (src/training_treespike.py).

Output:
  variance:  results/journal_r2/variance/<ds>/data_<D>/seed_<S>/results.json
  factorial: results/journal_r2/factorial/<ds>/<enc>_<kd>_T<T>/seed_<S>/results.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/journal_r2"
BLOCK = {"ciciot2023": dict(epochs=20, select_on_val=True, record_precisions=True)}
ENC = {"tree": {"encoder_mode": "tree"}, "raw": {"encoder_mode": "raw_vanilla"}}
KD = {"kd": {"a_kd": 1.0, "a_rank": 0.3}, "nokd": {"a_kd": 0.0, "a_rank": 0.0}}


def production(ds: str) -> dict:
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    cfg.update(BLOCK.get(ds, {}))
    return cfg


def main():
    if any(os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise SystemExit("thread variables are set; the teacher would not be the production teacher")
    study, ds = sys.argv[1], sys.argv[2]
    cfg = production(ds)
    if study == "variance":
        data_seed, seed = int(sys.argv[3]), int(sys.argv[4])
        rd = OUT / f"variance/{ds}/data_{data_seed}"
        cfg.update(data_seed=data_seed)
        tag = dict(study=study, data_seed=data_seed, seed=seed)
    elif study == "factorial":
        enc, kd, T, seed = sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
        rd = OUT / f"factorial/{ds}/{enc}_{kd}_T{T}"
        cfg.update(ENC[enc]); cfg.update(KD[kd]); cfg.update(T=T)
        tag = dict(study=study, encoder=enc, distillation=kd, T=T, seed=seed)
    else:
        raise SystemExit(f"unknown study {study!r}")
    p = rd / f"seed_{seed}" / "results.json"
    if p.exists():
        print(f"[skip] {p}"); return
    from src.joblock import claim
    lock = OUT / f".locks/{study}_{ds}_{rd.name}_{seed}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with claim(lock) as got:
        if not got:
            print(f"[busy] {p}"); return
        from src.training_treespike import run_experiment
        cfg.update(seed=seed, results_dir=str(rd), summary_csv=str(rd / f"summary_seed_{seed}.csv"))
        rec = run_experiment(cfg)
        r = json.loads(p.read_text())
        r["journal_r2"] = dict(**tag, production_overrides=BLOCK.get(ds, {}),
                               preregistration="results/preregistration/2026-11-17_r2_steps4_6.md")
        p.write_text(json.dumps(r, indent=2))
        m = rec["test_plus"]
        print(f"[done] {study} {ds} {rd.name} seed {seed}: macro-F1 {100 * m['metrics']['f1_macro']:.2f} "
              f"FAR {100 * m['binary_ids_metrics']['false_alarm_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
