#!/usr/bin/env python3
"""Host latency of the validation-selected compact GBDT within 128 KiB (revision 2, step 3 models;
scripts/r2_gbdt_valsel.py), under the deployment timing protocol of scripts/deployment_measure.py:
batch size one, one prediction thread, 500 warm-up calls then 2,000 timed calls on the fixed common
subset (the first 2,000 test rows), p50 and p95 in milliseconds.

Phase "fit" (parallel-safe): refits each selected configuration exactly as r2_gbdt_valsel.py does
(same XGBoost settings, cost weights, rounds, depth, quantized leaves) and checks that its test
macro-F1 equals the stored selected value; saves the quantized booster.
Phase "time" (run alone): loads each booster, sets nthread = 1, times the r2_gbdt_valsel prediction
path (DataFrame -> DMatrix -> booster.predict -> argmax) per row with the DMatrix also built on one
thread (r2_gbdt_valsel.predict builds it on all cores, which spawns an OpenMP team per call), after
checking that its predictions equal r2_gbdt_valsel.predict on the subset; samples the one-minute load
average and the process's thread count every 2 s during the timing.

    .venv/bin/python scripts/r2_gbdt_latency.py fit <dataset> [seeds...]
    .venv/bin/python scripts/r2_gbdt_latency.py time
Output: results/deployment/valsel_latency/<ds>_seed<S>.json (booster: *_booster.json), latency.csv
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/deployment/valsel_latency"
DS = ["toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
SEEDS = [42, 43, 44, 45, 46]


def fit_one(ds, seed):
    import r2_gbdt_valsel as V
    from src.metrics import compute_metrics
    sel = json.loads((REPO / f"results/journal_r2/gbdt_valsel/{ds}/seed_{seed}.json").read_text())["selected"]
    b, (Xtr, ytr, Xva, yva, Xte, yte) = V.load(ds, seed)
    K = b.num_classes
    present = np.unique(ytr)
    remap = np.full(K, -1, np.int64); remap[present] = np.arange(len(present))
    cnt = np.bincount(remap[ytr], minlength=len(present)).astype(np.float32)
    w = (cnt.sum() / (cnt + 1e-6))[remap[ytr]]
    Kt = len(present); k_units = Kt if Kt > 2 else 1
    m = V.fit(Xtr, remap[ytr], w, sel["rounds"], sel["max_depth"], seed)
    qb = V.quantized_booster(m, sel["rounds"], k_units, sel["leaf_bits"])
    f1 = 100 * compute_metrics(yte, present[V.predict(qb, Xte, k_units)], class_names=b.class_names)["f1_macro"]
    if f1 != sel["test_macro_f1"]:
        raise SystemExit(f"{ds}/{seed}: refit compact GBDT {f1!r} is not the stored selected model {sel['test_macro_f1']!r}")
    OUT.mkdir(parents=True, exist_ok=True)
    qb.save_model(str(OUT / f"{ds}_seed{seed}_booster.json"))
    np.save(OUT / f"{ds}_seed{seed}_Xsub.npy", Xte[:2000])
    (OUT / f"{ds}_seed{seed}.json").write_text(json.dumps(dict(dataset=ds, seed=seed, k_units=k_units, n_features=int(Xte.shape[1]),
                                                               selected=sel, refit_test_macro_f1=f1), indent=1))
    print(f"[fit] {ds}/{seed}: {sel['n_estimators']}x{sel['max_depth']} gain_prune={sel['gain_prune']} {sel['leaf_bits']}b "
          f"rounds {sel['rounds']} ({sel['bytes'] / 1024:.1f} KiB); refit F1 equals stored ({f1:.2f})", flush=True)


def time_all():
    import xgboost as xgb
    import r2_gbdt_valsel as V
    from deployment_measure import time_fn, N_WARMUP, N_TIMED
    rows = []
    for ds in DS:
        for seed in SEEDS:
            meta = OUT / f"{ds}_seed{seed}.json"
            if not meta.exists():
                print(f"[skip] {ds}/{seed}: not fitted", flush=True); continue
            info = json.loads(meta.read_text())
            qb = xgb.Booster(); qb.load_model(str(OUT / f"{ds}_seed{seed}_booster.json"))
            qb.set_param({"nthread": 1})
            X = np.load(OUT / f"{ds}_seed{seed}_Xsub.npy")
            cols = [f"f{i}" for i in range(X.shape[1])]
            k = info["k_units"]

            def pred1(x):
                p = qb.predict(xgb.DMatrix(pd.DataFrame(x, columns=cols), nthread=1))
                return p.argmax(1) if k > 1 else (p >= 0.5).astype(np.int64)
            if not np.array_equal(pred1(X), V.predict(qb, X, k)):
                raise SystemExit(f"{ds}/{seed}: one-thread prediction path differs from r2_gbdt_valsel.predict")
            loads, nthr, stop = [], [], threading.Event()

            def sample():
                while not stop.is_set():
                    loads.append(os.getloadavg()[0]); nthr.append(len(os.listdir("/proc/self/task"))); stop.wait(2.0)
            th = threading.Thread(target=sample, daemon=True); th.start()
            lat = time_fn(pred1, X, N_WARMUP, N_TIMED)
            stop.set(); th.join()
            s = info["selected"]
            rows.append(dict(dataset=ds, seed=seed, n_estimators=s["n_estimators"], max_depth=s["max_depth"],
                             gain_prune=s["gain_prune"], leaf_bits=s["leaf_bits"], rounds=s["rounds"], bytes=s["bytes"],
                             n_warmup=N_WARMUP, **lat, load1_min=min(loads), load1_mean=float(np.mean(loads)), load1_max=max(loads), process_threads_max=max(nthr)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "latency.csv", index=False)


def main():
    if sys.argv[1] == "fit":
        for s in (map(int, sys.argv[3:]) if len(sys.argv) > 3 else SEEDS):
            fit_one(sys.argv[2], s)
    else:
        time_all()


if __name__ == "__main__":
    main()
