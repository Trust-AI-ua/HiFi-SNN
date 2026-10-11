#!/usr/bin/env python3
"""Reproduction guards for the B6 CICIoT2023 block code path (amendment A1), before any block run.

For each trainer (Tree-to-Spike family, DNN-Tree/DNN-TQ, Vanilla SNN, TASNN) on TON-IoT
seed 42, against the stored run:
  off  select_on_val and record_precisions off, 10 epochs: the test macro-F1 and the whole
       training curve (train loss, validation accuracy per epoch) equal the stored run's.
  on   select_on_val and record_precisions on, 10 epochs: the training curve equals the
       stored run's (the selector only observes); the captured precision that matches the
       pipeline's own test pass (per-tensor int8 for the quantized students, fp32 for the
       Vanilla SNN and TASNN) equals it; the selected epoch is recorded.

    python scripts/partB_block_guard.py
Output: results/journal_partB/B6_block_guard/<run_id>/{guard.json,<trainer>_<mode>/...}
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

DS, SEED = "toniot", 42
STORED = {"t2s": "results/Tree-to-Spike/{d}/seed_{s}/results.json",
          "dnn": "results/DNN-Tree/{d}/seed_{s}/results.json",
          "vanilla": "results/{d}/SNN_LeakyParallel/latency/seed_{s}/results.json",
          "tasnn": "results/TASNN/{d}/seed_{s}/results.json"}


def base_cfg(trainer):
    rec = json.loads((C.REPO / STORED[trainer].format(d=DS, s=SEED)).read_text())
    if trainer == "t2s":
        cfg = C.t2s_config(DS, SEED)
    elif trainer == "dnn":
        cfg = yaml.safe_load(open(C.REPO / f"experiments/DNN-Tree/{DS}/config.yaml"))
    elif trainer == "vanilla":
        cfg = yaml.safe_load(open(C.REPO / f"configs/{DS}/SNN_LeakyParallel/latency/config.yaml"))
        cfg.update(rec["config"])
    else:
        cfg = yaml.safe_load(open(C.REPO / f"experiments/TASNN/{DS}/config.yaml"))
        cfg.update(rec["config"])
    cfg.update(seed=SEED)
    return cfg, rec


def module(trainer):
    import importlib
    return importlib.import_module({"t2s": "src.training_treespike", "dnn": "src.training_dnn",
                                    "vanilla": "src.training", "tasnn": "src.training_tasnn"}[trainer])


def run(trainer, out):
    C.no_thread_caps()
    res = {}
    for mode in ("off", "on"):
        cfg, stored = base_cfg(trainer)
        d = Path(out) / f"{trainer}_{mode}"
        cfg.update(epochs=10, force=True, results_dir=str(d), summary_csv=str(d / "summary.csv"),
                   select_on_val=(mode == "on"), record_precisions=(mode == "on"))
        rec = module(trainer).run_experiment(cfg)
        a, b = rec["training"]["training_curve"], stored["training"]["training_curve"]
        curve = len(a) == len(b) and all(x["train_loss"] == y["train_loss"] and x["val_accuracy"] == y["val_accuracy"]
                                         for x, y in zip(a, b))
        f1, f1s = rec["test_plus"]["metrics"]["f1_macro"], stored["test_plus"]["metrics"]["f1_macro"]
        r = dict(curve_identical=bool(curve), test_f1=f1, stored_test_f1=f1s)
        if mode == "off":
            r["ok"] = bool(curve and f1 == f1s)
        else:
            sel = rec["training"]["selection"]
            key = "per_tensor_int8" if trainer in ("t2s", "dnn") else "fp32"
            prec = rec["test_plus"]["precisions"][key]["metrics"]["f1_macro"]
            r.update(selected_epoch=sel["selected_epoch"], precision_matches_pipeline=bool(prec == f1),
                     test_equal_if_epoch10=(f1 == f1s) if sel["selected_epoch"] == 10 else None)
            r["ok"] = bool(curve and prec == f1 and (sel["selected_epoch"] != 10 or f1 == f1s))
        res[mode] = r
        print(f"[{trainer} {mode}] {r}", flush=True)
    return trainer, res


def main():
    st = C.stamp("B6_block_guard")
    out = C.run_dir(st)
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = dict(ex.map(run, ["t2s", "dnn", "vanilla", "tasnn"], [str(out)] * 4))
    ok = all(r["ok"] for t in results.values() for r in t.values())
    (out / "guard.json").write_text(json.dumps(dict(stamp=st, ok=ok, results=results), indent=1, default=str))
    print("ALL BLOCK GUARDS PASS" if ok else "BLOCK GUARD FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
