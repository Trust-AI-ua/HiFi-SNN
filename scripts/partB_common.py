"""Shared helpers for the Part B analyses (results/preregistration/2026-09-28_journal_partB.md).

stamp()        provenance for every record: commit (and whether the tree was dirty),
               a unique run id, the script, UTC time
run_dir()      results/journal_partB/<item>/<run_id>/, never reused (append-only)
load_bundle()  the loader exactly as the stored Tree-to-Spike run used it, with the
               split guard: test size and per-class training counts must equal the
               stored record's
"""
from __future__ import annotations

import datetime
import inspect
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
GP = ["kddcup99", "nslkdd", "ctu13", "iscxids2012", "cicids2017", "cse-cic-ids2018", "unswnb15"]
ALL = IOT + GP
SEEDS = range(42, 47)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS")


def no_thread_caps():
    set_ = [v for v in THREAD_VARS if os.environ.get(v)]
    if set_:
        sys.exit(f"thread variables set ({set_}); the XGBoost teacher depends on them. Unset them.")


def stamp(item: str) -> dict:
    git = lambda *a: subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO).returncode != 0
    return dict(item=item, commit=git("rev-parse", "--short", "HEAD"), dirty=dirty,
                run_id=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6],
                script=Path(sys.argv[0]).name,
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))


def run_dir(st: dict) -> Path:
    d = REPO / "results/journal_partB" / st["item"] / st["run_id"]
    d.mkdir(parents=True, exist_ok=False)
    (d / "provenance.json").write_text(json.dumps(st, indent=1))
    return d


def stored_record(ds: str, seed: int) -> dict:
    return json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/results.json").read_text())


def t2s_config(ds: str, seed: int) -> dict:
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    cfg.update(seed=seed)
    return cfg


def loader_kwargs(ds: str, seed: int) -> dict:
    from src.data import DATASET_LOADERS
    cfg = t2s_config(ds, seed)
    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=seed)
    for o in ("subsample_fraction", "test_fraction", "label_mode"):
        if o in cfg and o in lp:
            kw[o] = cfg[o]
    return kw


def split_signature(n_test, class_counts_train) -> tuple:
    return int(n_test), json.dumps({str(k): int(v) for k, v in class_counts_train.items()}, sort_keys=True)


def load_bundle(ds: str, seed: int):
    """(bundle, arrays) with the split guard against the stored Tree-to-Spike run."""
    from src.data import get_dataset
    from src.training_treespike import _arrays_from_bundle
    b = get_dataset(ds, **loader_kwargs(ds, seed))
    arr = _arrays_from_bundle(b)
    rec = stored_record(ds, seed)
    got = split_signature(len(arr[5]), b.class_counts_train)
    want = split_signature(rec["test_plus"]["n_samples"], rec["data"]["class_counts_train"])
    if got != want:
        sys.exit(f"split guard failed for {ds}/{seed}: {got} vs stored {want}")
    return b, arr


def fit_teacher_guarded(ds: str, seed: int, bundle, arrays):
    """Refit the production teacher and require its test macro-F1 to equal the stored run's
    (teacher_upper_bound in the Tree-to-Spike record). Returns (teacher, present)."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score
    from src.training_treespike import _fit_teacher
    no_thread_caps()
    Xtr, ytr, _, _, Xte, yte = arrays
    teacher, _, present = _fit_teacher(Xtr, ytr, bundle.num_classes, t2s_config(ds, seed))
    pred = present[teacher.predict(pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])]))]
    got = float(f1_score(yte, pred, average="macro", zero_division=0))
    want = stored_record(ds, seed)["test_plus"]["teacher_upper_bound"]["f1_macro"]
    if abs(got - want) > 1e-12:
        sys.exit(f"teacher guard failed for {ds}/{seed}: {got!r} vs stored {want!r}")
    return teacher, present
