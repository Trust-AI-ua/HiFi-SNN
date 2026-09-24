#!/usr/bin/env python3
"""Loader provenance for runs made before trainers recorded their loader kwargs.

Runs before 2026-09-24 did not store subsample_fraction, test_fraction or label_mode
in results.json, so the config files are their only record. This writes, per model,
dataset and seed, the value each run's loader received: the model's config value, or
the loader's own default where the config omits it. It adds the config file's last
commit (the value cannot have changed after that) and the run's recorded split
evidence (test size, per-class training counts where stored), and checks that all
four models agree per dataset and seed. Existing results.json are NOT modified.

    python scripts/loader_provenance.py
Output: results/diagnostics/loader_provenance.csv
"""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATASETS = ["kddcup99", "nslkdd", "unswnb15", "cicids2017", "cse-cic-ids2018", "iscxids2012",
            "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
MODELS = {"Tree-to-Spike": ("experiments/Tree-to-Spike/{d}/config.yaml", "results/Tree-to-Spike/{d}/seed_{s}/results.json"),
          "DNN-Tree": ("experiments/DNN-Tree/{d}/config.yaml", "results/DNN-Tree/{d}/seed_{s}/results.json"),
          "Vanilla SNN": ("configs/{d}/SNN_LeakyParallel/latency/config.yaml", "results/{d}/SNN_LeakyParallel/latency/seed_{s}/results.json"),
          "TASNN": ("experiments/TASNN/{d}/config.yaml", "results/TASNN/{d}/seed_{s}/results.json")}


def main():
    from src.data import DATASET_LOADERS
    rows = []
    for ds in DATASETS:
        params = inspect.signature(DATASET_LOADERS[ds]).parameters
        default = {k: (params[k].default if k in params else "n/a") for k in ("subsample_fraction", "test_fraction", "label_mode")}
        for model, (cp, rp) in MODELS.items():
            cfgp = REPO / cp.format(d=ds)
            cfg = yaml.safe_load(open(cfgp))
            last = subprocess.check_output(["git", "log", "-1", "--format=%h %ad", "--date=short", "--", str(cfgp)],
                                           cwd=REPO, text=True).strip()
            for s in range(42, 47):
                r = json.loads((REPO / rp.format(d=ds, s=s)).read_text())
                cc = r.get("data", {}).get("class_counts_train")
                lk = r.get("loader_kwargs")          # recorded by runs made after 2026-09-24
                val = {k: (lk.get(k, default[k]) if lk else (cfg[k] if k in cfg and k in params else default[k]))
                       for k in default}
                src = "recorded loader_kwargs" if lk else \
                    ("config" if "subsample_fraction" in cfg and "subsample_fraction" in params else "loader default")
                rows.append(dict(model=model, dataset=ds, seed=s, **val, source=src,
                                 config_last_commit=last,
                                 test_n=r["test_plus"].get("n_samples") or (r["data"].get("splits") or {}).get("test_plus"),
                                 train_counts_sha=hashlib.sha256(json.dumps(cc, sort_keys=True).encode()).hexdigest()[:12] if cc else None))
    d = pd.DataFrame(rows)
    bad = []
    for (ds, s), g in d.groupby(["dataset", "seed"]):
        for col in ("subsample_fraction", "test_fraction", "label_mode", "test_n"):
            if g[col].astype(str).nunique() > 1:
                bad.append(f"{ds}/{s}: {col} differs across models: {dict(zip(g.model, g[col]))}")
        if g.train_counts_sha.dropna().nunique() > 1:
            bad.append(f"{ds}/{s}: training class counts differ across models")
    d.to_csv(REPO / "results/diagnostics/loader_provenance.csv", index=False)
    print(d.groupby("dataset")[["subsample_fraction", "test_fraction", "label_mode"]].first().to_string())
    print(f"\n{len(d)} runs; cross-model disagreements: {len(bad)}")
    for b in bad:
        print("  " + b)
    print("[wrote] results/diagnostics/loader_provenance.csv")


if __name__ == "__main__":
    main()
