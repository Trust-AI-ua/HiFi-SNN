#!/usr/bin/env python3
"""Feature-level leakage audit across all eleven benchmarks (read-only, CPU).

Triggered by the CSE-CIC-IDS2018 finding: 02-20-2018.csv carries four columns
(Flow ID, Src IP, Src Port, Dst IP) that the other nine daily files lack; the
loader concatenates the files, so those columns are zero-padded elsewhere and
Src Port becomes a capture-day indicator that rules out r2l/u2r.

Layer A (schema): for every loader that stacks several source files, the
  columns that are not present in every file. Such a column is filled with the
  pad value for the files that lack it, so its "is padded" pattern equals a
  source-file indicator.

Layer B (preprocessed arrays, what the models actually see): on the seed-42
  training split built with each dataset's Tree-to-Spike loader kwargs, for
  every feature j the indicator a_j = [x_j > min_j] ("off its floor") and
    * nmi      I(a_j; y) / H(y)
    * excl     classes (>= 100 train rows) with P(a_j | c) == 0 while
               0.05 <= P(a_j) <= 0.95  -> the indicator rules those classes out
    * incl     classes with P(a_j | c) == 1 under the same condition
    * constant features (never off the floor)
  Layer B flags consequences whatever their cause; a flagged feature is not
  leakage by itself (legitimate flags also zero out for some attacks), so every
  flag is reviewed by name.

    python scripts/leakage_feature_audit.py [--datasets ...]
Outputs: results/diagnostics/leakage_audit/schema.csv
         results/diagnostics/leakage_audit/features.csv
"""
from __future__ import annotations

import argparse
import glob
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mutual_info_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import get_dataset, DATASET_LOADERS  # noqa: E402
from src.training_treespike import _arrays_from_bundle  # noqa: E402

OUT = REPO / "results/diagnostics/leakage_audit"
DATASETS = ["kddcup99", "nslkdd", "unswnb15", "cicids2017", "cse-cic-ids2018",
            "iscxids2012", "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
D = REPO / "dataset"


def _source_files(ds):
    from src.data import cicids2017, cseids2018, iscxids2012
    if ds == "cicids2017":
        return [D / "cicids2017" / f for f in cicids2017.CIC_FILES], "csv"
    if ds == "cse-cic-ids2018":
        return [D / "cse-cic-ids2018" / f for f in cseids2018.CSE_FILES], "csv"
    if ds == "iscxids2012":
        base = D / "iscxids2012" / iscxids2012.CSV_SUBDIR
        return [base / f for f in iscxids2012.ISCX_FILES], "csv"
    if ds == "ciciot2023":
        return [Path(p) for p in sorted(glob.glob(str(D / "cic-iot2023/csv/*.csv")))], "csv"
    if ds == "ciciomt2024":
        return [Path(p) for p in sorted(glob.glob(str(D / "cic-iomt-2024/csv/*/*.csv")))], "csv"
    if ds == "ctu13":
        return [Path(p) for p in sorted(glob.glob(str(D / "ctu13/binetflow/scenario-*-*.binetflow")))], "csv"
    if ds == "unswnb15":
        return [D / "unsw-nb15" / "UNSW_NB15_training-set.csv",
                D / "unsw-nb15" / "UNSW_NB15_testing-set.csv"], "csv"
    if ds == "kddcup99":
        return [D / "kddcup99" / "kddcup.data_10_percent_corrected",
                D / "kddcup99" / "corrected" / "corrected"], "headerless"
    if ds == "nslkdd":
        return sorted(p for p in (D / "nslkdd").glob("KDD*.txt")), "headerless"
    return [], None      # single-file loaders (toniot, edgeiiot)


def layer_a(ds):
    files, kind = _source_files(ds)
    files = [f for f in files if f.exists()]
    if len(files) < 2:
        return [dict(dataset=ds, n_files=len(files), column="", present_in="", n_present=None,
                     note="single source file (or none found)")]
    cols = {}
    for f in files:
        if kind == "headerless":
            with open(f) as fh:
                cols[f.name] = [f"col{i}" for i in range(len(fh.readline().rstrip("\n").split(",")))]
        else:
            enc = "utf-8-sig"
            try:
                cols[f.name] = [c.strip() for c in pd.read_csv(f, nrows=0, encoding=enc).columns]
            except UnicodeDecodeError:
                cols[f.name] = [c.strip() for c in pd.read_csv(f, nrows=0, encoding="latin-1").columns]
    union = []
    for c in cols.values():
        union += [x for x in c if x not in union]
    rows = []
    for c in union:
        have = [n for n, cs in cols.items() if c in cs]
        if len(have) < len(files):
            rows.append(dict(dataset=ds, n_files=len(files), column=c,
                             present_in=";".join(have), n_present=len(have), note="not in every file"))
    if not rows:
        rows.append(dict(dataset=ds, n_files=len(files), column="", present_in="",
                         n_present=len(files), note="identical schema across files"))
    return rows


def _cfg(ds):
    p = REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"
    return yaml.safe_load(open(p))


def layer_b(ds, max_rows=1_000_000):
    cfg = _cfg(ds)
    params = inspect.signature(DATASET_LOADERS[ds]).parameters
    kw = dict(batch_size=cfg.get("batch_size", 128), val_fraction=cfg.get("val_fraction", 0.1), seed=42)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in params:
            kw[opt] = cfg[opt]
    b = get_dataset(ds, **kw)
    X, y = _arrays_from_bundle(b)[:2]
    if len(y) > max_rows:
        idx = np.random.default_rng(0).choice(len(y), max_rows, replace=False)
        X, y = X[idx], y[idx]
    names = b.class_names
    classes, counts = np.unique(y, return_counts=True)
    hy = mutual_info_score(y, y)
    rows = []
    for j in range(X.shape[1]):
        a = X[:, j] > X[:, j].min()
        p = a.mean()
        per = {int(c): a[y == c].mean() for c in classes}
        big = [int(c) for c, n in zip(classes, counts) if n >= 100]
        mid = 0.05 <= p <= 0.95
        excl = [names[c] for c in big if mid and per[c] == 0.0]
        incl = [names[c] for c in big if mid and per[c] == 1.0]
        rows.append(dict(dataset=ds, feature=j, n_features=X.shape[1], frac_active=p,
                         nmi=(mutual_info_score(a, y) / hy) if p not in (0.0, 1.0) else 0.0,
                         constant=bool(X[:, j].min() == X[:, j].max()),
                         excludes=";".join(excl), includes=";".join(incl),
                         per_class_active=";".join(f"{names[c]}={per[c]:.3f}" for c in classes)))
    del X, y, b
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sa, fb = [], []
    for ds in a.datasets:
        sa += layer_a(ds)
        pd.DataFrame(sa).to_csv(OUT / "schema.csv", index=False)
        print(f"[A] {ds}: " + "; ".join(f"{r['column']} ({r['n_present']}/{r['n_files']})"
                                         if r["column"] else r["note"] for r in sa if r["dataset"] == ds),
              flush=True)
        rb = layer_b(ds)
        fb += rb
        pd.DataFrame(fb).to_csv(OUT / "features.csv", index=False)
        df = pd.DataFrame(rb)
        flag = df[(df.excludes != "") | (df.includes != "")]
        print(f"[B] {ds}: {len(df)} features, {int(df.constant.sum())} constant, "
              f"{len(flag)} class-excluding/including; top nmi: "
              + ", ".join(f"f{r.feature}={r.nmi:.2f}" for r in df.nlargest(5, "nmi").itertuples()),
              flush=True)


if __name__ == "__main__":
    main()
