#!/usr/bin/env python3
"""Does the layout-only detector isolate the Edge-IIoTset artifact?

Same detector in three settings (depth-4 decision tree on per-column TYPE
indicators only, UDP-flood class vs rest, 3-fold stratified CV):

  1. Edge-IIoTset as published       (DDoS_UDP vs rest)
  2. Edge-IIoTset after src/data/repair.py (DDoS_UDP vs rest)
  3. CICIoT2023, a clean extraction  (DDoS-UDP_Flood vs rest)

Settings 1 and 2 use the identical column set: the features the loader keeps
after the repair (retained columns minus http.referer and dns.qry.name), and the
identical rows (MITM removed from both). If setting 2 and 3 are also near 1.0,
type-level separability is a property of UDP floods, not of the artifact, and the
metric must not be used as evidence of the artifact.

    python scripts/layout_detector_control.py
Output: results/diagnostics/leakage_audit/layout_detector_control.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data import repair as R  # noqa: E402
from src.data.edgeiiot import CSV_REL, DROP_COLUMNS  # noqa: E402

IP = r"^\d{1,3}(\.\d{1,3}){3}$"
TS = r"^\d{4} \d{2}:\d{2}:\d{2}\.\d+$"


def typ(s: pd.Series) -> np.ndarray:
    s = s.fillna("").astype(str).str.strip()
    out = np.full(len(s), 6, np.int8)                          # other text
    out[s.str.match(r"^-?\d+\.\d+(e-?\d+)?$").to_numpy()] = 2  # decimal
    out[s.str.match(r"^-?\d+$").to_numpy()] = 1                # integer
    out[s.str.match(r"^0x[0-9a-fA-F]+$").to_numpy()] = 3       # hex
    out[s.str.match(IP).to_numpy()] = 4
    out[s.str.match(TS).to_numpy()] = 5
    out[s.isin(["", "0", "0.0"]).to_numpy()] = 0               # empty / zero
    return out


def detect(df: pd.DataFrame, cols, target) -> dict:
    T = pd.DataFrame({c: typ(df[c]) for c in cols})
    X = pd.get_dummies(T.astype(str)).to_numpy(np.uint8)
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=4, random_state=0), X, target,
                             cv=StratifiedKFold(3, shuffle=True, random_state=0))
    p, r, f, _ = precision_recall_fscore_support(target, pred, average="binary")
    return {"precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(f), 4), "n": int(len(target)), "positives": int(target.sum())}


def main():
    res = {}
    raw = pd.read_csv(REPO / "dataset/edge-iiotset" / CSV_REL, dtype=str, low_memory=False)
    fixed, prov = R.repair_edgeiiot(raw)
    raw = raw.loc[fixed.index]                                 # same rows (MITM removed)
    cols = [c for c in R.EDGE_FEATURES if c not in DROP_COLUMNS + R.EDGE_EXTRA_DROP]
    y = (raw["Attack_type"].str.strip() == "DDoS_UDP").to_numpy()
    res["edgeiiot_published"] = detect(raw, cols, y)
    print("published", res["edgeiiot_published"], flush=True)
    res["edgeiiot_repaired"] = detect(fixed, cols, y)
    print("repaired ", res["edgeiiot_repaired"], flush=True)

    shards = sorted(glob.glob(str(REPO / "dataset/cic-iot2023/csv/*.csv")))[:10]
    ciot = pd.concat([pd.read_csv(f, dtype=str) for f in shards], ignore_index=True)
    if len(ciot) > 1_000_000:
        ciot = ciot.sample(1_000_000, random_state=0)
    ccols = [c for c in ciot.columns if c != "label"]
    res["ciciot2023_clean"] = detect(ciot, ccols, (ciot["label"] == "DDoS-UDP_Flood").to_numpy())
    res["ciciot2023_clean"]["shards"] = len(shards)
    print("ciciot   ", res["ciciot2023_clean"], flush=True)
    out = REPO / "results/diagnostics/leakage_audit/layout_detector_control.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
