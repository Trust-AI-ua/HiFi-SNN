#!/usr/bin/env python3
"""Edge-IIoTset row-layout audit on the published DNN-EdgeIIoT-dataset.csv.

(a) Row validity: semantic constraints every correctly aligned tshark row must
    satisfy (timestamp token in frame.time; IP-or-0 in address columns; 0/1 in
    flag columns; ports <= 65535). Violations are counted per class.
(b) Layout-only separability: a depth-limited decision tree trained on nothing but
    per-column TYPE indicators (empty/zero, integer, decimal, hex, IP, timestamp,
    other text) of all 61 feature columns, and of the 46 retained ones, predicting
    DDoS_UDP vs rest; 3-fold stratified CV.

    python scripts/edgeiiot_layout_audit.py
Output: results/diagnostics/leakage_audit/edgeiiot_layout.json
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.edgeiiot import CSV_REL, DROP_COLUMNS, LABEL_COLUMNS

raw = pd.read_csv(REPO / "dataset/edge-iiotset" / CSV_REL, dtype=str, low_memory=False)
feat_cols = [c for c in raw.columns if c not in LABEL_COLUMNS]
y = raw["Attack_type"].str.strip()
S = {c: raw[c].fillna("").str.strip() for c in feat_cols}

IP = r"^\d{1,3}(\.\d{1,3}){3}$"
TS = r"^\d{4} \d{2}:\d{2}:\d{2}\.\d+$"
num = lambda c: pd.to_numeric(S[c], errors="coerce")
checks = {
    "frame.time_not_timestamp": ~S["frame.time"].str.match(TS),
}
for c in ["ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4", "arp.src.proto_ipv4"]:
    checks[f"{c}_not_ip_or_0"] = ~(S[c].str.match(IP) | S[c].isin(["0", "0.0", ""]))
for c in ["dns.qry.qu", "dns.retransmission", "dns.retransmit_request", "tcp.connection.fin",
          "tcp.connection.rst", "tcp.connection.syn", "tcp.connection.synack", "tcp.flags.ack",
          "mqtt.conflag.cleansess"]:
    v = num(c); checks[f"{c}_not_01"] = v.notna() & ~v.isin([0, 1])
for c in ["udp.port", "tcp.srcport", "tcp.dstport"]:
    v = num(c); checks[f"{c}_not_port"] = S[c].ne("") & (v.isna() | (v > 65535) | (v < 0))
bad_any = np.zeros(len(raw), bool)
per_class = {}
for k, m in checks.items():
    bad_any |= m.to_numpy()
    vc = y[m].value_counts()
    if len(vc):
        per_class[k] = vc.to_dict()
tot = y.value_counts()
bad_by_class = y[bad_any].value_counts()
validity = {cls: {"rows": int(tot[cls]), "invalid": int(bad_by_class.get(cls, 0)),
                  "invalid_pct": round(100 * bad_by_class.get(cls, 0) / tot[cls], 3)} for cls in tot.index}

def typ(s):
    out = np.full(len(s), 6, np.int8)                          # other text
    out[s.str.match(r"^-?\d+\.\d+(e-?\d+)?$").to_numpy()] = 2  # decimal
    out[s.str.match(r"^-?\d+$").to_numpy()] = 1                # integer
    out[s.str.match(r"^0x[0-9a-fA-F]+$").to_numpy()] = 3       # hex
    out[s.str.match(IP).to_numpy()] = 4
    out[s.str.match(TS).to_numpy()] = 5
    out[s.isin(["", "0", "0.0"]).to_numpy()] = 0               # empty / zero
    return out

T = pd.DataFrame({c: typ(S[c]) for c in feat_cols})
retained = [c for c in feat_cols if c not in DROP_COLUMNS]
target = (y == "DDoS_UDP").to_numpy()
res = {}
for name, cols in [("all_61_columns", feat_cols), ("retained_46_columns", retained)]:
    X = pd.get_dummies(T[cols].astype(str)).to_numpy(np.uint8)
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=4, random_state=0), X, target,
                             cv=StratifiedKFold(3, shuffle=True, random_state=0))
    p, r, f, _ = precision_recall_fscore_support(target, pred, average="binary")
    res[name] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
    print(name, res[name], flush=True)

out = {"row_validity_by_class": validity, "violations_by_check": per_class,
       "layout_only_ddos_udp_detector": res}
(REPO / "results/diagnostics/leakage_audit/edgeiiot_layout.json").write_text(json.dumps(out, indent=1))
print(json.dumps(validity, indent=0))
print(json.dumps(per_class, indent=0))
