#!/usr/bin/env python3
"""Verify the Edge-IIoTset row repair (src/data/repair.py) against the pcaps.

Every decoded row of the published DNN-EdgeIIoT-dataset.csv is matched to its
packet in the shipped captures by (ip.src_host, ip.dst_host, udp.port,
udp.stream) and compared field by field with tshark's own extraction of the same
61 fields (first occurrence of multi-valued fields, as the repair canonicalises).

  DDoS_UDP  (decoded_udp_flood, decoded_icmp_reply)  <- Attack traffic/DDoS UDP Flood Attacks.pcap
  Normal UDP (decoded_normal_udp)                    <- Normal traffic/*/*.pcap (udp only)

Comparison per field: empty == 0; True/False == 1/0 (boolean rendering differs
between tshark versions); hex compared as integers; numbers with
relative tolerance 1e-6; text exactly after stripping. ``frame.time`` is not
compared (unrecoverable for the DDoS_UDP layouts, and the published date is
truncated for every row). The upstream-truncated Modbus fields are compared to
tshark (expected empty). The published (unrepaired) values of the same rows are
scored against the same packets, giving a per-field before/after.

    python scripts/verify_edgeiiot_repair.py [--skip-normal]
Outputs: results/diagnostics/leakage_audit/edgeiiot_repair_verification.{json,csv}
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import repair as R  # noqa: E402
from src.data.edgeiiot import CSV_REL  # noqa: E402

E = REPO / "dataset/edge-iiotset"
OUT = REPO / "results/diagnostics/leakage_audit"
KEY = ["ip.src_host", "ip.dst_host", "udp.port", "udp.stream"]


def norm(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    s = str(x).strip()
    if s in ("", "0", "0.0", "False"):     # tshark >= 4 prints booleans as True/False,
        return 0.0                          # the published extraction as 1/0
    if s == "True":
        return 1.0
    if s.lower().startswith("0x"):
        try:
            return float(int(s, 16))
        except ValueError:
            return s
    try:
        return float(s)
    except ValueError:
        return s


def same(a, b):
    a, b = norm(a), norm(b)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))
    return a == b


def key_of(vals):
    k = []
    for c in KEY:
        v = norm(vals[c])
        k.append(v if isinstance(v, str) else float(v))
    return tuple(k)


def tshark_rows(pcap, display_filter=None):
    cmd = ["tshark", "-r", str(pcap), "-T", "fields", "-E", "separator=\t",
           "-E", "occurrence=f"]
    if display_filter:
        cmd += ["-Y", display_filter]
    for f in R.EDGE_FEATURES:
        cmd += ["-e", f]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in p.stdout:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == len(R.EDGE_FEATURES):
            yield dict(zip(R.EDGE_FEATURES, parts))
    p.wait()


def verify_candidates(rows, pcaps, display_filter, skip_fields, published):
    """For flows with several packets under one key: a row is verified if SOME
    packet of its flow matches it on every field. Per-field rates use, for each
    row, the best-matching candidate (chosen separately for published values)."""
    want = defaultdict(list)
    for i, r in zip(rows.index, rows.to_dict("records")):
        want[key_of(r)].append(i)
    cands = defaultdict(list)
    for pcap in pcaps:
        for pkt in tshark_rows(pcap, display_filter):
            k = key_of(pkt)
            if k in want and len(cands[k]) < 5000:
                cands[k].append(pkt)
    fields = [f for f in R.EDGE_FEATURES if f not in skip_fields]
    rep = {f: 0 for f in fields}
    pub = {f: 0 for f in fields}
    n = full = unmatched = 0
    sizes = []
    for k, idx in want.items():
        if not cands.get(k):
            unmatched += len(idx)
            continue
        for i in idx:
            n += 1
            sizes.append(len(cands[k]))
            for vals, acc in ((rows.loc[i], rep), (published.loc[i], pub)):
                best = max(cands[k], key=lambda p: sum(same(vals[f], p[f]) for f in fields))
                hits = [same(vals[f], best[f]) for f in fields]
                for f, h in zip(fields, hits):
                    acc[f] += h
                if acc is rep:
                    full += all(hits)
    sizes = pd.Series(sizes)
    return dict(rows=len(rows), matched=n, unmatched=unmatched, rows_all_fields_match=full,
                candidates_per_row=dict(median=float(sizes.median()) if len(sizes) else None,
                                        max=int(sizes.max()) if len(sizes) else None),
                per_field={f: dict(match=rep[f], n=n, rate=rep[f] / n if n else None,
                                   published_match=pub[f],
                                   published_rate=pub[f] / n if n else None) for f in fields})


def verify(rows: pd.DataFrame, pcaps, display_filter, skip_fields, published=None):
    """Match repaired ``rows`` to packets; score repaired values and, if given,
    the published (unrepaired) values of the same rows against the packet."""
    want = defaultdict(list)
    for i, r in zip(rows.index, rows.to_dict("records")):
        want[key_of(r)].append(i)
    found = {}
    ambiguous = set()
    for pcap in pcaps:
        for pkt in tshark_rows(pcap, display_filter):
            k = key_of(pkt)
            if k in want:
                if k in found and any(not same(found[k][f], pkt[f]) for f in R.EDGE_FEATURES
                                      if f not in skip_fields):
                    ambiguous.add(k)
                found.setdefault(k, pkt)
    per_field = {f: [0, 0] for f in R.EDGE_FEATURES if f not in skip_fields}
    pub_field = {f: 0 for f in per_field}
    matched_rows = 0
    rows_all_fields = 0
    for k, idx in want.items():
        if k not in found or k in ambiguous:
            continue
        pkt = found[k]
        for i in idx:
            matched_rows += 1
            r = rows.loc[i]
            ok_all = True
            for f in per_field:
                ok = same(r[f], pkt[f])
                per_field[f][0] += ok
                per_field[f][1] += 1
                ok_all &= ok
                if published is not None:
                    pub_field[f] += same(published.at[i, f], pkt[f])
            rows_all_fields += ok_all
    return dict(rows=len(rows), matched=matched_rows,
                ambiguous_rows=sum(len(want[k]) for k in ambiguous),
                unmatched=len(rows) - matched_rows - sum(len(want[k]) for k in ambiguous),
                rows_all_fields_match=rows_all_fields,
                per_field={f: dict(match=m, n=n, rate=(m / n if n else None),
                                   published_match=(pub_field[f] if published is not None else None),
                                   published_rate=(pub_field[f] / n if (n and published is not None)
                                                   else None))
                           for f, (m, n) in per_field.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-normal", action="store_true")
    ap.add_argument("--only-normal", action="store_true",
                    help="rerun the Normal part only, merging into the existing report")
    a = ap.parse_args()
    raw = pd.read_csv(E / CSV_REL, dtype=str, low_memory=False)
    prov = R.classify_layout(raw)
    fixed = R.realign_rows(raw, prov)
    report = {"provenance_counts": prov.value_counts().to_dict()}
    rpt = OUT / "edgeiiot_repair_verification.json"
    if a.only_normal and rpt.exists():
        report.update({k: v for k, v in json.loads(rpt.read_text()).items()
                       if k in ("decoded_udp_flood", "decoded_icmp_reply")})

    for lay in ([] if a.only_normal else ["decoded_udp_flood", "decoded_icmp_reply"]):
        sub = fixed[(prov == lay).to_numpy()]
        report[lay] = verify(sub, [E / "Attack traffic/DDoS UDP Flood Attacks.pcap"],
                             "udp", skip_fields={"frame.time"}, published=raw.loc[sub.index])
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_field"}
                      for k, v in report.items() if k != "provenance_counts"}, indent=1), flush=True)
    if not a.skip_normal:
        normal = fixed[(prov == "decoded_normal_udp").to_numpy()]
        pcaps = sorted(glob.glob(str(E / "Normal traffic/*/*.pcap")))
        report["decoded_normal_udp"] = verify_candidates(
            normal, pcaps, "udp", skip_fields={"frame.time"}, published=raw.loc[normal.index])
        print(json.dumps({kk: vv for kk, vv in report["decoded_normal_udp"].items()
                          if kk != "per_field"}, indent=1), flush=True)

    (OUT / "edgeiiot_repair_verification.json").write_text(json.dumps(report, indent=1))
    recs = []
    for part in ("decoded_udp_flood", "decoded_icmp_reply", "decoded_normal_udp"):
        if part in report:
            for f, d in report[part]["per_field"].items():
                recs.append(dict(layout=part, field=f, **d))
    pd.DataFrame(recs).to_csv(OUT / "edgeiiot_repair_verification.csv", index=False)
    print("[wrote] edgeiiot_repair_verification.{json,csv}")


if __name__ == "__main__":
    main()
