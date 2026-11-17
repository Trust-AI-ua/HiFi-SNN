#!/usr/bin/env python3
"""B1, step 1: per-packet fields from one Edge-IIoTset capture (results/preregistration/2026-11-17_neuro_stream.md).

tshark 4.2.2, IP packets only, first occurrence of every field (an ICMP error quotes an inner IP
header; the outer one is kept). Name resolution off.

    python scripts/stream_extract.py <pcap>
Output: dataset/edge-iiotset/_stream/packets/<capture name>.parquet with
  t (float64 epoch s), src, dst (str), proto (IP protocol), sport, dport (TCP or UDP, else -1),
  length (IP total length), flags (TCP flags byte, else 0), frame (frame number)
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "dataset/edge-iiotset/_stream/packets"
FIELDS = ["frame.number", "frame.time_epoch", "ip.src", "ip.dst", "ip.proto", "tcp.srcport", "tcp.dstport",
          "udp.srcport", "udp.dstport", "ip.len", "tcp.flags"]


def main():
    pcap = Path(sys.argv[1])
    out = OUT / f"{pcap.stem}.parquet"
    if out.exists():
        print(f"[skip] {out}"); return
    cmd = ["tshark", "-n", "-r", str(pcap), "-Y", "ip", "-T", "fields", "-E", "separator=\t",
           "-E", "occurrence=f"] + sum([["-e", f] for f in FIELDS], [])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    parts = []
    names = ["frame", "t", "src", "dst", "proto", "tsp", "tdp", "usp", "udp", "length", "flags"]
    for chunk in pd.read_csv(p.stdout, sep="\t", header=None, names=names, dtype=str, chunksize=2_000_000,
                             keep_default_na=False):
        def num(c, fill=-1):
            return pd.to_numeric(chunk[c].replace("", np.nan), errors="coerce").fillna(fill).astype(np.int64)
        d = pd.DataFrame({
            "frame": num("frame"), "t": pd.to_numeric(chunk["t"], errors="coerce"),
            "src": chunk["src"].astype("category"), "dst": chunk["dst"].astype("category"),
            "proto": num("proto").astype(np.int16),
            "sport": np.where(chunk["tsp"] != "", num("tsp"), num("usp")).astype(np.int32),
            "dport": np.where(chunk["tdp"] != "", num("tdp"), num("udp")).astype(np.int32),
            "length": num("length", 0).astype(np.int32),
            "flags": chunk["flags"].map(lambda s: int(s, 16) if s else 0).astype(np.int16)})
        parts.append(d)
    err = p.stderr.read().decode(errors="replace")
    truncated = False
    if p.wait() != 0:
        # A capture damaged part-way (one record header is corrupt): tshark has emitted every
        # packet before the damage. Keep them and record the truncation beside the output.
        if "damaged or corrupt" not in err or not parts:
            raise SystemExit(f"tshark failed on {pcap}: {err[-2000:]}")
        truncated = True
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=names)
    for c in ("src", "dst"):
        d[c] = d[c].astype(str).astype("category")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    d.to_parquet(tmp, index=False)
    tmp.rename(out)
    if truncated:
        import json
        out.with_suffix(".truncated.json").write_text(json.dumps(dict(
            capture=str(pcap.resolve().relative_to(REPO)), packets_read=int(len(d)), last_frame=int(d["frame"].max()),
            tshark_error=err.strip()), indent=1))
    print(f"[done] {pcap.name}: {len(d):,} IP packets, {d['t'].isna().sum()} without time"
          f"{'; tshark stderr: ' + err.strip()[-300:] if err.strip() else ''}", flush=True)


if __name__ == "__main__":
    main()
