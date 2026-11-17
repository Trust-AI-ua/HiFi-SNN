#!/usr/bin/env python3
"""B1, step 2: the Edge-IIoTset stream dataset (results/preregistration/2026-11-17_neuro_stream.md,
amendment 1 for the labels).

Input: dataset/edge-iiotset/_stream/packets/*.parquet (scripts/stream_extract.py).
Flows: bidirectional (unordered endpoint pair, ports and IP protocol), a new flow after 120 s of
idle time, first K = 32 packets. Labels: amendment 1. At most 100,000 flows per class (rng 0).
Features after every packet (19, cumulative, all incrementally updatable): FEATURES below.

    python scripts/stream_build.py
Output: dataset/edge-iiotset/_stream/stream_K32.npz
          X [N, 32, 19] float32 (steps after the flow's last packet repeat its final state)
          length [N], y [N], capture [N] (index into captures), t0 [N] (flow start, epoch s)
        results/neuro/b1_stream/{endpoints.txt, class_counts.csv, build.json}
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PK = REPO / "dataset/edge-iiotset/_stream/packets"
OUT = REPO / "dataset/edge-iiotset/_stream/stream_K32.npz"
REP = REPO / "results/neuro/b1_stream"
K, IDLE, CAP = 32, 120.0, 100_000
VICTIM, ATTACKER = "192.168.0.128", "192.168.0.170"
CLASSES = ['Normal', 'Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP', 'Fingerprinting', 'Password',
           'Port_Scanning', 'Ransomware', 'SQL_injection', 'Uploading', 'Vulnerability_scanner', 'XSS']
NORMAL = ["Distance", "Flame_Sensor", "Heart_Rate", "IR_Receiver", "Modbus", "phValue", "Soil_Moisture",
          "Sound_Sensor", "Temperature_and_Humidity", "Water_Level"]
ATTACK = {"Backdoor_attack": "Backdoor", "DDoS HTTP Flood Attacks": "DDoS_HTTP", "DDoS ICMP Flood Attacks": "DDoS_ICMP",
          "DDoS TCP SYN Flood Attacks": "DDoS_TCP", "DDoS UDP Flood Attacks": "DDoS_UDP",
          "OS Fingerprinting attack": "Fingerprinting", "Password attacks": "Password",
          "Port Scanning attack": "Port_Scanning", "Ransomware attack": "Ransomware",
          "SQL injection attack": "SQL_injection", "Uploading attack": "Uploading",
          "Vulnerability scanner attack": "Vulnerability_scanner", "XSS attacks": "XSS"}
FLOODS = {"DDoS ICMP Flood Attacks", "DDoS TCP SYN Flood Attacks", "DDoS UDP Flood Attacks"}
FEATURES = ["pkt_count", "bytes", "len_min", "len_mean", "len_max", "len_sd", "fwd_share", "bwd_bytes",
            "iat_min", "iat_mean", "iat_max", "iat_sd", "duration", "syn", "ack", "fin", "rst", "psh", "proto"]
PROTO = {6: 1, 17: 2, 1: 3}            # TCP, UDP, ICMP; anything else 0


def flows_of(d: pd.DataFrame):
    """Sort packets into flows. Returns d sorted with columns fid (flow id) and pos (index in flow)."""
    src, dst = d["src"].astype(str).to_numpy(), d["dst"].astype(str).to_numpy()
    a_first = (src < dst) | ((src == dst) & (d["sport"].to_numpy() <= d["dport"].to_numpy()))
    ipA = np.where(a_first, src, dst); ipB = np.where(a_first, dst, src)
    pA = np.where(a_first, d["sport"], d["dport"]); pB = np.where(a_first, d["dport"], d["sport"])
    codes, uniq = pd.factorize(pd.Series(np.concatenate([ipA, ipB])))
    n = len(d)
    cA, cB = codes[:n], codes[n:]
    t = d["t"].to_numpy()
    order = np.lexsort((d["frame"].to_numpy(), t, d["proto"].to_numpy(), pB, cB, pA, cA))
    d = d.iloc[order].reset_index(drop=True)
    cA, cB, pA, pB, t = cA[order], cB[order], pA[order], pB[order], t[order]
    proto = d["proto"].to_numpy()
    new_key = np.ones(n, bool)
    new_key[1:] = (cA[1:] != cA[:-1]) | (cB[1:] != cB[:-1]) | (pA[1:] != pA[:-1]) | (pB[1:] != pB[:-1]) | (proto[1:] != proto[:-1])
    gap = np.zeros(n); gap[1:] = t[1:] - t[:-1]
    start = new_key | (gap > IDLE)
    fid = np.cumsum(start) - 1
    pos = np.arange(n) - np.flatnonzero(start)[fid]
    d["fid"], d["pos"] = fid, pos
    return d


def features(d: pd.DataFrame):
    """d: packets of the kept flows, sorted by (fid, pos), pos < K. Returns X [F, K, 19], length [F], fid order."""
    g = d.groupby("fid", sort=True)
    L = d["length"].to_numpy(np.float64)
    t = d["t"].to_numpy(np.float64)
    first_src = g["src"].transform("first").astype(str).to_numpy()
    fwd = (d["src"].astype(str).to_numpy() == first_src).astype(np.float64)
    fl = d["flags"].to_numpy(np.int64)
    cnt = d["pos"].to_numpy() + 1.0
    gk = d["fid"].to_numpy()
    cs = lambda x: pd.Series(x).groupby(gk).cumsum().to_numpy()
    cmin = lambda x: pd.Series(x).groupby(gk).cummin().to_numpy()
    cmax = lambda x: pd.Series(x).groupby(gk).cummax().to_numpy()
    s1, s2 = cs(L), cs(L * L)
    t0 = g["t"].transform("first").to_numpy()
    iat = np.where(d["pos"].to_numpy() > 0, t - np.r_[t[0], t[:-1]], np.nan)
    ni = cnt - 1
    i1, i2 = cs(np.nan_to_num(iat)), cs(np.nan_to_num(iat) ** 2)
    iat_min = pd.Series(iat).groupby(gk).cummin().fillna(0).to_numpy()
    iat_max = pd.Series(iat).groupby(gk).cummax().fillna(0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        imean = np.where(ni > 0, i1 / ni, 0.0)
        isd = np.where(ni > 1, np.sqrt(np.maximum(i2 / ni - imean ** 2, 0)), 0.0)
        lmean = s1 / cnt
        lsd = np.sqrt(np.maximum(s2 / cnt - lmean ** 2, 0))
    proto = pd.Series(d["proto"].to_numpy()).map(PROTO).fillna(0).to_numpy()
    cols = [cnt, s1, cmin(L), lmean, cmax(L), lsd, cs(fwd) / cnt, cs(L * (1 - fwd)), iat_min, imean, iat_max, isd,
            t - t0, cs((fl >> 1) & 1), cs((fl >> 4) & 1), cs(fl & 1), cs((fl >> 2) & 1), cs((fl >> 3) & 1), proto]
    P = np.stack(cols, 1).astype(np.float32)                         # [packets, 19]
    fids, first = np.unique(gk, return_index=True)
    length = np.diff(np.r_[first, len(gk)])
    X = np.empty((len(fids), K, len(FEATURES)), np.float32)
    row = np.repeat(np.arange(len(fids)), length)
    X[row, d["pos"].to_numpy()] = P
    last = P[first + length - 1]
    for k in range(K):                                                # pad: repeat the final state
        m = length <= k
        X[m, k] = last[m]
    return X, length, fids


def main():
    REP.mkdir(parents=True, exist_ok=True)
    normal_ips = set()
    for n in NORMAL:
        d = pd.read_parquet(PK / f"{n}.parquet", columns=["src", "dst"])
        normal_ips |= set(d["src"].astype(str)) | set(d["dst"].astype(str))
    caps = NORMAL + list(ATTACK)
    out_X, out_len, out_y, out_cap, out_t0 = [], [], [], [], []
    counts, ep = [], []
    for ci, cap in enumerate(caps):
        d = pd.read_parquet(PK / f"{cap}.parquet")
        d = flows_of(d)
        first = d[d["pos"] == 0]
        s, t = first["src"].astype(str).to_numpy(), first["dst"].astype(str).to_numpy()
        if cap in NORMAL:
            cls = "Normal"
            keep = np.isin(s, list(normal_ips)) & np.isin(t, list(normal_ips))
        else:
            cls = ATTACK[cap]
            other = np.where(s == VICTIM, t, np.where(t == VICTIM, s, None))
            has_v = other != None  # noqa: E711
            if cap in FLOODS:
                att = np.array([o is not None and o not in normal_ips for o in other])
            else:
                att = other == ATTACKER
            keep = has_v & att
            c = collections.Counter(np.r_[s, t])
            ep.append(f"{cap}: {len(first):,} flows, kept {keep.sum():,}; top endpoints {c.most_common(5)}; "
                      f"dropped flows' endpoint pairs (top 5): "
                      f"{collections.Counter(zip(s[~keep], t[~keep])).most_common(5)}")
        kept_fids = first["fid"].to_numpy()[keep]
        dk = d[d["fid"].isin(kept_fids) & (d["pos"] < K)]
        X, length, fids = features(dk)
        y = np.full(len(fids), CLASSES.index(cls))
        t0 = d.loc[d["pos"] == 0].set_index("fid").loc[fids, "t"].to_numpy()
        counts.append(dict(capture=cap, cls=cls, packets=len(d), flows=int(len(first)), flows_kept=int(len(fids)),
                           one_packet_share=float(np.mean(length == 1)), flows_at_cap_K=float(np.mean(length == K))))
        print(counts[-1], flush=True)
        out_X.append(X); out_len.append(length); out_y.append(y); out_cap.append(np.full(len(fids), ci)); out_t0.append(t0)
    X, length, y = np.concatenate(out_X), np.concatenate(out_len), np.concatenate(out_y)
    cap_i, t0 = np.concatenate(out_cap), np.concatenate(out_t0)
    rng = np.random.default_rng(0)
    keep = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(y == c)
        keep.append(np.sort(rng.choice(idx, CAP, replace=False)) if len(idx) > CAP else idx)
    keep = np.concatenate(keep)
    uncapped = np.bincount(y, minlength=len(CLASSES))
    X, length, y, cap_i, t0 = X[keep], length[keep], y[keep], cap_i[keep], t0[keep]
    np.savez(OUT, X=X, length=length, y=y, capture=cap_i, t0=t0, captures=np.array(caps),
             class_names=np.array(CLASSES), features=np.array(FEATURES))
    (REP / "endpoints.txt").write_text("\n".join(ep) + "\n")
    pd.DataFrame(counts).to_csv(REP / "captures.csv", index=False)
    cc = pd.DataFrame(dict(cls=CLASSES, flows_uncapped=uncapped, flows=np.bincount(y, minlength=len(CLASSES)),
                           one_packet_share=[float(np.mean(length[y == c] == 1)) for c in range(len(CLASSES))],
                           median_length=[float(np.median(length[y == c])) for c in range(len(CLASSES))]))
    cc.to_csv(REP / "class_counts.csv", index=False)
    (REP / "build.json").write_text(json.dumps(dict(K=K, idle_timeout_s=IDLE, cap_per_class=CAP, n_flows=int(len(y)),
                                                    features=FEATURES, victim=VICTIM, attacker=ATTACKER,
                                                    output=str(OUT.relative_to(REPO))), indent=1))
    print(cc.to_string(index=False))


if __name__ == "__main__":
    main()
