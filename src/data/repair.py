"""Row- and column-level repairs for the two benchmarks the leakage audit
(results/diagnostics/leakage_audit/README.md) found contaminated.

Edge-IIoTset (``DNN-EdgeIIoT-dataset.csv``)
------------------------------------------
The published CSV was produced from tshark field output split on every comma
without quoting. ``frame.time`` ("Nov 24, 2021 22:27:10 UTC") and multi-valued
fields (``udp.port`` is always "src,dst"; ``ip.src_host``/``ip.dst_host`` carry
the inner header on ICMP errors) therefore add columns, and everything after
them sits one or two columns off. Four row layouts exist, each verified against
the shipped pcaps (scripts/verify_edgeiiot_repair.py):

  original            every row without UDP fields: aligned (the date's first
                      token was consumed upstream; ``frame.time`` is dropped).
  decoded_udp_flood   DDoS_UDP attacker packets: the whole date is missing, so
                      fields ip.src_host..udp.port sit one column LEFT; the two
                      udp.port values fill (tcp.srcport, udp.port); aligned from
                      udp.stream on. ``frame.time`` is unrecoverable.
  decoded_icmp_reply  DDoS_UDP victim ICMP port-unreachable replies: date
                      missing (-1), two-valued ip.src_host (+1) and ip.dst_host
                      (+1) and udp.port (+1). The last two fields
                      (mbtcp.trans_id, mbtcp.unit_id) were truncated upstream and
                      are filled with 0 (UDP rows carry no Modbus/TCP).
                      ``frame.time`` is unrecoverable.
  decoded_normal_udp  Normal UDP packets (DNS, NTP, NetBIOS, mDNS, DHCP): aligned
                      up to udp.port, whose second value pushes the rest one
                      column RIGHT; mbtcp.unit_id truncated, filled with 0.
  dropped_mitm        MITM rows: garbled beyond a verifiable decode; removed.

Multi-valued fields are canonicalised to their FIRST occurrence (tshark
``-E occurrence=f``): udp.port = source port of the (inner) UDP header, and
ip.*_host = outer header.

Value repairs: the zero placeholder is spelled "0" or "0.0" depending on the
source capture (in four columns the spelling alone determines Normal vs attack),
so "0.0" is canonicalised to "0" before any encoding; hex renderings
("0x00000000") are converted to decimal. ``http.referer`` (payload text,
including a Shellshock string on 185 rows) and ``dns.qry.name`` (a hostname
once the rows are aligned) are dropped as features, like the other
payload/identifier columns.

CSE-CIC-IDS2018
---------------
``Flow ID``, ``Src IP``, ``Src Port``, ``Dst IP`` exist only in 02-20-2018.csv;
concatenation zero-pads them for the other nine days, making ``Src Port`` a
capture-day indicator. They are dropped.
"""
from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CSE-CIC-IDS2018
CSE_DAY_ONLY_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]


def drop_cse_day_only_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in CSE_DAY_ONLY_COLUMNS if c in df.columns])


# ---------------------------------------------------------------- Edge-IIoTset
EDGE_FEATURES: List[str] = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4", "arp.opcode",
    "arp.hw.size", "arp.src.proto_ipv4", "icmp.checksum", "icmp.seq_le",
    "icmp.transmit_timestamp", "icmp.unused", "http.file_data", "http.content_length",
    "http.request.uri.query", "http.request.method", "http.referer",
    "http.request.full_uri", "http.request.version", "http.response", "http.tls_port",
    "tcp.ack", "tcp.ack_raw", "tcp.checksum", "tcp.connection.fin", "tcp.connection.rst",
    "tcp.connection.syn", "tcp.connection.synack", "tcp.dstport", "tcp.flags",
    "tcp.flags.ack", "tcp.len", "tcp.options", "tcp.payload", "tcp.seq", "tcp.srcport",
    "udp.port", "udp.stream", "udp.time_delta", "dns.qry.name", "dns.qry.name.len",
    "dns.qry.qu", "dns.qry.type", "dns.retransmission", "dns.retransmit_request",
    "dns.retransmit_request_in", "mqtt.conack.flags", "mqtt.conflag.cleansess",
    "mqtt.conflags", "mqtt.hdrflags", "mqtt.len", "mqtt.msg_decoded_as", "mqtt.msg",
    "mqtt.msgtype", "mqtt.proto_len", "mqtt.protoname", "mqtt.topic", "mqtt.topic_len",
    "mqtt.ver", "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id",
]
EDGE_EXTRA_DROP = ["http.referer", "dns.qry.name"]
_I = {c: i for i, c in enumerate(EDGE_FEATURES)}
UDP_PORT = _I["udp.port"]            # 35
N_FEAT = len(EDGE_FEATURES)          # 61

PROVENANCE = ["original", "decoded_udp_flood", "decoded_icmp_reply",
              "decoded_normal_udp", "dropped_mitm"]

_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_HEX = re.compile(r"^0x[0-9a-fA-F]+$")
_ZERO = {"", "0", "0.0"}


def _is_ip(s: pd.Series) -> pd.Series:
    return s.fillna("").str.strip().str.match(_IP)


def _nonzero(s: pd.Series) -> pd.Series:
    return ~s.fillna("").str.strip().isin(_ZERO)


def classify_layout(df: pd.DataFrame) -> pd.Series:
    """Provenance label per row. Raises if a row matches no verified layout."""
    y = df["Attack_type"].astype(str).str.strip()
    prov = pd.Series("original", index=df.index, dtype=object)
    udp_like = _nonzero(df["udp.port"]) | _nonzero(df["udp.stream"])

    udp = y == "DDoS_UDP"
    if udp.any():
        if not _is_ip(df.loc[udp, "frame.time"]).all():
            raise ValueError("DDoS_UDP row without an address in frame.time: unknown layout")
        reply = udp & _is_ip(df["arp.dst.proto_ipv4"])
        prov[udp & ~reply] = "decoded_udp_flood"
        prov[reply] = "decoded_icmp_reply"
    prov[y == "MITM"] = "dropped_mitm"
    normal_udp = (y == "Normal") & udp_like
    prov[normal_udp] = "decoded_normal_udp"

    stray = udp_like & (prov == "original")
    if stray.any():
        raise ValueError(f"{int(stray.sum())} UDP rows in classes with no verified layout: "
                         f"{y[stray].value_counts().to_dict()}")
    return prov


def _decode_udp_flood(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    out[:, 0] = np.nan                           # frame.time: unrecoverable
    out[:, 1:UDP_PORT] = v[:, 0:UDP_PORT - 1]    # ip.src_host .. tcp.srcport
    out[:, UDP_PORT] = v[:, UDP_PORT - 1]        # udp.port := source port
    # udp.stream .. end already aligned
    return out


def _decode_icmp_reply(v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    out[:, 0] = np.nan                           # frame.time: unrecoverable
    out[:, 1] = v[:, 0]                          # ip.src_host (outer)
    out[:, 2] = v[:, 2]                          # ip.dst_host (outer)
    out[:, 3:UDP_PORT] = v[:, 4:UDP_PORT + 1]    # arp.dst.proto_ipv4 .. tcp.srcport
    out[:, UDP_PORT] = v[:, UDP_PORT + 1]        # udp.port := inner source port
    out[:, UDP_PORT + 1:N_FEAT - 2] = v[:, UDP_PORT + 3:N_FEAT]
    out[:, N_FEAT - 2:] = "0"                    # mbtcp.trans_id, unit_id: truncated upstream
    return out


def _decode_normal_udp(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    # frame.time .. udp.port first value already aligned
    out[:, UDP_PORT + 1:N_FEAT - 1] = v[:, UDP_PORT + 2:N_FEAT]
    out[:, N_FEAT - 1] = "0"                     # mbtcp.unit_id: truncated upstream
    return out


_DECODERS = {"decoded_udp_flood": _decode_udp_flood,
             "decoded_icmp_reply": _decode_icmp_reply,
             "decoded_normal_udp": _decode_normal_udp}


def realign_rows(df: pd.DataFrame, prov: pd.Series) -> pd.DataFrame:
    """Apply the verified per-layout column mapping; other rows are untouched."""
    out = df.copy()
    for label, fn in _DECODERS.items():
        m = (prov == label).to_numpy()
        if m.any():
            out.loc[m, EDGE_FEATURES] = fn(df.loc[m, EDGE_FEATURES].to_numpy(dtype=object))
    return out


def canonical_value(x):
    """'0.0' -> '0', hex -> decimal string; everything else stripped, unchanged."""
    if not isinstance(x, str):
        return x
    s = x.strip()
    if s in ("0.0", "0"):
        return "0"
    if _HEX.match(s):
        return str(int(s, 16))
    return s


def normalise_values(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(canonical_value)
    return out


def repair_edgeiiot(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Full Edge-IIoTset repair on the string-typed published CSV.

    Returns (repaired string frame without the dropped rows, provenance for the
    kept rows). Feature columns are EDGE_FEATURES minus EDGE_EXTRA_DROP; the
    caller applies its own identifier drops and type inference."""
    missing = [c for c in EDGE_FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(f"unexpected Edge-IIoTset schema, missing {missing}")
    prov = classify_layout(raw)
    df = realign_rows(raw, prov)
    keep = (prov != "dropped_mitm").to_numpy()
    df, prov = df[keep], prov[keep]
    df = normalise_values(df, EDGE_FEATURES)
    df = df.drop(columns=EDGE_EXTRA_DROP)
    return df, prov


def infer_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Columns whose non-null values all parse as numbers become float; the rest
    stay text (and are one-hot encoded by the loader)."""
    out = df.copy()
    for c in cols:
        v = pd.to_numeric(out[c], errors="coerce")
        if v.notna().sum() == out[c].notna().sum():
            out[c] = v
    return out


# --------------------------------------------------------- post-repair validation
def row_violations(df: pd.DataFrame) -> pd.Series:
    """Boolean per row: violates a constraint every aligned tshark row satisfies.
    Only checks columns present in ``df``."""
    bad = pd.Series(False, index=df.index)
    s = lambda c: df[c].fillna("").astype(str).str.strip()
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    for c in ["ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4", "arp.src.proto_ipv4"]:
        if c in df:
            bad |= ~(s(c).str.match(_IP) | s(c).isin(_ZERO))
    for c in ["dns.qry.qu", "dns.retransmission", "dns.retransmit_request",
              "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn",
              "tcp.connection.synack", "tcp.flags.ack", "mqtt.conflag.cleansess"]:
        if c in df:
            v = num(c)
            bad |= v.notna() & ~v.isin([0, 1])
    for c in ["udp.port", "tcp.srcport", "tcp.dstport"]:
        if c in df:
            v = num(c)
            bad |= s(c).ne("") & (v.isna() | (v > 65535) | (v < 0))
    if "dns.qry.name.len" in df:
        bad |= num("dns.qry.name.len").isna() & s("dns.qry.name.len").ne("")
    return bad
