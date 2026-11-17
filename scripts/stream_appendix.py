#!/usr/bin/env python3
"""B1 appendix: the shortcut screen's flags with a reason for each, class counts and one-packet
shares (results/preregistration/2026-11-17_neuro_stream.md, amendment 2).

Reads results/neuro/b1_stream/{shortcut_screen.csv, class_counts.csv} (seed-42 training split).
A flag: indicator a = [feature > its training minimum] with 0.05 <= P(a) <= 0.95 that is never
(rules out) or always (rules in) active for a class with >= 100 training flows.

    python scripts/stream_appendix.py
Output: results/neuro/b1_stream/appendix_shortcut_flags.{md,tex}
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
B1 = REPO / "results/neuro/b1_stream"
STATE = {"after_1": "1", "after_4": "4", "after_last": "last"}
SIZE = ("the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; "
        "flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B)")
REASON = {
    "bytes": SIZE, "len_min": SIZE, "len_mean": SIZE, "len_max": SIZE,
    "syn": "TCP SYN seen: UDP and ICMP floods carry no TCP; SYN floods, scans and these attack sessions open with SYN",
    "pkt_count": "a second packet: these attacks are TCP sessions with the attacker (at least 2 packets); floods are 1-packet flows",
    "iat_min": "defined only once a second packet arrives (same indicator as the packet count)",
    "iat_mean": "defined only once a second packet arrives (same indicator as the packet count)",
    "iat_max": "defined only once a second packet arrives (same indicator as the packet count)",
    "duration": "non-zero only once a second packet arrives (same indicator as the packet count)",
    "len_sd": "packet sizes vary within the flow: floods and scans repeat one fixed-size packet; sessions carry payload",
    "bwd_bytes": "the responder sends bytes: the UDP flood gets no reply; the attack sessions always get one",
    "iat_sd": "at least three packets with uneven timing: interactive sessions vary, floods and scans do not",
    "ack": "TCP ACK seen: UDP and ICMP floods carry no TCP; established sessions acknowledge",
    "rst": "TCP RST seen: probes to closed ports are reset (port scan); UDP and ICMP floods carry no TCP; "
           "the injection sessions end with FIN",
    "psh": "TCP PSH (payload push) seen: floods, scans and fingerprinting probes carry no payload",
    "fin": "TCP FIN seen: completed web sessions (SQL injection, XSS) close with FIN; floods, scans and "
           "fingerprinting never complete a session",
    "proto": "protocol code",
}


def main():
    sc = pd.read_csv(B1 / "shortcut_screen.csv").fillna("")
    cc = pd.read_csv(B1 / "class_counts.csv")
    flag = sc[(sc.excludes != "") | (sc.includes != "")]
    md = ["# Appendix: stream-dataset shortcut screen", "",
          "Seed-42 training split (384,375 flows). Indicator a = [feature > its training minimum], after packet 1, "
          "after packet 4, and after the flow's last packet. A flag: 0.05 <= P(a) <= 0.95 and, for a class with at "
          "least 100 training flows, P(a | class) = 0 (rules out) or 1 (rules in). NMI = I(a; y) / H(y). "
          "No feature was dropped; every flag has the traffic reason given.", "",
          "| After packet | Feature | P(a) | NMI | Rules out (P(a \\| c) = 0) | Rules in (P(a \\| c) = 1) | Reason |",
          "|---|---|--:|--:|---|---|---|"]
    tex = [r"\begin{longtable}{@{}llrrp{3.2cm}p{3.6cm}p{5.2cm}@{}}",
           r"\caption{Stream-dataset shortcut screen: every flagged indicator (seed-42 training split).}\\",
           r"\toprule After & Feature & $P(a)$ & NMI & Rules out & Rules in & Reason \\ \midrule \endhead"]
    esc = lambda s: s.replace("_", r"\_").replace(";", ", ")
    for r in flag.itertuples():
        reason = REASON.get(r.feature, "")
        md.append(f"| {STATE[r.state]} | {r.feature} | {r.frac_active:.3f} | {r.nmi:.3f} | "
                  f"{r.excludes.replace(';', ', ')} | {r.includes.replace(';', ', ')} | {reason} |")
        tex.append(f"{STATE[r.state]} & {esc(r.feature)} & {r.frac_active:.3f} & {r.nmi:.3f} & {esc(r.excludes)} & "
                   f"{esc(r.includes)} & {esc(reason)} \\\\")
    tex += [r"\bottomrule", r"\end{longtable}", "",
            r"\begin{table}[t]\centering\caption{Stream dataset: flows per class (cap 100{,}000) and one-packet share.}",
            r"\begin{tabular}{@{}lrrrr@{}}\toprule Class & Flows (uncapped) & Flows & One-packet (\%) & Median length \\ \midrule"]
    md += ["", "## Class counts and one-packet share", "",
           "| Class | Flows (uncapped) | Flows (capped at 100,000) | One-packet flows (%) | Median length (packets) |",
           "|---|--:|--:|--:|--:|"]
    for r in cc.itertuples():
        md.append(f"| {r.cls} | {r.flows_uncapped:,} | {r.flows:,} | {100 * r.one_packet_share:.2f} | {r.median_length:g} |")
        tex.append(f"{esc(r.cls)} & {r.flows_uncapped:,} & {r.flows:,} & {100 * r.one_packet_share:.2f} & {r.median_length:g} \\\\")
    tot = cc.flows.sum()
    md.append(f"| All | {cc.flows_uncapped.sum():,} | {tot:,} | "
              f"{100 * (cc.flows * cc.one_packet_share).sum() / tot:.2f} | |")
    tex += [r"\bottomrule\end{tabular}\end{table}"]
    (B1 / "appendix_shortcut_flags.md").write_text("\n".join(md) + "\n")
    (B1 / "appendix_shortcut_flags.tex").write_text("\n".join(tex) + "\n")
    print(f"{len(flag)} flagged rows, {len(cc)} classes")


if __name__ == "__main__":
    main()
