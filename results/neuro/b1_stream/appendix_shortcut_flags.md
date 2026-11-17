# Appendix: stream-dataset shortcut screen

Seed-42 training split (384,375 flows). Indicator a = [feature > its training minimum], after packet 1, after packet 4, and after the flow's last packet. A flag: 0.05 <= P(a) <= 0.95 and, for a class with at least 100 training flows, P(a | class) = 0 (rules out) or 1 (rules in). NMI = I(a; y) / H(y). No feature was dropped; every flag has the traffic reason given.

| After packet | Feature | P(a) | NMI | Rules out (P(a \| c) = 0) | Rules in (P(a \| c) = 1) | Reason |
|---|---|--:|--:|---|---|---|
| 1 | bytes | 0.644 | 0.314 | DDoS_ICMP | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| 1 | len_min | 0.644 | 0.314 | DDoS_ICMP | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| 1 | len_mean | 0.644 | 0.314 | DDoS_ICMP | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| 1 | len_max | 0.644 | 0.314 | DDoS_ICMP | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| 1 | syn | 0.622 | 0.344 | DDoS_ICMP, DDoS_UDP | DDoS_TCP, Port_Scanning, Uploading, XSS | TCP SYN seen: UDP and ICMP floods carry no TCP; SYN floods, scans and these attack sessions open with SYN |
| 4 | pkt_count | 0.611 | 0.243 |  | Backdoor, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | a second packet: these attacks are TCP sessions with the attacker (at least 2 packets); floods are 1-packet flows |
| 4 | bytes | 0.683 | 0.249 |  | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| 4 | len_min | 0.644 | 0.314 | DDoS_ICMP | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| 4 | len_mean | 0.644 | 0.314 | DDoS_ICMP | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| 4 | len_max | 0.644 | 0.314 | DDoS_ICMP | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| 4 | len_sd | 0.531 | 0.295 | DDoS_ICMP, DDoS_UDP, Fingerprinting, Port_Scanning | Vulnerability_scanner | packet sizes vary within the flow: floods and scans repeat one fixed-size packet; sessions carry payload |
| 4 | bwd_bytes | 0.592 | 0.241 | DDoS_UDP | Backdoor, Password, Port_Scanning, SQL_injection, Uploading, Vulnerability_scanner, XSS | the responder sends bytes: the UDP flood gets no reply; the attack sessions always get one |
| 4 | iat_min | 0.611 | 0.243 |  | Backdoor, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | defined only once a second packet arrives (same indicator as the packet count) |
| 4 | iat_mean | 0.611 | 0.243 |  | Backdoor, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | defined only once a second packet arrives (same indicator as the packet count) |
| 4 | iat_max | 0.611 | 0.243 |  | Backdoor, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | defined only once a second packet arrives (same indicator as the packet count) |
| 4 | iat_sd | 0.405 | 0.350 | DDoS_UDP, Port_Scanning | Password, SQL_injection, XSS | at least three packets with uneven timing: interactive sessions vary, floods and scans do not |
| 4 | duration | 0.611 | 0.243 |  | Backdoor, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | non-zero only once a second packet arrives (same indicator as the packet count) |
| 4 | syn | 0.622 | 0.344 | DDoS_ICMP, DDoS_UDP | DDoS_TCP, Port_Scanning, Uploading, XSS | TCP SYN seen: UDP and ICMP floods carry no TCP; SYN floods, scans and these attack sessions open with SYN |
| 4 | ack | 0.552 | 0.294 | DDoS_ICMP, DDoS_UDP | Backdoor, Password, Port_Scanning, SQL_injection, Uploading, Vulnerability_scanner, XSS | TCP ACK seen: UDP and ICMP floods carry no TCP; established sessions acknowledge |
| 4 | rst | 0.173 | 0.166 | DDoS_ICMP, DDoS_UDP, SQL_injection, XSS | Port_Scanning | TCP RST seen: probes to closed ports are reset (port scan); UDP and ICMP floods carry no TCP; the injection sessions end with FIN |
| 4 | psh | 0.339 | 0.272 | DDoS_ICMP, DDoS_TCP, DDoS_UDP, Fingerprinting, Port_Scanning |  | TCP PSH (payload push) seen: floods, scans and fingerprinting probes carry no payload |
| last | pkt_count | 0.611 | 0.243 |  | Backdoor, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | a second packet: these attacks are TCP sessions with the attacker (at least 2 packets); floods are 1-packet flows |
| last | bytes | 0.683 | 0.249 |  | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| last | len_min | 0.644 | 0.314 | DDoS_ICMP | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| last | len_mean | 0.644 | 0.314 | DDoS_ICMP | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| last | len_max | 0.644 | 0.314 | DDoS_ICMP | Normal, Backdoor, DDoS_HTTP, DDoS_TCP, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the ICMP flood sends only minimum-size (28-byte) packets, every other class's first packet is larger; flood-tool packet sizes are fixed (TCP SYN flood 160 B, port-scan probe 40 B) |
| last | len_sd | 0.537 | 0.294 | DDoS_ICMP, DDoS_UDP, Port_Scanning | Password, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | packet sizes vary within the flow: floods and scans repeat one fixed-size packet; sessions carry payload |
| last | bwd_bytes | 0.596 | 0.238 | DDoS_UDP | Backdoor, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | the responder sends bytes: the UDP flood gets no reply; the attack sessions always get one |
| last | iat_min | 0.611 | 0.242 |  | Backdoor, Fingerprinting, Port_Scanning, Ransomware, SQL_injection, Vulnerability_scanner, XSS | defined only once a second packet arrives (same indicator as the packet count) |
| last | iat_mean | 0.611 | 0.243 |  | Backdoor, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | defined only once a second packet arrives (same indicator as the packet count) |
| last | iat_max | 0.611 | 0.243 |  | Backdoor, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | defined only once a second packet arrives (same indicator as the packet count) |
| last | iat_sd | 0.405 | 0.350 | DDoS_UDP, Port_Scanning | Password, SQL_injection, XSS | at least three packets with uneven timing: interactive sessions vary, floods and scans do not |
| last | duration | 0.611 | 0.243 |  | Backdoor, Fingerprinting, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | non-zero only once a second packet arrives (same indicator as the packet count) |
| last | syn | 0.622 | 0.344 | DDoS_ICMP, DDoS_UDP | DDoS_TCP, Password, Port_Scanning, Uploading, XSS | TCP SYN seen: UDP and ICMP floods carry no TCP; SYN floods, scans and these attack sessions open with SYN |
| last | ack | 0.556 | 0.292 | DDoS_ICMP, DDoS_UDP | Backdoor, Password, Port_Scanning, Ransomware, SQL_injection, Uploading, Vulnerability_scanner, XSS | TCP ACK seen: UDP and ICMP floods carry no TCP; established sessions acknowledge |
| last | fin | 0.374 | 0.321 | DDoS_ICMP, DDoS_TCP, DDoS_UDP, Fingerprinting, Port_Scanning, Ransomware | SQL_injection, XSS | TCP FIN seen: completed web sessions (SQL injection, XSS) close with FIN; floods, scans and fingerprinting never complete a session |
| last | rst | 0.367 | 0.247 | DDoS_ICMP, DDoS_UDP | Port_Scanning | TCP RST seen: probes to closed ports are reset (port scan); UDP and ICMP floods carry no TCP; the injection sessions end with FIN |
| last | psh | 0.346 | 0.272 | DDoS_ICMP, DDoS_TCP, DDoS_UDP, Fingerprinting, Port_Scanning | XSS | TCP PSH (payload push) seen: floods, scans and fingerprinting probes carry no payload |

## Class counts and one-packet share

| Class | Flows (uncapped) | Flows (capped at 100,000) | One-packet flows (%) | Median length (packets) |
|---|--:|--:|--:|--:|
| Normal | 673,988 | 100,000 | 0.27 | 16 |
| Backdoor | 833 | 833 | 0.00 | 2 |
| DDoS_HTTP | 12,113 | 12,113 | 0.45 | 7 |
| DDoS_ICMP | 2,405,720 | 100,000 | 78.98 | 1 |
| DDoS_TCP | 1,176,011 | 100,000 | 28.41 | 2 |
| DDoS_UDP | 3,215,222 | 100,000 | 100.00 | 1 |
| Fingerprinting | 138 | 138 | 0.00 | 2 |
| Password | 94,498 | 94,498 | 0.00 | 12 |
| Port_Scanning | 9,987 | 9,987 | 0.00 | 2 |
| Ransomware | 572 | 572 | 0.00 | 2 |
| SQL_injection | 4,383 | 4,383 | 0.00 | 11 |
| Uploading | 7,580 | 7,580 | 0.00 | 2 |
| Vulnerability_scanner | 2,603 | 2,603 | 0.00 | 32 |
| XSS | 1,148 | 1,148 | 0.00 | 13 |
| All | 7,604,796 | 533,855 | 38.91 | |
