# B1 checks (seed-42 split)

Flows: train 384,375, validation 42,709, test 106,771.

## Shortcut screen: flagged indicators

| State | Feature | P(active) | NMI | Rules out | Rules in |
|---|---|--:|--:|---|---|
| after_1 | bytes | 0.644 | 0.314 | DDoS_ICMP | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_1 | len_min | 0.644 | 0.314 | DDoS_ICMP | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_1 | len_mean | 0.644 | 0.314 | DDoS_ICMP | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_1 | len_max | 0.644 | 0.314 | DDoS_ICMP | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_1 | syn | 0.622 | 0.344 | DDoS_ICMP;DDoS_UDP | DDoS_TCP;Port_Scanning;Uploading;XSS |
| after_4 | pkt_count | 0.611 | 0.243 |  | Backdoor;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | bytes | 0.683 | 0.249 |  | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | len_min | 0.644 | 0.314 | DDoS_ICMP | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | len_mean | 0.644 | 0.314 | DDoS_ICMP | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | len_max | 0.644 | 0.314 | DDoS_ICMP | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | len_sd | 0.531 | 0.295 | DDoS_ICMP;DDoS_UDP;Fingerprinting;Port_Scanning | Vulnerability_scanner |
| after_4 | bwd_bytes | 0.592 | 0.241 | DDoS_UDP | Backdoor;Password;Port_Scanning;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | iat_min | 0.611 | 0.243 |  | Backdoor;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | iat_mean | 0.611 | 0.243 |  | Backdoor;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | iat_max | 0.611 | 0.243 |  | Backdoor;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | iat_sd | 0.405 | 0.350 | DDoS_UDP;Port_Scanning | Password;SQL_injection;XSS |
| after_4 | duration | 0.611 | 0.243 |  | Backdoor;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | syn | 0.622 | 0.344 | DDoS_ICMP;DDoS_UDP | DDoS_TCP;Port_Scanning;Uploading;XSS |
| after_4 | ack | 0.552 | 0.294 | DDoS_ICMP;DDoS_UDP | Backdoor;Password;Port_Scanning;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_4 | rst | 0.173 | 0.166 | DDoS_ICMP;DDoS_UDP;SQL_injection;XSS | Port_Scanning |
| after_4 | psh | 0.339 | 0.272 | DDoS_ICMP;DDoS_TCP;DDoS_UDP;Fingerprinting;Port_Scanning |  |
| after_last | pkt_count | 0.611 | 0.243 |  | Backdoor;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | bytes | 0.683 | 0.249 |  | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | len_min | 0.644 | 0.314 | DDoS_ICMP | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | len_mean | 0.644 | 0.314 | DDoS_ICMP | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | len_max | 0.644 | 0.314 | DDoS_ICMP | Normal;Backdoor;DDoS_HTTP;DDoS_TCP;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | len_sd | 0.537 | 0.294 | DDoS_ICMP;DDoS_UDP;Port_Scanning | Password;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | bwd_bytes | 0.596 | 0.238 | DDoS_UDP | Backdoor;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | iat_min | 0.611 | 0.242 |  | Backdoor;Fingerprinting;Port_Scanning;Ransomware;SQL_injection;Vulnerability_scanner;XSS |
| after_last | iat_mean | 0.611 | 0.243 |  | Backdoor;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | iat_max | 0.611 | 0.243 |  | Backdoor;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | iat_sd | 0.405 | 0.350 | DDoS_UDP;Port_Scanning | Password;SQL_injection;XSS |
| after_last | duration | 0.611 | 0.243 |  | Backdoor;Fingerprinting;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | syn | 0.622 | 0.344 | DDoS_ICMP;DDoS_UDP | DDoS_TCP;Password;Port_Scanning;Uploading;XSS |
| after_last | ack | 0.556 | 0.292 | DDoS_ICMP;DDoS_UDP | Backdoor;Password;Port_Scanning;Ransomware;SQL_injection;Uploading;Vulnerability_scanner;XSS |
| after_last | fin | 0.374 | 0.321 | DDoS_ICMP;DDoS_TCP;DDoS_UDP;Fingerprinting;Port_Scanning;Ransomware | SQL_injection;XSS |
| after_last | rst | 0.367 | 0.247 | DDoS_ICMP;DDoS_UDP | Port_Scanning |
| after_last | psh | 0.346 | 0.272 | DDoS_ICMP;DDoS_TCP;DDoS_UDP;Fingerprinting;Port_Scanning | XSS |

## Highest indicator NMI per state

- after_1: syn 0.34, bytes 0.31, len_min 0.31, len_mean 0.31, len_max 0.31
- after_4: iat_sd 0.35, syn 0.34, len_min 0.31, len_mean 0.31, len_max 0.31
- after_last: iat_sd 0.35, syn 0.34, fin 0.32, len_min 0.31, len_mean 0.31

## Test flows whose final state and length equal a training flow's

All classes: 58.8%. Per class: Normal 0.2%, Backdoor 52.1%, DDoS_HTTP 0.4%, DDoS_ICMP 99.6%, DDoS_TCP 99.8%, DDoS_UDP 100.0%, Fingerprinting 92.6%, Password 0.0%, Port_Scanning 99.4%, Ransomware 19.3%, SQL_injection 0.0%, Uploading 49.4%, Vulnerability_scanner 0.0%, XSS 0.0%
## Review of the flags (recorded before any B2 model)

Every flag has a traffic explanation; none is a capture artifact, and no feature is dropped.
- Packet-length indicators after packet 1: the flood tools send fixed-size packets, and the
  ICMP flood's packets are the smallest in the data, so "larger than the minimum" rules
  DDoS_ICMP out and every other class in.
- SYN, ACK, FIN, RST, PSH counts: protocol semantics (UDP and ICMP floods carry no TCP flags;
  SYN floods and port scans open with SYN; scans close with RST).
- Packet count, interarrival and duration after packet 4: flow length (scans and floods are
  one- or two-packet flows; the attack sessions are longer).
- No feature is an identifier (no IPs, ports, TTLs, capture timestamps).

The exact-duplicate share (test flows whose final state and length equal a training flow's)
is 58.8%, almost all in the floods and the port scan, whose flows are identical packets; it is
0.2% for Normal. This is a property of the traffic, as in the tabular Edge-IIoTset, and is
reported beside the B2 results.
