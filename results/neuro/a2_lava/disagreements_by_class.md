# A2: Lava vs integer NumPy disagreements by true class (2,000-flow sample, seed 42)

Subtractive reset (the P5 variant): every disagreeing flow.

| Dataset | Disagreeing flows | True class (sample size) | Integer right, Lava wrong | Integer wrong, Lava right | Both wrong | Lava predicted |
|---|--:|---|--:|--:|--:|---|
| toniot | 2 | ddos (192) | 0 | 2 | 0 | ddos:2 |
| toniot | 8 | mitm (56) | 3 | 4 | 1 | dos:2; mitm:4; password:1; xss:1 |
| toniot | 2 | scanning (192) | 1 | 1 | 0 | dos:1; scanning:1 |
| edgeiiot | 2 | DDoS_HTTP (83) | 0 | 1 | 1 | Normal:1; DDoS_HTTP:1 |
| edgeiiot | 2 | Ransomware (56) | 2 | 0 | 0 | Backdoor:2 |
| edgeiiot | 1 | Vulnerability_scanner (84) | 1 | 0 | 0 | XSS:1 |
| edgeiiot | 1 | XSS (60) | 1 | 0 | 0 | DDoS_HTTP:1 |
| ciciomt2024 | 1 | DDoS (1172) | 1 | 0 | 0 | DoS:1 |
| ciciomt2024 | 2 | Spoofing (52) | 0 | 2 | 0 | Spoofing:2 |
| ciciot2023 | 1 | BruteForce (50) | 1 | 0 | 0 | Spoofing:1 |
| ciciot2023 | 1 | Recon (62) | 0 | 0 | 1 | DoS:1 |
| ciciot2023 | 1 | Spoofing (67) | 0 | 1 | 0 | Spoofing:1 |
| ciciot2023 | 2 | Web (51) | 0 | 1 | 1 | Benign:1; Web:1 |

Zero reset: share of each dataset's disagreements by true class (top 3), and the share of that class that disagrees.

| Dataset | Disagreeing flows | Top classes: share of disagreements (share of class) |
|---|--:|---|
| toniot | 1269 | normal 25% (79%); injection 14% (95%); ransomware 14% (94%) |
| edgeiiot | 973 | Normal 40% (40%); DDoS_TCP 9% (100%); DDoS_ICMP 8% (76%) |
| ciciomt2024 | 189 | DoS 67% (26%); Recon 22% (53%); Spoofing 7% (25%) |
| ciciot2023 | 146 | DoS 21% (9%); Web 21% (59%); BruteForce 18% (54%) |
