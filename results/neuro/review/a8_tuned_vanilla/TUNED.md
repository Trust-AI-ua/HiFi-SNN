# Validation-tuned fixed-point Vanilla SNN (amendment 8, part 2)

Five candidates per seed (per-tensor |W| percentile mapped to 127: 100 = the amendment-4 rule, 99.99, 99.9, 99.5, 99), selected on the whole validation split by the Lava emulation's macro-F1, frozen (`selection.json`, committed) before any test evaluation. Selected networks through the amendment-6 chain under the guard (Lava = emulation bit for bit, SANA-FE gate 100% on every seed). Full-test macro-F1; five-seed mean (SD).

## edgeiiot

| Seed | p | Float record | Original fixed point | Tuned fixed point | Change | Agreement tuned/float | Energy original / tuned (nJ) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 | 100 | 77.04 | 62.12 | 62.12 | +0.00 | 92.1% | 521.5 / 521.5 |
| 43 | 99.99 | 78.11 | 66.79 | 66.60 | -0.19 | 95.7% | 545.5 / 546.3 |
| 44 | 99.99 | 73.38 | 54.50 | 60.64 | +6.14 | 94.9% | 559.8 / 557.0 |
| 45 | 99.5 | 74.10 | 56.34 | 64.71 | +8.36 | 84.5% | 580.1 / 572.8 |
| 46 | 100 | 74.29 | 62.79 | 62.79 | +0.00 | 82.7% | 532.1 / 532.1 |
| mean | | 75.39 (2.06) | 60.51 (5.02) | 63.37 (2.32) | +2.86 | 90.0% | 547.8 / 545.9 |

- Recovery (tuned - original) / (float - original): 0.19 (not most of the loss); float minus original +14.88 points; agreement with the float record 89.1% (original), 90.0% (tuned).
- Footprint: 9984 int8 weight bytes, unchanged; nonzero synapses 7037 (original) and 7529 (tuned); SANA-FE maps every dense connection, zeros included.
- Amendment-5 rule against the tuned Vanilla SNN: T10 at 437.0 nJ and 81.47 macro-F1 against 545.9 nJ and 63.37: dominates on five-seed means; per-seed wins 5/5.

| Class | Support (mean) | Float | Original | Tuned | Change |
|---|--:|--:|--:|--:|--:|
| Uploading | 7361 | 32.66 | 8.35 | 5.57 | -2.78 |
| XSS | 3013 | 60.89 | 20.87 | 25.21 | +4.34 |
| Ransomware | 1938 | 56.06 | 26.64 | 30.85 | +4.21 |
| SQL_injection | 10165 | 62.45 | 29.23 | 38.58 | +9.35 |
| Password | 9987 | 52.87 | 38.85 | 43.84 | +5.00 |
| Backdoor | 4805 | 85.13 | 52.60 | 70.96 | +18.36 |
| DDoS_HTTP | 9709 | 67.94 | 53.23 | 51.77 | -1.46 |
| Fingerprinting | 171 | 62.66 | 57.49 | 58.90 | +1.41 |
| Port_Scanning | 3995 | 85.69 | 80.85 | 82.54 | +1.69 |
| Vulnerability_scanner | 10005 | 92.07 | 85.78 | 85.37 | -0.41 |
| Normal | 272800 | 97.13 | 94.11 | 94.30 | +0.19 |
| DDoS_TCP | 10012 | 99.99 | 99.30 | 99.48 | +0.18 |
| DDoS_ICMP | 13588 | 99.85 | 99.83 | 99.83 | -0.01 |
| DDoS_UDP | 24314 | 99.99 | 99.97 | 99.97 | +0.00 |

## ciciomt2024

| Seed | p | Float record | Original fixed point | Tuned fixed point | Change | Agreement tuned/float | Energy original / tuned (nJ) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 42 | 100 | 73.49 | 73.98 | 73.98 | +0.00 | 82.9% | 422.9 / 422.9 |
| 43 | 99.9 | 72.86 | 74.27 | 73.04 | -1.22 | 98.0% | 408.4 / 407.5 |
| 44 | 99.9 | 71.39 | 69.84 | 70.12 | +0.28 | 98.3% | 415.0 / 414.7 |
| 45 | 99.9 | 73.63 | 71.32 | 74.88 | +3.56 | 97.6% | 414.7 / 414.9 |
| 46 | 99 | 71.67 | 74.09 | 73.92 | -0.17 | 96.5% | 404.1 / 405.3 |
| mean | | 72.61 (1.03) | 72.70 (2.01) | 73.19 (1.83) | +0.49 | 94.7% | 413.0 / 413.0 |

- Recovery (tuned - original) / (float - original): not defined: the original fixed-point network was already within 1 point of the float record; float minus original -0.09 points; agreement with the float record 89.5% (original), 94.7% (tuned).
- Footprint: 6528 int8 weight bytes, unchanged; nonzero synapses 6196 (original) and 6302 (tuned); SANA-FE maps every dense connection, zeros included.
- Amendment-5 rule against the tuned Vanilla SNN: t2s_exit at 308.8 nJ and 88.57 macro-F1 against 413.0 nJ and 73.19: dominates on five-seed means; per-seed wins 4/5.

| Class | Support (mean) | Float | Original | Tuned | Change |
|---|--:|--:|--:|--:|--:|
| Spoofing | 418 | 32.94 | 33.06 | 31.91 | -1.15 |
| DoS | 104339 | 31.74 | 35.37 | 37.37 | +2.00 |
| DDoS | 266456 | 85.98 | 83.05 | 85.10 | +2.06 |
| Benign | 9425 | 92.09 | 92.08 | 92.03 | -0.05 |
| Recon | 6963 | 94.20 | 93.91 | 94.03 | +0.13 |
| MQTT | 15945 | 98.70 | 98.74 | 98.69 | -0.05 |

