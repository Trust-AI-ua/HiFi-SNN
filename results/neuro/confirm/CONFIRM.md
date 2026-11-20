# Amendments 6 and 7: five-seed chip simulation against the fixed-point Vanilla SNN

SANA-FE energy and latency per inference (each seed's 2,000-flow sample; about 10% calibration caveat) and test macro-F1 of the simulated network (bit-exact emulation of Lava's arithmetic, whole test split). Each Tree-to-Spike row is compared with the fixed-point Vanilla SNN of the same seed.

## edgeiiot: T2S, T = 10 against the fixed-point Vanilla (amendment 6)

| Seed | T2S, T = 10 energy (nJ) / macro-F1 | Vanilla, fixed point energy (nJ) / macro-F1 | Win |
|--:|--:|--:|---|
| 42 | 430 / 81.36 | 521 / 62.12 | yes |
| 43 | 452 / 82.04 | 545 / 66.79 | yes |
| 44 | 425 / 81.75 | 560 / 54.50 | yes |
| 45 | 437 / 81.12 | 580 / 56.34 | yes |
| 46 | 442 / 81.08 | 532 / 62.79 | yes |

Five-seed means: T2S, T = 10 437 ± 11 nJ, 81.47 ± 0.42; Vanilla 548 ± 23 nJ, 60.51 ± 5.02. Energy ratio 0.80 (margin +20.2% of Vanilla's energy; SANA-FE calibration error about 10%); macro-F1 difference +20.96.

**Dominance on five-seed means: yes.** Per-seed wins: 5 of 5.

Context (descriptive), mean over the seeds run:

| Row | Seeds | Energy (nJ) | Latency (µs) | Macro-F1 |
|---|--:|--:|--:|--:|
| T2S, early exit (C1 gate) | 5 | 698 | 56.4 | 83.83 |
| T2S, full window (T = 20) | 5 | 738 | 59.3 | 83.95 |

## ciciomt2024: T2S, early exit (C1 gate) against the fixed-point Vanilla (amendment 6)

| Seed | T2S, early exit (C1 gate) energy (nJ) / macro-F1 | Vanilla, fixed point energy (nJ) / macro-F1 | Win |
|--:|--:|--:|---|
| 42 | 246 / 89.80 (gate 0.95) | 423 / 73.98 | yes |
| 43 | 244 / 87.99 (gate 0.99) | 408 / 74.27 | yes |
| 44 | 223 / 83.45 (gate 0.95) | 415 / 69.84 | yes |
| 45 | 576 / 90.12 (gate disabled: full window) | 415 / 71.32 | no |
| 46 | 255 / 91.49 (gate 0.99) | 404 / 74.09 | yes |

Five-seed means: T2S, early exit (C1 gate) 309 ± 150 nJ, 88.57 ± 3.12; Vanilla 413 ± 7 nJ, 72.70 ± 2.01. Energy ratio 0.75 (margin +25.2% of Vanilla's energy; SANA-FE calibration error about 10%); macro-F1 difference +15.87.

**Dominance on five-seed means: yes.** Per-seed wins: 4 of 5.

Context (descriptive), mean over the seeds run:

| Row | Seeds | Energy (nJ) | Latency (µs) | Macro-F1 |
|---|--:|--:|--:|--:|
| T2S, full window (T = 20) | 5 | 597 | 49.9 | 89.46 |

## toniot: T2S, full window (T = 20) against the fixed-point Vanilla (amendment 7)

| Seed | T2S, full window (T = 20) energy (nJ) / macro-F1 | Vanilla, fixed point energy (nJ) / macro-F1 | Win |
|--:|--:|--:|---|
| 42 | 1101 / 92.76 | 532 / 90.56 | no |
| 43 | 1082 / 93.00 | 538 / 91.63 | no |
| 44 | 1023 / 93.40 | 543 / 93.20 | no |
| 45 | 1081 / 93.19 | 550 / 91.26 | no |
| 46 | 1003 / 92.87 | 541 / 92.93 | no |

Five-seed means: T2S, full window (T = 20) 1058 ± 43 nJ, 93.04 ± 0.25; Vanilla 541 ± 6 nJ, 91.91 ± 1.12. Energy ratio 1.96 (margin -95.7% of Vanilla's energy; SANA-FE calibration error about 10%); macro-F1 difference +1.13.

**Dominance on five-seed means: no.** Per-seed wins: 0 of 5.

## toniot: T2S, early exit (C1 gate) against the fixed-point Vanilla (amendment 7)

| Seed | T2S, early exit (C1 gate) energy (nJ) / macro-F1 | Vanilla, fixed point energy (nJ) / macro-F1 | Win |
|--:|--:|--:|---|
| 42 | 871 / 92.76 (gate 0.99) | 532 / 90.56 | no |
| 43 | 747 / 92.31 (gate 0.95) | 538 / 91.63 | no |
| 44 | 774 / 92.64 (gate 0.95) | 543 / 93.20 | no |
| 45 | 886 / 92.68 (gate 0.99) | 550 / 91.26 | no |
| 46 | 881 / 92.86 (gate 0.99) | 541 / 92.93 | no |

Five-seed means: T2S, early exit (C1 gate) 832 ± 66 nJ, 92.65 ± 0.21; Vanilla 541 ± 6 nJ, 91.91 ± 1.12. Energy ratio 1.54 (margin -53.8% of Vanilla's energy; SANA-FE calibration error about 10%); macro-F1 difference +0.73.

**Dominance on five-seed means: no.** Per-seed wins: 0 of 5.

## toniot: T2S, T = 10 against the fixed-point Vanilla (amendment 7)

| Seed | T2S, T = 10 energy (nJ) / macro-F1 | Vanilla, fixed point energy (nJ) / macro-F1 | Win |
|--:|--:|--:|---|
| 42 | 733 / 93.61 | 532 / 90.56 | no |
| 43 | 738 / 93.04 | 538 / 91.63 | no |
| 44 | 740 / 94.15 | 543 / 93.20 | no |
| 45 | 694 / 93.84 | 550 / 91.26 | no |
| 46 | 696 / 93.24 | 541 / 92.93 | no |

Five-seed means: T2S, T = 10 720 ± 23 nJ, 93.58 ± 0.45; Vanilla 541 ± 6 nJ, 91.91 ± 1.12. Energy ratio 1.33 (margin -33.1% of Vanilla's energy; SANA-FE calibration error about 10%); macro-F1 difference +1.66.

**Dominance on five-seed means: no.** Per-seed wins: 0 of 5.

## ciciot2023: T2S, full window (T = 20) against the fixed-point Vanilla (amendment 7)

| Seed | T2S, full window (T = 20) energy (nJ) / macro-F1 | Vanilla, fixed point energy (nJ) / macro-F1 | Win |
|--:|--:|--:|---|
| 42 | 758 / 79.18 | 446 / 62.48 | no |
| 43 | 836 / 74.71 | 459 / 60.62 | no |
| 44 | 824 / 72.09 | 455 / 61.28 | no |
| 45 | 726 / 75.70 | 464 / 61.18 | no |
| 46 | 770 / 74.39 | 454 / 61.86 | no |

Five-seed means: T2S, full window (T = 20) 783 ± 46 nJ, 75.22 ± 2.58; Vanilla 455 ± 7 nJ, 61.48 ± 0.71. Energy ratio 1.72 (margin -71.9% of Vanilla's energy; SANA-FE calibration error about 10%); macro-F1 difference +13.73.

**Dominance on five-seed means: no.** Per-seed wins: 0 of 5.

## ciciot2023: T2S, early exit (C1 gate) against the fixed-point Vanilla (amendment 7)

| Seed | T2S, early exit (C1 gate) energy (nJ) / macro-F1 | Vanilla, fixed point energy (nJ) / macro-F1 | Win |
|--:|--:|--:|---|
| 42 | 437 / 78.55 (gate 0.9) | 446 / 62.48 | yes |
| 43 | 836 / 74.71 (gate disabled: full window) | 459 / 60.62 | no |
| 44 | 718 / 71.99 (gate 0.95) | 455 / 61.28 | no |
| 45 | 417 / 75.28 (gate 0.95) | 464 / 61.18 | yes |
| 46 | 570 / 74.12 (gate 0.99) | 454 / 61.86 | no |

Five-seed means: T2S, early exit (C1 gate) 595 ± 180 nJ, 74.93 ± 2.38; Vanilla 455 ± 7 nJ, 61.48 ± 0.71. Energy ratio 1.31 (margin -30.8% of Vanilla's energy; SANA-FE calibration error about 10%); macro-F1 difference +13.45.

**Dominance on five-seed means: no.** Per-seed wins: 2 of 5.

## Seed 42 rerun against the frontier and A3 values

- ciciomt2024 t2s_exit: energy 246.2755 vs 246.2755, macro-F1 89.7988 vs 89.7988: identical
- ciciomt2024 t2s_full: energy 602.3421 vs 602.3421, macro-F1 90.7303 vs 90.7303: identical
- ciciomt2024 vanilla_fixed: energy 422.8865 vs 422.8865, macro-F1 73.9850 vs 73.9850: identical
- ciciot2023 t2s_exit: energy 437.1399 vs 437.1399, macro-F1 78.5501 vs 78.5501: identical
- ciciot2023 t2s_full: energy 757.6392 vs 757.6392, macro-F1 79.1774 vs 79.1774: identical
- ciciot2023 vanilla_fixed: energy 446.0033 vs 446.0033, macro-F1 62.4829 vs 62.4829: identical
- edgeiiot T10: energy 429.6242 vs 429.6242, macro-F1 81.3581 vs 81.3581: identical
- edgeiiot t2s_exit: energy 731.8614 vs 731.8614, macro-F1 84.1119 vs 84.1119: identical
- edgeiiot t2s_full: energy 731.8614 vs 731.8614, macro-F1 84.1119 vs 84.1119: identical
- edgeiiot vanilla_fixed: energy 521.4703 vs 521.4703, macro-F1 62.1196 vs 62.1196: identical
- toniot T10: energy 732.6166 vs 732.6166, macro-F1 93.6091 vs 93.6091: identical
- toniot t2s_exit: energy 871.1273 vs 871.1273, macro-F1 92.7573 vs 92.7573: identical
- toniot t2s_full: energy 1101.0820 vs 1101.0820, macro-F1 92.7621 vs 92.7621: identical
- toniot vanilla_fixed: energy 532.1431 vs 532.1431, macro-F1 90.5564 vs 90.5564: identical

