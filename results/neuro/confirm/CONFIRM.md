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

## Seed 42 rerun against the frontier and A3 values

- ciciomt2024 t2s_exit: energy 246.2755 vs 246.2755, macro-F1 89.7988 vs 89.7988: identical
- ciciomt2024 t2s_full: energy 602.3421 vs 602.3421, macro-F1 90.7303 vs 90.7303: identical
- ciciomt2024 vanilla_fixed: energy 422.8865 vs 422.8865, macro-F1 73.9850 vs 73.9850: identical
- edgeiiot T10: energy 429.6242 vs 429.6242, macro-F1 81.3581 vs 81.3581: identical
- edgeiiot t2s_exit: energy 731.8614 vs 731.8614, macro-F1 84.1119 vs 84.1119: identical
- edgeiiot t2s_full: energy 731.8614 vs 731.8614, macro-F1 84.1119 vs 84.1119: identical
- edgeiiot vanilla_fixed: energy 521.4703 vs 521.4703, macro-F1 62.1196 vs 62.1196: identical

