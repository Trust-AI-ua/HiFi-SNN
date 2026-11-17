# Matched false-alarm check (revision 2, step 3b): results

Pre-registration: `results/preregistration/2026-11-17_r2_matched_far.md` (rule R5; amendment 1 adds (D), (E)).
Five-seed means (42 to 46), test split, percent. Thresholds are set on validation; test is scored once per threshold.

## (A) to (D), per dataset

| Dataset | T2S argmax | GBDT argmax | GBDT (A) matched | GBDT (B) val FAR<=1% | T2S (D) val FAR<=1% | GBDT (C) at T2S test FAR |
|---|--:|--:|--:|--:|--:|--:|
| CICIoMT2024 | 89.42 / 0.87 / 99.78 | 89.21 / 7.97 / 99.96 | 93.77 / 1.59 / 99.88 | 91.31 / 4.24 / 99.91 | 89.61 / 3.46 / 99.84 | 94.62 / 0.84 / 99.87 |
| CICIoT2023 | 75.20 / 0.70 / 99.46 | 77.11 / 9.18 / 99.88 | 81.26 / 0.74 / 99.52 | 80.21 / 1.10 / 99.59 | 75.36 / 1.13 / 99.52 | 81.23 / 0.70 / 99.53 |
| Edge-IIoTset | 83.94 / 0.04 / 88.54 | 80.15 / 16.12 / 99.90 | 86.07 / 0.03 / 87.63 | 85.91 / 0.10 / 87.82 | 83.70 / 0.36 / 89.01 | 86.07 / 0.04 / 87.65 |
| TON-IoT | 92.98 / 0.58 / 99.53 | 94.96 / 0.73 / 99.84 | 95.02 / 0.63 / 99.81 | 94.86 / 0.99 / 99.89 | 92.91 / 1.05 / 99.70 | 95.06 / 0.58 / 99.81 |
| IoT mean | 85.39 / 0.55 / 96.83 | 85.36 / 8.50 / 99.89 | 89.03 / 0.75 / 96.71 | 88.07 / 1.61 / 96.80 | 85.40 / 1.50 / 97.02 | 89.24 / 0.54 / 96.71 |

Cells are macro-F1 / FAR / DR. (C) is a readout on the test ROC curve, not a deployable operating point.

Benign-versus-attack test ROC AUC (score 1 - p_benign), percent:

| Dataset | GBDT | Tree-to-Spike |
|---|--:|--:|
| CICIoMT2024 | 99.99 | 99.98 |
| CICIoT2023 | 99.96 | 99.94 |
| Edge-IIoTset | 99.32 | 99.31 |
| TON-IoT | 99.98 | 99.95 |
| IoT mean | 99.81 | 99.80 |

## Rule R5

- CICIoMT2024: matched GBDT 93.77 vs Tree-to-Spike 89.42 (+4.35); advantage does not hold; Tree-to-Spike ahead on 0/5 seeds.
- CICIoT2023: matched GBDT 81.26 vs Tree-to-Spike 75.20 (+6.06); advantage does not hold; Tree-to-Spike ahead on 0/5 seeds.
- Edge-IIoTset: matched GBDT 86.07 vs Tree-to-Spike 83.94 (+2.13); advantage does not hold; Tree-to-Spike ahead on 0/5 seeds.
- TON-IoT: matched GBDT 95.02 vs Tree-to-Spike 92.98 (+2.04); advantage does not hold; Tree-to-Spike ahead on 0/5 seeds.
- IoT mean: matched GBDT 89.03 vs Tree-to-Spike 85.39 (+3.64).

**Verdict:** the false-alarm advantage holds on 0/4 datasets and not on the IoT mean. The claim "with far fewer false alarms" is not supported; per R5 the paper reports the matched-FAR results and drops it as a general claim.

## (E) Macro-F1 against FAR (amendment 1, descriptive)

Both models under the same benign-threshold rule, tuned on validation to each FAR target. Cells are test macro-F1 / test FAR; the last column counts seeds on which Tree-to-Spike has the higher macro-F1.

### CICIoMT2024

| Target FAR | GBDT | Tree-to-Spike | Difference (T2S - GBDT) | T2S ahead |
|--:|--:|--:|--:|--:|
| 0.1% | 95.01 / 0.36 | 88.07 / 0.19 | -6.93 | 0/5 |
| 0.25% | 94.19 / 1.08 | 89.96 / 0.65 | -4.24 | 0/5 |
| 0.5% | 92.92 / 2.25 | 90.25 / 1.68 | -2.67 | 0/5 |
| 1% | 91.31 / 4.24 | 89.61 / 3.46 | -1.70 | 0/5 |
| 2% | 90.34 / 5.69 | 89.08 / 5.05 | -1.26 | 2/5 |

### CICIoT2023

| Target FAR | GBDT | Tree-to-Spike | Difference (T2S - GBDT) | T2S ahead |
|--:|--:|--:|--:|--:|
| 0.1% | 82.73 / 0.09 | 73.22 / 0.13 | -9.51 | 0/5 |
| 0.25% | 82.07 / 0.26 | 74.26 / 0.30 | -7.80 | 0/5 |
| 0.5% | 81.26 / 0.59 | 74.82 / 0.55 | -6.44 | 0/5 |
| 1% | 80.21 / 1.10 | 75.36 / 1.13 | -4.86 | 0/5 |
| 2% | 79.40 / 2.09 | 75.85 / 2.19 | -3.55 | 0/5 |

### Edge-IIoTset

| Target FAR | GBDT | Tree-to-Spike | Difference (T2S - GBDT) | T2S ahead |
|--:|--:|--:|--:|--:|
| 0.1% | 86.06 / 0.07 | 83.92 / 0.04 | -2.14 | 0/5 |
| 0.25% | 85.91 / 0.10 | 83.89 / 0.07 | -2.02 | 0/5 |
| 0.5% | 85.91 / 0.10 | 83.89 / 0.07 | -2.02 | 0/5 |
| 1% | 85.91 / 0.10 | 83.70 / 0.36 | -2.21 | 0/5 |
| 2% | 85.72 / 0.49 | 83.58 / 0.59 | -2.14 | 0/5 |

### TON-IoT

| Target FAR | GBDT | Tree-to-Spike | Difference (T2S - GBDT) | T2S ahead |
|--:|--:|--:|--:|--:|
| 0.1% | 92.30 / 0.15 | 85.89 / 0.08 | -6.42 | 1/5 |
| 0.25% | 94.93 / 0.26 | 90.71 / 0.22 | -4.22 | 0/5 |
| 0.5% | 95.08 / 0.56 | 92.86 / 0.46 | -2.22 | 0/5 |
| 1% | 94.86 / 0.99 | 92.91 / 1.05 | -1.95 | 0/5 |
| 2% | 94.38 / 1.97 | 92.66 / 2.08 | -1.72 | 0/5 |

### IoT mean

| Target FAR | GBDT | Tree-to-Spike | Difference (T2S - GBDT) | T2S ahead |
|--:|--:|--:|--:|--:|
| 0.1% | 89.02 / 0.17 | 82.78 / 0.11 | -6.25 | 1/20 |
| 0.25% | 89.28 / 0.43 | 84.71 / 0.31 | -4.57 | 0/20 |
| 0.5% | 88.79 / 0.88 | 85.46 / 0.69 | -3.34 | 0/20 |
| 1% | 88.07 / 1.61 | 85.40 / 1.50 | -2.68 | 0/20 |
| 2% | 87.46 / 2.56 | 85.29 / 2.48 | -2.17 | 2/20 |
