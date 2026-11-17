# Early-exit calibration and the repeated wrong-exit audit (revision 2, step 4)

Pre-registration: `results/preregistration/2026-11-17_r2_steps4_6.md` (R6). Tree-to-Spike, IoT four, seeds 42-46, test split, five-seed means. Wrong-exit = share of all test samples that cross the threshold with a wrong label (the published audit). Harmful = exits before T that are wrong where the full window is right.

Operating points: U = uncalibrated, theta 0.90 (published audit); U-gate = published validation gate; C1 = per-step temperature, theta 0.90; C1-gate = validation gate on C1 confidence; C1g = global temperature, theta 0.90; C2 = per-class thresholds.

## CICIoT2023

| Operating point | Wrong-exit | Harmful | Exit before T | Mean exit step | SOP reduction | Macro-F1 | Change vs full | FAR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| U | 51.19% | 50.69% | 99.2% | 5.98 | 67.9% | 55.38 | -19.83 | 12.88 |
| U-gate | 0.00% | 0.00% | 0.0% | 20.00 | 0.0% | 75.20 | +0.00 | 0.70 |
| C1 | 5.11% | 4.61% | 99.2% | 11.74 | 46.4% | 72.41 | -2.79 | 1.75 |
| C1-gate | 0.81% | 0.50% | 78.5% | 14.55 | 33.5% | 74.93 | -0.27 | 1.03 |
| C1g | 12.62% | 12.45% | 79.4% | 12.58 | 41.0% | 70.32 | -4.89 | 0.91 |
| C2 | 0.52% | 0.10% | 81.0% | 12.43 | 40.1% | 73.97 | -1.24 | 1.36 |

## CICIoMT2024

| Operating point | Wrong-exit | Harmful | Exit before T | Mean exit step | SOP reduction | Macro-F1 | Change vs full | FAR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| U | 19.25% | 19.11% | 99.6% | 4.77 | 69.7% | 81.12 | -8.31 | 0.79 |
| U-gate | 0.23% | 0.18% | 98.3% | 7.64 | 60.7% | 89.17 | -0.25 | 0.85 |
| C1 | 1.64% | 1.43% | 99.8% | 5.61 | 65.8% | 86.87 | -2.56 | 1.79 |
| C1-gate | 0.85% | 0.75% | 79.4% | 8.99 | 51.3% | 88.56 | -0.86 | 1.08 |
| C1g | 0.30% | 0.24% | 98.4% | 7.38 | 61.3% | 89.17 | -0.25 | 0.86 |
| C2 | 0.25% | 0.05% | 88.8% | 7.58 | 57.2% | 89.01 | -0.41 | 0.92 |

## Edge-IIoTset

| Operating point | Wrong-exit | Harmful | Exit before T | Mean exit step | SOP reduction | Macro-F1 | Change vs full | FAR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| U | 18.83% | 15.96% | 87.6% | 9.31 | 57.9% | 59.95 | -23.99 | 14.84 |
| U-gate | 0.00% | 0.00% | 0.0% | 20.00 | 0.0% | 83.94 | +0.00 | 0.04 |
| C1 | 5.32% | 2.64% | 89.0% | 13.51 | 47.1% | 74.67 | -9.27 | 0.50 |
| C1-gate | 0.28% | 0.02% | 26.5% | 18.97 | 9.7% | 83.83 | -0.11 | 0.04 |
| C1g | 5.05% | 4.03% | 63.7% | 14.06 | 35.2% | 73.75 | -10.19 | 2.61 |
| C2 | 1.91% | 0.42% | 72.3% | 13.68 | 41.4% | 82.03 | -1.91 | 0.06 |

## TON-IoT

| Operating point | Wrong-exit | Harmful | Exit before T | Mean exit step | SOP reduction | Macro-F1 | Change vs full | FAR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| U | 17.08% | 15.63% | 92.3% | 8.98 | 49.6% | 75.90 | -17.08 | 6.98 |
| U-gate | 0.23% | 0.16% | 19.9% | 18.49 | 8.3% | 92.81 | -0.18 | 0.58 |
| C1 | 4.81% | 3.73% | 92.4% | 13.23 | 39.7% | 89.03 | -3.95 | 0.72 |
| C1-gate | 0.83% | 0.38% | 75.1% | 16.07 | 28.9% | 92.59 | -0.39 | 0.59 |
| C1g | 4.22% | 3.78% | 53.7% | 14.41 | 26.4% | 88.77 | -4.21 | 0.77 |
| C2 | 1.52% | 0.40% | 73.2% | 12.92 | 34.3% | 92.48 | -0.50 | 0.84 |

## IoT mean

| Operating point | Wrong-exit | Harmful | Exit before T | Mean exit step | SOP reduction | Macro-F1 | Change vs full | FAR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| U | 26.59% | 25.35% | 94.7% | 7.26 | 61.3% | 68.09 | -17.30 | 8.87 |
| U-gate | 0.11% | 0.09% | 29.5% | 16.53 | 17.3% | 85.28 | -0.11 | 0.55 |
| C1 | 4.22% | 3.10% | 95.1% | 11.02 | 49.7% | 80.75 | -4.64 | 1.19 |
| C1-gate | 0.69% | 0.41% | 64.9% | 14.64 | 30.8% | 84.98 | -0.41 | 0.69 |
| C1g | 5.55% | 5.13% | 73.8% | 12.11 | 41.0% | 80.50 | -4.89 | 1.29 |
| C2 | 1.05% | 0.25% | 78.8% | 11.65 | 43.2% | 84.37 | -1.01 | 0.79 |

## Calibration of the prefix posterior (test, IoT mean over datasets and seeds)

| t | ECE before | ECE after C1 | Brier before | Brier after | NLL before | NLL after |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 32.48 | 18.95 | 0.8426 | 0.7413 | 2.0172 | 1.7152 |
| 5 | 32.49 | 20.43 | 0.6935 | 0.5735 | 1.6875 | 1.2408 |
| 10 | 16.45 | 11.64 | 0.3872 | 0.3324 | 1.0208 | 0.7062 |
| 20 | 2.24 | 0.40 | 0.0393 | 0.0355 | 0.0935 | 0.0810 |

## R6

- **C1**: wrong-exit lower than U on 4/4 datasets, IoT mean 4.22% vs 26.59%; SOP reduction 49.7% (>= 10% required); worst macro-F1 change -9.27 (>= -1.0 required). **Fails.**
- **C1-gate**: wrong-exit lower than U on 4/4 datasets, IoT mean 0.69% vs 26.59%; SOP reduction 30.8% (>= 10% required); worst macro-F1 change -0.86 (>= -1.0 required). **Passes.**
- **C1g**: wrong-exit lower than U on 4/4 datasets, IoT mean 5.55% vs 26.59%; SOP reduction 41.0% (>= 10% required); worst macro-F1 change -10.19 (>= -1.0 required). **Fails.**
- **C2**: wrong-exit lower than U on 4/4 datasets, IoT mean 1.05% vs 26.59%; SOP reduction 43.2% (>= 10% required); worst macro-F1 change -1.91 (>= -1.0 required). **Fails.**

**Decision: early exit is kept, with C1-gate** (lowest IoT-mean wrong-exit among passing calibrators).
