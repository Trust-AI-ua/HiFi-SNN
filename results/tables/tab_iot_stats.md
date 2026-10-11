| Dataset | Tree-to-Spike | DNN-Tree | Vanilla SNN | TASNN |
|---|---|---|---|---|
| CICIoT2023 | 75.20 ± 2.62 (1) | 74.75 ± 0.66 (2) | 62.72 ± 0.94 (3) | 60.00 ± 0.28 (4) |
| CICIoMT2024 | 89.42 ± 3.24 (1) | 89.24 ± 2.72 (2) | 72.61 ± 1.03 (4) | 74.18 ± 4.80 (3) |
| Edge-IIoTset | 83.94 ± 0.17 (1) | 80.60 ± 5.42 (2) | 75.39 ± 2.06 (4) | 78.37 ± 1.56 (3) |
| TON-IoT | 92.98 ± 0.23 (2) | 94.06 ± 1.17 (1) | 92.04 ± 1.00 (3) | 90.37 ± 2.68 (4) |
| **Mean rank** | 1.25 | 1.75 | 3.50 | 3.50 |

Friedman chi2_F = 9.90 (df 3), exact p = 0.0062 (asymptotic 0.0194); Iman-Davenport F_F = 14.14, p = 0.0009. Macro-F1, five-seed mean ± SD (rank). No post-hoc test at N = 4.

| Comparison | Wins | Mean diff | CICIoT2023 | CICIoMT2024 | Edge-IIoTset | TON-IoT |
|---|---|---|---|---|---|---|
| Tree-to-Spike vs DNN-Tree | 3/4 | +0.72 | +0.45 | +0.18 | +3.34 | -1.08 |
| Tree-to-Spike vs Vanilla SNN | 4/4 | +9.70 | +12.48 | +16.82 | +8.56 | +0.95 |
| Tree-to-Spike vs TASNN | 4/4 | +9.66 | +15.21 | +15.24 | +5.57 | +2.61 |
