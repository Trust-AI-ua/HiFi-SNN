| Dataset | Tree-to-Spike | DNN-Tree | Vanilla SNN | TASNN |
|---|---|---|---|---|
| CICIoT2023 | 72.22 ± 1.81 (2) | 72.43 ± 0.51 (1) | 62.82 ± 0.84 (3) | 59.09 ± 0.82 (4) |
| CICIoMT2024 | 89.42 ± 3.24 (1) | 89.24 ± 2.72 (2) | 72.61 ± 1.03 (4) | 74.18 ± 4.80 (3) |
| Edge-IIoTset | 83.94 ± 0.17 (1) | 80.60 ± 5.42 (2) | 75.39 ± 2.06 (4) | 78.37 ± 1.56 (3) |
| TON-IoT | 92.98 ± 0.23 (2) | 94.06 ± 1.17 (1) | 92.04 ± 1.00 (3) | 90.37 ± 2.68 (4) |
| **Mean rank** | 1.50 | 1.50 | 3.50 | 3.50 |

Friedman chi2_F = 9.60 (df 3), exact p = 0.0069 (asymptotic 0.0223); Iman-Davenport F_F = 12.00, p = 0.0017. Macro-F1, five-seed mean ± SD (rank). No post-hoc test at N = 4.

| Comparison | Wins | Mean diff | CICIoT2023 | CICIoMT2024 | Edge-IIoTset | TON-IoT |
|---|---|---|---|---|---|---|
| Tree-to-Spike vs DNN-Tree | 2/4 | +0.56 | -0.22 | +0.18 | +3.34 | -1.08 |
| Tree-to-Spike vs Vanilla SNN | 4/4 | +8.93 | +9.39 | +16.82 | +8.56 | +0.95 |
| Tree-to-Spike vs TASNN | 4/4 | +9.14 | +13.12 | +15.24 | +5.57 | +2.61 |
