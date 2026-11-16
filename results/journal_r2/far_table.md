# Macro-F1, FAR and DR, IoT four (five seeds)

Mean ± sample SD over seeds 42-46, test split, in percent. FAR = benign flows flagged as attacks; DR = attack flows flagged as attacks. CICIoT2023 neural models are the 20-epoch block (A1).

## CICIoT2023

| System | Precision | Macro-F1 | FAR | DR |
|---|---|--:|--:|--:|
| Vanilla SNN | fp32 | 62.72 ± 0.94 | 5.62 ± 2.05 | 99.30 ± 0.08 |
| Tree-to-Spike | int8 weights | 75.20 ± 2.62 | 0.70 ± 0.49 | 99.46 ± 0.09 |
| DNN-Tree | int8 weights | 74.75 ± 0.66 | 0.67 ± 0.47 | 99.54 ± 0.07 |
| TASNN | fp32 | 60.00 ± 0.28 | 7.30 ± 0.82 | 99.45 ± 0.03 |
| Full GBDT | native | 82.95 ± 0.38 | 6.96 ± 0.13 | 99.88 ± 0.01 |
| Compact GBDT (validation-selected rounds) | native | 82.08 ± 0.36 | 7.23 ± 0.19 | 99.88 ± 0.01 |
| PLE-MLP | fp32 | 83.89 ± 0.72 | 3.57 ± 0.96 | 99.79 ± 0.02 |
| PLE-MLP | per-tensor int8 | 83.70 ± 1.03 | 3.90 ± 1.09 | 99.80 ± 0.02 |
| PLE-MLP | per-channel int8 | 83.79 ± 0.76 | 3.53 ± 1.04 | 99.79 ± 0.02 |

## CICIoMT2024

| System | Precision | Macro-F1 | FAR | DR |
|---|---|--:|--:|--:|
| Vanilla SNN | fp32 | 72.61 ± 1.03 | 7.61 ± 1.86 | 99.80 ± 0.03 |
| Tree-to-Spike | int8 weights | 89.42 ± 3.24 | 0.87 ± 1.28 | 99.78 ± 0.05 |
| DNN-Tree | int8 weights | 89.24 ± 2.72 | 0.10 ± 0.07 | 99.80 ± 0.05 |
| TASNN | fp32 | 74.18 ± 4.80 | 7.15 ± 3.95 | 99.77 ± 0.04 |
| Full GBDT | native | 90.54 ± 0.23 | 5.40 ± 0.32 | 99.95 ± 0.00 |
| Compact GBDT (validation-selected rounds) | native | 89.36 ± 0.18 | 7.13 ± 0.33 | 99.96 ± 0.00 |
| PLE-MLP | fp32 | 94.58 ± 1.30 | 1.63 ± 1.17 | 99.92 ± 0.01 |
| PLE-MLP | per-tensor int8 | 94.83 ± 1.11 | 1.54 ± 1.26 | 99.92 ± 0.02 |
| PLE-MLP | per-channel int8 | 94.46 ± 1.36 | 1.64 ± 1.13 | 99.92 ± 0.01 |

## Edge-IIoTset

| System | Precision | Macro-F1 | FAR | DR |
|---|---|--:|--:|--:|
| Vanilla SNN | fp32 | 75.39 ± 2.06 | 0.02 ± 0.01 | 85.26 ± 1.76 |
| Tree-to-Spike | int8 weights | 83.94 ± 0.17 | 0.04 ± 0.01 | 88.54 ± 0.13 |
| DNN-Tree | int8 weights | 80.60 ± 5.42 | 4.47 ± 6.96 | 89.49 ± 0.89 |
| TASNN | fp32 | 78.37 ± 1.56 | 0.03 ± 0.02 | 88.90 ± 0.37 |
| Full GBDT | native | 74.75 ± 0.26 | 15.67 ± 0.08 | 99.95 ± 0.00 |
| Compact GBDT (validation-selected rounds) | native | 75.50 ± 1.03 | 16.94 ± 1.61 | 99.95 ± 0.02 |
| PLE-MLP | fp32 | 89.49 ± 0.17 | 0.01 ± 0.00 | 89.35 ± 0.14 |
| PLE-MLP | per-tensor int8 | 88.84 ± 0.34 | 0.01 ± 0.00 | 89.34 ± 0.15 |
| PLE-MLP | per-channel int8 | 89.53 ± 0.25 | 0.01 ± 0.00 | 89.35 ± 0.14 |

## TON-IoT

| System | Precision | Macro-F1 | FAR | DR |
|---|---|--:|--:|--:|
| Vanilla SNN | fp32 | 92.04 ± 1.00 | 0.62 ± 0.12 | 99.53 ± 0.16 |
| Tree-to-Spike | int8 weights | 92.98 ± 0.23 | 0.58 ± 0.13 | 99.53 ± 0.09 |
| DNN-Tree | int8 weights | 94.06 ± 1.17 | 3.29 ± 6.19 | 99.59 ± 0.25 |
| TASNN | fp32 | 90.37 ± 2.68 | 0.70 ± 0.13 | 99.58 ± 0.11 |
| Full GBDT | native | 96.67 ± 0.28 | 0.34 ± 0.05 | 99.95 ± 0.01 |
| Compact GBDT (validation-selected rounds) | native | 95.75 ± 0.38 | 0.50 ± 0.06 | 99.94 ± 0.02 |
| PLE-MLP | fp32 | 96.28 ± 0.27 | 0.31 ± 0.05 | 99.89 ± 0.02 |
| PLE-MLP | per-tensor int8 | 96.26 ± 0.25 | 0.31 ± 0.05 | 99.88 ± 0.03 |
| PLE-MLP | per-channel int8 | 96.29 ± 0.26 | 0.31 ± 0.04 | 99.89 ± 0.02 |
