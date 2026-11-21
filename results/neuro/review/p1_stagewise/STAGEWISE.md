# Stagewise predictions, Edge-IIoTset, seeds 42-46 (amendment 8, part 1)

Macro-F1, five-seed mean (SD). Every flow of the split. Checks (test): the float stages reproduce the records, integer Q = 8 equals the saved integer predictions, Lava equals the confirmation runs (full-test macro-F1 and Lava/SANA-FE's predictions on the 2,000-flow sample); the student's fp32 checkpoint projects onto the deployed one.

## test split

| Model | Stage | Seeds | Macro-F1 | Change from previous stage | Agreement with first stage |
|---|---|--:|--:|--:|--:|
| vanilla | float (GPU, record) | 5 | 75.39 (2.06) |  | 100.0% |
| vanilla | float64 (CPU) | 5 | 75.50 (2.16) | 0.12 (0.22) | 99.9% |
| vanilla | int8-projected weights | 5 | 60.39 (4.29) | -15.11 (4.00) | 88.4% |
| vanilla | projected + 12-bit leak | 5 | 60.40 (4.34) | 0.01 (0.08) | 88.4% |
| vanilla | integer Q = 8 | 5 | 59.78 (5.04) | -0.62 (1.07) | 87.3% |
| vanilla | Lava fixed point | 5 | 60.51 (5.02) | 0.73 (3.08) | 89.1% |
| vanilla | Lava, no saturation | 5 | 60.51 (5.02) |  | 89.1% |
| t2s | fp32 (before projection) | 5 | 84.27 (0.45) |  | 100.0% |
| t2s | int8-projected weights (deployed) | 5 | 83.94 (0.17) | -0.33 (0.53) | 99.6% |
| t2s | integer Q = 8 | 5 | 83.97 (0.22) | 0.03 (0.08) | 99.6% |
| t2s | Lava fixed point | 5 | 83.95 (0.18) | -0.02 (0.06) | 99.6% |
| t2s_T10 | int8-projected weights (deployed) | 5 | 81.46 (0.38) |  | 100.0% |
| t2s_T10 | integer Q = 8 | 5 | 81.38 (0.43) | -0.08 (0.11) | 100.0% |
| t2s_T10 | Lava fixed point | 5 | 81.47 (0.42) | 0.09 (0.10) | 99.9% |

### Tree-to-Spike minus Vanilla SNN at matched stages (test)

| Stage | Vanilla | Tree-to-Spike | Gap | Gap per seed |
|---|--:|--:|--:|---|
| float, before projection | 75.39 (2.06) | 84.27 (0.45) | 8.89 (2.12) | 42: 7.13, 43: 6.23, 44: 11.11, 45: 9.48, 46: 10.48 |
| int8-projected weights, float dynamics | 60.39 (4.29) | 83.94 (0.17) | 23.55 (4.27) | 42: 27.31, 43: 17.95, 44: 28.09, 45: 23.48, 46: 20.93 |
| integer Q = 8 | 59.78 (5.04) | 83.97 (0.22) | 24.19 (5.04) | 42: 28.06, 43: 17.28, 44: 29.16, 45: 25.68, 46: 20.76 |
| Lava fixed point | 60.51 (5.02) | 83.95 (0.18) | 23.44 (4.93) | 42: 21.99, 43: 17.12, 44: 29.17, 45: 27.66, 46: 21.27 |

### Per-class F1 (test, five-seed mean): where the Vanilla SNN loses

| Class | Test support (mean) | Vanilla float | Vanilla projected | Change at projection | Vanilla Lava | Student fp32 | Student Lava |
|---|--:|--:|--:|--:|--:|--:|--:|
| XSS | 3013 | 60.89 | 20.73 | -40.16 | 20.87 | 73.21 | 72.72 |
| SQL_injection | 10165 | 62.45 | 22.91 | -39.54 | 29.23 | 75.63 | 75.16 |
| Backdoor | 4805 | 85.13 | 52.87 | -32.26 | 52.60 | 91.49 | 91.04 |
| Ransomware | 1938 | 56.06 | 28.83 | -27.24 | 26.64 | 81.49 | 80.22 |
| Uploading | 7361 | 32.66 | 6.39 | -26.27 | 8.35 | 71.20 | 70.43 |
| Password | 9987 | 52.87 | 38.75 | -14.12 | 38.85 | 77.06 | 76.93 |
| DDoS_HTTP | 9709 | 67.94 | 56.35 | -11.59 | 53.23 | 70.61 | 70.97 |
| Vulnerability_scanner | 10005 | 92.07 | 83.64 | -8.43 | 85.78 | 94.34 | 93.85 |
| Normal | 272800 | 97.13 | 93.54 | -3.59 | 94.11 | 97.74 | 97.74 |
| Port_Scanning | 3995 | 85.69 | 82.21 | -3.48 | 80.85 | 91.22 | 91.22 |
| Fingerprinting | 171 | 62.66 | 59.84 | -2.82 | 57.49 | 56.26 | 55.47 |
| DDoS_TCP | 10012 | 99.99 | 99.57 | -0.42 | 99.30 | 99.99 | 100.00 |
| DDoS_UDP | 24314 | 99.99 | 99.97 | -0.02 | 99.97 | 99.91 | 99.90 |
| DDoS_ICMP | 13588 | 99.85 | 99.84 | -0.01 | 99.83 | 99.63 | 99.65 |

## val split

| Model | Stage | Seeds | Macro-F1 | Change from previous stage | Agreement with first stage |
|---|---|--:|--:|--:|--:|
| vanilla | float (GPU, record) | 5 | 75.69 (2.08) |  | 100.0% |
| vanilla | float64 (CPU) | 5 | 75.81 (2.20) | 0.12 (0.26) | 99.9% |
| vanilla | int8-projected weights | 5 | 60.38 (4.41) | -15.43 (4.17) | 88.4% |
| vanilla | projected + 12-bit leak | 5 | 60.40 (4.48) | 0.02 (0.09) | 88.3% |
| vanilla | integer Q = 8 | 5 | 59.81 (5.18) | -0.59 (1.02) | 87.2% |
| vanilla | Lava fixed point | 5 | 60.47 (5.10) | 0.66 (3.34) | 89.1% |
| vanilla | Lava, no saturation | 5 | 60.47 (5.10) |  | 89.1% |

- seed 42: Lava hidden-membrane saturation on 0.000% of test flows; output spike-count ties in the float record on 1.73%; weight scales s_h 0.1294, s_o 0.4615
- seed 43: Lava hidden-membrane saturation on 0.000% of test flows; output spike-count ties in the float record on 1.70%; weight scales s_h 0.0908, s_o 0.5423
- seed 44: Lava hidden-membrane saturation on 0.000% of test flows; output spike-count ties in the float record on 2.13%; weight scales s_h 0.0823, s_o 0.5875
- seed 45: Lava hidden-membrane saturation on 0.000% of test flows; output spike-count ties in the float record on 1.91%; weight scales s_h 0.1163, s_o 0.4670
- seed 46: Lava hidden-membrane saturation on 0.000% of test flows; output spike-count ties in the float record on 1.58%; weight scales s_h 0.1156, s_o 0.5055
