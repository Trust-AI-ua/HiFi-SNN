| dataset | dnn_level_fp32 | dnn_tq_fp32 | t2s_fp32 | gap_level_fp32 | gap_tq_fp32 | chg_fp32 | dnn_level_pt | dnn_tq_pt | t2s_pt | gap_level_pt | gap_tq_pt | chg_pt | dnn_level_pc | dnn_tq_pc | t2s_pc | gap_level_pc | gap_tq_pc | chg_pc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ciciot2023 | 75.29 | 75.71 | 75.18 | +0.11 | +0.53 | +0.42 | 74.75 | 75.45 | 75.20 | -0.45 | +0.24 | +0.69 | 75.17 | 75.69 | 75.22 | -0.05 | +0.47 | +0.52 |
| ciciomt2024 | 89.54 | 89.06 | 89.18 | +0.36 | -0.12 | -0.48 | 89.24 | 89.16 | 89.42 | -0.18 | -0.27 | -0.08 | 89.59 | 89.10 | 89.05 | +0.54 | +0.05 | -0.49 |
| edgeiiot | 87.13 | 85.32 | 84.27 | +2.86 | +1.05 | -1.81 | 80.60 | 83.99 | 83.94 | -3.34 | +0.05 | +3.39 | 85.49 | 85.36 | 84.33 | +1.16 | +1.03 | -0.13 |
| toniot | 94.44 | 94.23 | 93.02 | +1.42 | +1.21 | -0.21 | 94.06 | 93.91 | 92.98 | +1.08 | +0.92 | -0.16 | 93.97 | 94.26 | 93.03 | +0.94 | +1.23 | +0.29 |
| IoT mean | 86.60 | 86.08 | 85.41 | +1.19 | +0.67 | -0.52 | 84.66 | 85.62 | 85.39 | -0.72 | +0.24 | +0.96 | 86.05 | 86.10 | 85.41 | +0.65 | +0.69 | +0.05 |

Columns: five-seed means of macro-F1 (%); level = production DNN-Tree input, tq = spike-time input; gap = dense minus Tree-to-Spike; chg = tq minus level. Outcome decided on fp32.

Pre-registered outcome: C: input resolution explains 44% of the gap

Single-dataset dominance (> half of a change) and sign notes:
- the fp32 change has mixed signs across datasets (ciciot2023 +0.42, ciciomt2024 -0.48, edgeiiot -1.81, toniot -0.21); its IoT mean is a net
- edgeiiot accounts for 87% of the fp32 change (-1.81 of the four-dataset sum -2.08; IoT mean -0.52)
- the per-tensor int8 change has mixed signs across datasets (ciciot2023 +0.69, ciciomt2024 -0.08, edgeiiot +3.39, toniot -0.16); its IoT mean is a net
- edgeiiot accounts for 88% of the per-tensor int8 change (+3.39 of the four-dataset sum +3.84; IoT mean +0.96)
- the per-channel int8 change has mixed signs across datasets (ciciot2023 +0.52, ciciomt2024 -0.49, edgeiiot -0.13, toniot +0.29); its IoT mean is a net
- ciciot2023 accounts for 273% of the per-channel int8 change (+0.52 of the four-dataset sum +0.19; IoT mean +0.05)
- toniot accounts for 151% of the per-channel int8 change (+0.29 of the four-dataset sum +0.19; IoT mean +0.05)
