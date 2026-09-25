| dataset | dnn_level_fp32 | dnn_tq_fp32 | t2s_fp32 | gap_level_fp32 | gap_tq_fp32 | chg_fp32 | dnn_level_pt | dnn_tq_pt | t2s_pt | gap_level_pt | gap_tq_pt | chg_pt | dnn_level_pc | dnn_tq_pc | t2s_pc | gap_level_pc | gap_tq_pc | chg_pc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ciciot2023 | 72.30 | 72.66 | 72.22 | +0.08 | +0.45 | +0.36 | 72.43 | 72.56 | 72.22 | +0.22 | +0.34 | +0.13 | 72.17 | 72.39 | 72.27 | -0.10 | +0.13 | +0.22 |
| ciciomt2024 | 89.54 | 89.06 | 89.18 | +0.36 | -0.12 | -0.48 | 89.24 | 89.16 | 89.42 | -0.18 | -0.27 | -0.08 | 89.59 | 89.10 | 89.05 | +0.54 | +0.05 | -0.49 |
| edgeiiot | 86.81 | 85.32 | 84.46 | +2.35 | +0.86 | -1.49 | 80.83 | 83.99 | 83.95 | -3.11 | +0.04 | +3.15 | 85.59 | 85.36 | 84.42 | +1.16 | +0.93 | -0.23 |
| toniot | 94.44 | 94.23 | 93.02 | +1.42 | +1.21 | -0.21 | 94.06 | 93.91 | 92.98 | +1.08 | +0.92 | -0.16 | 93.97 | 94.26 | 93.03 | +0.94 | +1.23 | +0.29 |
| IoT mean | 85.77 | 85.32 | 84.72 | +1.05 | +0.60 | -0.45 | 84.14 | 84.90 | 84.64 | -0.50 | +0.26 | +0.76 | 85.33 | 85.28 | 84.69 | +0.64 | +0.59 | -0.05 |

Columns: five-seed means of macro-F1 (%); level = production DNN-Tree input, tq = spike-time input; gap = dense minus Tree-to-Spike; chg = tq minus level. Outcome decided on fp32.

Pre-registered outcome: C: input resolution explains 43% of the gap

Single-dataset dominance (> half of a change) and sign notes:
- the fp32 change has mixed signs across datasets (ciciot2023 +0.36, ciciomt2024 -0.48, edgeiiot -1.49, toniot -0.21); its IoT mean is a net
- edgeiiot accounts for 82% of the fp32 change (-1.49 of the four-dataset sum -1.82; IoT mean -0.45)
- the per-tensor int8 change has mixed signs across datasets (ciciot2023 +0.13, ciciomt2024 -0.08, edgeiiot +3.15, toniot -0.16); its IoT mean is a net
- edgeiiot accounts for 104% of the per-tensor int8 change (+3.15 of the four-dataset sum +3.04; IoT mean +0.76)
- the per-channel int8 change has mixed signs across datasets (ciciot2023 +0.22, ciciomt2024 -0.49, edgeiiot -0.23, toniot +0.29); its IoT mean is a net
- ciciomt2024 accounts for 238% of the per-channel int8 change (-0.49 of the four-dataset sum -0.21; IoT mean -0.05)
- edgeiiot accounts for 111% of the per-channel int8 change (-0.23 of the four-dataset sum -0.21; IoT mean -0.05)
