### Encoder ablation - what does the teacher actually contribute?

Mean over 5 seeds. **SOPs include the input->layer-1 term** (channels x hidden[0]); **Energy = SOPs × 0.9 pJ.** Arms: **tree** = XGBoost split edges + importance-pruned/budgeted channels (ours). **quantile** = equal-frequency edges with channels IDENTICAL to tree (same pruning/budget/jitter; per-feature edge count matched) - isolates whether the teacher's split placement (its 'ruler') matters. **nokd** = tree encoder trained WITHOUT the teacher-posterior loss terms (a_kd = a_rank = 0) - isolates the teacher's soft labels. **uniform** = tree edges, 1 channel per feature-with-edges (NOTE: changes pruning AND allocation together, so it is confounded; kept for completeness). **all** = tree edges, max channels per feature (dense upper bound). **GBDT (teacher)** = the XGBoost teacher's own test accuracy/F1, the reference ceiling (not a spiking model; no energy columns). **TTFS** = mean encoder time-to-first-spike (/20, lower = earlier). **Exit step** = mean early-exit timestep of the trained student (/20, lower = decides sooner).

| Dataset | Encoder | Channels | SOPs/sample | Energy (pJ) | Accuracy | Macro-F1 | TTFS (/20) | Exit step (/20) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| KDDCup99 | **tree** | 70 | 8,407 | 7,567 | 92.6 | 61.8 | 16.1 | 1.3 |
| KDDCup99 | quantile | 70 | 15,044 | 13,540 | 92.2 | 52.7 | 3.0 | 2.3 |
| KDDCup99 | nokd | 70 | 7,853 | 7,068 | 92.5 | 61.1 | 16.1 | 1.1 |
| KDDCup99 | uniform | 77 | 9,230 | 8,307 | 92.5 | 61.1 | 16.6 | 1.4 |
| KDDCup99 | all | 231 | 19,450 | 17,505 | 92.5 | 63.2 | 16.1 | 1.9 |
| KDDCup99 | GBDT (teacher) | -- | -- | -- | 92.7 | 63.1 | -- | -- |
| NSL-KDD | **tree** | 68 | 10,708 | 9,637 | 77.3 | 58.5 | 16.4 | 2.5 |
| NSL-KDD | quantile | 68 | 15,281 | 13,753 | 76.6 | 50.8 | 3.6 | 3.8 |
| NSL-KDD | nokd | 68 | 11,436 | 10,293 | 76.8 | 56.6 | 16.4 | 1.9 |
| NSL-KDD | uniform | 78 | 12,222 | 10,999 | 77.7 | 56.3 | 16.8 | 2.1 |
| NSL-KDD | all | 235 | 24,548 | 22,093 | 78.4 | 59.8 | 16.3 | 2.8 |
| NSL-KDD | GBDT (teacher) | -- | -- | -- | 78.4 | 59.9 | -- | -- |
| CTU-13 | **tree** | 24 | 2,922 | 2,630 | 100.0 | 100.0 | 8.5 | 1.1 |
| CTU-13 | quantile | 24 | 3,894 | 3,505 | 100.0 | 100.0 | 8.3 | 2.6 |
| CTU-13 | nokd | 24 | 2,060 | 1,854 | 100.0 | 100.0 | 8.5 | 1.1 |
| CTU-13 | uniform | 27 | 3,285 | 2,956 | 100.0 | 100.0 | 8.8 | 1.1 |
| CTU-13 | all | 82 | 6,639 | 5,975 | 100.0 | 100.0 | 8.8 | 1.0 |
| CTU-13 | GBDT (teacher) | -- | -- | -- | 100.0 | 100.0 | -- | -- |
| ISCX-IDS2012 | **tree** | 52 | 7,810 | 7,029 | 99.7 | 98.0 | 16.0 | 1.8 |
| ISCX-IDS2012 | quantile | 52 | 12,838 | 11,555 | 99.6 | 96.6 | 1.9 | 2.9 |
| ISCX-IDS2012 | nokd | 52 | 7,089 | 6,380 | 99.7 | 97.7 | 16.0 | 1.1 |
| ISCX-IDS2012 | uniform | 63 | 8,275 | 7,447 | 99.7 | 98.0 | 17.0 | 1.7 |
| ISCX-IDS2012 | all | 190 | 16,891 | 15,202 | 99.8 | 98.5 | 16.5 | 2.8 |
| ISCX-IDS2012 | GBDT (teacher) | -- | -- | -- | 99.8 | 98.8 | -- | -- |
| CIC-IDS2017 | **tree** | 62 | 12,002 | 10,802 | 99.8 | 98.4 | 14.4 | 7.5 |
| CIC-IDS2017 | quantile | 62 | 13,112 | 11,801 | 99.8 | 98.5 | 10.6 | 6.9 |
| CIC-IDS2017 | nokd | 62 | 10,493 | 9,444 | 99.8 | 98.1 | 14.4 | 4.8 |
| CIC-IDS2017 | uniform | 68 | 12,513 | 11,261 | 99.8 | 98.4 | 14.6 | 7.9 |
| CIC-IDS2017 | all | 204 | 20,219 | 18,197 | 99.9 | 98.8 | 14.3 | 7.4 |
| CIC-IDS2017 | GBDT (teacher) | -- | -- | -- | 99.9 | 99.6 | -- | -- |
| CSE-CIC-IDS2018 | **tree** | 66 | 13,595 | 12,236 | 98.0 | 75.9 | 14.1 | 12.7 |
| CSE-CIC-IDS2018 | quantile | 66 | 14,492 | 13,043 | 98.2 | 73.8 | -- | 15.7 |
| CSE-CIC-IDS2018 | nokd | 66 | 15,032 | 13,529 | 98.0 | 69.0 | -- | 6.3 |
| CSE-CIC-IDS2018 | uniform | 71 | 13,251 | 11,926 | 95.4 | 74.9 | 14.5 | 13.3 |
| CSE-CIC-IDS2018 | all | 213 | 23,686 | 21,318 | 98.2 | 77.3 | 14.1 | 12.3 |
| CSE-CIC-IDS2018 | GBDT (teacher) | -- | -- | -- | 82.7 | 70.4 | -- | -- |
| UNSW-NB15 | **tree** | 156 | 24,532 | 22,079 | 74.5 | 41.7 | 17.4 | 9.1 |
| UNSW-NB15 | quantile | 156 | 35,600 | 32,040 | 74.0 | 40.0 | -- | 12.6 |
| UNSW-NB15 | nokd | 156 | 23,953 | 21,558 | 77.6 | 40.8 | -- | 5.1 |
| UNSW-NB15 | uniform | 190 | 26,029 | 23,426 | 76.4 | 43.0 | 17.8 | 6.6 |
| UNSW-NB15 | all | 570 | 54,234 | 48,811 | 75.1 | 43.4 | 17.2 | 11.4 |
| UNSW-NB15 | GBDT (teacher) | -- | -- | -- | 67.9 | 50.6 | -- | -- |
| CICIoT2023 | **tree** | 39 | 12,830 | 11,547 | 99.1 | 72.2 | 14.8 | 4.7 |
| CICIoT2023 | quantile | 39 | 15,631 | 14,068 | 98.7 | 61.4 | -- | 10.2 |
| CICIoT2023 | nokd | 39 | 11,893 | 10,704 | 99.0 | 72.8 | -- | 2.1 |
| CICIoT2023 | uniform | 40 | 12,770 | 11,493 | 98.2 | 69.5 | 14.5 | 1.4 |
| CICIoT2023 | all | 119 | 16,681 | 15,013 | 99.3 | 72.2 | 14.2 | 5.6 |
| CICIoT2023 | GBDT (teacher) | -- | -- | -- | 99.5 | 82.9 | -- | -- |
| CICIoMT2024 | **tree** | 37 | 8,355 | 7,519 | 99.7 | 89.4 | 14.5 | 4.8 |
| CICIoMT2024 | quantile | 37 | 10,048 | 9,043 | 99.7 | 87.7 | -- | 8.4 |
| CICIoMT2024 | nokd | 37 | 7,934 | 7,141 | 99.6 | 88.0 | -- | 2.3 |
| CICIoMT2024 | uniform | 42 | 8,562 | 7,706 | 99.7 | 89.4 | 14.8 | 4.6 |
| CICIoMT2024 | all | 127 | 13,983 | 12,585 | 99.7 | 91.9 | 14.4 | 2.5 |
| CICIoMT2024 | GBDT (teacher) | -- | -- | -- | 99.7 | 90.5 | -- | -- |
| Edge-IIoTset | **tree** | 41 | 9,703 | 8,732 | 98.0 | 88.8 | 14.6 | 7.9 |
| Edge-IIoTset | quantile | 41 | 11,058 | 9,952 | 96.8 | 82.3 | -- | 8.0 |
| Edge-IIoTset | nokd | 41 | 5,570 | 5,013 | 97.9 | 88.1 | -- | 4.4 |
| Edge-IIoTset | uniform | 46 | 10,040 | 9,036 | 98.0 | 88.8 | 15.1 | 9.0 |
| Edge-IIoTset | all | 139 | 15,736 | 14,163 | 98.3 | 90.5 | 14.7 | 6.1 |
| Edge-IIoTset | GBDT (teacher) | -- | -- | -- | 98.4 | 90.8 | -- | -- |
| TON-IoT | **tree** | 73 | 18,043 | 16,239 | 96.9 | 93.0 | 14.7 | 9.0 |
| TON-IoT | quantile | 73 | 24,687 | 22,218 | 95.3 | 89.3 | -- | 10.8 |
| TON-IoT | nokd | 73 | 18,709 | 16,838 | 96.9 | 92.7 | -- | 5.3 |
| TON-IoT | uniform | 71 | 16,708 | 15,037 | 96.8 | 92.6 | 14.7 | 8.4 |
| TON-IoT | all | 212 | 34,492 | 31,042 | 97.0 | 92.5 | 14.3 | 10.6 |
| TON-IoT | GBDT (teacher) | -- | -- | -- | 99.0 | 96.7 | -- | -- |
