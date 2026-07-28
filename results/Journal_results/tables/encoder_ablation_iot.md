### Encoder ablation - what does the teacher actually contribute?

Mean over 5 seeds. **SOPs include the input->layer-1 term** (channels x hidden[0]); **Energy = SOPs × 0.9 pJ.** Arms: **tree** = XGBoost split edges + importance-pruned/budgeted channels (ours). **quantile** = equal-frequency edges with channels IDENTICAL to tree (same pruning/budget/jitter; per-feature edge count matched) - isolates whether the teacher's split placement (its 'ruler') matters. **nokd** = tree encoder trained WITHOUT the teacher-posterior loss terms (a_kd = a_rank = 0) - isolates the teacher's soft labels. **uniform** = tree edges, 1 channel per feature-with-edges (NOTE: changes pruning AND allocation together, so it is confounded; kept for completeness). **all** = tree edges, max channels per feature (dense upper bound). **GBDT (teacher)** = the XGBoost teacher's own test accuracy/F1, the reference ceiling (not a spiking model; no energy columns). **TTFS** = mean encoder time-to-first-spike (/20, lower = earlier). **Exit step** = mean early-exit timestep of the trained student (/20, lower = decides sooner).

| Dataset | Encoder | Channels | SOPs/sample | Energy (pJ) | Accuracy | Macro-F1 | TTFS (/20) | Exit step (/20) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CICIoT2023 | **tree** | 39 | 12,830 | 11,547 | 99.1 | 72.2 | 14.8 | 4.7 |
| CICIoT2023 | quantile | 39 | 15,631 | 14,068 | 98.7 | 61.4 | 9.4 | 10.2 |
| CICIoT2023 | nokd | 39 | 11,893 | 10,704 | 99.0 | 72.8 | 14.8 | 2.1 |
| CICIoT2023 | uniform | 40 | 12,770 | 11,493 | 98.2 | 69.5 | 14.5 | 1.4 |
| CICIoT2023 | all | 119 | 16,681 | 15,013 | 99.3 | 72.2 | 14.2 | 5.6 |
| CICIoT2023 | GBDT (teacher) | -- | -- | -- | 99.5 | 82.9 | -- | -- |
| CICIoMT2024 | **tree** | 37 | 8,355 | 7,519 | 99.7 | 89.4 | 14.5 | 4.8 |
| CICIoMT2024 | quantile | 37 | 10,048 | 9,043 | 99.7 | 87.7 | 11.4 | 8.4 |
| CICIoMT2024 | nokd | 37 | 7,934 | 7,141 | 99.6 | 88.0 | 14.5 | 2.3 |
| CICIoMT2024 | uniform | 42 | 8,562 | 7,706 | 99.7 | 89.4 | 14.8 | 4.6 |
| CICIoMT2024 | all | 127 | 13,983 | 12,585 | 99.7 | 91.9 | 14.4 | 2.5 |
| CICIoMT2024 | GBDT (teacher) | -- | -- | -- | 99.7 | 90.5 | -- | -- |
| Edge-IIoTset | **tree** | 41 | 9,703 | 8,732 | 98.0 | 88.8 | 14.6 | 7.9 |
| Edge-IIoTset | quantile | 41 | 11,058 | 9,952 | 96.8 | 82.3 | 3.0 | 8.0 |
| Edge-IIoTset | nokd | 41 | 5,570 | 5,013 | 97.9 | 88.1 | 14.6 | 4.4 |
| Edge-IIoTset | uniform | 46 | 10,040 | 9,036 | 98.0 | 88.8 | 15.1 | 9.0 |
| Edge-IIoTset | all | 139 | 15,736 | 14,163 | 98.3 | 90.5 | 14.7 | 6.1 |
| Edge-IIoTset | GBDT (teacher) | -- | -- | -- | 98.4 | 90.8 | -- | -- |
| TON-IoT | **tree** | 73 | 18,043 | 16,239 | 96.9 | 93.0 | 14.7 | 9.0 |
| TON-IoT | quantile | 73 | 24,687 | 22,218 | 95.3 | 89.3 | 2.2 | 10.8 |
| TON-IoT | nokd | 73 | 18,709 | 16,838 | 96.9 | 92.7 | 14.7 | 5.3 |
| TON-IoT | uniform | 71 | 16,708 | 15,037 | 96.8 | 92.6 | 14.7 | 8.4 |
| TON-IoT | all | 212 | 34,492 | 31,042 | 97.0 | 92.5 | 14.3 | 10.6 |
| TON-IoT | GBDT (teacher) | -- | -- | -- | 99.0 | 96.7 | -- | -- |
