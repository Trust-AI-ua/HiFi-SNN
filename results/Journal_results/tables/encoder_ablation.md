### Encoder ablation - tree-guided vs uniform vs all-channel

Mean over 5 seeds. **SOPs include the input->layer-1 term** (channels x hidden[0]); **Energy = SOPs × 0.9 pJ.** The all-channel encoder costs ~2x for negligible accuracy gain - the importance-guided channel budget is the efficiency mechanism.

| Dataset | Encoder | Channels | SOPs/sample | Energy (pJ) | Accuracy | Macro-F1 |
|---|---|--:|--:|--:|--:|--:|
| KDDCup99 | **tree** | 70 | 8,407 | 7,567 | 92.6 | 61.8 |
| KDDCup99 | uniform | 77 | 9,230 | 8,307 | 92.5 | 61.1 |
| KDDCup99 | all | 231 | 19,450 | 17,505 | 92.5 | 63.2 |
| NSL-KDD | **tree** | 68 | 10,708 | 9,637 | 77.3 | 58.5 |
| NSL-KDD | uniform | 78 | 12,222 | 10,999 | 77.7 | 56.3 |
| NSL-KDD | all | 235 | 24,548 | 22,093 | 78.4 | 59.8 |
| CTU-13 | **tree** | 24 | 2,922 | 2,630 | 100.0 | 100.0 |
| CTU-13 | uniform | 27 | 3,285 | 2,956 | 100.0 | 100.0 |
| CTU-13 | all | 82 | 6,639 | 5,975 | 100.0 | 100.0 |
| ISCX-IDS2012 | **tree** | 52 | 7,810 | 7,029 | 99.7 | 98.0 |
| ISCX-IDS2012 | uniform | 63 | 8,275 | 7,447 | 99.7 | 98.0 |
| ISCX-IDS2012 | all | 190 | 16,891 | 15,202 | 99.8 | 98.5 |
| CIC-IDS2017 | **tree** | 62 | 12,002 | 10,802 | 99.8 | 98.4 |
| CIC-IDS2017 | uniform | 68 | 12,513 | 11,261 | 99.8 | 98.4 |
| CIC-IDS2017 | all | 204 | 20,219 | 18,197 | 99.9 | 98.8 |
| CSE-CIC-IDS2018 | **tree** | 66 | 13,595 | 12,236 | 98.0 | 75.9 |
| CSE-CIC-IDS2018 | uniform | 71 | 13,251 | 11,926 | 95.4 | 74.9 |
| CSE-CIC-IDS2018 | all | 213 | 23,686 | 21,318 | 98.2 | 77.3 |
| UNSW-NB15 | **tree** | 156 | 24,532 | 22,079 | 74.5 | 41.7 |
| UNSW-NB15 | uniform | 190 | 26,029 | 23,426 | 76.4 | 43.0 |
| UNSW-NB15 | all | 570 | 54,234 | 48,811 | 75.1 | 43.4 |
| CICIoT2023 | **tree** | 39 | 12,830 | 11,547 | 99.1 | 72.2 |
| CICIoT2023 | uniform | 40 | 12,770 | 11,493 | 98.2 | 69.5 |
| CICIoT2023 | all | 119 | 16,681 | 15,013 | 99.3 | 72.2 |
| CICIoMT2024 | **tree** | 37 | 8,355 | 7,519 | 99.7 | 89.4 |
| CICIoMT2024 | uniform | 42 | 8,562 | 7,706 | 99.7 | 89.4 |
| CICIoMT2024 | all | 127 | 13,983 | 12,585 | 99.7 | 91.9 |
| Edge-IIoTset | **tree** | 41 | 9,703 | 8,732 | 98.0 | 88.8 |
| Edge-IIoTset | uniform | 46 | 10,040 | 9,036 | 98.0 | 88.8 |
| Edge-IIoTset | all | 139 | 15,736 | 14,163 | 98.3 | 90.5 |
| TON-IoT | **tree** | 73 | 18,043 | 16,239 | 96.9 | 93.0 |
| TON-IoT | uniform | 71 | 16,708 | 15,037 | 96.8 | 92.6 |
| TON-IoT | all | 212 | 34,492 | 31,042 | 97.0 | 92.5 |
