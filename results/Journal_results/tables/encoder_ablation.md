### Encoder ablation — tree-guided vs uniform vs all-channel

Mean over 5 seeds. **SOPs include the input->layer-1 term** (channels x hidden[0]); **Energy = SOPs × 0.9 pJ.** The all-channel encoder costs ~2x for negligible accuracy gain — the importance-guided channel budget is the efficiency mechanism.

| Dataset | Encoder | Channels | SOPs/sample | Energy (pJ) | Accuracy | Macro-F1 |
|---|---|--:|--:|--:|--:|--:|
| NSL-KDD | **tree** | 68 | 10,708 | 9,637 | 77.3 | 58.5 |
| NSL-KDD | uniform | 78 | 12,222 | 10,999 | 77.7 | 56.3 |
| NSL-KDD | all | 235 | 24,548 | 22,093 | 78.4 | 59.8 |
| UNSW-NB15 | **tree** | 156 | 24,532 | 22,079 | 74.5 | 41.7 |
| UNSW-NB15 | uniform | 190 | 26,029 | 23,426 | 76.4 | 43.0 |
| UNSW-NB15 | all | 570 | 54,234 | 48,811 | 75.1 | 43.4 |
| TON-IoT | **tree** | 73 | 18,043 | 16,239 | 96.9 | 93.0 |
| TON-IoT | uniform | 71 | 16,708 | 15,037 | 96.8 | 92.6 |
| TON-IoT | all | 212 | 34,492 | 31,042 | 97.0 | 92.5 |
| CIC-IDS2017 | **tree** | 62 | 12,002 | 10,802 | 99.8 | 98.4 |
| CIC-IDS2017 | uniform | 68 | 12,513 | 11,261 | 99.8 | 98.4 |
| CIC-IDS2017 | all | 204 | 20,219 | 18,197 | 99.9 | 98.8 |
