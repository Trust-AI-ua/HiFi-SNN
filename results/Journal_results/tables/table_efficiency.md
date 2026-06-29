### Efficiency (mean across datasets)

| Model | Params | Energy@32b (pJ) | Energy@int8 (pJ) | Spike sparsity | Latency (ms) |
|---|--:|--:|--:|--:|--:|
| Vanilla SNN | 14,028 | 15,399 | 513 | 0.925 | 0.060 |
| Tree-to-Spike | 8,661 | 10,547 | 352 | 0.888 | 0.754 |
| DNN-Tree | 8,661 | 39,225 | 1,961 | -- | 0.028 |
| TASNN | 547,727 | 231,814,392† | 11,565,239† | 0.995 | 0.247 |

† TASNN energy is an analytic FLOP estimate (transformer MACs×4.6pJ + spiking-core ACs×0.9pJ); the other models report measured SOPs/MACs.
