### Efficiency (mean across datasets)

| Model | Params | Energy (pJ/sample) | Spike sparsity | Latency (ms) |
|---|--:|--:|--:|--:|
| Vanilla SNN | 14,028 | 3,551 | 0.925 | 0.060 |
| Tree-to-Spike | 8,661 | 6,941 | 0.888 | 0.754 |
| DNN-Tree | 8,661 | 39,225 | -- | 0.029 |
| TASNN | 547,727 | 231,814,392† | 0.995 | 0.247 |

† TASNN energy is an analytic FLOP estimate (transformer MACs×4.6pJ + spiking-core ACs×0.9pJ); the other models report measured SOPs/MACs.
