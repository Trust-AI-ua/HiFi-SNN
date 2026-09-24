### Efficiency (mean across datasets)

| Model | Params | Energy@32b (pJ) | Energy@int8 (pJ) | Spike sparsity | Latency (ms) |
|---|--:|--:|--:|--:|--:|
| Vanilla SNN | 13,609 | 3,825 | 128 | 0.923 | 0.063 |
| Tree-to-Spike | 8,622 | 10,624 | 354 | 0.885 | 0.814 |
| DNN-Tree | 8,622 | 39,043 | 1,952 | -- | 0.029 |
| TASNN | 544,448 | 224,210,956† | 11,185,653† | 0.995 | 0.245 |

† TASNN energy is an analytic FLOP estimate (transformer MACs×4.6pJ + spiking-core ACs×0.9pJ); the other models report measured SOPs/MACs.
