| dataset | tree | tree_sd | entropy | entropy_sd | d_tree_minus_entropy | tree_ksops | entropy_ksops | fallback_share | edge_shortfall_per_run | reading |
|---|---|---|---|---|---|---|---|---|---|---|
| ciciot2023 | 72.22 | 1.81 | 65.60 | 2.96 | +6.61 | 12.83 | 10.97 | 0.00 | 0.40 | against supervised entropy placement |
| ciciomt2024 | 89.42 | 3.24 | 85.27 | 3.21 | +4.16 | 8.35 | 8.73 | 0.00 | 0.00 | against supervised entropy placement |
| edgeiiot | 83.94 | 0.17 | 84.50 | 1.42 | -0.56 | 10.53 | 10.75 | 0.01 | 0.20 | against supervised entropy placement |
| toniot | 92.98 | 0.23 | 92.63 | 0.58 | +0.35 | 18.04 | 18.52 | 0.00 | 0.00 | against supervised entropy placement |
| IoT mean | 84.64 | 1.36 | 82.00 | 2.04 | +2.64 | 12.44 | 12.24 | 0.00 | 0.15 |  |

Pre-registered outcome: A: tree clearly wins (tree-derived placement is the mechanism)

Scope: E2 isolates edge placement only; channel selection, the per-feature edge budget and pruning stay tree-derived in both arms.

Fallback reading:
- ciciot2023: fallback share 0.00 -> against supervised entropy placement
- ciciomt2024: fallback share 0.00 -> against supervised entropy placement
- edgeiiot: fallback share 0.01 -> against supervised entropy placement
- toniot: fallback share 0.00 -> against supervised entropy placement
