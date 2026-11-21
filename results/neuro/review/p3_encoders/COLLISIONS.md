# Multichannel collisions of the deployed Tree-to-Spike encoders (amendment 8, part 4)

A retained feature's level k (0 to |E_j|) emits one spike time per channel; two levels collide when every channel emits the same time. The single-channel bound |E_j| >= T is the criterion behind the paper's 45.9% (computed there on the diagnostic encoder tables, pooled IoT); here it is recomputed on the production encoders. Every table was checked to reproduce the deployed spike times on the whole test split.

| Encoder | Dataset | Seeds | Retained features | Bound violated | Features with a collision | Level distinctions lost | ... with the centre channel alone | Test values in a merged tuple (mean over features) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| T10 (T = 10) | ciciomt2024 | 1 | 33.0 | 97.0% | 97.0% | 89.9% | 90.5% | 90.9% |
| T10 (T = 10) | ciciot2023 | 1 | 31.0 | 71.0% | 71.0% | 92.2% | 93.4% | 69.2% |
| T10 (T = 10) | edgeiiot | 5 | 31.8 | 23.9% | 23.9% | 87.9% | 87.9% | 22.7% |
| T10 (T = 10) | toniot | 5 | 56.6 | 17.7% | 17.7% | 85.4% | 85.4% | 17.7% |
| production (T = 20) | ciciomt2024 | 5 | 33.2 | 92.8% | 92.2% | 80.2% | 81.7% | 83.1% |
| production (T = 20) | ciciot2023 | 5 | 31.8 | 71.1% | 71.1% | 84.9% | 87.5% | 69.6% |
| production (T = 20) | edgeiiot | 5 | 31.8 | 22.0% | 21.4% | 81.4% | 81.5% | 18.9% |
| production (T = 20) | toniot | 5 | 56.6 | 17.7% | 17.7% | 78.2% | 78.2% | 17.7% |

Pooled over the four IoT datasets and five seeds (production, T = 20): 767 retained features; 45.9% violate the single-channel bound; 45.6% have at least one collision among their emitted tuples; 81.8% of all level distinctions are lost (83.4% with the centre channel alone); a mean 42.8% of a feature's test values fall in a merged tuple.
Features with more than one channel: 120 (15.6%); among them 30.8% collide, and the extra channels recover 757 of 4105 levels the centre channel alone would merge.
