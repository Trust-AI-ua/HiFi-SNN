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

## Definitions

For a retained feature j with |E_j| thresholds and m_j channels (offsets o_c):

- **Level** k in 0 .. |E_j|: the number of thresholds at or below the value; the feature has |E_j| + 1 levels.
- **Emitted tuple** of level k: (t_1(k), ..., t_m(k)), t_c(k) = round((1 - clip(k/|E_j| + o_c, 0, 1)) (T - 1)) (float32 level, NumPy round half to even).
- **Distinct tuples** D_j: the number of different tuples over all |E_j| + 1 levels.
- **Bound violated**: |E_j| >= T (a single channel has T spike times, so |E_j| + 1 > T levels must share a time). Share = violating features / retained features, pooled over features (the 45.9% definition).
- **Collides**: D_j < |E_j| + 1, i.e. at least two levels emit the same tuple on every channel. Share = colliding features / retained features, pooled over features.
- **Levels lost** for feature j: |E_j| + 1 - D_j. **Level distinctions lost** = sum over features of levels lost / sum over features of |E_j| + 1, pooled over features (so features with many thresholds dominate).
- **... with the centre channel alone**: the same count using only the channel with the smallest |o_c|, i.e. what one channel would lose; the difference is what the extra channels recover.
- **Test values in a merged tuple**: for feature j, the share of test flows whose level shares its tuple with at least one other level; the table reports the mean over features. The stricter share, where the other level also occurs in the test split, is in the JSONs: pooled production mean 42.80% against 42.83%, largest per-feature difference 2.14 points.
- Counts are of the encoder function, not of classification loss: two merged levels cost accuracy only if they should be told apart.
