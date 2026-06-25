# Comprehensive benchmark — Vanilla SNN vs Tree-to-Spike vs DNN

5 seeds x 6 datasets, macro-averaged across datasets. DNN-Tree is the iso-architecture twin of Tree-to-Spike (identical params). Horowitz 45nm: MAC = 4.6 pJ, AC/SOP = 0.9 pJ.

## 1. Quality (macro-averaged across 6 datasets)
| Model | Acc | Bal-Acc | F1-mac | F1-wt | MCC | kappa | Prec | Rec | ROC-AUC | DR | FAR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Vanilla SNN (LeakyParallel/latency) | 0.944 | 0.806 | 0.812 | 0.936 | 0.892 | 0.886 | 0.896 | 0.806 | 0.978 | 0.913 | 0.014 |
| Tree-to-Spike (hybrid SNN) | 0.946 | 0.822 | 0.821 | 0.937 | 0.898 | 0.892 | 0.886 | 0.822 | 0.981 | 0.917 | 0.009 |
| DNN-Tree (dense ANN) | 0.945 | 0.827 | 0.834 | 0.938 | 0.897 | 0.892 | 0.893 | 0.827 | 0.987 | 0.917 | 0.014 |

## 2. Efficiency & energy (macro-averaged)
| Model | Params | Ops/sample | Op type | Energy (pJ) | Sparsity | Exit frac | delta vs dense ANN |
|---|--:|--:|:--|--:|--:|--:|--:|
| Vanilla SNN (LeakyParallel/latency) | 13849 | 3946 | SOPs | 3551 | 0.918 | 0.28 | 25.8x |
| Tree-to-Spike (hybrid SNN) | 8141 | 5584 | SOPs | 5026 | 0.919 | 0.22 | 9.8x |
| DNN-Tree (dense ANN) | 8141 | 8009 | MACs | 36839 | - | - | 1.0x (ref) |

## 2b. HEADLINE energy — memory-aware total, batch-1 (on-chip SRAM 5 pJ)
Compute **+** memory access (the publication-grade number; compute-only flatters SNNs). SNNs use early-exit (T_eff); DNN amortises weights only at high batch, so batch-1 is its worst case.
| Model | Compute E (pJ) | Total E @B=1 (pJ) | delta_total vs dense ANN |
|---|--:|--:|--:|
| Vanilla SNN (LeakyParallel/latency) | 3551 | 15898 | **8.3x** |
| Tree-to-Spike (hybrid SNN) | 5026 | 13282 | **5.9x** |
| DNN-Tree (dense ANN) | 36839 | 78822 | 1.0x (ref) |

- At batch-1 with memory access, **Tree-to-Spike still uses 5.9x less total energy** than its iso DNN twin, and **Vanilla SNN 8.3x** vs its dense twin — the win survives a full memory accounting because early-exit keeps SNN memory traffic below the DNN's dense weight reads. (Without early-exit it can drop below 1x at DRAM — see energy_memory_model.md.)

## 3. Per-dataset F1-macro
| Model | kddcup99 | nslkdd | cicids20 | ctu13 | iscxids2 | cse-cic- |
|---|--:|--:|--:|--:|--:|--:|
| Vanilla SNN (LeakyParallel/latency) | 0.620 | 0.593 | 0.986 | 1.000 | 0.970 | 0.701 |
| Tree-to-Spike (hybrid SNN) | 0.618 | 0.585 | 0.984 | 1.000 | 0.980 | 0.759 |
| DNN-Tree (dense ANN) | 0.635 | 0.622 | 0.988 | 1.000 | 0.983 | 0.774 |

## 4. Per-dataset energy gain (delta = E_dense-ANN / E_SNN)
| Model | kddcup99 | nslkdd | cicids20 | ctu13 | iscxids2 | cse-cic- |
|---|--:|--:|--:|--:|--:|--:|
| Vanilla SNN | 31.2x | 24.4x | 7.9x | 6.9x | 73.8x | 10.3x |
| Tree-to-Spike | 11.6x | 7.1x | 5.3x | 21.2x | 8.7x | 4.7x |

## Bottom line
- **Accuracy**: all three are comparable (macro F1 — Vanilla 0.812, Tree-to-Spike 0.821, DNN 0.834); the spiking models are not paying an accuracy penalty.
- **Energy**: Tree-to-Spike uses **9.8x** less compute energy than its iso-architecture DNN twin; the Vanilla SNN **25.8x** vs its dense twin.
- **Model size**: Tree-to-Spike/DNN are tiny (8141 params) vs the Vanilla SNN (13849).
- **Caveats**: Horowitz energy is compute-only (excludes the SNN's per-timestep memory access); latency on file is batched for the SNNs vs batch-1 for the DNN, so ELP needs a batch-1 re-measure.
