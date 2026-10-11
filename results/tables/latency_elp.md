# Batch-1 latency & Energy-Latency Product (ELP) — publication re-measure

CPU single-thread, batch-1, 200 reps after 20 warmup, seeds [42, 43, 44] averaged. Energy = recorded ops x Horowitz (SOP 0.9 pJ / MAC 4.6 pJ). ELP = energy_pj x latency_ms.

## Latency (ms/sample, batch-1)
| Model | kddcup99 | nslkdd | cicids20 | ctu13 | iscxids2 | cse-cic- | mean |
|---|--:|--:|--:|--:|--:|--:|--:|
| Vanilla SNN | 1.309 | 1.303 | 1.296 | 1.306 | 1.341 | 1.333 | 1.315 |
| Tree-to-Spike | 2.733 | 2.669 | 2.692 | 2.719 | 2.728 | 2.736 | 2.713 |
| DNN-Tree | 0.013 | 0.013 | 0.013 | 0.013 | 0.013 | 0.013 | 0.013 |

## ELP (pJ·ms/sample, batch-1) — lower is better
| Model | kddcup99 | nslkdd | cicids20 | ctu13 | iscxids2 | cse-cic- | mean |
|---|--:|--:|--:|--:|--:|--:|--:|
| Vanilla SNN | 2951 | 3956 | 7997 | 4451 | 1930 | 6641 | 4654 |
| Tree-to-Spike | 9532 | 15311 | 20404 | 3251 | 10766 | 23191 | 13742 |
| DNN-Tree | 548 | 536 | 523 | 349 | 465 | 531 | 492 |

## Summary (macro-avg)
| Model | Energy (pJ) | Batch-1 latency (ms) | ELP (pJ·ms) |
|---|--:|--:|--:|
| Vanilla SNN | 3548 | 1.315 | 4665 |
| Tree-to-Spike | 5070 | 2.713 | 13755 |
| DNN-Tree | 36833 | 0.013 | 492 |

## How to read this (honest framing)
- **Energy** is the hardware-agnostic SNN win (SOPs x 0.9 pJ vs MACs x 4.6 pJ).
- **Batch-1 latency on a CPU favours the one-shot DNN**: the SNNs run T sequential timesteps (Vanilla T=25, Tree-to-Spike T=20), so wall-clock is ~T x higher. This is a von-Neumann artifact — on event-driven neuromorphic hardware the SNN timesteps run concurrently and this latency collapses.
- **ELP therefore reflects the conventional-hardware trade-off**, not the neuromorphic target. Report energy (delta) as the headline efficiency claim and ELP with this caveat.
