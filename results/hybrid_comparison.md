# Hybrid SNN vs Vanilla (LeakyParallel / latency) — comparison

Aggregated over 5 seeds (42–46) per (model, dataset), then macro-averaged across the
6 datasets (kddcup99, nslkdd, cicids2017, ctu13, iscxids2012, cse-cic-ids2018).
Baseline = the best Phase-1 vanilla SNN operating point (LeakyParallel neuron + latency
encoding). Per-flow models share the same test instances; **Spiking-SSM is evaluated on
sequence *windows* (label = last record), a different task — its numbers are not strictly
comparable.** Vanilla cse-cic uses 4 seeds (seed 46 still running); all others 5.

## 1. Comprehensive metrics (macro-averaged across 6 datasets)

| Model | Acc | Bal-Acc | F1-macro | F1-wt | MCC | κ | Prec-mac | Rec-mac | ROC-AUC | DR | FAR | Sparsity | Exit |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **Vanilla LP/latency** | 0.944 | 0.805 | 0.812 | 0.936 | 0.892 | 0.886 | 0.896 | 0.805 | 0.978 | 0.913 | 0.014 | 0.918 | 7.7/25 |
| **Tree-to-Spike** | **0.946** | **0.822** | **0.821** | **0.937** | **0.898** | **0.892** | 0.886 | **0.822** | 0.981 | **0.917** | **0.009** | 0.919 | 4.5/20 |
| **S-xLSTM** | 0.942 | 0.791 | 0.787 | 0.934 | 0.891 | 0.884 | 0.851 | 0.791 | **0.982** | 0.910 | 0.010 | 0.739 | 2.1/25 |
| **Spiking-Griffin** | 0.944 | 0.793 | 0.787 | 0.936 | 0.889 | 0.884 | 0.843 | 0.793 | 0.977 | 0.914 | 0.014 | 0.872 | 1.4/25 |
| **Spiking-SSM** | 0.800 | 0.717 | 0.617 | 0.806 | 0.622 | 0.595 | 0.609 | 0.717 | 0.904 | 0.887 | 0.176 | 0.862 | 8.5/16 |

`Sparsity` = fraction of non-spikes (higher = fewer spikes = lower energy). `Exit` = mean
early-exit step / total steps T (lower = decides sooner; note T differs: 25 for
LP/S-xLSTM/Griffin, 20 for Tree-to-Spike, 16 for SSM).

## 2. Per-dataset F1-macro

| Model | kddcup99 | nslkdd | cicids2017 | ctu13 | iscxids2012 | cse-cic | macro |
|---|--:|--:|--:|--:|--:|--:|--:|
| Vanilla LP/latency | 0.620 | 0.593 | 0.986 | 1.000 | 0.970 | 0.701 | 0.812 |
| Tree-to-Spike | 0.618 | 0.585 | 0.984 | 1.000 | **0.980** | **0.759** | **0.821** |
| S-xLSTM | 0.542 | 0.529 | 0.974 | 1.000 | 0.971 | 0.707 | 0.787 |
| Spiking-Griffin | 0.546 | 0.535 | 0.970 | 1.000 | 0.963 | 0.711 | 0.787 |
| Spiking-SSM | 0.506 | 0.146 | 0.570 | 0.997 | 0.841 | 0.638 | 0.617 |

## 3. Per-dataset MCC

| Model | kddcup99 | nslkdd | cicids2017 | ctu13 | iscxids2012 | cse-cic | macro |
|---|--:|--:|--:|--:|--:|--:|--:|
| Vanilla LP/latency | 0.829 | 0.649 | 0.993 | 1.000 | 0.941 | 0.937 | 0.892 |
| Tree-to-Spike | 0.831 | **0.670** | **0.995** | 1.000 | **0.961** | 0.932 | **0.898** |
| S-xLSTM | 0.825 | 0.644 | 0.991 | 1.000 | 0.944 | **0.940** | 0.891 |
| Spiking-Griffin | 0.826 | 0.651 | 0.992 | 0.999 | 0.927 | 0.941 | 0.889 |
| Spiking-SSM | 0.803 | -0.009 | 0.666 | 0.994 | 0.714 | 0.567 | 0.622 |

## 4. Ranking (quality composite = mean of F1-macro, MCC, balanced-accuracy)

| Rank | Model | F1-macro | MCC | Bal-Acc | Acc | Composite | ΔF1m vs vanilla |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | **Tree-to-Spike** | 0.821 | 0.898 | 0.822 | 0.946 | **0.847** | **+0.009** |
| 2 | Vanilla LP/latency (baseline) | 0.812 | 0.892 | 0.805 | 0.944 | 0.836 | — |
| 3 | Spiking-Griffin | 0.787 | 0.889 | 0.793 | 0.944 | 0.823 | −0.024 |
| 4 | S-xLSTM | 0.787 | 0.891 | 0.791 | 0.942 | 0.823 | −0.025 |
| 5 | Spiking-SSM | 0.617 | 0.622 | 0.717 | 0.800 | 0.652 | −0.195 |

## Bottom line
- **Tree-to-Spike is the only hybrid that beats the vanilla baseline** on quality
  (F1-macro, MCC, balanced-accuracy, DR, FAR) while matching its spike sparsity — the
  best accuracy-and-energy package. Biggest gains: cse-cic-ids2018 (F1m 0.759 vs 0.701)
  and iscxids2012; competitive on the hard KDD-family.
- **S-xLSTM and Spiking-Griffin essentially tie the vanilla** on accuracy/MCC but lose
  ~2.5 F1-macro points (weaker minority-class recall on the KDD-family). Their real edge
  is **latency**: they early-exit far sooner (Griffin 1.4/25, S-xLSTM 2.1/25 vs vanilla
  7.7/25), i.e. lower inference energy for near-baseline quality.
- **Spiking-SSM underperforms** and collapses on NSL-KDD (MCC ≈ 0); it also runs a
  different (windowed, class-weighted) task, so treat it separately, not as a per-flow peer.

Per-(model, dataset, seed)-mean values: `results/hybrid_comparison_by_dataset.csv`.
