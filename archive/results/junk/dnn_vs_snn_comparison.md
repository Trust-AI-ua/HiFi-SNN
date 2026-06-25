# Fair DNN-vs-SNN comparison — DNN-Tree (dense ANN) vs Tree-to-Spike (spiking)

Iso-architecture twins: identical Linear topology, same XGBoost teacher + tree-guided encoder + distillation; only the compute paradigm differs.
Horowitz 45nm: MAC = 4.6 pJ, AC/SOP = 0.9 pJ. delta = E_DNN / E_SNN (energy gain of the SNN).

| Dataset | Params (=) | Acc DNN/SNN | F1m DNN/SNN | MCC DNN/SNN | MACs (DNN) | SOPs (SNN) | E_DNN pJ | E_SNN pJ | delta (x) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| kddcup99 | 9042 = | 0.926/0.926 | 0.635/0.618 | 0.829/0.831 | 8909 | 3914 | 40980 | 3523 | 11.6 |
| nslkdd | 8927 = | 0.763/0.773 | 0.622/0.585 | 0.653/0.670 | 8794 | 6330 | 40451 | 5697 | 7.1 |
| cicids2017 | 8504 = | 0.999/0.998 | 0.988/0.984 | 0.996/0.995 | 8371 | 8047 | 38508 | 7242 | 5.3 |
| ctu13 | 5890 = | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 5760 | 1386 | 26496 | 1247 | 21.2 |
| iscxids2012 | 7708 = | 0.998/0.997 | 0.983/0.980 | 0.965/0.961 | 7578 | 4457 | 34857 | 4011 | 8.7 |
| **macro-avg** | = | 0.937/0.939 | 0.845/0.833 | 0.889/0.891 | — | — | — | — | **10.8** |

## Batch-1 latency & Energy-Latency Product (ELP)
| Dataset | DNN lat (ms, batch-1) | SNN lat (ms, batched*) | ELP_DNN | ELP_SNN |
|---|--:|--:|--:|--:|
| kddcup99 | 0.029 | 0.773 | 1201 | 2723 |
| nslkdd | 0.028 | 1.686 | 1137 | 9606 |
| cicids2017 | 0.029 | 0.416 | 1126 | 3013 |
| ctu13 | 0.030 | 0.398 | 785 | 496 |
| iscxids2012 | 0.029 | 0.371 | 1019 | 1487 |

\* SNN latency on file is batched (bs=128), not batch-1 — re-measure for a strict ELP.

## Notes
- **Energy gain delta = 10.8x** on average: the spiking student does fewer, cheaper (addition-only) ops than the dense ANN's MACs.
- **Same parameters, comparable accuracy** -> the gap is the paradigm, not capacity (iso-architecture principle).
- Horowitz SOP energy is **compute-only**; the SNN's T-step membrane/weight memory access is extra (a unit-latency T=1 SNN would remove it). State this in the paper.
- Edge realism: DNN measured at **batch-1** (neuromorphic HW is batch-1 optimised).
