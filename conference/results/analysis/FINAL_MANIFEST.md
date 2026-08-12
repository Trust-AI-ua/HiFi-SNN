# Evidence package — regenerated 2026-08-12 17:09:53

Produced by `scripts/finalize.py`. Every manuscript number must come
from re-running this, not from memory.

## Coverage

```
39 runs on disk across 8 (protocol, variant) cells

protocol                 variant                      seeds  artifacts
ctu13_v2_f0              LeakyParallel/latency            4          4  <- INCOMPLETE
ctu13_v2_f0              LeakyParallel/rate               5          5
kddcup99_v2              LeakyParallel/delta              5          5
kddcup99_v2              LeakyParallel/latency            5          5
kddcup99_v2              LeakyParallel/rate               5          5
nslkdd_v2                LeakyParallel/delta              5          5
nslkdd_v2                LeakyParallel/latency            5          5
nslkdd_v2                LeakyParallel/rate               5          5

1 cell(s) below 5 seeds
```

## Steps

- OK       screening statistics (v1, 27 configurations)
- OK       strict confirmation — Axis A, encoding (neuron fixed)
- OK       strict confirmation — Axis B, neuron (encoding fixed)
- OK       mechanism — early readout, T95/T99, spike saving
- OK       mechanism — spike decomposition and SOPs
- OK       protocol robustness — split audit
- OK       protocol robustness — deduplication sensitivity
- OK       boundary — fixed FAR, per class, duplicate-free
- OK       boundary — NSL-KDD known vs unseen attack types
- OK       LaTeX tables
