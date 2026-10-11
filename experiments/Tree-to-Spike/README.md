# Tree-to-Spike (Design C) — GBDT-guided spike encoder + compact SNN student

The method of this repository. It is **tabular** (no sequence windowing): a
cost-sensitive gradient-boosted-tree teacher guides both a fixed spike encoder
and a compact LIF MLP student trained by distillation. At deployment **only the
encoder + SNN student run** (no tree), int8-quantized, with confidence-triggered
early exit.

## Architecture

![Tree-to-Spike (Design C) two-phase architecture. Training: network flow features feed a GBDT teacher and a tree-guided spike encoder, which feed a compact SNN student trained with CE+KD+ranking+sparsity+FP. Deployment: the fixed encoder and the quantized spiking network with early exit produce a decision, with no tree at runtime.](architecture.svg)

Two phases. **Training** (top): the GBDT teacher (amber, train-time only) supplies soft labels, split thresholds, and feature importances; the tree-guided spike encoder turns those into the spike representation; the compact SNN student is distilled with `CE + KD + ranking + spike-sparsity + FP-aware` loss. **Deployment** (bottom): only the fixed encoder + int8-quantized SNN run — no tree at runtime — with a confidence-triggered early exit. Teal = spiking/SNN, amber = teacher (training only), grey = I/O.

## Pipeline

```
TRAIN TIME
  tabular features [B, F]  (shared KDDCup99 preprocessing: log1p + one-hot + MinMax)
    -> cost-sensitive XGBoost teacher        (sample weights up-weight rare classes)
    -> TreeGuidedSpikeEncoder.fit(teacher):
         split thresholds  -> per-feature latency bins
         feature importance -> per-feature channel budget (population) + pruning
    -> SNN student trained by distillation:
         loss = CE + a_kd*KD + a_rank*ranking + a_sparse*spike_rate + a_fp*FP-aware

DEPLOY TIME (no tree)
    encoder -> int8-quantized SNN student -> confidence-triggered early exit
    -> attack class / alert
```

The contribution is the **encoder**: tree split thresholds define the spike
bins, and feature importance allocates the per-feature channel/timestep budget
(and prunes weak features). The tree is a *fixed function* after fitting and is
never executed at inference.

## Layout

```
experiments/Tree-to-Spike/
└── kddcup99/    config.yaml + run.py
```

Shared code in `src/`:
- `src/models/tree_to_spike.py` — `TreeGuidedSpikeEncoder`, `SNNStudent`
  (+ early-exit + spike/SynOps accounting), `quantize_int8_`.
- `src/training_treespike.py` — teacher fit + encoder fit + distillation loop
  (multi-seed, rich metrics, early-exit, teacher upper bound), reusing
  `src/data` and `src/metrics`.

## Running (per-dataset positional, like the other models)

```bash
# one dataset, default seed (42)
python experiments/Tree-to-Spike/kddcup99/run.py

# one dataset, multiple seeds
python scripts/run_treespike.py kddcup99 --seeds 42 43 44 45 46


# dry-run / force
python scripts/run_treespike.py kddcup99 --dry-run
python scripts/run_treespike.py kddcup99 --seeds 42 --force
```

## Results (stored separately)

```
results/Tree-to-Spike/<dataset>/seed_<N>/results.json   # rich metrics + early-exit + teacher-upper-bound blocks
results/Tree-to-Spike/<dataset>/seed_<N>/checkpoint.pt  # SNN student weights
results/summary_treespike.csv                           # separate summary, has a `model` column
```

`results.json` carries the shared metric schema (accuracy, balanced accuracy,
F1 macro/weighted, per-class P/R/F1, MCC, Cohen's κ, ROC-AUC, DR/FAR, spike
energy + sparsity, parameter count, timing) **plus** Design-C-specific blocks:
`early_exit` (accuracy/F1, mean exit step, uncertain fraction), a SynOps proxy
(`synops_per_sample`), the kept `encoder_channels`, and `teacher_upper_bound`
(the XGBoost teacher's accuracy/F1/MCC on the test set, as a reference ceiling).

Headline metrics are computed on the **standard forward** of the (quantized)
student so ROC-AUC is comparable to the other models; early-exit is reported as
an auxiliary block.

## Key configuration

| Setting | Default | Note |
|---|---|---|
| `T` | 20 | encoder latency window |
| `pop_max` / `prune_quantile` / `jitter` | 3 / 0.20 / 0.12 | population budget, feature pruning, latency spread |
| `hidden` | [64, 64] | SNN student MLP |
| `beta` / `leaky` | 0.9 / true | LIF decay; `leaky: false` = non-leaky IF ablation |
| `teacher_n_estimators` / `max_depth` / `lr` | 200 / 6 / 0.1 | XGBoost teacher |
| `tau` / `a_kd` / `a_rank` / `a_sparse` / `a_fp` | 2.0 / 1.0 / 0.3 / 1e-3 / 0.5 | distillation loss weights |
| `quantize` | true | int8 fake-quant the student before eval (deploy scenario) |
| `theta_exit` / `theta_uncertain` | 0.9 / 0.5 | early-exit + uncertainty thresholds |
| `epochs` / `batch_size` / `lr` | 10 / 128 / 2e-3 | student training |

## Dependencies and caveats

- Requires **xgboost** (installed in the venv). The teacher is trained per seed.
- The tree-guided encoder reads split thresholds from
  `teacher.get_booster().trees_to_dataframe()` and importances from
  `teacher.feature_importances_`; thresholds live in the MinMax-scaled feature
  space (the teacher trains on the same scaled features the encoder bins).
- int8 quantization is a per-tensor fake-quant stand-in for QAT; for production
  use `torch.ao.quantization`. The energy numbers (spikes/sample, SynOps,
  early-exit step) are **proxies**, not measured hardware energy.
- Unlike the Vanilla SNN (LeakyParallel + latency), this design uses its own encoder and
  `snn.Leaky` (fast-sigmoid surrogate); it is intentionally not constrained to
  the LeakyParallel/latency convention.
