# Spiking-SSM (Design A) — event-driven spiking state-space NIDS

The **third Phase-2 hybrid**, alongside [S-xLSTM](../S-xLSTM/) and
[Spiking-Griffin](../Spiking-Griffin/). Unlike those two (tabular, latency
encoding + `LeakyParallel`), Design A is a **sequence model**: it consumes flow
windows and keeps its native design — a learned spiking front-end, selective
spiking state-space blocks (Mamba-style, spike-gated), and supervised +
self-supervised heads with confidence-triggered early exit.

## Pipeline

```
consecutive-record window  [B, L, F]
  -> spiking front-end (embed + LIF, per step)
  -> stack of selective spiking state-space blocks x N:
         input-dependent Delta/B/C  (Mamba-style selectivity)
         diagonal state scan, spike-gated (event-driven) updates, LIF, O(L)
  -> supervised classifier head        (mean output membrane over time)
  -> self-supervised next-window head  (anomaly score = prediction error)
  -> confidence-triggered early exit
```

## Sequence construction (KDDCup99)

KDDCup99's 41-feature format has **no source-IP / destination-IP / timestamp**,
so genuine per-host windows are impossible. We therefore use **consecutive-record
sliding windows**: window i = rows `[i, i+L)` in the file's (temporal) order,
with the window label = the class of its **last** row. Windows are built within
the official train file and within the test file separately (no window straddles
the split). Preprocessing is identical to the tabular KDDCup99 loader (log1p,
one-hot of the three categoricals over the train+test union, MinMax to [0,1] fit
on train), so the features match the SNN / S-xLSTM / Spiking-Griffin pipeline —
only windowed.

`seq_len` (L) and `stride` are configurable; `stride: 16` (= `seq_len`) gives
non-overlapping windows. Lower stride → more overlap and more training windows
(and more compute).

## Layout

```
experiments/Spiking-SSM/
└── kddcup99/    config.yaml + run.py
```

Shared code in `src/`:
- `src/data/sequence_windows.py` — consecutive-record windowing
  (`get_sequence_dataset`, `SEQ_DATASET_LOADERS`); the extension point for more datasets.
- `src/models/spiking_ssm.py` — `SpikingFrontEnd`, `SpikingSSMBlock`, `SpikingSSM`
  (+ early-exit + self-supervised score).
- `src/training_ssm.py` — multi-task training/eval loop (multi-seed, rich metrics,
  early-exit), reusing `src/metrics`.

## Running (per-dataset positional, like the other models)

```bash
# one dataset, default seed (42)
python experiments/Spiking-SSM/kddcup99/run.py

# one dataset, multiple seeds
python scripts/run_ssm.py kddcup99 --seeds 42 43 44 45 46

# detached, survives disconnect (session hisnn-ssm-<dataset>)
./scripts/run_ssm_tmux.sh kddcup99 --seeds 42 43 44 45 46

# dry-run / force
python scripts/run_ssm.py kddcup99 --dry-run
python scripts/run_ssm.py kddcup99 --seeds 42 --force
```

## Results (stored separately from SNN / S-xLSTM / Spiking-Griffin)

```
results/Spiking-SSM/<dataset>/seed_<N>/results.json   # rich metrics + early-exit + self-supervised blocks
results/Spiking-SSM/<dataset>/seed_<N>/checkpoint.pt
results/summary_ssm.csv                               # separate summary, has a `model` column
```

`results.json` carries the shared metric schema (accuracy, balanced accuracy,
F1 macro/weighted, per-class P/R/F1, MCC, Cohen's κ, ROC-AUC, DR/FAR, spike
energy + sparsity, parameter count, timing) **plus** Design-A-specific blocks:
`early_exit` (mean exit step out of `seq_len`, fraction flagged uncertain) and
`self_supervised` (mean next-window anomaly score on benign vs attack rows).

## Multi-task training

```
loss = CE(classification)
       + a_pred  * mean next-window MSE   (self-supervised predictor)
       + a_sparse * spike rate            (energy regulariser)
```

Inverse-frequency class weighting is **on** by default (`class_weighted: true`),
which is Design A's native choice for KDDCup99's heavy imbalance. Set it to
`false` for an unweighted comparison with the other models.

## Key configuration

| Setting | Default | Note |
|---|---|---|
| `seq_len` (L) | 16 | window length |
| `stride` | 16 | non-overlapping; lower = more overlap/data |
| `d_model` / `d_state` / `n_blocks` | 32 / 8 / 2 | from the reference prototype |
| `beta` | 0.9 | LIF membrane decay |
| `a_pred` / `a_sparse` | 0.3 / 1e-3 | multi-task loss weights |
| `theta_exit` / `theta_uncertain` | 0.9 / 0.5 | early-exit confidence + uncertainty flag |
| `batch_size` / `epochs` / `lr` | 128 / 10 / 2e-3 | (reference used 2e-3) |
| `class_weighted` | true | inverse-frequency CE for imbalance |

## Notes and caveats

- The selective scan is an explicit O(L) Python loop for clarity; a
  parallel/associative scan is a drop-in speed optimisation.
- The mean exit step (out of `seq_len`) and spike counts are **energy proxies**,
  not measured hardware energy. The self-supervised score is reported for the
  benign-vs-attack separation it provides, not as a tuned detector.
- The LRAM / in-context-support / dual-mode pieces from the design figures are
  not implemented; this is the core spiking-SSM backbone.
