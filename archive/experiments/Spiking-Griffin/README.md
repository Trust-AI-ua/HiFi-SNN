# Spiking-Griffin — spiking Griffin hybrid for NIDS

The **second Phase-2 hybrid**, alongside [S-xLSTM](../S-xLSTM/). It is a spiking
adaptation of Griffin: a backbone that **interleaves** a gated linear recurrence
with windowed local attention, each followed by a `LeakyParallel` spike readout.
It deliberately reuses the exact same front-end (latency / TTFS encoding), the
same `LeakyParallel` readout, and the **same training loop** as S-xLSTM, so the
two hybrids differ only in the backbone and can be compared head-to-head with
everything else held constant.

## Architecture

![Spiking-Griffin end-to-end pipeline: windowed traffic into per-modality encoders, latency spike encoding, an N-times interleaved backbone of a gated linear recurrence (RG-LRU) and a windowed spike-driven local-attention block each with a LeakyParallel readout, then a readout-with-early-exit head and an attack-class / alert output.](architecture.svg)

The backbone interleaves, `N` times: a **gated linear recurrence** (RG-LRU gating, teal) → `LeakyParallel` readout, then a **windowed spike-driven local attention** block (purple, `|i-j| <= window`) → `LeakyParallel` readout. The purple local-attention block is the one component that is **new relative to S-xLSTM**; everything else (latency front-end, `LeakyParallel` readout, training loop) is shared so the two hybrids are directly comparable. Coral = spiking readouts, teal = gated recurrence, purple = local attention, grey = I/O.

## Pipeline

```
feature vector x in [0,1]^d   (per-modality preprocessing handled in src/data)
  -> latency (TTFS) encoding                       spikegen.latency  -> [T, B, F]
  -> Spiking-Griffin backbone x N (interleaved):
         gated linear recurrence (RG-LRU gating)    -> LeakyParallel readout
         windowed spike-driven local attention      -> LeakyParallel readout
  -> readout + early-exit                            (accumulate spikes, decide)
  -> attack class / alert
```

- **Gated linear recurrence (teal):** a diagonal RG-LRU-style recurrence with an
  input-dependent decay gate (the selective leak). Linear in the state, hence
  parallelisable; run sequentially over the short latency window for clarity.
- **Windowed spike attention (purple):** softmax-free local attention on
  spike-form Q/K/V emitted by `LeakyParallel` neurons; scores are spike
  coincidence counts restricted to a local band `|i - j| <= window` and averaged
  over the window. Cost is `O(T * window)` with spike operands. This is what is
  **new relative to S-xLSTM**.
- The spike layer sits **after** each sub-block, keeping the recurrence linear
  and the attention windowed/local, so training is parallelisable and inference
  stays sub-quadratic.

## Layout

```
experiments/Spiking-Griffin/
└── kddcup99/    config.yaml + run.py     (the worked example for now)
```

Shared code in `src/`:
- `src/models/spiking_griffin.py` — `GatedLinearRecurrence`, `WindowedSpikeAttention`,
  `RecurrenceSubBlock`, `AttentionSubBlock`, `SpikingGriffin` (+ early-exit).
- `src/training_griffin.py` — training/eval loop (multi-seed, rich metrics,
  early-exit), reusing the `src/data` loaders and `src/metrics`. The loop logic
  is identical to `src/training_sxlstm.py` by construction, so S-xLSTM and
  Spiking-Griffin are trained and evaluated the same way.

Additional datasets (nslkdd, cicids2017, ctu13, ...) are a one-directory copy:
duplicate `kddcup99/`, change `dataset:` and `results_dir:` in `config.yaml`
(add `subsample_fraction:` for the large datasets). The loaders and `run.py`
are dataset-agnostic.

## Running (per-dataset positional, like the SNN and S-xLSTM scripts)

```bash
# one dataset, default seed (42)
python experiments/Spiking-Griffin/kddcup99/run.py

# one dataset, multiple seeds
python scripts/run_griffin.py kddcup99 --seeds 42 43 44 45 46

# detached, survives disconnect (session hisnn-griffin-<dataset>)
./scripts/run_griffin_tmux.sh kddcup99 --seeds 42 43 44 45 46

# dry-run / force
python scripts/run_griffin.py kddcup99 --dry-run
python scripts/run_griffin.py kddcup99 --seeds 42 --force
```

## Results (stored separately from SNN and S-xLSTM)

```
results/Spiking-Griffin/<dataset>/seed_<N>/results.json   # same schema as S-xLSTM + early-exit block
results/Spiking-Griffin/<dataset>/seed_<N>/checkpoint.pt
results/summary_griffin.csv                               # separate from summary.csv / summary_sxlstm.csv
```

Multi-seed isolation, skip-guard, and `(model, dataset, seed)` dedup behave
exactly like the SNN and S-xLSTM scripts. The energy proxy (spikes/sample)
counts **every** `LeakyParallel` emission — the two sub-block readouts plus the
spike-form Q/K/V inside the windowed attention.

## Key configuration

| Setting | Default | Note |
|---|---|---|
| `num_steps` (T) | 25 | latency window; matches the SNN ablation and S-xLSTM |
| `d_model` | 64 | hidden width |
| `num_layers` | 1 | each layer = 1 recurrence sub-block + 1 attention sub-block |
| `window` | 4 | local-attention radius (`|i-j| <= window`) |
| `beta` | 0.9 | LeakyParallel initial membrane decay (learned) |
| `batch_size` / `epochs` / `lr` | 128 / 10 / 1e-3 | match the SNN ablation and S-xLSTM |
| `weight_decay` | 1e-4 | mild regularisation |
| `class_weighted` | false | off by default for fair comparison; set `true` for inverse-frequency CE |
| `early_exit_conf` | 0.9 | confidence threshold for early exit |

## Notes and caveats

- The recurrence and windowed attention run sequentially / band-restricted over
  the short latency window for clarity; chunked parallel scan and exact
  spike-driven self-attention are drop-in optimisations.
- The windowed attention here uses spike-coincidence scores with a windowed
  mean (no softmax); it is a simplification of strict addition-only
  Spike-Driven Self-Attention.
- The LRAM / in-context-support and dual parallel-train/recurrent-infer modes in
  the design figure are **not** implemented; they are future extensions. This
  directory implements the core interleaved Spiking-Griffin backbone.
