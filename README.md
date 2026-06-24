# HiFi-SNN
Hybrid Spiking Neural Networks for High Fidelity Intrustion Detection (HiFi-SNN).

---

# Project Roadmap

The project runs in three phases. **Phase 1** (the 162-run SNN ablation) is
complete; **Phase 2** (modern datasets, hybrids, the iso-architecture DNN twin,
and the TASNN SOTA comparison) is in progress; **Phase 3** (adversarial
robustness) is deferred.

### Phase 1 — Comprehensive SNN ablation *(complete)*

Identify which **(neuron model, spike encoding)** combination trains and
generalises best on intrusion-detection traffic.

- **9 neuron models** from `snntorch`: Alpha, Lapicque, Leaky, LeakyParallel,
  RLeaky, RSynaptic, SConv2dLSTM, SLSTM, Synaptic
- **3 spike encodings**: rate coding, latency coding, delta modulation
- **6 classic IDS datasets**: KDDCup99, NSL-KDD, CTU-13, ISCXIDS2012,
  CIC-IDS2017, CSE-CIC-IDS2018
- **9 × 3 × 6 = 162 runs** — one script + one config + one results directory per run

Winning operating point: **`LeakyParallel` neuron + `latency` (TTFS) encoding**,
carried forward as the SNN backbone for everything below.

### Phase 2 — Modern datasets, hybrids, and a fair DNN / SOTA comparison *(in progress)*

Built on the Phase-1 winner.

- **5 modern IoT / IIoT / IoMT datasets** at the winning operating point:
  **UNSW-NB15, CICIoT2023, CICIoMT2024, Edge-IIoTset, TON-IoT** — taxonomy /
  split / preprocessing matched to each source paper (see
  [SOTA_DATASETS.md](SOTA_DATASETS.md)).
- **Hybrid SNNs**: **Tree-to-Spike** (GBDT-guided spike encoder + distilled,
  int8 + early-exit SNN student — our headline model), plus **S-xLSTM**,
  **Spiking-Griffin**, **Spiking-SSM**.
- **Iso-architecture DNN twin** (`DNN-Tree`): a dense ReLU MLP with the *identical*
  topology and parameter count as the Tree-to-Spike student, for a fair
  SNN-vs-DNN energy comparison (SOPs × 0.9 pJ vs MACs × 4.6 pJ, Horowitz 45 nm).
- **SOTA competitor** (`TASNN`): faithful reimplementation of the transformer–
  spiking hybrid (Sci. Rep. 2026) run on our *shared* features (no privileged
  preprocessing) for a fair head-to-head — see [experiments/TASNN/README.md](experiments/TASNN/README.md).

All models share one metric set, so `results/summary*.csv` are directly comparable.

### Phase 3 — Adversarial robustness *(deferred)*

FGSM, PGD, C&W, BIM, MIM, plus Membership Inference and Model Stealing. The
directory layout and shared library are designed so this plugs in without
disturbing earlier results.

---

# Model Comparison (4-way)

Phase 2's headline result is a **4-way comparison on identical data**, run across the
datasets (6 classic + 5 modern — see [SOTA_DATASETS.md](SOTA_DATASETS.md)):

| Model | What it is | Params* | Summary CSV |
|---|---|--:|---|
| **Vanilla SNN** | `LeakyParallel` neuron + `latency` (TTFS) encoding — the Phase-1 winner | ~16 k | `results/summary.csv` |
| **Tree-to-Spike** | GBDT-guided spike encoder + distilled, int8 + early-exit SNN student (**headline model**) | ~15 k | `results/summary_treespike.csv` |
| **DNN-Tree** | iso-architecture dense ReLU twin of Tree-to-Spike (**identical topology + param count**) | = Tree-to-Spike | `results/summary_dnn.csv` |
| **TASNN** | transformer–spiking SOTA competitor (Sci. Rep. 2026), on our features, no privileged preprocessing | ~0.6 M | `results/summary_tasnn.csv` |

*Per-dataset (varies with feature dimensionality); Tree-to-Spike and DNN-Tree are
parameter-identical for a given dataset.

**Fair-comparison basis.** Every model sees the *same* features (one-hot + log1p +
MinMax→[0,1]), the *same* train/val/test splits, the *same* `subsample_fraction` /
`label_mode`, **no resampling**, and **5 seeds (42–46)** — so differences reflect the model,
not the pipeline. DNN-Tree is parameter-identical to Tree-to-Spike (only LIF→ReLU differs),
isolating the spiking-vs-dense effect; TASNN is the heavyweight transformer baseline run on
the same footing (its PAAN / Port-Role / CPSMOTE-Tomek preprocessing is deliberately
excluded — see [experiments/TASNN/README.md](experiments/TASNN/README.md)).

## Metrics reported for every model

All produced by [src/metrics.py](src/metrics.py), so every row is apples-to-apples.

**Detection quality** — `accuracy`, `balanced_accuracy`; `f1_macro`, `f1_weighted` (+ per-class F1);
`precision_macro`, `recall_macro`; **`matthews_corrcoef` (MCC)** and `cohen_kappa`
(gold-standard for imbalanced multi-class); `roc_auc_weighted` (one-vs-rest, support-weighted).

**IDS-canonical (binary normal-vs-attack)** — **`detection_rate`** (DR = TP/(TP+FN)) and
**`false_alarm_rate`** (FAR = FP/(FP+TN)), the primary report-card metrics in the IDS
literature; plus `binary_accuracy`, `binary_f1`.

**Efficiency / energy** — *the headline of the SNN-vs-DNN argument:*
- `energy_spikes_per_sample`, `spike_sparsity` (spiking models) vs `macs_per_sample` (DNN)
- **operation energy**: SOPs × 0.9 pJ (spiking) vs MACs × 4.6 pJ (dense) — Horowitz 45 nm
- **memory-aware total energy** (batch-1 SRAM accounting) — built to survive a hostile reviewer
- **early-exit** `mean_exit_step` (effective T ≪ T → where the SNN energy win comes from)
- `trainable_parameters`, `inference_ms_per_sample` (batch-1 CPU latency)

**The defensible claim:** *comparable detection quality at materially lower energy than the
parameter-matched DNN-Tree twin* — **not** "higher accuracy." The accuracy–energy Pareto and
the 3-/4-way radar are generated by [scripts/benchmark_3way.py](scripts/benchmark_3way.py)
(TASNN folds in as the fourth model).

---

# Phase 1 Design

### What varies, what doesn't

The only things that differ across the 162 runs are:

1. **Dataset** — different feature dimensionality, class count, scale.
2. **Neuron model** — a one-line swap in the SNN definition.
3. **Encoding** — the function that maps a `[batch, features]` tensor to a
   `[timesteps, batch, features]` spike train.

Everything else — training loop, optimiser, loss, metrics, logging — is shared.

### Dataset split (NSL-KDD)

Splits follow `notebooks/exploration/02_nslkdd_exploration.ipynb`:

| Split | Source | Use | Size |
|---|---|---|---|
| train | KDDTrain+.txt → stratified 90% | gradient updates | ~113 k |
| val | KDDTrain+.txt → stratified 10% | per-epoch monitoring | ~12.6 k |
| test+ | KDDTest+.txt | **challenging** (includes novel attack types) | 22 544 |
| test-21 | KDDTest-21.txt | **standard** (excludes difficulty-21 trivial records) | 11 850 |

Preprocessing pipeline (in [src/data/nslkdd.py](src/data/nslkdd.py)):
1. Drop `difficulty` metadata column.
2. `log1p` on heavy-tailed numerics (`duration`, `src_bytes`, `dst_bytes`,
   the byte / count columns) — without this, MinMax collapses 99% of mass
   to ~0 because `src_bytes` spans 0 → 1.4 B.
3. One-hot `{protocol_type, service, flag}` against the union of all splits.
4. Map labels → 5-class taxonomy `{normal, dos, probe, r2l, u2r}` (all 17
   novel KDDTest+ attack types are placed into the existing 4 attack buckets
   per standard NSL-KDD literature).
5. MinMax scale to [0, 1] — fit on train, transform-and-clip val / test+ / test-21.

### Metrics (computed on both test sets per run)

| Category | Metric | Notes |
|---|---|---|
| Accuracy | `accuracy`, `balanced_accuracy` | balanced accuracy is the imbalance-robust pair |
| F1 | `f1_macro`, `f1_weighted` | plus `per_class[*].f1` |
| Precision / Recall | `precision_macro`, `recall_macro` | plus `per_class[*].precision` / `.recall` |
| Multi-class agreement | `matthews_corrcoef`, `cohen_kappa` | MCC = gold-standard for imbalanced multi-class |
| Ranking | `roc_auc_weighted` | one-vs-rest, weighted by support, softmax-normalised spike counts |
| Confusion | `confusion_matrix` | as 2-D list |
| **IDS-canonical (binary normal-vs-attack)** | `detection_rate`, `false_alarm_rate`, `binary_accuracy`, `binary_f1`, `binary_confusion_matrix` | DR = TP/(TP+FN); FAR = FP/(FP+TN). The IDS literature treats these as the *primary* report-card metrics. |
| Energy / efficiency | `energy_spikes_per_sample`, `spike_sparsity`, `total_spikes` | sparsity = 1 − (spikes / sites) |
| Cost | `trainable_parameters`, `train_time_s`, `inference_ms_per_sample` | |
| Per-epoch | `training_curve[t] = {train_loss, val_loss, val_accuracy}` | for diagnostic plots |

All produced by [src/metrics.py](src/metrics.py) so cross-run comparison is apples-to-apples.

### Alignment with prior NSL-KDD work

Our setup is deliberately compatible with the dominant evaluation conventions in
the NSL-KDD literature, so headline numbers from this sweep can be placed
directly alongside published baselines:

| Aspect | Our choice | Aligns with |
|---|---|---|
| Class taxonomy | 5-class {normal, dos, probe, r2l, u2r} | Zhou et al. 2020 (arXiv:2010.07803); TASNN 2025 (PMC12920788); community baselines |
| Feature dim | 41 → 122 (one-hot of `protocol_type`/`service`/`flag`) | matches Zhou et al. 2020 *exactly* |
| Test sets | Both KDDTest+ (challenging) and KDDTest-21 (standard) | Zhou et al. 2020 reports both |
| Numeric scaling | `log1p` on heavy-tailed columns → MinMax to [0,1] | log1p per our own exploration notebook; MinMax required by spike encoders |
| IDS binary metrics | `detection_rate` + `false_alarm_rate` | thinline72 community baseline flags these as "most important"; ubiquitous in IDS papers |

Two deliberate departures, with rationale:

| | Common in prior work | Our choice | Rationale |
|---|---|---|---|
| Train/val split | 80/20 (community) or KDDTrain+ entire with internal CV (Zhou et al.) | 90/10 stratified | More training data for the heavily-minority R2L (~900 samples) and U2R (~50 samples) classes; the 10 % val is still ~12 600 samples for monitoring. |
| Seeds | 1 (Zhou) or 5 (TASNN) | 1 for the 162-run ablation; **5 seeds (42–46)** for the carried-forward models | Single seed keeps the full 9×3×6 ablation tractable; the winning combo, the hybrids, the DNN twin, and TASNN are each run over 5 seeds for mean ± std / CI reporting. |

Things we **add** beyond prior work: 9-neuron × 3-encoding ablation grid;
spike energy proxy + sparsity (rare in IDS papers but essential for the SNN
efficiency claim); MCC + Cohen's Kappa (gold-standard imbalanced-multi-class
metrics that most prior IDS papers omit).

### Neuron models *(directory name → `snntorch` class)*

| Directory | snntorch class | Description |
|---|---|---|
| `Alpha` | `snn.Alpha` | Double-exponential (alpha) synaptic conductance |
| `Lapicque` | `snn.Lapicque` | RC-circuit LIF with explicit membrane time constant |
| `Leaky` | `snn.Leaky` | Standard Leaky Integrate-and-Fire |
| `LeakyParallel` | `snn.LeakyParallel` | Parallelised LIF for faster training across timesteps |
| `RLeaky` | `snn.RLeaky` | Recurrent LIF with self-feedback |
| `RSynaptic` | `snn.RSynaptic` | Recurrent synaptic neuron with feedback |
| `SConv2dLSTM` | `snn.SConv2dLSTM` | Spiking ConvLSTM (spatial-temporal feature extraction) |
| `SLSTM` | `snn.SLSTM` | Spiking LSTM cell |
| `Synaptic` | `snn.Synaptic` | First-order synaptic current model |

### Spike encodings

| Directory | Strategy | Description |
|---|---|---|
| `rate` | Rate coding | Spike probability ∝ input magnitude over T timesteps |
| `latency` | Latency coding | Stronger input → earlier spike; one spike per neuron per sample |
| `delta` | Delta modulation | Spike on significant change in input value (temporal contrast) |

---

# Project Structure

## Organising Philosophy

Code is organised **by concern, not by dataset**. With 11 datasets, 9 neuron
models, 3 encoding strategies, and several model families (SNN ablation,
hybrids, the DNN twin, the TASNN competitor) the alternative would be hundreds
of near-duplicate copies of the same training loop — a bug fix would need to be
applied in every copy.

The solution is to isolate what actually *differs* between datasets — loading
and preprocessing — in `src/data/`, and keep everything else (neurons,
encodings, model wiring, metrics) shared and dataset-agnostic. Once data
leaves `src/data/`, all downstream components receive a uniform tensor
interface and never need to know which dataset they came from.

**Experiment configs** (`configs/`) are the glue: a single YAML file binds a
dataset, neuron model, and encoding together for one reproducible run.

## Directory Overview

```
HiSNN/
├── dataset/          raw data files (gitignored — see Datasets section)
├── src/              all source code, organised by concern
├── configs/          YAML experiment definitions (one file = one reproducible run)
├── experiments/      entry-point scripts that read a config and execute a run
├── results/          all outputs (gitignored except summary.csv)
└── notebooks/        dataset exploration and results-analysis notebooks
```

---

## `src/` — Shared Library

```
src/
├── data/         one module per dataset: loading, cleaning, normalisation
├── encoding/     spike encoding strategies (rate, latency, delta)
├── neurons/      thin wrappers / factory around snntorch neuron classes
├── models/       SNN architecture(s) that consume spike tensors
└── metrics.py    unified metrics computation shared across all experiments
```

### `src/data/`

The **only** place dataset names appear in source code. Each module exposes a
consistent interface: given a file path and a split (`"train"` or `"test"`),
return a normalised PyTorch `DataLoader`. Input dimensionality and class count
are exposed as properties so models can be configured dynamically from the
loader rather than from hardcoded constants. `base.py` defines the abstract
interface that every loader must implement.

### `src/encoding/`

Converts a continuous feature vector into a spike train consumed by the SNN.
Each module is a standalone function that takes a `[batch, features]` tensor
and returns a spike tensor of shape `[timesteps, batch, features]`.

| Module | Strategy |
|---|---|
| `rate.py` | Rate coding |
| `latency.py` | Latency coding |
| `delta.py` | Delta modulation |

### `src/neurons/`

Neuron models are used directly from the `snntorch` library. This module is a
thin factory mapping a directory name (e.g. `"Leaky"`) to the corresponding
`snn.<Class>` constructor with the right defaults, so swapping neuron models
is a one-line change driven by the config.

### `src/models/`

The shared SNN architecture that every SNN run uses, plus the hybrid models
(`tree_to_spike.py`, `s_xlstm.py`, `spiking_griffin.py`, `spiking_ssm.py`), the
iso-architecture DNN twin (`dnn_student.py`), and the TASNN competitor
(`tasnn.py`). The neuron type is a constructor argument, not a hardcoded class,
so one SNN definition covers all nine variants.

### `src/metrics.py`

Single shared module for all metrics: Accuracy, F1, Precision, Recall,
ROC-AUC, an energy proxy (total spike count), and wall-clock inference time.
Every experiment script calls this module — never inline — to guarantee
consistency across every comparison row.

---

## Unified Experiment Hierarchy

The entire experiment lifecycle — configuration, execution, results — follows
the same directory structure. For any combination, the config, the training
script, and the saved results all live under the same relative path:

```
configs/nslkdd/SNN_Leaky/rate/      ← hyperparameters (config.yaml)
experiments/nslkdd/SNN_Leaky/rate/  ← run.py (thin wrapper)
results/nslkdd/SNN_Leaky/rate/      ← results.json, checkpoint.pt
```

The full shape:

```
configs/   {dataset}/SNN_{neuron}/{rate|latency|delta}/   ← Phase-1 ablation grid
experiments/
└── {dataset}/SNN_{neuron}/{rate|latency|delta}/          Phase-1 SNN ablation
└── Tree-to-Spike/{dataset}/        GBDT-guided hybrid (headline model)
└── DNN-Tree/{dataset}/             iso-architecture dense ANN twin
└── TASNN/{dataset}/                transformer-spiking SOTA competitor
└── {S-xLSTM,Spiking-Griffin,Spiking-SSM}/{dataset}/      other hybrids
results/   mirrors the above; per-model summary CSVs:
           summary.csv (SNN) · summary_treespike.csv · summary_dnn.csv · summary_tasnn.csv · …
```

**Datasets (11).** Classic IDS: `kddcup99`, `nslkdd`, `ctu13`, `iscxids2012`,
`cicids2017`, `cse-cic-ids2018`. Modern IoT/IIoT/IoMT: `unswnb15`, `ciciot2023`,
`ciciomt2024`, `edgeiiot`, `toniot`. The full 9×3 ablation grid is run on the 6
classic datasets; the hybrids, DNN twin, and TASNN run on all 11 at the
Phase-1 winning operating point (`LeakyParallel` + `latency`).

> The per-model top-level directories (`SNN_<neuron>`, `Tree-to-Spike`,
> `DNN-Tree`, `TASNN`, …) all sit under `experiments/` and share the same
> `{dataset}/` leaves and metric set, so a new model family plugs in without
> renaming anything.

### `configs/`

Each leaf directory holds a `config.yaml` with all hyperparameters for that
run (learning rate, batch size, timesteps, neuron-specific constants, etc.).
The path itself encodes what is being run, so there is no ambiguity about
what a config produces.

### `experiments/`

Each leaf directory holds the training and evaluation script for that
combination. The shared library (`src/`) provides all reusable components; the
per-experiment script is only responsible for wiring them together, running
the loop, and writing outputs to the matching `results/` path.

### `results/`

Each leaf directory receives two artifacts per completed run:

- `results.json` — the full record (config, data-split sizes, model size,
  training curve, full metric set for **both** test sets, confusion matrices,
  per-class breakdown, energy, sparsity, timing). Top-level schema:

```json
{
  "dataset": "nslkdd",
  "neuron": "Leaky",
  "encoding": "rate",
  "config":  { "num_steps": 25, "hidden_size": 128, "batch_size": 256,
               "epochs": 5, "lr": 0.001, "val_fraction": 0.1, "seed": 0,
               "device": "cuda" },
  "data":    { "num_features": 122, "num_classes": 5,
               "class_names": ["normal", "dos", "probe", "r2l", "u2r"],
               "class_counts_train": { "normal": 60608, "dos": 41334, ... },
               "splits": { "train": 113375, "val": 12598,
                           "test_plus": 22544, "test_minus_21": 11850 } },
  "model_size":    { "total_parameters": 16389, "trainable_parameters": 16389 },
  "training":      { "train_time_s": 11.7,
                     "training_curve": [
                       { "epoch": 1, "train_loss": 0.110,
                         "val_loss": 0.025, "val_accuracy": 0.991 }, ... ] },
  "test_plus":     { "n_samples": 22544,
                     "metrics": { "accuracy": ..., "balanced_accuracy": ...,
                                  "f1_macro": ..., "f1_weighted": ...,
                                  "precision_macro": ..., "recall_macro": ...,
                                  "matthews_corrcoef": ..., "cohen_kappa": ...,
                                  "roc_auc_weighted": ...,
                                  "per_class": { "normal": { "precision":...,
                                                  "recall":..., "f1":...,
                                                  "support":... }, ... },
                                  "confusion_matrix": [[...]],
                                  "class_names": [...] },
                     "spike_stats":  { "energy_spikes_per_sample": 301.5,
                                       "total_spikes": ..., "spike_density": 0.091,
                                       "spike_sparsity": 0.909 },
                     "inference_time_s": 0.54,
                     "inference_ms_per_sample": 0.024,
                     "test_loss": 0.094 },
  "test_minus_21": { ... same shape as test_plus, for KDDTest-21 ... }
}
```

- `checkpoint.pt` — the trained model state dict.

Both are gitignored. The **only** results artifact tracked in git is
`results/summary.csv` — one row appended per completed run, written by
[src.training.run_experiment](src/training.py). Schema:

```
dataset, neuron, encoding,
accuracy, balanced_accuracy, f1_macro, f1_weighted,
precision_macro, recall_macro, roc_auc_weighted,
matthews_corrcoef, cohen_kappa,
test21_accuracy, test21_f1_macro, test21_matthews_corrcoef,
energy_spikes_per_sample, spike_sparsity, trainable_parameters,
train_time_s, inference_ms_per_sample,
epochs, num_steps, hidden_size, batch_size, lr, seed
```

This CSV is the direct source for every comparison table and figure across
the 162 runs.

---

## Running an experiment

### Single combination
Each leaf script is a thin self-contained wrapper:

```bash
python experiments/nslkdd/SNN_Leaky/rate/run.py
```

### Full sweep — use the orchestrator
[scripts/run_sweep.py](scripts/run_sweep.py) discovers every script for a dataset, runs them
sequentially, and prints per-run status:

```bash
# All 27 NSL-KDD combinations
python scripts/run_sweep.py nslkdd

# Subset (by neuron and/or encoding)
python scripts/run_sweep.py nslkdd --only Leaky Alpha --encodings rate latency

# Re-run everything from scratch (ignore skip-guard)
python scripts/run_sweep.py nslkdd --force

# Just print the plan
python scripts/run_sweep.py nslkdd --dry-run
```

### Training hyperparameters (and prior-work alignment)

| Parameter | Our setting | Source |
|---|---|---|
| Optimiser | Adam | matches Zhou 2020 (arXiv:2010.07803), TASNN 2025, Attention-CNN-LSTM (Nature 2025) |
| Learning rate | 1e-3 | identical to all three |
| **Batch size** | **128** | identical to Zhou 2020 and Attention-CNN-LSTM |
| **Epochs** | **10** | identical to Attention-CNN-LSTM; within the 10–50 range typical for SNNs |
| Training mode | mini-batch SGD (Adam) | identical to all SNN prior work — no full-batch GD |
| Hidden size | 128 | our choice; not standardised across papers |
| Timesteps T | 25 | mid-range for snntorch defaults |

### Reproducibility
- Every config sets `seed: 42`. [src/training.py](src/training.py) seeds `numpy`, `torch`,
  CUDA, and forces `torch.backends.cudnn.deterministic = True`. Two runs of
  the same script with the same code produce identical numbers.
- **Multi-seed runs do not overwrite each other.** Each run writes to
  `results/{dataset}/SNN_{neuron}/{encoding}/seed_{N}/`, so seeds 42, 43, 44 all
  coexist. `results/summary.csv` keys on `(dataset, neuron, encoding, seed)`
  for dedup, so each seed contributes its own row.
- The orchestrator's `--seeds` flag fans out the grid over multiple seeds in
  one invocation:
  ```bash
  python scripts/run_sweep.py nslkdd --seeds 42 43 44   # 27 × 3 = 81 runs
  ```
- For variance reporting, the plotting script auto-detects multi-seed runs
  and produces `ranked_by_f1_macro_multiseed.csv` with mean ± std per combo.

### GPU contention — how to make training-time comparable

Wall-clock training time is the **one** metric in our sweep that's sensitive
to GPU sharing. Accuracy, F1, MCC, AUC, DR/FAR, spike sparsity, parameter
count — all of those are bit-deterministic regardless of GPU contention.
Inference-per-sample timing is bracketed with `torch.cuda.synchronize()` so
it reflects compute rather than queue depth, but it can still drift on a
busy GPU.

[scripts/run_sweep.py](scripts/run_sweep.py) prints an `nvidia-smi` snapshot at the start of
every sweep so you can see the GPU state before committing.

**Recommendations, in order of preference:**

1. **Best — exclusive GPU.** Wait for the GPU to idle, then run the whole
   162-combo sweep in one go (~30–90 min depending on neuron mix). All
   `train_time_s` values are then mutually comparable.
2. **Good — pin to an idle GPU.** If the host has multiple GPUs and one is
   free, pin to it:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python scripts/run_sweep.py nslkdd
   ```
   The orchestrator records `cuda_visible_devices` and `gpu_name` in every
   `results.json` so post-hoc you can confirm all rows used the same device.
3. **Acceptable — shared GPU, single sweep, single sitting.** All 27 runs
   under the same contention level → training times are noisy but
   *internally* comparable (the ordering between neurons is preserved, the
   absolute seconds aren't). The 1-epoch smoke test showed a 7.5 s → 16.4 s
   slowdown for SNN_Leaky/rate when the GPU went from idle to 100 %
   utilisation, i.e. roughly 2× under heavy contention.
4. **Worst — shared GPU, run interleaved with other workloads.** The
   contention level changes between runs so `train_time_s` becomes
   uninterpretable. Avoid.

If you cannot get exclusive access for the full sweep, our recommendation is
to **report model-quality metrics from the contested run** (those are
deterministic) and **defer the training-time analysis** until you can re-run
the few top performers on an idle GPU.

### No silent overwrites
- A run that finds an existing `results.json` in its leaf directory **skips**
  (returns the existing record). Pass `--force` (or set `force: true` in the
  config) to re-run.
- `results/summary.csv` dedups on `(dataset, neuron, encoding, seed)`: a re-run
  *replaces* its previous row rather than appending a duplicate.
- Each leaf writes its own `results.json` + `checkpoint.pt` under
  `results/{dataset}/SNN_{neuron}/{encoding}/` — combinations never overwrite
  each other.

Each script (and the orchestrator):
1. Reads `configs/{dataset}/SNN_{neuron}/{encoding}/config.yaml`
2. Calls `src.training.run_experiment(cfg)`
3. Writes `results/{dataset}/SNN_{neuron}/{encoding}/{results.json, checkpoint.pt}`
4. Updates the row for this combination in `results/summary.csv`

## Result figures and tables

After a sweep completes, generate the analysis figures with one command:

```bash
python scripts/make_plots.py nslkdd
```

Outputs land in `results/{dataset}/figures/` as both **PDF** (vector,
paper-ready) and PNG (preview). Styling uses an IEEE-compatible
**colorblind-safe palette** — the Wong 2011 8-colour set for categorical
data and `viridis` for sequential / heatmap data (both perceptually uniform,
readable in greyscale, and accepted in IEEE accessibility guidelines).

| File | What it shows |
|---|---|
| `01_accuracy_heatmap.pdf` | Accuracy across 9 neurons × 3 encodings |
| `02_f1_macro_heatmap.pdf` | Macro-F1 same grid |
| `03_mcc_heatmap.pdf` | MCC same grid |
| `04_detection_rate_heatmap.pdf` | IDS detection rate (binary) |
| `05_far_heatmap.pdf` | IDS false alarm rate (binary) |
| `06_spike_sparsity_heatmap.pdf` | Spike sparsity (efficiency) |
| `07_per_class_f1.pdf` | Per-class F1 bars, 3×3 grid by neuron, grouped by encoding |
| `08_dr_vs_far.pdf` | Detection Rate vs False Alarm Rate scatter (IDS ROC view) |
| `09_accuracy_vs_energy.pdf` | Accuracy vs energy proxy (Pareto trade-off) |
| `10_training_curves.pdf` | Per-epoch validation loss, 3×3 grid by neuron |
| `ranked_by_f1_macro.csv` | Top-to-bottom ranking of all 27 combinations |

---

## `notebooks/`

```
notebooks/
├── exploration/         dataset inspection: feature distributions, class balance, correlation
└── results_analysis/    load summary.csv → generate paper figures and LaTeX tables
```

---

# Datasets

All datasets are stored locally under `dataset/` and are gitignored due to size.

### 1. KDD Cup 1999
Early benchmark for intrusion detection, derived from the DARPA evaluation experiment.
- Source: [kdd.org](https://www.kdd.org/kdd-cup/view/kdd-cup-1999/Data) · [Kaggle](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)
- Local path: `dataset/kddcup99/`
- Format: CSV (raw + 10% subset, labeled test set `corrected`)

### 2. NSL-KDD
Refined version of KDD Cup 1999 — removes duplicate records and rebalances attack classes.
- Source: [UNB CIC](http://www.unb.ca/cic/datasets/nsl.html) · [Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd)
- Local path: `dataset/nslkdd/`
- Format: TXT / ARFF (`KDDTrain+`, `KDDTest+`, 20% subset, `KDDTest-21`)

### 3. CTU-13
Botnet traffic dataset with 13 scenarios covering 7 distinct botnets in a university-like environment.
- Source: [Stratosphere IPS](https://www.stratosphereips.org/datasets-ctu13) · [Kaggle](https://www.kaggle.com/datasets/dhoogla/ctu13)
- Local path: `dataset/ctu13/`
- Format: Parquet (one file per scenario/botnet)

### 4. ISCXIDS2012
Network traffic with labeled normal and attack flows across multiple attack types.
- Source: [UNB CIC](https://www.unb.ca/cic/datasets/ids.html)
- Local path: `dataset/iscxids2012/`
- Format: tar.gz archive — extract before use: `tar -xzf dataset/iscxids2012/iscxids2012-master.tar.gz -C dataset/iscxids2012/`

### 5. CIC-IDS2017
Machine-learning-focused IDS dataset covering DoS, DDoS, port scanning, and more.
- Source: [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) · [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
- Local path: `dataset/cicids2017/`
- Format: CSV (one file per day of the week)

### 6. CSE-CIC-IDS2018
CSE–CIC joint dataset extending CIC-IDS2017 with additional attack scenarios.
- Source: [UNB CIC](https://www.unb.ca/cic/datasets/ids-2018.html) · [Kaggle](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv)
- Local path: `dataset/cse-cic-ids2018/`
- Format: CSV (one file per date)

---

## SOTA-comparison datasets (added) — sources & taxonomy verification

Five modern IoT/IIoT/IoMT benchmarks added for the head-to-head against recent hybrid
SNN-NIDS work. **The class taxonomies and evaluation config of each loader were verified
against the dataset's source paper** (label scenarios, train/test split, preprocessing,
metrics). The per-dataset alignment table — including the deliberate, documented deviations
(MinMax instead of StandardScaler, required by the spike encoders) — is in
[SOTA_DATASETS.md](SOTA_DATASETS.md). Each loader exposes a `label_mode`
(`binary` | `category` | `full`) switch so we can match whatever a given competitor reports.

| Dataset | Loader key / path | Class taxonomy (verified vs paper) | Source paper |
|---|---|---|---|
| **CICIoT2023** | `ciciot2023` · `dataset/cic-iot2023/` | binary / **8-category** / 34-fine (Benign + DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce) | Neto, Dadkhah, Ferreira, Zohourian, Lu, Ghorbani, *"CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment,"* **Sensors** 23(13):5941, 2023 — [paper](https://www.mdpi.com/1424-8220/23/13/5941) · [UNB CIC](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| **CICIoMT2024** | `ciciomt2024` · `dataset/cic-iomt-2024/` | binary / **6-category** / 19-fine (Benign + DDoS, DoS, Recon, MQTT, Spoofing) | Dadkhah, Neto, et al., *"CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT,"* **Internet of Things**, 2024 — [paper](https://www.sciencedirect.com/science/article/pii/S2542660524002920) · [UNB CIC](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) |
| **Edge-IIoTset** | `edgeiiot` · `dataset/Edge-IIoTset/` | binary / **6-category** / 15-fine (Normal + DDoS, Information-Gathering, Injection, MITM, Malware) | Ferrag, Friha, Hamouda, Maglaras, Janicke, *"Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning,"* **IEEE Access**, 2022, DOI [10.1109/ACCESS.2022.3165809](https://ieeexplore.ieee.org/document/9751703) · [Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) |
| **UNSW-NB15** | `unswnb15` · `dataset/unsw-nb15/` | 10-class `attack_cat` (Normal + 9 attacks); official train/test split | Moustafa & Slay, *"UNSW-NB15: a comprehensive data set for network intrusion detection systems,"* **MilCIS**, 2015 — [project](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| **TON-IoT** (Network) | `toniot` · `dataset/ton-iot/` | binary / **10-class `type`** (normal + backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss); Zeek/Bro flow features | Moustafa, *"A new distributed architecture for evaluating AI-based security systems ... Network TON_IoT datasets,"* **Sustainable Cities and Society**, 2021 — [project](https://research.unsw.edu.au/projects/toniot-datasets) |

The category mappings (e.g. CICIoT2023 *VulnerabilityScan → Recon*; the six web attacks →
*Web*) were taken directly from each paper, not assumed. Splits follow the source where one
is provided (CICIoMT2024 ships `csv/train`/`csv/test`; UNSW-NB15 ships training/testing files)
and the paper's ratio otherwise (80/20). See [SOTA_DATASETS.md](SOTA_DATASETS.md) for full detail.

---

# Additional Datasets
# Tier 1 Datasets — SNN vs DNN Energy Consumption Benchmarking

> **Strongly Recommended.** Large, well-established, and widely cited benchmarks ideal for energy benchmarking, enabling direct comparison with existing literature.

---

## 1. KDD Cup '99

**Why Applicable:** Historical standard; large scale; enables comparison with decades of IDS literature.

---

## 2. NSL-KDD

**Why Applicable:** Improved KDD; balanced classes; very commonly used ML baseline.

---

## 3. UNSW-NB15

**Why Applicable:** Modern, realistic, 9 attack families; widely used in recent DNN/ML IDS papers.

---

## 4. CIC-IDS2017

**Why Applicable:** De facto modern benchmark; 5-day simulation; high feature richness.

---

## 5. CSE-CIC-IDS2018

**Why Applicable:** Expanded CIC benchmark; larger testbed; good for scaling energy tests.

---

## 6. Improved CIC-IDS2017 / CSE-CIC-IDS2018

**Why Applicable:** Corrected labeling; more trustworthy for performance-energy tradeoff analysis.

---

## 7. LYCOS-IDS2017

**Why Applicable:** Corrected feature extraction; cleaner for fair algorithmic comparison.

---

## 8. BoT-IoT

**Why Applicable:** 72M+ records; massive scale ideal for energy cost measurement.

---

## 9. ToN-IoT

**Why Applicable:** Multi-modal telemetry; large and diverse; commonly used in recent SNN/IDS work.

---

## 10. N-BaIoT

**Why Applicable:** NetFlow-based; IoT botnet focus; well-structured for ML pipelines.

---

## 11. NF-UQ-NIDS / NF-UQ-NIDS-v2

**Why Applicable:** NetFlow standardized; re-generated from multiple benchmarks; great for cross-dataset comparison.

---

## 12. CTU-13

**Why Applicable:** 13 real-world botnet scenarios; temporal flow data favorable for SNNs.

---

## 13. ISCX-IDS-2012

**Why Applicable:** One of the earliest realistic IDS datasets; still widely used as a baseline.

---

## Summary Table

| # | Dataset | Why Applicable |
|---|---------|---------------|
| 1 | KDD Cup '99 | Historical standard; large scale; enables comparison with decades of IDS literature |
| 2 | NSL-KDD | Improved KDD; balanced classes; very commonly used ML baseline |
| 3 | UNSW-NB15 | Modern, realistic, 9 attack families; widely used in recent DNN/ML IDS papers |
| 4 | CIC-IDS2017 | De facto modern benchmark; 5-day simulation; high feature richness |
| 5 | CSE-CIC-IDS2018 | Expanded CIC benchmark; larger testbed; good for scaling energy tests |
| 6 | Improved CIC-IDS2017 / CSE-CIC-IDS2018 | Corrected labeling; more trustworthy for performance-energy tradeoff analysis |
| 7 | LYCOS-IDS2017 | Corrected feature extraction; cleaner for fair algorithmic comparison |
| 8 | BoT-IoT | 72M+ records; massive scale ideal for energy cost measurement |
| 9 | ToN-IoT | Multi-modal telemetry; large and diverse; commonly used in recent SNN/IDS work |
| 10 | N-BaIoT | NetFlow-based; IoT botnet focus; well-structured for ML pipelines |
| 11 | NF-UQ-NIDS / NF-UQ-NIDS-v2 | NetFlow standardized; re-generated from multiple benchmarks; great for cross-dataset comparison |
| 12 | CTU-13 | 13 real-world botnet scenarios; temporal flow data favorable for SNNs |
| 13 | ISCX-IDS-2012 | One of the earliest realistic IDS datasets; still widely used as a baseline |