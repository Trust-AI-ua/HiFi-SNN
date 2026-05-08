# HiSNN
Hybrid Spiking Neural Networks (HiSNN) for Intrusion Detection Systems.


# Project Architecture
1. Propose 2 hybrid approaches with SNN: one with **KAN** and one with **AutoEncoder + Energy-Based Model Head**
2. SNN Ablation Study — different rate encodings, different neuron models
3. 6 different datasets — various sizes, partial vs. full data, binary vs. multi-class classification
4. Use datasets and models from prior surveys for benchmarking and comparison
5. Baselines: SNN, KAN, Deep Neural Network, RNN
6. Metrics: Accuracy, Neuron Firing Rate, Energy Consumption, F1, Precision, Recall, ROC-AUC
7. Mathematical formulation of model complexity and hardness
8. Trade-off analysis: training time, computational overhead, inference time
9. Attack robustness: Membership Inference, Model Stealing, FGSM, PGD, C&W, BIM, MIM


---

# Project Structure

## Organizing Philosophy

Code is organized **by concern, not by dataset**. With 6 datasets, 3+ neuron models, 3+
encoding strategies, and 5+ model types, organizing by dataset would duplicate every model
~6 times — a bug fix or architecture change would need to be applied to each copy
independently.

The solution is to isolate what actually *differs* between datasets — loading and
preprocessing — in `src/data/`, and keep everything else (neurons, encodings, models,
attacks) shared and dataset-agnostic. Once data leaves `src/data/`, all downstream
components receive a uniform tensor interface and never need to know which dataset they came from.

**Experiment configs** (`configs/`) are the glue: a single YAML file binds a dataset, model,
neuron type, and encoding together for one reproducible run.

## Directory Overview

```
HiSNN/
├── dataset/          raw data files (gitignored — see Datasets section)
├── src/              all source code, organized by concern
├── configs/          YAML experiment definitions (one file = one reproducible run)
├── experiments/      entry-point scripts that read a config and execute a run
├── results/          all outputs (gitignored except summary.csv)
└── notebooks/        dataset exploration and results-analysis notebooks
```

---

## `src/` — Source Code

```
src/
├── data/             one module per dataset: loading, cleaning, normalization
├── encoding/         spike encoding strategies (rate, temporal, population)
├── neurons/          neuron model definitions (LIF, IF, AdLIF, ...)
├── models/           SNN and hybrid model architectures
│   └── baselines/    non-SNN baselines: DNN, RNN, KAN
├── attacks/          adversarial and membership-inference attack implementations
└── metrics.py        unified metrics computation shared across all experiments
```

### `src/data/`

The **only** place dataset names appear in source code. Each module exposes a consistent
interface: given a file path and a split (`"train"` or `"test"`), return a normalized
PyTorch `DataLoader`. Input dimensionality and class count are exposed as properties so
models can be configured dynamically from the loader rather than from hardcoded constants.
`base.py` defines the abstract interface that every loader must implement.

### `src/encoding/`

Converts a continuous feature vector into a spike train consumed by the SNN. Each module
is a standalone function that takes a `[batch, features]` tensor and returns a spike tensor
of shape `[timesteps, batch, features]`. The three strategies covered:

| Module | Strategy | Description |
|---|---|---|
| `rate.py` | Rate encoding | Spike probability ∝ input magnitude over T timesteps |
| `temporal.py` | Temporal / latency | Earlier spike = stronger input; single spike per neuron |
| `population.py` | Population encoding | Input represented by a bank of Gaussian receptive fields |

### `src/neurons/`

Neuron models are used directly from the `snntorch` library — no custom wrappers needed.
See the **Neuron models** table in the Unified Experiment Hierarchy section for the full
list. Swapping neuron models in any architecture is a one-line change in the config or
experiment script.

### `src/models/`

All models accept spike tensors from `src/encoding/` and produce logits, so they are
fully interchangeable in the experiment runner.

| Module | Description |
|---|---|
| `snn.py` | Baseline SNN (fully configurable neuron type) |
| `hybrid_kan.py` | SNN encoder + KAN classifier head |
| `hybrid_ae_ebm.py` | SNN encoder + AutoEncoder + Energy-Based Model head |
| `baselines/dnn.py` | Standard feedforward DNN |
| `baselines/rnn.py` | Recurrent baseline (LSTM / GRU) |
| `baselines/kan.py` | Pure Kolmogorov–Arnold Network |

### `src/attacks/`

Each attack module takes a model and a batch of inputs and returns adversarial examples.
White-box attacks (FGSM, PGD, C&W, BIM, MIM) and black-box attacks
(Membership Inference, Model Stealing) are kept separate.

### `src/metrics.py`

Single shared module for all metrics: Accuracy, F1, Precision, Recall, ROC-AUC,
an energy proxy (total spike count), and wall-clock inference time. Every experiment
script calls this module — never inline — to guarantee consistency across comparisons.

---

## Unified Experiment Hierarchy

The entire experiment lifecycle — configuration, execution, and results — follows the
same dataset-first directory structure. For any given combination the config, the
training script, and the saved results all live under the same relative path:

```
configs/nslkdd/snn/Leaky/rate/     ← hyperparameters (config.yaml)
experiments/nslkdd/snn/Leaky/rate/ ← training & evaluation script
results/nslkdd/snn/Leaky/rate/     ← metrics.json, plots, checkpoints
```

The shared shape across all three trees is:

```
{configs | experiments | results}/
└── {dataset}/
    ├── snn/
    │   ├── {neuron}/          one directory per snntorch neuron model (9 total)
    │   │   ├── rate/
    │   │   ├── latency/
    │   │   └── delta/
    │   └── ...
    ├── hybrid_kan/
    │   └── {neuron}/ {rate, latency, delta}
    ├── hybrid_ae_ebm/
    │   └── {neuron}/ {rate, latency, delta}
    ├── dnn/
    ├── rnn/
    └── kan/
```

SNN-based models (`snn`, `hybrid_kan`, `hybrid_ae_ebm`) have four levels because their
behaviour depends on both the neuron model and the encoding strategy. Baselines (`dnn`,
`rnn`, `kan`) are flat at the model level — they use neither.

**Neuron models** (all from the `snntorch` library):

| Directory name | snntorch class | Description |
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

**Spike encodings:**

| Directory name | Strategy | Description |
|---|---|---|
| `rate` | Rate coding | Spike probability ∝ input magnitude over T timesteps |
| `latency` | Latency coding | Stronger input → earlier spike; one spike per neuron per sample |
| `delta` | Delta modulation | Spike on significant change in input value (temporal contrast) |

Datasets: `kddcup99`, `nslkdd`, `ctu13`, `cicids2017`, `cse-cic-ids2018`, `iscxids2012`.

### `configs/`

Each leaf directory holds a `config.yaml` with all hyperparameters for that run
(learning rate, batch size, timesteps, neuron-specific constants, etc.). The path itself
encodes what is being run, so there is no ambiguity about what a config produces.

### `experiments/`

Each leaf directory holds the training and evaluation script for that combination. A
single shared library (`src/`) provides all reusable components; the per-experiment
script is only responsible for wiring them together, running the loop, and writing
outputs to the matching `results/` path.

### `results/`

Each leaf directory receives the outputs of one completed run: `metrics.json`,
`confusion_matrix.png`, `roc_curve.png`, and model checkpoints. All of these are
gitignored. The **only** results artifact tracked in git is `results/summary.csv` — one
row appended per completed run — which is the direct source for every comparison table
and figure in the paper:

```
dataset, model,  neuron, encoding, accuracy, f1,    precision, recall, roc_auc, energy, train_time_s, inference_ms
nslkdd,  snn,    Leaky,  rate,     0.982,    0.978, 0.981,     0.975,  0.997,   4.2e-4, 312.4,        1.7
nslkdd,  snn,    Alpha,  rate,     0.985,    0.981, 0.984,     0.979,  0.998,   4.5e-4, 318.1,        1.8
nslkdd,  snn,    Leaky,  delta,    0.979,    0.974, 0.977,     0.971,  0.996,   3.8e-4, 309.2,        1.6
nslkdd,  dnn,    -,      -,        0.976,    0.971, 0.974,     0.968,  0.995,   -,      145.3,        0.4
```

---

## `src/` — Shared Library

```
src/
├── data/             one module per dataset: loading, cleaning, normalization
├── encoding/         spike encoding strategies (rate, temporal, population)
├── neurons/          neuron model definitions (LIF, IF, AdLIF, ...)
├── models/           SNN and hybrid model architectures
│   └── baselines/    non-SNN baselines: DNN, RNN, KAN
├── attacks/          adversarial and membership-inference attack implementations
└── metrics.py        unified metrics computation shared across all experiments
```

`src/` is organized by concern rather than by dataset because the neuron models,
encoders, and architectures are identical regardless of which dataset is used — only the
data loading differs. One implementation of each component means a fix or improvement
applies to every experiment automatically. It is the reusable foundation that the
dataset-centric experiment layer (`configs/`, `experiments/`, `results/`) is built on.

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
- Source: [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) · [Kaggle](https://www.kaggle.com/datasets/cicdataset/cicids2017)
- Local path: `dataset/cicids2017/`
- Format: CSV (one file per day of the week)

### 6. CSE-CIC-IDS2018
CSE–CIC joint dataset extending CIC-IDS2017 with additional attack scenarios.
- Source: [UNB CIC](https://www.unb.ca/cic/datasets/ids-2018.html)
- Local path: `dataset/cse-cic-ids2018/`
- Format: CSV (one file per date)
