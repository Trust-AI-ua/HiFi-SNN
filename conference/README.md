# Rethinking Network Anomaly Detection with Spiking Neural Networks

Self-contained code and results for the conference paper: a controlled ablation
pairing **9 snnTorch neuron models × 3 spike encodings = 27 variants**, trained
on four intrusion-detection benchmarks (NSL-KDD, KDDCup99, CIC-IDS2017, CTU-13)
over five seeds.

Everything needed to reproduce the study lives in this directory. The **only**
external dependency is the raw datasets, which are not redistributed here.

## Scope

This is a **controlled intra-SNN design-space study implemented entirely in
snnTorch**. Every model here is a spiking network; the axes of variation are the
neuron model and the spike encoding, and the question is which configuration
gives the strongest quality–efficiency operating point for intrusion detection.

Deliberately out of scope, with zero runs in the plan: XGBoost, Random Forest,
Logistic Regression, conventional DNN/MLP baselines, Tree-to-Spike, DNN-Tree,
TASNN, GBDT teachers, knowledge distillation, and teacher–student comparisons.
The efficiency analysis stays inside the spiking design space as well — there is
no SNN-vs-ANN MAC comparison. `src/models/` exports `SNN` and nothing else, by
design rather than by omission. See `PREREGISTRATION.md` §0.

Those model families do exist in the parent HiSNN repository; they belong to the
journal line of work, which asks a different question.

```
conference/
├── PREREGISTRATION.md     frozen ranking rules — read this before the results
├── src/                   loaders, encoders, neurons, model, trainer, metrics
├── configs/               one config.yaml per (protocol, neuron, encoding)
├── experiments/           matching thin run.py wrappers
├── scripts/               sweep driver + analysis
├── results/               manifests, run artifacts, analysis outputs
└── paper/                 generated tables and figures
```

## Datasets — the one external reference

Point the code at your dataset root, in this order of precedence:

```bash
export HISNN_DATASET_ROOT=/path/to/datasets     # 1. explicit
ln -s /path/to/datasets conference/dataset      # 2. symlink inside this repo
                                                # 3. ../dataset (in-place layout)
```

Expected layout, matching the Kaggle redistributions cited in the paper:

```
<root>/nslkdd/{KDDTrain+.txt, KDDTest+.txt, KDDTest-21.txt}
<root>/kddcup99/{kddcup.data_10_percent_corrected, corrected/corrected}
<root>/cicids2017/*.pcap_ISCX.csv                     (8 per-day capture files)
<root>/ctu13/binetflow/scenario-*-*.binetflow         (13 scenarios)
```

Nothing writes to the dataset root. Derived caches go to `results/cache/`.

## Setup

```bash
pip install -r requirements.txt
python -c "import sys; sys.path.insert(0,'.'); from src.data import DATASET_LOADERS as D; print(len(D))"
```

## Two protocol generations

The registry carries both, and the name tells you which you get. **Never mix
them in one table** — the difference between them is a result, not noise.

| | v1 (`nslkdd`, `ctu13`, …) | v2 (`nslkdd_v2`, `ctu13_v2_f0`, …) |
| --- | --- | --- |
| Categorical vocabulary | union of all splits | training rows only, `<unk>` bucket |
| Scaler | train-only | train-only |
| Partition | random, **keyed on the model seed** for CIC-IDS2017 / CTU-13 | frozen independently of the seed |
| Group structure | ignored | capture (CIC-IDS2017), scenario (CTU-13) |
| CTU-13 host aggregates | whole-scenario | partition-local causal |
| Audit | none | gate refuses to train on a violated claim |

v1 is preserved so the submitted paper's table stays reproducible. v2 is what
the revision reports.

## Reproducing

```bash
# Gate 1 — build every v2 protocol and audit it. Nothing trains until this is clean.
python scripts/split_audit.py

# a sweep (one protocol, 27 variants × 5 seeds, 4 concurrent workers)
python scripts/run_sweep.py nslkdd_v2 --seeds 42 43 44 45 46 --workers 4

# the pre-registered ranking analysis
python scripts/rank_analysis.py --input results/v1_submitted/sweep_540.csv --tag v1
```

Results land in `results/runs/<protocol>/SNN_<Neuron>/<encoding>/seed_<S>/` and
append to `results/runs/summary.csv`. Runs are idempotent — re-running the same
command resumes rather than repeats.

## What is already in `results/`

| Path | Contents |
| --- | --- |
| `v1_submitted/sweep_540.csv` | the 540 runs behind the submitted paper (9×3×4×5) |
| `v1_submitted/runs/` | per-run JSON + checkpoints for `LeakyParallel/latency` on all four datasets |
| `v1_submitted/ctu13_strict_vanilla.csv` | 60 `LeakyParallel/latency` runs under the three CTU-13 leakage conditions |
| `analysis/v1/` | pre-registered ranking output for the v1 sweep |
| `analysis/split_audit.csv` | Gate-1 audit table |
| `manifests/` | one frozen split manifest per v2 protocol |

The CTU-13 champion artifacts under `v1_submitted/runs/ctu13/` are marked
`STALE_DO_NOT_USE` in the parent HiSNN checkout. That marker is correct *there* —
it warns the journal line's leakage experiment not to mistake them for its own
output. They are phase-1 artifacts, which is to say they are exactly the runs
this paper reports, and here they are the primary record.

## Relationship to the parent repository

This tree was split out of `HiSNN/` so it can be handed to reviewers on its own.
`src/` is a **snapshot**, not a symlink: the revision changes preprocessing and
splits, so the two lines are expected to diverge, and freezing the conference
code is the point. Modules ported unchanged (`ctu13_causal.py`,
`ctu13_conditions.py`, `train_only.py`, `stats.py`) are noted as such in their
docstrings.
