# HiSNN — Implementation TODO

Tracking remaining engineering work. Items are ordered by priority/dependency.
Status legend: ⬜ not started · 🟡 in progress · ✅ done.

Current model families and where their code lives:

| Family | Model | Training | Experiments | Summary CSV |
|---|---|---|---|---|
| SNN ablation | `src/models/snn.py` | `src/training.py` | `experiments/<ds>/SNN_<neuron>/<enc>/` | `results/summary.csv` |
| S-xLSTM | `src/models/s_xlstm.py` | `src/training_sxlstm.py` | `experiments/S-xLSTM/<ds>/` | `results/summary_sxlstm.csv` |
| Spiking-Griffin | `src/models/spiking_griffin.py` | `src/training_griffin.py` | `experiments/Spiking-Griffin/<ds>/` | `results/summary_griffin.csv` |
| Spiking-SSM | `src/models/spiking_ssm.py` | `src/training_ssm.py` | `experiments/Spiking-SSM/<ds>/` | `results/summary_ssm.csv` |
| Tree-to-Spike | `src/models/tree_to_spike.py` | `src/training_treespike.py` | `experiments/Tree-to-Spike/<ds>/` | `results/summary_treespike.csv` |

---

## 1. Energy / efficiency metrics — the "four green additions"  🟡

Add the analytical efficiency metrics agreed for the paper, computed **post-hoc**
from each run's `results.json` (no retraining). Implemented in
`scripts/efficiency_analysis.py`.

**The four metrics (Horowitz 45 nm CMOS model: MAC = 4.6 pJ, AC = 0.9 pJ):**

1. **Horowitz inference energy** `E_SNN = N_AC × 0.9 pJ`
   (the analog encoding layer, if any, is charged at 4.6 pJ/MAC; our latency/rate/delta
   encoders feed *binary spikes* into the first linear layer, so that layer is AC, not MAC).
2. **Iso-architecture MAC count** `N_MAC = Σ_layers (in × out) × T` — the *same* network
   run densely as a non-spiking ANN, used as the energy reference (`E_ANN = N_MAC × 4.6 pJ`).
3. **Energy-efficiency gain** `α = E_ANN / E_SNN` (SNN vs its own dense-ANN twin).
4. **Energy-Latency Product** `ELP = E_SNN × inference_ms`  and  **FPS** `= 1000 / inference_ms`.

**Output:** `results/efficiency/efficiency_metrics.csv` (per variant) + a
Horowitz-energy-vs-accuracy Pareto figure (IEEE palette, via `src/plotting.py`).

### Prerequisites / sub-tasks
- ⬜ **Unify the `synops_per_sample` (total SOPs) definition across ALL families.**
  `N_AC` for the energy MUST be the *total* synaptic-accumulate count over *all*
  layers so it is comparable to the all-layers iso-ANN MAC count. Current state:
  - ✅ **SNN** — `predict_early_exit` returns total SOPs = `input_spikes×hidden + hidden_spikes×num_classes`
    (SConv2dLSTM layer-1 fan-out approximated by `hidden_size`; documented caveat).
  - 🟡 **Tree-to-Spike** — currently counts the two hidden LIF layers
    (`n_spk1×fanout1 + n_spk2×fanout2`) but **omits the input→layer-1 term**; make it
    consistent with the SNN definition (add the input-spike × fan-out term).
  - ⬜ **S-xLSTM / Spiking-Griffin / Spiking-SSM** — do **not** record `synops_per_sample`
    at all. Add a `synops`/total-SOP computation to each model (mirror the SNN: sum
    `spikes_in × fan_out` over the readout(s) and the recurrence/attention spike layers),
    return it from the eval path, and write it into `results.json` + the summary CSV.
    These hybrids only have KDDCup99 results so far, so this needs a re-run of their
    KDDCup99 seeds (small).
- 🟡 **Per-model iso-ANN MAC formulas.** `efficiency_analysis.py` needs a MAC function
  per architecture (reads dims from `results.json` `config`/`data`):
  - ✅ SNN: `T × (num_features×hidden + hidden×num_classes)`.
  - ⬜ S-xLSTM (mLSTM: Wq/Wk/Wv/Wo + gates), Spiking-Griffin (RG-LRU + windowed attention
    Q/K/V/proj), Spiking-SSM (front-end embed + dt/B/C/A/D projections + heads),
    Tree-to-Spike (encoder out_dim → hidden → hidden → classes). Document gate/conv
    terms as lower bounds where exact counts are awkward.
- ⬜ **Run `efficiency_analysis.py` after the SNN backfill (item 2) completes** so
  `synops_per_sample` is present for the SNN family.

---

## 2. SNN early-exit + SynOps backfill, incl. CSE-CIC-IDS2018  🟡

The SNN ablation originally recorded energy/sparsity/params/timing but not
**early-exit** or **SynOps**. `scripts/backfill_snn_metrics.py` replays each saved
`checkpoint.pt` (eval only, no retraining), augments `results.json` with the
`early_exit` block + `spike_stats.synops_per_sample`, and writes the companion
`results/summary_snn_efficiency.csv`.

- 🟡 **Run the backfill to completion** over all finished SNN runs (nslkdd, kddcup99,
  ctu13, cicids2017, and the **completed** cse-cic-ids2018 seeds). Idempotent/resumable.
- ⬜ **CSE-CIC-IDS2018 is still being swept** (the `hisnn-sweep-cse-cic-ids2018` tmux
  session is mid-run). Re-run the backfill with `--only cse-cic-ids2018` once its sweep
  finishes so the late seeds also get early-exit + SynOps.
- ⬜ **Fold the backfilled metrics into the main pipeline once the cse-cic sweep is done**
  (deferred now to avoid a schema race with the live sweep writing `summary.csv`):
  - Wire early-exit + total-SOP SynOps into `src/training.py`'s eval so **future** SNN
    runs capture them natively (mirror the hybrids' `_eval` early-exit path).
  - Extend `src/training.py`'s `SUMMARY_HEADER` with `early_exit_accuracy`,
    `early_exit_f1_macro`, `mean_exit_step`, `uncertain_frac`, `synops_per_sample`,
    then regenerate `results/summary.csv` from the augmented `results.json` files.

---

## 3. Extend the hybrid SNN models to all datasets  ⬜

The four hybrids currently have **KDDCup99 only**. Extend each to the other
in-scope datasets so they can be compared against the SNN ablation everywhere.

- ⬜ For each hybrid (S-xLSTM, Spiking-Griffin, Spiking-SSM, Tree-to-Spike), add
  `experiments/<Model>/<dataset>/{config.yaml, run.py}` for **nslkdd, cicids2017, ctu13**
  (and cse-cic-ids2018 if in scope). One-directory copy of the KDDCup99 dir; change
  `dataset:` and `results_dir:`, add `subsample_fraction:` for the large datasets.
- ⬜ **Spiking-SSM caveat:** it consumes *sequence windows* via
  `src/data/sequence_windows.py`, which currently only registers `kddcup99`
  (`SEQ_DATASET_LOADERS`). Add consecutive-record window loaders for the other datasets
  (NSL-KDD shares the schema; CIC/CTU need their own windowing over the cached arrays).
  Note CTU-13 *does* have SrcAddr/timestamps (raw `.binetflow`) so genuine per-host
  windows are possible there — see `dataset/ctu13/DOWNLOAD_SOURCES.md`.
- ⬜ **Tree-to-Spike caveat:** retrains an XGBoost teacher per run; budget extra wall-clock
  on the large datasets (CIC-IDS2017 ~1.78 M train rows).
- ⬜ Run multi-seed sweeps (`scripts/run_<model>.py <dataset> --seeds 42 43 44 45 46`)
  and confirm each writes to its own `results/<Model>/<dataset>/seed_<N>/` tree.

---

## 4. Add additional NIDS datasets for more robust evaluation  ⬜

Broaden beyond the current six datasets, using the modern NetFlow-standardised
benchmarks referenced in `README.md` / the reference designs.

- ⬜ **Candidate datasets** (NetFlow-v2 standardised, temporal splits recommended):
  NF-UNSW-NB15, NF-CSE-CIC-IDS2018, NF-ToN-IoT, NF-BoT-IoT, plus UNSW-NB15 itself.
- ⬜ For each: add `src/data/<dataset>.py` implementing the standard loader interface
  (`DatasetBundle`: train/val/test loaders, `num_features`, `num_classes`,
  `class_names`, `class_counts_train`); one-hot categoricals over the train+test union,
  log1p heavy-tailed numerics, MinMax to [0,1], stratified 90/10 train/val. Register in
  `src/data/__init__.py` `DATASET_LOADERS`.
- ⬜ Use **temporal** (not random) train/test splits where the dataset provides a time
  order, to test generalisation honestly (flag this as a methodological upgrade over the
  stratified random splits used for CIC/CTU).
- ⬜ Restructure `configs/experiments/results` trees to `SNN_<neuron>/<encoding>` for each
  new dataset and generate the 27 SNN configs (reuse the generator pattern from the
  existing datasets); add hybrid dirs as needed.
- ⬜ Update the `# Datasets` section of `README.md` with source URLs, local paths, and
  formats for the added datasets (mirror the existing entries; record exact download
  URLs like `dataset/ctu13/DOWNLOAD_SOURCES.md` did for CTU-13).

---

## Cross-cutting notes

- **Reproducibility:** all training seeds numpy/torch/CUDA and set
  `cudnn.deterministic=True`; multi-seed runs live in `seed_<N>/` subdirs and summary
  CSVs dedup on `(model, dataset, seed)`.
- **Energy numbers are proxies** (spike/SOP counts, Horowitz pJ, mean exit step,
  shared-GPU `inference_ms`) — state this in the paper's limitations. A measured-energy
  claim needs neuromorphic hardware (Loihi/Akida) or rigorous platform modelling, and a
  comparison against a quantized/pruned ANN of similar size.
- **Out of scope** (not feasible without extra hardware/instrumentation): PE utilization
  (needs Loihi 2), memory-access dataflow counts (needs a hardware dataflow simulator),
  dynamic power on the shared GPU (too noisy), spike-rate efficiency η = MI/N_spikes
  (mutual-information estimation on high-dim inputs is unstable).
