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

## 3. Extend the hybrid SNN models to all datasets  🟡

Scope: kddcup99, **nslkdd, cicids2017, ctu13, iscxids2012** at full data, **plus
cse-cic-ids2018 capped at `subsample_fraction: 0.5`** (the vanilla SNN sweep
`hisnn-sweep-cse-cic-ids2018` has been running since Jun 4, so the hybrids use only half
to limit load/contention). iscxids2012 uses `subsample_fraction: 1.0` to match the SNN
ablation. The orchestrators (`scripts/run_<model>.py`) auto-discover datasets from the
`experiments/<Model>/<dataset>/` dirs, so no orchestrator changes were needed.

**Bugfix (latency encoder):** S-xLSTM and Spiking-Griffin called `spikegen.latency`
directly without clamping; MinMax `fit_transform` emits a tiny float32 overshoot
(1.0000001) on ctu13's train set, which `spikegen.latency` rejects (it needs strict
[0, 1]). Both `_encode` methods now `x.clamp(0.0, 1.0)` first, matching the shared SNN
encoders in `src/encoding/`. Spiking-SSM (learned front-end) and Tree-to-Spike (own
encoder) were unaffected.

- ✅ **Experiment dirs generated** for all four hybrids on nslkdd/cicids2017/ctu13/
  iscxids2012/cse-cic-ids2018 (`experiments/<Model>/<dataset>/{config.yaml, run.py}`);
  kddcup99 already existed. cicids2017/ctu13/iscxids2012 carry `subsample_fraction: 1.0`,
  cse-cic carries `0.5` (nslkdd omits it).
- ✅ **Spiking-SSM sequence loaders added** in `src/data/sequence_windows.py`
  (`SEQ_DATASET_LOADERS` now registers kddcup99/nslkdd/cicids2017/ctu13):
  - **nslkdd** — consecutive-record windows over the official KDDTrain+/KDDTest+ files
    (KDDTest-21 → `test21_loader`), same preprocessing as the tabular loader.
  - **cicids2017** — consecutive-flow windows in file (Mon..Fri) order (no IP/time
    columns exist); windows split stratified 70/30 + 90/10, MinMax fit on train windows.
  - **cse-cic-ids2018** — same record-order windowing as cicids2017 (Timestamp dropped,
    no Source IP); reuses the existing 1.1 GB `_preprocessed_cache.npz` (no raw re-parse,
    so no contention with the live sweep). Loader default `subsample_fraction=0.5`.
    Memory note: the full cache is ~5 GB (16.2 M flows x 82 feats); building all windows
    is a ~10 GB transient peak before the 0.5 subsample is applied.
  - **iscxids2012** — same record-order windowing (source/destination IP dropped); binary
    (normal/attack); reuses the existing ~1 GB `_preprocessed_cache.npz`.
  - **ctu13** — genuine **per-host temporal windows**: L consecutive flows per
    (scenario, SrcAddr) from the raw `.binetflow`, same host-context features as the
    tabular CTU-13 loader. Cached separately as `dataset/ctu13/_seq_cache.npz`.
  - `src/training_ssm.py` now forwards `subsample_fraction`/`test_fraction` from cfg
    when present (no-op for the official-split datasets).
- ✅ **Smoke-tested** (1 epoch): Griffin/{nslkdd,ctu13}, Tree-to-Spike/nslkdd,
  SSM/{nslkdd,cicids2017(sub),ctu13(sub, per-host)}, S-xLSTM/{ctu13,iscxids2012(sub)},
  SSM/iscxids2012(sub) all run end-to-end; SSM/cse-cic-ids2018 + iscxids2012 loaders
  validated against their existing caches (read-only).
- ⬜ **Run the multi-seed sweeps** (deferred to the user — heavy):
  `scripts/run_<model>.py <dataset> --seeds 42 43 44 45 46`; each writes to its own
  `results/<Model>/<dataset>/seed_<N>/` tree. **Tree-to-Spike caveat:** retrains an
  XGBoost teacher per run — budget extra wall-clock on CIC-IDS2017 (~1.78 M train rows).
  **First CTU-13 SSM run** re-reads the 2.6 GB raw `.binetflow` to build `_seq_cache.npz`
  (~5-10 min, one-time).

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

## 5. NSL-KDD evaluation protocol — why we cap at ~76%, and optional comparison modes  ⬜

**Context / question raised:** our NSL-KDD accuracy (~0.75-0.77, macro-F1 ~0.58-0.60)
looks low next to Kaggle/blog notebooks that report ~99% (e.g. the eneskosar19
"Intrusion Detection System [NSL-KDD]" notebook). This section records *why* that gap
exists and what would be needed to report comparable numbers, so we can implement it
later if a reviewer asks for a head-to-head.

### Why our methodology never surpasses ~76% accuracy (and that is correct)

Our loader (`src/data/nslkdd.py`) trains on **KDDTrain+** and tests on the **official
KDDTest+** file. This is the *honest, hard* protocol the NSL-KDD authors designed:

1. **Train/test distribution shift by design.** KDDTest+ deliberately contains **17
   attack signatures that never appear in KDDTrain+** and a re-balanced class mix, so a
   model cannot win by memorising training patterns — it must *generalise* to unseen
   attacks. This is the entire reason NSL-KDD exists (it fixed KDDCup99's redundancy).
2. **The ceiling is a property of the split, not the model.** We verified this directly:
   the *same* features + a tuned `RandomForestClassifier(n_estimators=200)` give

   | Protocol | Accuracy | Macro-F1 |
   |---|---|---|
   | train KDDTrain+ -> test **official KDDTest+** (5-class) | **0.754** | 0.493 |
   | random 80/20 split of KDDTrain+ (**IID**, what 99% notebooks do) | **0.999** | 0.960 |
   | **binary** (normal vs attack) on official KDDTest+ | 0.777 | 0.776 |

   So a RandomForest scores **0.754** on our protocol — *below* our SNN (0.769) and
   S-xLSTM (~0.74-0.78). Model choice barely moves the headline; the ~75-80% ceiling is
   the distribution shift, and it caps every classifier (RF, XGBoost, DNN, SNN alike).
3. **Two minority classes do the damage** (per-class F1 on KDDTest+, SNN Synaptic/latency):
   `normal` 0.79, `dos` 0.89, `probe` 0.70, but **`r2l` 0.26 (recall 0.15)** and
   **`u2r` 0.34 (recall 0.22)**. KDDTest+ is dominated by *novel* R2L/U2R attacks
   (r2l support 2885, u2r 67 with ~1 training example), so recall on them is intrinsically
   low. This is why macro-F1 (~0.60) sits far below weighted-F1 (~0.74).
4. **KDDTest-21 is even harder** (acc ~0.56): it keeps only the records that *all* 21
   baseline learners misclassified, i.e. the hardest tail.

### What the ~99% papers/notebooks actually do (none of these are wrong, just different)

- **IID random split** of KDDTrain+ (or concatenate KDDTrain+ + KDDTest+ then split) ->
  train and test are the same distribution -> ~99% (row 2 above). This measures
  memorisation, not intrusion-detection generalisation. **This is "Suggested Step 1".**
- **Binary** task (normal vs attack) instead of 5-class -> easier boundary (row 3, ~0.78
  honest; ~0.99 if also IID-split).
- **Preprocessing leakage:** fitting the scaler / one-hot / feature selector on the full
  dataset (train+test) before splitting.
- Honest KDDTest+ numbers in the literature for strong models land at **~76-82%
  (multiclass)** / ~80-83% (binary) — exactly our range (see thinline72/nsl-kdd, and
  arXiv:1904.01352, arXiv:2512.03200).

### Deferred implementation options (only if we need comparable headline numbers)

- ⬜ **Add an opt-in IID / binary eval mode** to `src/data/nslkdd.py` (e.g.
  `split="official" | "iid"`, `task="multiclass" | "binary"`), reported as a clearly
  labelled *secondary* table — never replacing the official KDDTest+ headline, which is
  what keeps our cross-dataset story scientifically honest.
- ⬜ **Legitimately lift macro-F1 on the official split** (the real, defensible
  improvement): our SNN runs are **unweighted** — try `class_weighted: true` (flag already
  in the hybrids), focal loss, or train-side oversampling/SMOTE to raise R2L/U2R recall.
  Accuracy will stay ~80% (the shift), but macro-F1 should rise; our 0.60 already beats
  RF's 0.49, so there is headroom.
- ⬜ **Paper framing:** state the protocol explicitly in Methodology and note that
  headline gaps vs Kaggle-style reports are a train/test-split artefact, with the RF
  table above as the one-line evidence.

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
