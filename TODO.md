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

## 1. Energy / efficiency metrics — the "four green additions"  ✅ (superseded/extended)

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
Horowitz-energy-vs-accuracy Pareto figure.

### Status — realized, then superseded by the rigorous DNN comparison
- ✅ **`scripts/efficiency_analysis.py` implemented** (Horowitz E, single-pass iso-ANN MACs,
  α gain, ELP+FPS, Pareto) and **SNN SynOps backfilled** (see item 2).
- ✅ **The energy story is now carried by the (more rigorous) DNN-vs-SNN comparison**, which
  replaced the original analytic plan:
  - `DNN-Tree` — a **measured** iso-architecture dense-ANN twin of Tree-to-Spike (identical
    params), so the MAC reference is a real trained network, not an analytic formula
    (`src/models/dnn_student.py`, `src/training_dnn.py`, `experiments/DNN-Tree/`).
  - `scripts/benchmark_3way.py` — Vanilla SNN vs Tree-to-Spike vs DNN across all metrics +
    the **memory-aware total-energy** headline + accuracy-vs-total-energy Pareto + radar.
  - `scripts/energy_memory_model.py` — compute **+** memory-access total energy, swept over
    memory cost and batch size (the reviewer-proof accounting; shows early-exit is what
    keeps the SNN ahead).
  - `scripts/latency_elp.py` — batch-1 CPU latency + ELP re-measure (`results/latency_elp.{md,csv}`).
- ❎ **Dropped as moot:** unifying SynOps + analytic iso-MAC formulas for **S-xLSTM /
  Spiking-Griffin / Spiking-SSM** — those three are now **archived** (Tree-to-Spike is the
  lead), and Tree-to-Spike's energy reference is the measured `DNN-Tree`, not an analytic MAC
  count. (Minor residual, low priority: Tree-to-Spike's `synops` omits the input→layer-1
  term; add it for a fully-consistent SOP count if a reviewer asks.)

---

## 2. SNN early-exit + SynOps backfill, incl. CSE-CIC-IDS2018  🟡

The SNN ablation originally recorded energy/sparsity/params/timing but not
**early-exit** or **SynOps**. `scripts/backfill_snn_metrics.py` replays each saved
`checkpoint.pt` (eval only, no retraining), augments `results.json` with the
`early_exit` block + `spike_stats.synops_per_sample`, and writes the companion
`results/summary_snn_efficiency.csv`.

- ✅ **Backfill complete for nslkdd, kddcup99, ctu13, cicids2017, iscxids2012** (135/135
  each have `early_exit` + `synops_per_sample`). iscxids2012 was backfilled Jun 17 (needed a
  temporary `results/iscxids2012` symlink because the dataset now lives under
  `SNN_Model_compare_Neuron_vs_Encoding/` — see the deferred repath note below).
- 🟡 **CSE-CIC-IDS2018: 90/122 backfilled.** Its full 27-combo seed-46 ablation is still being
  swept (`hisnn-sweep-cse-cic-ids2018`), so new combos keep appearing. Re-run
  `backfill_snn_metrics.py --only cse-cic-ids2018` once that sweep finishes to cover the late
  combos. (LeakyParallel/latency — the cell the 3-way benchmark uses — is already done.)
- ⬜ **Fold the backfilled metrics into the main pipeline once the cse-cic sweep is done**
  (deferred now to avoid a schema race with the live sweep writing `summary.csv`):
  - Wire early-exit + total-SOP SynOps into `src/training.py`'s eval so **future** SNN
    runs capture them natively (mirror the hybrids' `_eval` early-exit path).
  - Extend `src/training.py`'s `SUMMARY_HEADER` with `early_exit_accuracy`,
    `early_exit_f1_macro`, `mean_exit_step`, `uncertain_frac`, `synops_per_sample`,
    then regenerate `results/summary.csv` from the augmented `results.json` files.
- ⬜ **Deferred directory repath (user-parked).** The vanilla SNN results moved to
  `results/SNN_Model_compare_Neuron_vs_Encoding/<ds>/`, but **cse-cic-ids2018 stays at
  `results/cse-cic-ids2018/`** (the live sweep writes there). The path-dependent tooling
  (`backfill_snn_metrics.py`, `efficiency_analysis.py`, `make_plots.py`, `conference_analysis.py`,
  `run_sweep.py`) and the SNN `run.py` `parents[]` still assume the old `experiments/<ds>/`
  layout. When the cse-cic sweep finishes: consolidate cse-cic results under
  `SNN_Model_compare_Neuron_vs_Encoding/`, fix the tooling globs + `run.py` repo-root, and
  repoint the `experiments/Tree-to-Spike/README.md` links to the archived siblings.

---

## 3. Extend the hybrid SNN models to all datasets  ✅

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
- ✅ **Multi-seed sweeps complete** — all four hybrids ran on all 6 datasets over seeds
  42–46 (`results/<Model>/<dataset>/seed_<N>/`). **Tree-to-Spike is the lead model** (best
  accuracy + cleanest energy story); **S-xLSTM, Spiking-Griffin, Spiking-SSM are now archived
  under `archive/experiments/` and `archive/results/`.** The Tree-to-Spike/cse-cic-ids2018
  XGBoost teacher class-remap bug (datasets missing a class → `[0,1,3,4]`) was fixed in
  `src/training_treespike.py`.

---

## 4. Add additional NIDS datasets for more robust evaluation  ⬜

Broaden beyond the current six datasets, using the modern NetFlow-standardised
benchmarks referenced in `README.md` / the reference designs.

- ⬜ **Datasets the SOTA hybrid SNN-NIDS papers actually use** (add these so we can do a
  *same-dataset* head-to-head — see item 6): **CICIoT2023, Edge-IIoTset, CICIoMT2024,
  TON-IoT**, and the raw CIC-IDS / UNSW-NB15 CSVs. TASNN (our closest competitor) reports
  on CICIoT2023 / Edge-IIoTset / CICIoMT2024; the recurrent+spiking hybrid on
  CICIoT2023 / TON-IoT.
- ⬜ **Feature-set incomparability gotcha (reviewers are now alert to this).** The NF-v2
  datasets use a *different, fixed* NetFlow feature set than the raw CIC-IDS / UNSW-NB15
  CSVs the baselines use, so we **cannot quote a baseline's published accuracy and claim
  parity** — the inputs differ. Resolve by ONE of: (a) re-run the baselines on the same
  NF-v2 datasets; (b) also report on the datasets they use; or (c) state the feature-set
  difference very explicitly. Pick (a) or (b) for the head-to-head; (c) is the minimum.
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

## 6. SOTA comparison vs prior hybrid SNN-NIDS + bulletproof energy methodology  ⬜

**Framing (the defensible claim):** *comparable accuracy at lower energy than a matched
DNN* — NOT "higher accuracy" (sidesteps the GBDT-teacher accuracy ceiling). This is what
the SNN-NIDS field rewards, but the subfield has turned **skeptical of SNN energy claims**,
so the energy comparison must survive a hostile reviewer. We already have the iso-DNN twin
(`DNN-Tree`), the memory-aware total-energy model (`scripts/energy_memory_model.py`), and a
batch-1 latency/ELP re-measure — this item lists what is still needed to make it bulletproof
and who to benchmark against.

### 6a. Hybrid SNN-NIDS baseline set (2024-2026) to position against
- ⬜ **TASNN** — transformer-augmented SNN for IDS (Sci Rep 2026); energy + rare-class focus
  on **CICIoT2023, Edge-IIoTset, CICIoMT2024**. **Primary head-to-head** (positioning
  overlaps ours most).
- ⬜ **Hybrid spiking-conv + ELM** (Li, Sci Rep 2026) — itself benchmarks vs **Mehrabi &
  Van Schaik 2024** (neuromorphic IDS, industrial) and **spiking residual ShuffleNet for
  IoT** (Jacob & Sultana, IEEE Access 2025) — useful secondary baselines.
- ⬜ **Hybrid recurrent + spiking** (Frontiers in AI 2025) — CICIoT2023, TON-IoT.
- ⬜ **Feature-optimised hybrid SNN for IoT IDS** (J. AI & Technology 2026).
- ⬜ **Convolutional-SNN NIDS** (Wang et al., Sci Rep 2024); **IDS-SNNDT** (decision-tree +
  SNN — our **nearest conceptual neighbour**, cite/differentiate explicitly); **SURFS**
  (hierarchical federated SNN IDS); spiking energy-efficient NIDS for medical-IoT.
- ⬜ Build a baseline comparison table keyed to these papers **and** the dataset they each
  use (resolve the feature-set incomparability per item 4).

### 6b. Energy claim — make it survive the skeptics (this is where the paper lives or dies)
A reviewer will almost certainly cite the recent "SNNs aren't as efficient as claimed" wave:
- **"Are Conventional SNNs Really Efficient? A Perspective from Network Quantization"
  (CVPR 2024)** — SNN timesteps ≈ quantized activation bit-widths; **SynOps should be
  replaced by a "bit budget"** accounting. A **SynOps-only proxy (what our first README
  used) is exactly what they say inflates SNN efficiency.**
- **STEP benchmark** — quantized ANNs are more efficient than assumed (bit-serial execution
  turns MACs into AC sequences exploiting bit sparsity); **memory-access energy is usually
  ignored** even though SNNs maintain/update high-precision membrane potentials every step.
  Once memory is counted, a spiking transformer's *total* energy came out **higher** than the
  quantized equivalent. Reference per-op numbers: **E_MAC ≈ 4.6 pJ, E_AC ≈ 0.9 pJ,
  E_mem ≈ 3.1 pJ** (STEP, 45nm).
- **Analytical fairness mapping** — a rate-encoded SNN with T timesteps ↔ a quantized net
  with **ceil(log2(T+1)) bits** (matched representational capacity); ignoring data
  movement / memory access yields misleading efficiency conclusions.
- **Acute risk for us:** our student is a *tiny 3-layer MLP*, not a deep conv net, so we do
  **not** have the depth-driven sparsity that usually favours SNNs. The naive
  `E_SNN = FLOPs × E_AC × T` (load-bearing ×T, ignores memory) is precisely the inflated
  estimate to avoid.

**To make the claim bulletproof (concrete, deferred):**
- ⬜ **Quantized-AND-pruned DNN baseline of matched capacity** — not FP32, not a strawman.
  Our int8 student is the precedent; apply the *same* compression to the `DNN-Tree`
  baseline so both get sparsity-awareness.
- ⬜ **Fair energy model that counts memory access** (already started in
  `energy_memory_model.py`); also report a **bit-budget** accounting alongside SynOps so
  reviewers see we know the gap. Report **both ways**.
- ⬜ **Hold accuracy fixed / energy-accuracy Pareto** — since the claim is "comparable, not
  superior", define comparability **statistically**: bounded accuracy gap **plus
  overlapping confidence intervals across the 5 seeds**, else a reviewer says we just traded
  accuracy for energy.
- ⬜ **Substrate-fair hardware** — if we ever measure on Loihi/Akida, do **not** compare
  SNN-on-Loihi vs DNN-on-GPU (the canonical unfair comparison); run both on comparable
  hardware or use one consistent analytical model for both.

### 6c. Foreground the tree-guided encoder as the *mechanism* behind the energy number
This is our genuine asset — convert it from a novelty into the measured source of savings:
- ⬜ Report **actual per-layer firing rates** and **post-pruning channel counts** explicitly
  (importance-based pruning shrinks input channels; TTFS fires ~1 spike/channel → very low
  firing rate by construction).
- ⬜ **Ablation**: tree-guided encoder vs **uniform-bin** and **all-channel** variants —
  show the alternatives fire more and cost more (the make-or-break ablation already noted
  in the cross-cutting section). This makes "we used an SNN" into "the encoder is *why* it
  is efficient".

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
