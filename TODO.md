# HiSNN — Remaining Tasks

Open work only (completed items removed — see git history / `README.md` / `SOTA_DATASETS.md`
for what's done). **Current study:** 4-way comparison — **Vanilla SNN (LeakyParallel/latency),
Tree-to-Spike, DNN-Tree, TASNN** across 11 datasets × 5 seeds; artifacts in
`results/Journal_results/` (regenerate with `scripts/benchmark_4way.py`).
Headline claim: *comparable detection quality at lower energy than the parameter-matched
dense DNN; competitive with a 548K-param transformer SOTA (TASNN) at ~60× fewer params.*

Status: ⬜ not started · 🟡 in progress.

---

> **Recently completed (artifacts in `results/Journal_results/`):** encoder ablation
> (`tree|uniform|all`; all-channel costs ~2× tree-guided for ≤1.7 F1 → the importance-guided
> channel budget is the verified efficiency mechanism); DNN-Tree int8 (matched precision →
> Tree-to-Spike 79.8 ≈ DNN-Tree 79.9, tied); matched-precision Energy@32b/int8 **with the
> input→layer-1 SOP term counted for both paradigms** → honest ratio: Tree-to-Spike uses
> **3.7× (32b) / 5.6× (int8)** less compute energy than the matched DNN-Tree.

## 1. Bit-budget energy accounting ⬜

Pre-empt the CVPR-2024 *"Are SNNs really efficient?"* critique: a T-step SNN ≈ a
⌈log₂(T+1)⌉-bit quantized net. Report a **bit-budget** number alongside SynOps (report both
ways) so reviewers see the gap is acknowledged. Reference per-op: E_MAC≈4.6pJ, E_AC≈0.9pJ,
E_mem≈3.1pJ (STEP, 45nm). Our student is a tiny 3-layer MLP (no depth-driven sparsity), so
avoid the naive `E = FLOPs × E_AC × T`.

## 2. Lift macro-F1 on the hard official splits (optional, legitimate) ⬜

NSL-KDD (R2L/U2R) and UNSW-NB15 (Worms/Shellcode/Analysis) macro-F1 is capped by minority
recall. SNN runs are **unweighted** — try `class_weighted: true` (flag exists) or focal loss
to raise minority recall. Accuracy stays ~the same (distribution shift), macro-F1 should rise.

## 3. Writeup / related-work positioning ⬜

- ⬜ **Differentiate IDS-SNNDT** (Zarzoor et al., IJECE 2023) — DT selects *samples*; ours is
  a GBDT-importance-guided *encoder* + KD. Foreground "no prior GBDT-guided spike-encoding +
  KD for tabular IDS." Cite Tier-2 SNN-NIDS peers, Tier-4 KD-for-SNN precedents, Tier-5
  non-SNN SOTA (full tiered list in this file's git history).
- ⬜ **Methodology paragraph:** state the official-split protocol; ~76% NSL-KDD / UNSW numbers
  are honest (distribution shift), with the RF-on-official-split control (below) as evidence.
- ⬜ **Feature-set incomparability:** when quoting a competitor's published number, state
  whether it used the same features/split (NF-v2 vs raw CSV); never claim parity across
  different inputs.

## 4. Lower priority / optional ⬜

- ⬜ **Temporal (not random) splits** for datasets with a time order — an honesty upgrade
  over stratified random splits.
- ⬜ **IID / binary eval mode** for `src/data/nslkdd.py` (`split=official|iid`,
  `task=multiclass|binary`) as a clearly-labelled *secondary* table only.

---

## Reference (kept — supports the writeup tasks)

**NSL-KDD ~76% ceiling is correct, not a weakness.** Same features + tuned RandomForest:

| Protocol | Accuracy | Macro-F1 |
|---|---|---|
| train KDDTrain+ → official KDDTest+ (5-class) | 0.754 | 0.493 |
| random 80/20 IID split (what ~99% notebooks do) | 0.999 | 0.960 |
| binary (normal vs attack), official KDDTest+ | 0.777 | 0.776 |

RF scores 0.754 on our protocol — *below* our SNN (0.769); the ceiling is the train/test
distribution shift, not the model. UNSW-NB15 is the same story (Normal 31.94%→44.94%
train→test; the 97–99% literature numbers use an IID merge and/or SMOTE). Full detail in
`SOTA_DATASETS.md` §8.

**Energy numbers are proxies** (SOP/spike counts, Horowitz pJ, mean-exit-step, shared-GPU
`inference_ms`) — state in limitations. A measured-energy claim needs neuromorphic hardware
(Loihi/Akida) or rigorous platform modelling. **Out of scope:** PE utilization, hardware
dataflow memory counts, GPU dynamic power, spike-rate efficiency η = MI/N_spikes.
