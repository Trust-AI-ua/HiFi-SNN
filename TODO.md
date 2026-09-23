# HiSNN — Remaining Tasks

The **experimental study is complete** — 4-way comparison (**Vanilla SNN, Tree-to-Spike,
DNN-Tree, TASNN**) across 11 datasets × 5 seeds, all artifacts in `results/Journal_results/`
(regenerate with `scripts/benchmark_4way.py`). What remains is mostly the **paper writeup**
plus optional enhancements. (Completed items removed — see git history / `README.md` /
`SOTA_DATASETS.md`.)

Status: ⬜ not started · 🟡 in progress.

> **Recently completed (artifacts in `results/Journal_results/`):**
> - **Encoder ablation** — `tree|quantile|nokd|uniform|all` + GBDT-alone reference. The
>   all-channel encoder costs ~2× tree-guided for up to +2.5 F1 (budget = the cost lever);
>   uniform ≈ tree shows importance *weighting* alone is not the mechanism. The `quantile`
>   (teacher's ruler at matched channels) and `nokd` (teacher's posterior) arms are the
>   controlled tests of whether the teacher is load-bearing — read conclusions off the
>   regenerated table, do not restate the old claim.
> - **DNN-Tree int8 (matched-precision accuracy)** — Tree-to-Spike 79.8 ≈ DNN-Tree 79.9 (tied).
> - **Matched-precision energy (32b + int8) + input→layer-1 SOP correction** — honest ratio:
>   Tree-to-Spike uses **3.16× (32b) / 4.74× (int8)** less compute energy than the matched
>   DNN-Tree on the IoT four (pooled over all eleven: 3.67× / 5.51×). [updated 2026-09-23]
>   (Together with the memory-aware total-energy model, this addresses the CVPR-2024
>   "are SNNs really efficient?" critique.)
> - **Statistical significance** — Friedman–Nemenyi CD diagrams (Tree-to-Spike tied with DNN-Tree,
>   both beat TASNN); mean-rank plots.
> - **Learned-representation viz** — per-dataset UMAP cluster panels + silhouette table.

---

## 1. Writeup / related-work positioning ⬜

The main remaining work — turn the results into the paper.
- ⬜ **Differentiate IDS-SNNDT** (Zarzoor et al., IJECE 2023) — DT selects *samples*; ours is a
  GBDT-importance-guided *encoder* + KD. Foreground "no prior GBDT-guided spike-encoding + KD
  for tabular IDS." Cite the SNN-NIDS peers, KD-for-SNN precedents, and non-SNN SOTA
  (tiered list in this file's git history).
- ⬜ **Methodology paragraph:** state the official-split protocol; ~76% NSL-KDD / UNSW numbers
  are honest (distribution shift), with the RF-on-official-split control (below) as evidence.
- ⬜ **Feature-set incomparability:** when quoting a competitor's published number, state whether
  it used the same features/split (NF-v2 vs raw CSV); never claim parity across different inputs.
- ⬜ **Cross-verify competitor numbers** before any comparison table — pull each paper's own
  reported official-split figure rather than relying on secondary summaries.

## 2. Optional enhancements ⬜

Nice-to-haves; none block the core claims.
- ⬜ **Lift macro-F1 on the hard splits** (NSL-KDD R2L/U2R, UNSW Worms/Shellcode/Analysis): the
  SNN runs are **unweighted** — try `class_weighted: true` (flag exists) or focal loss to raise
  minority recall (accuracy stays ~flat; macro-F1 should rise).
- ⬜ **Bit-budget energy accounting** — report a `⌈log₂(T+1)⌉`-bit equivalent alongside SOPs as an
  extra reviewer-proofing angle (the matched-precision int8 + memory-aware model already cover
  the main critique).
- ⬜ **Temporal (not random) splits** for datasets with a time order — an honesty upgrade.
- ⬜ **IID / binary eval mode** for `src/data/nslkdd.py` (`split=official|iid`,
  `task=multiclass|binary`) as a clearly-labelled *secondary* table only.

---

## Reference (kept — supports the writeup tasks)

**Honest-protocol controls live in `SOTA_DATASETS.md` §8.** RandomForest control tables for
**both NSL-KDD and UNSW-NB15** (official / IID / binary protocols) show the ~76% ceiling is a
property of the official-split distribution shift, *not* a model weakness (RF official: NSL-KDD
0.754, UNSW 0.756 — at/below our SNN). Use that as the one-line evidence in Methodology.

**Energy methodology (state precisely in limitations).** The **operation counts are empirical**:
SOPs (spikes × fan-out) are *measured during evaluation* on the real test data, MACs are computed
from the trained model topology, and `mean_exit_step` is measured from early-exit. **Energy in pJ
is an analytical model** — those real counts × published **Horowitz 45 nm** per-operation costs
(MAC 4.6 pJ, AC 0.9 pJ; reported at 32-bit and int8) — the standard SNN-NIDS accounting, **not** a
hardware power measurement. A measured-joule claim would need neuromorphic hardware (Loihi/Akida)
or rigorous platform modelling. Latency (`inference_ms`) is wall-clock on a *shared* GPU
(indicative). **Out of scope:** PE utilization, hardware dataflow memory counts, GPU dynamic power,
spike-rate efficiency η = MI/N_spikes.
