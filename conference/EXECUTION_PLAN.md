# Execution plan to 15 August 2026 — SNN-only

Written 2026-08-12, revised the same day to remove the conventional-baseline
tier. Three days, one idle H200, 16 CPU cores.

## Scope, fixed

This paper is a **controlled intra-SNN design-space study implemented entirely
in snnTorch**. The research question is about neuron dynamics and spike-encoding
choices, not about whether spiking networks beat tabular classifiers.

**Excluded from every conference experiment**, with zero runs budgeted:
XGBoost, Random Forest, Logistic Regression, conventional DNN/MLP baselines,
Tree-to-Spike, DNN-Tree, TASNN, GBDT teachers, knowledge distillation,
tree-derived encodings, teacher–student comparisons.

The external review did recommend LR/RF/XGBoost/MLP baselines, including an
architecture-matched MLP control. That recommendation is **declined for this
venue**, deliberately and on the record: it answers a different question than
the one this paper asks. The manuscript should state its scope in the
introduction so the omission reads as a boundary rather than a gap. The
efficiency analysis stays inside the spiking design space too — no SNN-vs-ANN
MAC comparison (§7).

## The constraint that decides everything

A full clean 27-variant v2 re-sweep costs **198 GPU-hours**, measured from the
v1 sweep's own `train_time_s` column, not estimated:

| Dataset | GPU-h for 9 neurons × 3 encodings × 5 seeds |
| --- | --- |
| CIC-IDS2017 | 123.5 |
| CTU-13 | 41.4 |
| KDDCup99 | 27.5 |
| NSL-KDD | 5.8 |
| **Total** | **198.2** |

At four workers that is ~50 h of wall clock and leaves nothing for analysis or
writing. **The full v2 re-sweep is not feasible by the 15th.** Hence the
two-axis confirmation design (§3) rather than a second full grid.

---

## Already done (2026-08-12, no GPU)

**Reorganisation.** `conference/` is self-contained; the only external reference
is the dataset root, via `src/paths.py`. The tree contains no non-SNN model code
— `src/models/` exports `SNN` and nothing else.

**Pre-registration frozen** (`PREREGISTRATION.md`) before any statistic was
computed.

**Gate 1 passed** on all eight v2 protocols (`results/analysis/split_audit.csv`):

| Dataset | v2 protocol | Group overlap | Duplicate test rows |
| --- | --- | --- | --- |
| NSL-KDD | official split | 0 | 2.68% |
| KDDCup99 | official split | 0 | **57.53%** |
| CIC-IDS2017 | within-capture order-disjoint | n/a | 10.86% |
| CIC-IDS2017 | day-disjoint, binary | 0 days | 4.91% |
| CTU-13 | scenario-disjoint × 4 folds | 0 scenarios | ~0.00% |

Nothing is fitted on test in any v2 protocol.

**Pre-registered analysis of the existing 540-run screening sweep**
(`results/analysis/v1/`):

- `LeakyParallel/latency` is **rank 1 of 27** by mean rank (4.03).
- Friedman χ² = 263.9, p = 3.2e−41; Nemenyi CD = 9.28.
- **14 of 27 variants lie within one critical difference of the leader** →
  "top-ranked", never "statistically best".
- LP/latency vs LP/rate: 16–4 in blocks, median +0.70 pp, Holm p = 0.0084,
  bootstrap +3.79 pp [+0.11, +10.19]. Carried by CIC-IDS2017 (+13.3 pp), near
  zero elsewhere — report the variation, not a universal claim.
- LP/latency vs Leaky/latency: n.s. (p = 0.33). Co-leaders.
- Encoding level: latency 1.25 < rate 1.90 < delta 2.85; latency-vs-rate
  Wilcoxon significant (Holm p = 0.027) but the dataset-level bootstrap CI
  [−0.18, +11.30] includes zero.

One clarification worth putting in the rebuttal: the review's observation that
"rate slightly wins on KDDCup99" is true **of accuracy** (−0.032 pp for
latency). On the pre-registered primary metric, macro-F1, latency wins there by
**+0.379 pp**. Fixing the primary metric in advance is what makes that answer
credible rather than convenient.

---

## Prerequisite: trainer instrumentation (~half a day, no GPU)

Tier B is only worth running once the trainer persists what the zero-retraining
analyses need. `src/models/snn.py` already carries `_output_spike_train` and
`predict_early_exit`, which return the per-timestep output-spike stack `[T,B,C]`
plus hidden spike totals, so the model side is done. Missing, all in
`src/training.py` and `src/metrics.py`:

- per-sample test scores and predictions written to disk;
- prefix output-spike counts for every `t = 1…T`;
- spike decomposition into input / hidden / output rather than one total;
- `SOP = Σ_{ℓ,t} s_ℓ(t) · fanout_ℓ`, and AUPRC alongside ROC-AUC;
- the split manifest hash and protocol name in every result record.

Do this first. Re-running 60 jobs because the scores were not saved costs more
than writing the instrumentation carefully once.

---

## Tier B — encoding axis, with LeakyParallel fixed

`LeakyParallel × {latency, rate, delta} × 5 seeds`, under v2, saving everything
above.

| Protocol set | Runs | GPU-h |
| --- | ---: | ---: |
| `nslkdd_v2`, `kddcup99_v2`, `cicids2017_v2` | 45 | **10.7** |
| `ctu13_v2_f{0,1}` (the extreme folds: 99.9 and 77.0) | 30 | 6.6 |
| `ctu13_v2_f{2,3}` | 30 | 6.6 |
| `cicids2017_daydisjoint` (binary transfer) | 15 | 8.4 |

Core Tier B = the first two rows: **75 runs, 17.3 GPU-h, ~5 h wall at 4
workers.** This one run set unlocks, with **no further training**:

- early-readout / anytime inference: F1(t), MCC(t), DR(t), T95, T99 (§13 of the
  revised brief);
- fixed-FAR security: validation-selected thresholds at 1% and 0.1%, frozen,
  then DR and realised FAR on test;
- per-class precision/recall/F1 with support, R2L and U2R especially;
- NSL-KDD known-vs-unseen attack DR;
- KDDCup99 duplicate-free test sensitivity;
- SOP and spike decomposition, LP/latency vs LP/rate vs LP/delta.

Six analyses, one run set. This is the highest return in the plan and should
start first.

## Tier C — neuron axis, with latency fixed

`9 neuron families × latency × 5 seeds` under v2. Purpose is narrow: does
`LeakyParallel` stay highly ranked when the protocol tightens? Not a second full
design-space ranking.

| Protocol | Runs | GPU-h | Priority |
| --- | ---: | ---: | --- |
| `nslkdd_v2` + `kddcup99_v2` | 90 | **11.0** | C1 — run |
| `ctu13_v2_f0` + `ctu13_v2_f1` | 90 | 27.4 | C2 — run if B lands clean |
| `cicids2017_v2` | 45 | 40.6 | C3 — droppable, decide Aug 14 |
| `ctu13_v2_f2` + `ctu13_v2_f3` | 90 | 27.4 | C4 — stretch only |

**Do not queue C3 and C4 together.** B core + C1 + C2 is 55.7 GPU-h (~14 h wall
at 4 workers), which fits with slack for analysis and writing. Adding C3 takes
it to 96 GPU-h (~24 h wall) and consumes the buffer.

Report Tier C per-dataset as mean macro-F1 ± **seed** SD, plus mean ranks over
the nine neurons.

## CTU-13 — already answered, zero new runs

The journal line's 420-run matrix included the vanilla `LeakyParallel/latency`
arm; those 60 runs are ported to
`results/v1_submitted/ctu13_strict_vanilla.csv`:

| Condition | macro-F1 |
| --- | --- |
| whole-scenario aggregates + random split (**the submitted protocol**) | 100.00 ± 0.00 |
| causal aggregates + random split | 99.69 ± 0.23 |
| causal aggregates + **scenario-disjoint** | **92.01 ± 9.72**, folds 77.0 / 91.2 / 99.9 / 99.9 |

**The 92.01 result becomes the primary CTU-13 number.** The 100.00 is retained
only as a protocol comparison, labelled as such, with the explanation that the
random flow split let scenario- and host-related information cross the
evaluation boundary. Never quote 92.01 without the fold range beside it.

One wording constraint carried over from the fold audit: the scenario protocol
is **not host-disjoint** — 76.2% of fold-1 test hosts also appear in training
(recorded in each CTU-13 manifest). The paper must not describe it as
host-disjoint.

## Optional, in priority order if time remains

| Experiment | Runs | GPU-h | Note |
| --- | ---: | ---: | --- |
| Preprocessing sensitivity: LP × {latency, rate} × {MinMax, quantile, clipped-z} × 2 datasets × 5 seeds | 60 | **4.5** | Cheap and targeted at the latency confound. Best value of the optional tier. |
| T-sensitivity: LP × 3 enc × T ∈ {5,10,50} | 180 | ~36 | Only if early-readout lands well; it answers "least temporal budget for best quality". |
| Threshold probe: θ ∈ {0.1,0.25,0.5,1.0}, gated families only | ~48 | ~19 | Deferred by default. |
| Full 432-run threshold grid | 432 | ~170 | **Cut.** |
| Temporal windows for a fair delta evaluation | — | large | **Cut.** State the mismatch instead. |

## Order and gates

```
instrumentation → Tier B core → analyses (6, no GPU) → C1 → C2 → [C3 | preprocessing]
```

- **Gate 2** — after the first Tier B seed: no NaNs, no spike quiescence, no
  majority-class collapse, and the saved prefix at `t = T` must reproduce the
  terminal prediction exactly. Not used to select anything.
- **Gate 3** — after C1, run `rank_analysis.py --tag v2_latency` over the nine
  neurons. If `LeakyParallel` is no longer highly ranked, do not spend C2/C3
  trying to restore it; the pre-registration already says what to write.
- **Gate 4** — after the early-readout analysis. If latency reaches near-terminal
  quality at a materially smaller T than rate, that becomes a headline finding
  and the encoding claim no longer has to rest on terminal macro-F1 alone.

## The claim this plan supports

Not "92.11% average accuracy". Something closer to:

> Across a controlled 27-configuration snnTorch design-space evaluation over 20
> paired dataset × seed blocks, `LeakyParallel` with latency encoding attained
> the lowest mean macro-F1 rank, with 14 configurations inside one critical
> difference. Focused leakage-resistant experiments confirmed the selection
> along both design axes — encoding with the neuron fixed, neuron with the
> encoding fixed — and temporal readout analysis characterised how spike
> encoding governs the rate at which discriminative evidence accumulates. Under
> scenario-disjoint evaluation with causal host features, CTU-13 falls from a
> perfect 100.00 to 92.01 with a 77–100 fold range, and we report the latter.

**Top-ranked**, not *statistically best* and not *universally superior*. That
wording is frozen regardless of how the remaining experiments turn out.
