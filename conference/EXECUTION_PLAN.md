# Execution plan to 15 August 2026

Written 2026-08-12. Three days, one idle H200, 16 CPU cores. What follows is
what fits, what does not, and why.

## The constraint that decides everything

A full clean 27-variant re-sweep costs **198 GPU-hours**, measured from the v1
sweep's own `train_time_s` column, not estimated:

| Dataset | GPU-hours for 9 neurons × 3 encodings × 5 seeds |
| --- | --- |
| CIC-IDS2017 | 123.5 |
| CTU-13 | 41.4 |
| KDDCup99 | 27.5 |
| NSL-KDD | 5.8 |
| **Total** | **198.2** |

At four concurrent workers that is ~50 hours of wall clock, consuming nearly the
entire window and leaving nothing for analysis, figures or writing. **The full
re-sweep under new splits is not feasible by the 15th.** Everything below is
built around that fact.

The good news is that the four highest-value review items — statistics,
leakage, baselines, early exit — do not all require it.

---

## Already done (2026-08-12, no GPU)

**Reorganisation.** `conference/` is self-contained: `src`, `configs`,
`experiments`, `scripts`, `results`, `paper`. The only external reference is the
dataset root, resolved through `src/paths.py`.

**Pre-registration frozen** (`PREREGISTRATION.md`) before any statistic was
computed. Primary metric macro-F1, unit = (dataset × seed) block, primary
statistic = mean rank, tests specified in advance, wording mapped to outcomes,
and an explicit ban on the pooled cross-dataset SD and the ad-hoc consistency
score.

**Gate 1 passed.** All eight v2 protocols build, and the audit gate refuses to
train on a violated disjointness claim (`results/analysis/split_audit.csv`):

| Dataset | v2 protocol | Group overlap | Duplicate test rows |
| --- | --- | --- | --- |
| NSL-KDD | official split | 0 | 2.68% |
| KDDCup99 | official split | 0 | **57.53%** |
| CIC-IDS2017 | within-capture chronological | n/a | 10.86% |
| CIC-IDS2017 | day-disjoint, binary | 0 days | 4.91% |
| CTU-13 | scenario-disjoint × 4 folds | 0 scenarios | ~0.00% |

Nothing is fitted on test in any v2 protocol. Two findings worth reporting on
their own: **57.5% of the KDDCup99 official test set occurs verbatim in
training**, and the v1 union-of-splits vocabulary turned out to be empirically
harmless — 0 NSL-KDD and 2 KDDCup99 test rows carry an unseen category — so that
criticism can be answered with a measurement instead of a rerun.

**Pre-registered ranking analysis of the existing 540 runs**
(`results/analysis/v1/`). The champion survives, with an important
qualification:

- `LeakyParallel/latency` is **rank 1 of 27** by mean rank (4.03).
- Friedman rejects, χ²=263.9, p=3.2e−41; Nemenyi CD = 9.28.
- **14 of 27 variants sit within one critical difference of the leader.** Per
  the pre-registered wording table this licenses "top-ranked" and
  "co-leading", not "best-performing".
- `LeakyParallel/latency` vs `LeakyParallel/rate`: 16–4 in blocks, median
  +0.70 pp, Holm-adjusted p = 0.0084, bootstrap +3.79 pp [+0.11, +10.19].
  The closest-competitor challenge is answered — but the effect is carried by
  CIC-IDS2017 (+13.3 pp) and is near zero on the other three.
- `LeakyParallel/latency` vs `Leaky/latency`: not significant (p = 0.33). They
  are co-leaders and the paper should say so.
- Encoding level: latency 1.25 < rate 1.90 < delta 2.85. latency vs rate is
  significant by Wilcoxon (Holm p = 0.027) but its **dataset-level bootstrap CI
  [−0.18, +11.30] includes zero** — with four datasets there is not enough
  evidence to call latency generally superior to rate. Say that plainly.

---

## Tier B — cheap GPU, launch first (~20 GPU-h, ~6 h wall at 4 workers)

**B1. `LeakyParallel` × 3 encodings × 4 datasets × 5 seeds = 60 runs**, saving
per-timestep prefix logits, per-layer spike decomposition and test scores.
≈14 GPU-h. One run set unlocks **five** review items with no further training:
early-exit T99, DR@1%/0.1% FAR, per-class P/R/F1, NSL-KDD known-vs-unseen
attacks, and SOP/energy accounting.

**B2. Baselines — LR, RF, XGBoost, MLP × 4 datasets × 5 seeds = 80 runs** on the
identical frozen features. Mostly CPU and GPU-hist; ≈4–6 h wall, runs alongside
B1. This is the reviewer's single biggest ask and the cheapest to satisfy.

Optional: add `Leaky` and `Synaptic` latency for second and third early-exit
comparators (+12 GPU-h).

## Tier C — the leakage answer (~21 GPU-h, ~6 h wall)

**CTU-13 is already answered and needs zero new runs.** The journal line's
420-run matrix included the vanilla `LeakyParallel/latency` arm, and those 60
runs are ported to `results/v1_submitted/ctu13_strict_vanilla.csv`:

| Condition | macro-F1 |
| --- | --- |
| whole-scenario aggregates + random split (**the submitted protocol**) | **100.00 ± 0.00** |
| causal aggregates + random split | 99.69 ± 0.23 |
| causal aggregates + **scenario-disjoint** | **92.01 ± 9.72**, folds 77.0 / 91.2 / 99.9 / 99.9 |

The paper's `1.0000` on CTU-13 is a leakage artifact and we can prove it from
committed results. This must be corrected in the manuscript regardless of what
else gets done — it is the item most likely to sink the paper if a reviewer
finds it independently. Never quote the 92.01 mean without the fold range.

**C1. Champion family under v2 on the remaining datasets**: `LeakyParallel` ×
3 encodings × 5 seeds on `nslkdd_v2`, `kddcup99_v2`, `cicids2017_v2`, plus the
4 CTU-13 folds. ≈21 GPU-h. Establishes that the ranking is not an artifact of
the v1 preprocessing, for the configurations the claim actually rests on.

**C2 (stretch).** All 27 variants under `ctu13_v2_f*` at 5 seeds — ~40 GPU-h,
~10 h wall. Only if C1 lands early.

## Deferred, and the paper should say so

| Review item | Cost | Verdict |
| --- | --- | --- |
| Full clean 540-run v2 sweep | 198 GPU-h | **Not feasible.** Report v1 as the primary sweep under its own name, with v2 as a targeted robustness check. |
| T-sensitivity, T ∈ {5,10,25,50} | ~36 GPU-h | Stretch. High value if the early-exit result lands, since it answers "which encoding reaches quality with the least temporal budget". |
| Preprocessing robustness (quantile / clipped-z) | ~10 GPU-h | Stretch, 2 datasets only. |
| Threshold fairness, full θ grid, 432 runs | ~170 GPU-h | **Cut.** Replace with the honest limitation already written into PREREGISTRATION.md §8, plus a reduced θ ∈ {0.1, 0.5, 1.0} probe on the two gated families if time allows (~19 GPU-h). |
| Genuinely temporal windows for a fair delta evaluation | large | Cut. State the representational mismatch instead. |

## Order of execution and decision gates

```
Gate 1 (done) → Tier B1+B2 in parallel → statistics → Tier C1 → C2/T-sweep if time
```

- **Gate 2** — after the first seed of B1, check for NaNs, spike quiescence,
  majority-class collapse and that the prefix logits reconstruct the terminal
  prediction. Do not use it to pick a champion.
- **Gate 3** — after C1, re-run `rank_analysis.py --tag v2`. If the champion is
  no longer competitive, do not spend runs trying to restore it; the
  pre-registration already says what to write instead.
- **Gate 4** — after baselines, pick the claim: SNN wins both, SNN matches on
  quality and wins on event-compute, or SNN trades quality for efficiency. All
  three are publishable; only the third needs the framing changed.

## The claim this plan supports

Not "92.11% average accuracy". Something closer to:

> Across 27 configurations evaluated over 20 paired dataset × seed blocks,
> `LeakyParallel/latency` attains the lowest mean macro-F1 rank, though 14
> configurations lie within one critical difference of it. Against its closest
> competitor, `LeakyParallel/rate`, it wins 16 of 20 blocks
> (Holm-adjusted p = 0.008), while at the encoding level latency and rate are
> not separable at the dataset level. Under scenario-disjoint evaluation with
> causal host features, CTU-13 falls from a perfect 100.00 to 92.01 with a
> 77–100 fold range, and we report that rather than the leaked figure.

That survives leakage, reseeding, and the closest-competitor challenge, and it
does not depend on delta coding being bad.
