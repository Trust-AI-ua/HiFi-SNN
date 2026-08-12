# Pre-registration — ranking rules for the SNN neuron × encoding study

**Frozen 2026-08-12, before any statistic in `results/analysis/` was computed.**
Nothing below may be changed after seeing an analysis output. If a rule turns out
to be wrong, the fix is a new dated section marked as an amendment, with the
reason, not an edit to the frozen text.

The point of freezing this is narrow and specific. The submitted paper names
`LeakyParallel/latency` the best configuration. Under a stricter protocol that
claim may not survive, and the temptation when it does not is to adjust the
metric, the aggregation, or the dataset set until it does. These rules remove
that freedom.

---

## 1. Primary hypothesis

> Under leakage-resistant evaluation and a common training protocol,
> `LeakyParallel` with latency encoding provides the strongest overall
> quality–efficiency operating point among the 27 evaluated SNN configurations.

Note what this does **not** say. It does not say latency beats rate on terminal
macro-F1, and the paper must not be built on that. The submitted table already
shows rate is close, and on KDDCup99 rate is marginally ahead of latency for
`LeakyParallel` (0.9260 vs 0.9257). Much of the apparent latency advantage in
v1 came from delta being weak, which is a separate and weaker claim (§7).

## 2. Metrics, with roles fixed in advance

| Role | Metric | Used for |
| --- | --- | --- |
| **Primary quality** | macro-F1 | the ranking, all headline claims |
| Supporting quality | MCC | corroboration; must not contradict the primary |
| Security | DR, FAR, DR@1% FAR, DR@0.1% FAR | operational framing |
| Efficiency | SOPs/sample, spikes/sample, T99, inference ms | the cost axis of the Pareto analysis |

Macro-F1 is primary because every dataset here is severely imbalanced and
accuracy is inflatable by a majority-class predictor. Accuracy may appear in
tables; it may not carry a claim.

## 3. Experimental unit and aggregation

One **block** = one (dataset × seed) pair. Within each block all 27
configurations are ranked by macro-F1, rank 1 = best.

The primary ranking statistic is **mean rank across blocks**, not the mean of raw
macro-F1 across datasets. Raw cross-dataset means are dominated by how different
the datasets are from each other rather than by how the configurations differ,
which is the same defect that makes the submitted paper's `0.7998 ± 0.1989`
uninterpretable.

**Dispersion is always reported within a dataset**, as mean ± SD over seeds.
Pooled cross-dataset standard deviations are banned from the manuscript.

## 4. Statistical tests, specified before the data

1. **Friedman test** over all 27 configurations across blocks. If it does not
   reject, no configuration may be called better than another on quality.
2. If it rejects, **Nemenyi** post-hoc with a critical-difference diagram.
3. **Pre-specified focused comparison**, decided now and not chosen from the
   results: `LeakyParallel/latency` vs `LeakyParallel/rate`, paired Wilcoxon
   signed-rank over the blocks.
4. `LeakyParallel/latency` vs the best non-`LeakyParallel` configuration, with
   the family of focused tests corrected by **Holm**.
5. Effect size and bootstrap 95% CI for Δmacro-F1 (latency − rate), resampling
   datasets first and seeds within datasets, matching
   `scripts/statistics_utils.py`. α = 0.05 throughout.

## 5. How the result maps to manuscript wording

Decided in advance so the wording follows the evidence rather than the reverse.

| Outcome | Permitted wording |
| --- | --- |
| Lowest mean rank **and** significant advantage | "best-performing / top-ranked configuration" |
| Lowest mean rank, not significantly ahead of another variant | "top-ranked" or "co-leading configuration" |
| Comparable quality, clearly better efficiency | "strongest quality–efficiency operating point" |
| Another configuration clearly wins | report that. The champion claim is dropped. |

## 6. Protocol generations

`v1` is the submitted protocol, kept reproducible and reported as such. `v2` is
leakage-resistant: train-only fitting of every learned transform, a partition
frozen independently of the model seed, group structure respected where the
benchmark has one, and an audit that refuses to train on a split violating a
disjointness it claims. Per dataset:

| Dataset | v2 primary evaluation |
| --- | --- |
| NSL-KDD | official train/test split |
| KDDCup99 | official train/test split |
| CIC-IDS2017 | within-capture chronological 70/10/20 (5-class); day-disjoint Mon–Wed / Thu–Fri as a binary transfer test |
| CTU-13 | scenario-disjoint frozen 4-fold, partition-local causal host aggregates |

v1 and v2 numbers are never mixed inside one table. Their difference is a
result — the size of the leakage effect — and is reported as one.

## 7. Delta encoding

Delta modulation encodes temporal contrast, and the input here is a static
per-flow feature vector, so a weak delta result is at least partly a
representational mismatch rather than evidence that delta is intrinsically
inferior. Delta's weakness may be reported as a secondary observation, phrased
as a property of this formulation. **It may not be used as the primary evidence
that latency encoding is preferable.** The primary encoding comparison is
latency vs rate.

## 8. Threshold confound, stated not hidden

`SLSTM` and `SConv2dLSTM` run at firing threshold 0.1 while the other seven
families run at 1.0, because at 1.0 the gated cells emit no spikes at all and
collapse to the majority class. This means the sweep is not a pure
neuron-only ablation, and the manuscript must say so wherever it claims
otherwise. Whether it changes the ranking is an empirical question
(`scripts/threshold_sensitivity.py`); until that evidence exists the limitation
is stated, not argued away.

## 9. Things that are forbidden

- Selecting a subset of datasets after seeing the ranking.
- Re-running with a different seed set and reporting the better one.
- Choosing an operating threshold on test data. Fixed-FAR thresholds are
  selected on validation and then frozen.
- Reporting the consistency "Score" of the submitted Table III. It is an
  arbitrary statistic, and it was already reported inconsistently (text says
  "7 of the 8 (dataset, metric) cells", the table column says `7/12`). It is
  replaced by a Pareto analysis over quality and cost.
- Quoting a mean across CTU-13 scenario folds without the fold range beside it.
  The folds span roughly 77 to 100 macro-F1 and the mean hides that entirely.
