# Addendum 1 — full confirmation sweep

**Frozen 2026-08-14, before any run of the full confirmation sweep completed and
before any statistic derived from it was computed.** The original
`PREREGISTRATION.md` is unchanged and remains in force; this document extends it
to a population that the original explicitly declined to evaluate on cost
grounds.

Everything below must be checkable against the commit that introduced it. If a
rule here turns out to be wrong, the fix is a further dated addendum stating
why, not an edit to this text.

---

## 1. Why this addendum exists

`PREREGISTRATION.md` §4a adopted a two-axis confirmation design because a
complete 27-configuration re-sweep was outside the budget available before the
original deadline. That constraint has lifted. 235 of the 675 cells of the full
confirmation sweep are already complete, so the remainder is 440 runs rather
than 675.

The two-axis design remains the **confirmatory** analysis, because it is the one
that was registered before any confirmation result existed. The full sweep is
reported alongside it under this addendum, and its status is stated wherever it
appears: registered in advance of its own results, but after the two-axis
results were known.

## 2. Population

All **27 configurations** (9 neuron families × 3 encodings) on all **5
confirmation protocols** (`nslkdd_v2`, `kddcup99_v2`, `cicids2017_v2`,
`ctu13_v2_f0`, `ctu13_v2_f1`) at **5 seeds** {42, 43, 44, 45, 46} = 675 cells.
Population identifier: `full_27_confirmation`, registered in
`src/populations.py` and enforced by the same guard that refuses mixed fields.

## 3. Metric, unit, and statistics — unchanged from the original

- **Primary metric:** macro-F1 over the **fixed label universe**. On
  CIC-IDS2017 this remains capped at 0.80 because U2R has no test support; the
  test-present variant stays descriptive and is never used for ranking.
- **Block:** one (protocol × seed) pair, giving 25 blocks.
- **Ranking statistic:** mean rank across blocks. Ties within a block receive
  average ranks (`rankdata(..., method="average")`).
- **Omnibus:** Friedman across the 27 configurations, with Nemenyi post-hoc
  reported **descriptively**. With 27 configurations and 25 blocks the critical
  difference is large; failure to separate is reported as limited all-pairs
  resolution and never as evidence of equivalence.
- **Cross-distribution summary:** the protocol-level collapse (one mean per
  protocol before ranking, 5 units) is the **primary** cross-distribution
  statistic, because five seeds on one protocol share a dataset, a partition and
  a preprocessing fit and are not five independent observations.
- **Intervals:** hierarchical bootstrap resampling protocols first, then seeds
  within protocols, as in `src/stats.py`.

## 4. Ranking-stability statistic, specified now

**Kendall's τ** between the screening ranking and the full confirmation ranking
over all 27 configurations, with a permutation p-value (10,000 permutations,
RNG seed 20260814). Reported with the configurations whose rank changes most in
absolute position. This is the quantity that answers the paper's own question —
how far an optimistic protocol displaces design-space conclusions — and it is
fixed here so it cannot be selected afterwards from among several possible
agreement measures.

## 5. Outcome to permitted wording

Decided before the result exists, including the branch in which the
recommendation is withdrawn. Writing that branch down is what makes the others
worth believing.

| Outcome under the full confirmation sweep | Permitted wording |
| --- | --- |
| `LeakyParallel/latency` holds rank 1 | "top-ranked under both protocols" |
| Another **latency** configuration takes rank 1 | "the leading group is stable; the leader within it is not", naming the new leader, and reporting `LeakyParallel/latency`'s position explicitly |
| A **non-latency** configuration takes rank 1 | the encoding conclusion is reported as protocol-dependent and **the recommendation is withdrawn** |

In every branch the screening result is reported unchanged alongside, and the
difference between the two rankings is the finding.

Additional constraints carried forward from the original: no dataset or protocol
may be dropped after seeing the ranking; no seed set may be re-drawn; the
primary metric may not be swapped; and pooled cross-protocol standard deviations
remain prohibited.

## 6. What this does not change

The pre-registered two-axis tests are **not** recomputed over the expanded
population. Their p-values, mean ranks and block counts stand as registered.
Adding configurations after seeing results and then re-running the registered
tests over the larger set would destroy exactly the property the registration
exists to provide.

The post-hoc analyses stay post-hoc and stay labelled: the threshold-matched
encoding ablation, the delta-equivalence derivation, the CTU-13 covariate-shift
diagnostic, and any T-sweep.

## 7. Cost, recorded for honesty about what was affordable

Measured from the 235 completed runs rather than estimated: the remaining 440
cells cost **266.9 GPU-hours**, dominated by CIC-IDS2017 (157.3 h for 120 runs)
and by the slower neuron families (Alpha 3.00×, SConv2dLSTM 2.19× and Lapicque
1.93× the `LeakyParallel` reference). At the measured throughput of 5.17 GPU-h
per wall-hour at six-way concurrency, that is roughly 52 wall-clock hours.
