# Revision 2, steps 1-3: results and pre-registered decisions

Pre-registration: `results/preregistration/2026-11-16_r2_integer_gbdt.md`. IoT four, seeds 42-46.

## Step 2. Integer-only execution

Macro-F1 (%), five-seed means. Float = the reported deployed model (int8-grid weights, float arithmetic), reproduced exactly from each checkpoint. Integer = int8 weights, int32 accumulators, Q = 8, leak (v*230)>>8.

| Dataset | T2S float | T2S integer | Change | Worst seed | Agreement | T2S FAR int | DNN float | DNN integer | Change | Agreement |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CICIoT2023 | 75.20 | 75.16 | -0.05 | -0.15 | 99.96% | 0.68 | 74.75 | 74.83 | +0.07 | 99.93% |
| CICIoMT2024 | 89.42 | 89.34 | -0.08 | -0.24 | 99.99% | 0.86 | 89.24 | 89.38 | +0.14 | 99.99% |
| Edge-IIoTset | 83.94 | 83.97 | +0.03 | -0.07 | 99.88% | 0.04 | 80.60 | 80.77 | +0.18 | 99.66% |
| TON-IoT | 92.98 | 92.96 | -0.02 | -0.26 | 99.47% | 0.62 | 94.06 | 94.04 | -0.02 | 99.94% |
| IoT mean | 85.39 | 85.36 | -0.03 | -0.18 | 99.83% | 0.55 | 84.66 | 84.76 | +0.09 | 99.88% |

Variants (Tree-to-Spike, IoT-mean change): 16-bit leak +0.00, Q = 0 -3.97. Largest integer magnitude: Tree-to-Spike 11,303,584, DNN-Tree 90,839 (int32 limit 2,147,483,647).

Tree-to-Spike minus DNN-Tree, IoT mean: +0.72 with the reported models, +0.60 under integer execution.

**R2:** IoT-mean change -0.03, worst dataset -0.08. Rule: hurts if <= -0.5 or any dataset <= -1.0. **Integer execution does not hurt the spiking model.**

## Step 3. Validation-selected compact GBDT within 128 KiB

Selected on validation macro-F1 among the 48 configurations within 128 KiB (packed format); test scored once. "more-rounds" is the sweep's mislabelled gain-prune arm.

| Dataset | GBDT (val-selected, <=128 KiB) | Tree-to-Spike | GBDT minus T2S | Seeds GBDT ahead | GBDT FAR | T2S FAR | Mean KiB | Configurations |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| CICIoT2023 | 77.11 ± 0.23 | 75.20 ± 2.62 | +1.91 | 4/5 | 9.18 | 0.70 | 126.6 | 50x3 more-rounds 8b |
| CICIoMT2024 | 89.21 ± 0.14 | 89.42 ± 3.24 | -0.21 | 2/5 | 7.97 | 0.87 | 125.8 | 100x4 more-rounds 8b; 50x4 more-rounds 16b |
| Edge-IIoTset | 80.15 ± 0.49 | 83.94 ± 0.17 | -3.79 | 0/5 | 16.12 | 0.04 | 64.2 | 100x3 8b; 50x3 8b; 50x4 8b |
| TON-IoT | 94.96 ± 0.25 | 92.98 ± 0.23 | +1.98 | 5/5 | 0.73 | 0.58 | 105.3 | 25x3 more-rounds 16b; 25x6 16b; 25x6 8b |
| IoT mean | 85.36 | 85.39 | -0.03 | | 8.50 | 0.55 | 105.5 | |

Guard (seed 42, all four datasets, 96 configurations): round counts equal 96/96, packed bytes equal 96/96, largest test macro-F1 difference 0.0005 (the published values have three decimals).

**R3:** the GBDT is ahead on 2 of 4 datasets; IoT mean lower. Rule: wins on more benchmarks if >= 2 of 4 or a higher IoT mean. **The tree wins on more benchmarks; the framing needs reassessment.**

## Deviation from the pre-registered implementation

The per-depth packed-prediction check first compared labels of the in-model quantized booster and `src/serialize.predict_packed`. On Edge-IIoTset seed 45, depth 6, one of 300 rows differed: the packed sums of classes 7 and 10 tie exactly at 8-bit leaves, and XGBoost's per-class intercept (which `predict_packed` omits) breaks the tie the other way. The check was replaced by a stricter one (identical leaf indices, and margins equal up to a constant per class) and that cell was rerun; the other 19 cells had passed the label check. Selection and scoring are unchanged. The intercepts (4 B per class, at most 56 B) are not in the packed price; every selected model keeps at least 378 B of headroom under 128 KiB.
