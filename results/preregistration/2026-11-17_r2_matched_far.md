# Pre-registration: matched false-alarm check (revision 2, step 3b)

Written 2026-11-17 on branch `journal-r2` (after commit 17f247d0), before any of the runs
below. It decides how the compact GBDT appears in the paper under framing 1 ("matches a
compact tree on macro-F1, with far fewer false alarms").

**Why.** At the argmax operating point the validation-selected GBDT within 128 KiB
(`results/journal_r2/gbdt_valsel`) has an IoT-mean FAR of 8.50% against 0.55% for
Tree-to-Spike. The training setups differ. Tree-to-Spike has a false-positive term
(`a_fp = 0.5`) and no class weights. The GBDT has inverse-frequency cost weights, which give
the large benign class little weight. Part of the gap may therefore be the operating point,
not the model.

## Systems and data

- IoT four, seeds 42 to 46. Same splits as the stored records (same loader arguments).
- **GBDT.** The configuration each seed selected in step 3, refit, leaves quantized as
  packed. Guard: its argmax test macro-F1 equals the stored `selected.test_macro_f1`.
- **Tree-to-Spike.** The stored checkpoint, floating-point path (the reported model; integer
  execution agrees on 99.8% of test flows). Guard: the refit teacher equals the stored
  teacher, and the test macro-F1 equals the record exactly. Its validation predictions are
  computed here; none are stored.
- Thread variables unset.

## Benign-threshold rule (GBDT)

With `p_b` the softmax probability of the benign class (class 0): predict benign if
`p_b >= t`, otherwise the most probable attack class. FAR(t) is the share of benign flows
with `p_b < t`. For a target FAR `f`, `t` is the largest threshold with validation
FAR <= `f`. The test split is scored once per threshold.

## Reported, per dataset and seed (five-seed means in the summary)

- **(A) Validation-matched FAR.** `f` = Tree-to-Spike's validation FAR for that seed.
  Report the GBDT's test macro-F1, FAR and DR beside Tree-to-Spike's test macro-F1, FAR and
  DR at its own argmax.
- **(B) Requirement operating point.** `f` = 1%. Report the GBDT's test macro-F1, FAR and DR.
  These are the values for the requirements tables.
- **(C) Descriptive, threshold-free.** Benign-versus-attack ROC AUC of both models on test
  (score `1 - p_b`), and the GBDT's test DR at Tree-to-Spike's test FAR (a readout on the
  test ROC curve, not a deployable operating point).

## Decision rule R5

Per dataset, on the five-seed means of (A):

- **The false-alarm advantage holds** if the GBDT, with its FAR matched to Tree-to-Spike's on
  validation, has a lower test macro-F1 than Tree-to-Spike.
- Otherwise the gap is an operating point: the tree can match Tree-to-Spike's false-alarm
  rate without losing macro-F1.

The claim "matches a compact tree on macro-F1, with far fewer false alarms" is supported if
the advantage holds on at least 3 of the 4 datasets and on the IoT mean. If it holds on fewer
datasets, the paper reports the matched-FAR results and drops "far fewer false alarms" as a
general claim. DR at matched FAR is reported beside macro-F1 but does not enter the rule.

## Amendment 1 (2026-11-17, after (A) to (C) were scored)

(A) tuned the GBDT's benign threshold and left Tree-to-Spike at its argmax, so the comparison
is asymmetric. Added, descriptive, with no new decision rule (R5 stands as scored):

- **(D) Symmetric threshold.** Tree-to-Spike gets the same benign-threshold rule on its own
  softmax (`p_b` from the summed output membrane), tuned on validation to FAR <= 1%. Its test
  macro-F1, FAR and DR are reported beside the GBDT's (B).
- **(E) Macro-F1 against FAR.** For both models, thresholds tuned on validation to FAR targets
  0.1, 0.25, 0.5, 1 and 2%; test macro-F1, FAR and DR at each target.

Same guards; output `results/journal_r2/matched_far_sym/`.
