# Pre-committed reporting policy: DNN-Tree instability on CSE-CIC-IDS2018

Written BEFORE inspecting per-seed values, confusion matrices, unit activity,
quantization deltas, or training curves for this cell.

Current state: DNN-Tree mean macro-F1 (11 datasets) = 79.86. The abstract claim
"within 0.1 points of the dense twin" depends on this cell.

1. Artifact (fixable: learning rate, initialization, quantization, budget)
   - Apply the fix UNIFORMLY to every model and every dataset.
   - Retrain, regenerate all aggregates, and replace the abstract's within-N-points
     claim with the corrected number, whatever it is.
   - Describe the fix in Section V-F.
   - Prefer initialization or learning-rate changes over changing the nonlinearity,
     since changing ReLU alters what the iso-architecture twin is.

2. Genuine instability (reproducible across initializations)
   - Keep the reported number.
   - Report per-seed values, and state it as a finding about dense-model
     sensitivity on this dataset.
   - The abstract claim stands, but must also cite the median.

3. Either way
   - Per-seed values for this cell go in the appendix.
   - If anything changes, regenerate: DNN-Tree aggregate macro-F1, the abstract
     claim, M9 Friedman ranks and CD diagram, and the Section VI-E paired Wilcoxon.
written 2026-09-16T14:42:47Z
