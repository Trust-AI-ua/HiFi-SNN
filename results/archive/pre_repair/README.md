# Pre-repair results (contaminated; do not cite)

Produced before the leakage audit (results/diagnostics/leakage_audit/README.md).
CSE-CIC-IDS2018 used `Src Port`, a capture-day indicator; Edge-IIoTset used the
`0`/`0.0` spelling artifact and column-shifted rows. Kept for before/after
comparison and for the provenance-based sensitivity check only.

* `Tree-to-Spike/`, `DNN-Tree/`, `TASNN/`, `<dataset>/SNN_LeakyParallel/latency/`
  — the four headline models on the two affected datasets, superseded by the
  re-runs in `results/`.
* `phase1_contaminated/cse-cic-ids2018/` — the Phase-1 neuron x encoding sweep on
  the contaminated corpus (27 configs x 5 seeds). Never reported: the published
  Phase-1 evaluation [11] covers KDDCup99, NSL-KDD, CIC-IDS2017 and CTU-13, all
  of which the audit found clean. The "LeakyParallel + latency is strongest and
  most consistent" claim holds on every subset: mean rank 1st of 27 over all six
  internal-sweep datasets (3.17), over the five excluding CSE-CIC-IDS2018 (2.20),
  and over the four datasets in [11] (2.00). CSE-CIC-IDS2018 was the dataset where
  that configuration ranked *worst* (8th), so the claim does not rest on it.

NOTE: `results/tables/summary.csv` still holds the 135 contaminated Phase-1 rows
for cse-cic-ids2018 plus the pre-repair rows for both datasets; the re-runs only
replace the (dataset, neuron, encoding, seed) keys they touch.
