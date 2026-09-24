# HANDOFF: Tree-to-Spike (HiFi-SNN) manuscript for IEEE TNNLS

> **Precedence:** Reconciliation part 3 (the final section) is authoritative wherever it conflicts with an earlier part.

Current as of 21 September 2026. Written so a new assistant can continue with
exactly the same scope, constraints, and state of knowledge. Sections 0-19 are
the author's document, stored verbatim; the repo-side reconciliation is appended
at the end and is the only part written from this machine.

## 0. Synopsis

Tree-to-Spike: a cost-sensitive XGBoost teacher, used only during training,
guides a compact LIF student through (1) a structural pathway (split thresholds
become TTFS bin edges; gain importances allocate and prune channels) and (2) a
supervisory pathway (class posterior drives distillation and ranking losses).
The tree leaves the inference path; only compiled threshold tables and the
student remain.

Premise: close to a DNN in accuracy, less energy on edge devices, favorable
against previous SNN architectures.

Agreed framing: on the four IoT benchmarks Tree-to-Spike is on par with the
dense twin (84.6 vs 84.1 macro-F1) and beats both SNN baselines on every IoT
dataset (+8.9, +9.1). Operation-priced compute energy advantage over the dense
twin is robust (about 3.3x on IoT). Memory-inclusive total energy advantage is
conditional on the memory hierarchy and on early exit. No efficiency claim
against the GBDT teacher; that analysis answers "why an SNN at all" (scope: a
spiking substrate is required). "State-of-the-art" softened to "exceeds the
evaluated SNN baselines".

Scope: the 4 IoT/IIoT/IoMT benchmarks are the primary evaluation; the 7
general-purpose benchmarks are an appendix generalization study.

## 1. Working agreement

Act as a rigorous TNNLS peer reviewer and co-analyst. Raj runs all experiments.
NO EM DASHES, EVER, in any text written for him; en dashes for numeric ranges
and compounds are fine. TNNLS: 10 pages plus 5 extra; abstract at most 250
words, no math symbols. Academic scientific story, not a claim list. Do not
compute headline numbers that could be stale; regenerate from scripts. Pre-commit
to how a result will be reported. Report negative and conditional findings.
Aggregation rule: predictive columns are 5-seed mean plus/minus SD; deployment
measurement columns are single-run at seed 42; label which applies per column
group. Skewed data: lead with median and p90, paired tests, explicit
conditioning. Deliverables: complete paste-ready LaTeX blocks. Process lessons:
PID-level kills, check no process holds a path before mv/rm, append-with-run-ID
rather than overwrite, stamp artifacts with path, run ID, git SHA, source script.

## 2. Manuscript basics

Title (current): "HiFi-SNN: Hybrid Spiking Neural Networks for High Fidelity
Network Intrusion Detection in IoT". Method is Tree-to-Spike everywhere else;
mismatch unresolved. Double-blind. Prior work [patel2026evaluation] =
arXiv:2606.01442 (Phase-1: LeakyParallel + latency strongest on KDDCup99,
NSL-KDD, CIC-IDS2017, CTU-13). TASNN baseline: Karthik et al., Scientific
Reports 2026.

RQ-1 (IoT): exceeds SNN baselines in macro-F1; mean within one point of the
dense twin? RQ-2: contribution of threshold placement, channel allocation,
posterior distillation. RQ-3: trade-offs in operations, energy, artifact size,
latency, early exit, precision, and under which accounting assumptions an
efficiency advantage holds.

Section VI order: A Classification; Statistical Analysis; B Encoder-Variant
Ablation; C Compute Energy Under Operation Pricing; D Operation-Priced Energy
Does Not Survive Measurement; E The Teacher as a Deployment Candidate; F
Early-Exit Behavior and Its Failure Modes; G How Analytical Proxies Fail; H
Cross-Cutting Observations. Quantization subsection (subsec:quantcost) merges
after E.

## 3. Method and configuration

Teacher: 200 rounds, depth 6, lr 0.1, subsample 0.9/0.8, histogram, gain
importance, cost power gamma = 1.0. Encoder: q = 0.20, M_max = 3, jitter 0.12,
T = 20; channels added in raw feature-column order, not importance order.
Distillation: tau 2.0, mu 0.05, a_kd 1.0, a_rank 0.3, a_sp 1e-3, a_fp 0.5.
Student: C-64-64-K LIF, beta 0.9, threshold 1.0, subtractive reset,
fast-sigmoid slope 25; Adam, batch 128, 10 epochs, val 0.1. LEARNING-RATE
CONFLICT: Section V-F says 1e-3, hyperparameter table says 2e-3. Unresolved.
Deployment: post-training symmetric per-tensor int8 grid, no QAT, no calibration;
stored and executed as float32; no integer-only claim. Early exit: theta_exit
selected per dataset and seed on validation, disabled when none qualifies (26 of
55 runs); theta_unc = 0.5.

Theory: Lemma 1, Theorem 1, Corollary 1 (cor:distill), Lemma 2 (exact recovery
only if T >= |E_j| + 1), Prop. 1-3, Remarks. REVIEWER M5 OPEN: at T = 20 many
features likely violate T >= max|E_j| + 1; scope theory claims to the level code,
not spike timing.

## 4. Systems compared

Tree-to-Spike (artifact 53,398 B mean, fp32 storage); DNN-Tree (iso-architecture
dense twin, same int8 grid); Vanilla SNN (ONE hidden layer of 128 LeakyParallel,
T = 25, raw-feature TTFS, arctan surrogate; parameter-count inconsistency between
appendix and repaired table must be unified); TASNN (480-646K parameters,
privileged preprocessing removed, Bernoulli spikes at inference, no early exit);
GBDT full; GBDT compact (THREE CONFLICTING DEFINITIONS in the paper: int8
thresholds/quantized leaves, boosting-round prefix, and the sweep; reconcile).

## 5. Datasets

Primary: CICIoT2023 (8), CICIoMT2024 (6), Edge-IIoTset (14 after repair, width
64), TON-IoT (10). Appendix: KDDCup99, NSL-KDD, CTU-13 (binary, ceiling),
ISCXIDS2012, CIC-IDS2017, CSE-CIC-IDS2018 (Probe empty, width 78), UNSW-NB15.
UNSW-NB15 is general-purpose; Section V-A still wrongly groups it with IoT and
still says Edge-IIoTset is 15-class.

## 6. Leakage audit and repair

CSE-CIC-IDS2018: Src Port and three identifier columns exist only in
02-20-2018.csv; Src Port excludes r2l and u2r on 48.1% of rows; teacher ranked it
most important; all four dropped (82 to 78). DNN-Tree gained 9.77 macro-F1 after
removal: the leaked feature drew a first-layer outlier weight that broke
per-tensor quantization.

Edge-IIoTset: (a) zero written 0 or 0.0 by capture, one-hot as text made the
spelling determine Normal vs attack (NMI 0.48 vs 0.25 for tcp.flags);
http.referer carried payload. (b) Unquoted commas shifted all 121,568 DDoS_UDP
rows (two layouts), all 1,214 MITM rows, and 2,227 benign UDP rows. Repair:
canonicalize, drop http.referer and dns.qry.name, decode three layouts, drop
MITM (15 to 14), per-row provenance flag. Verified against pcaps: 123,795 rows,
all 60 comparable fields. Width 95 to 64 = 18 (spelling/hex) + 12 (dropped) + 1
(MITM level); realignment width-neutral. Layout probe uninformative (1.000 /
0.9994 / 0.701).

ISCXIDS2012: direction is address-derived and informative; disclosed, not removed.

Audit method: Layer A schema, Layer B per-feature indicator vs label. Cannot see
value-based shortcuts. Phase-1 check: LeakyParallel-latency ranks first of 27 in
every subset; contaminated runs archived.

## 7. Final headline results (repaired, 5-seed macro-F1)

IoT: CICIoT2023 62.82 / 72.22 / 72.43 / 59.09; CICIoMT2024 72.61 / 89.42 / 89.24
/ 74.18; Edge-IIoTset 75.39 / 83.94 / 80.60 / 78.37; TON-IoT 92.04 / 92.98 /
94.06 / 90.37 (Vanilla / T2S / DNN / TASNN). Means 75.72 / 84.64 / 84.08 / 75.50.
T2S beats both SNNs on all four (+8.92, +9.14). T2S minus DNN: -0.21, +0.18,
+3.34, -1.08. Edge lead is DNN's int8 fragility (fp32 DNN 86.81 vs T2S 84.46):
say "on par", not "exceeds". N = 4: Friedman chi2 9.6 (p = 0.022), CD 2.345
exceeds the maximum observed gap, so no pairwise significance attainable.

General-purpose: means 76.16 / 76.07 / 77.51 / 72.33. T2S TIES the Vanilla SNN
and trails DNN by 1.44; the main text must say so.

Pooled eleven: T2S 79.19, DNN 79.90, Vanilla 75.99, TASNN 73.48; ranks 1.55 /
2.18 / 2.45 / 3.82; chi2 18.16; CD 1.414; TASNN vs DNN and TASNN vs T2S
significant, Vanilla vs TASNN NOT; TOST p 0.31, CI [-1.74, +0.36], Wilcoxon 0.12,
paired SD 1.86; equivalence fails on location, not dispersion.

Pre vs post repair: CSE T2S 75.94 to 74.02, DNN 65.93 to 75.70, TASNN 69.56 to
69.55, Vanilla 70.10 to 71.71. Edge T2S 88.78 to 83.94, DNN 89.87 to 80.60,
TASNN 81.57 to 78.37, Vanilla 82.13 to 75.39. Keep the 15-to-14 class effect
separate from the leakage effect. Pre-repair aggregates are obsolete.

## 8. Quantization (M3, M6 closed)

Clean-dataset mean delta: T2S +0.06, DNN -0.12, TASNN 0.00, Vanilla -0.47.
Repaired: CSE DNN -1.28 (worst -3.60), T2S -1.58; Edge DNN -5.97 (worst -13.20),
T2S -0.51, Vanilla -14.91 (worst -20.18). Paired T2S vs DNN over 11: +0.615,
Wilcoxon p 0.083. Two quantizer columns (Linear-only vs all weight matrices).
Mechanism: fc1 outlier sets the per-tensor scale. Two-cause framing: intrinsic
dense fragility, amplified by the leak on CSE. Do NOT write the encoder-sparsity
or high-index-channel explanations (refuted).

## 9. Energy

Horowitz 45 nm: 4.6 / 0.9 pJ (int8 0.23 / 0.03), SOPs include the input layer.
Operation-priced T2S vs DNN: IoT 3.32x (32-bit), 4.98x (int8), lower on every
IoT dataset. Memory-inclusive: memory is 95.7 to 99.9% of total; band 0.94 to
2.06x spans a modeling choice; IoT-only band not computed. Early exit is
load-bearing: without it T2S executes more operations than DNN on IoT. Loihi-class
repricing moves T2S from rank 2 to 6 of 7; report only as an idealization caveat.
GBDT arithmetic: compact 0.49x T2S, full 1.22x. Latency (CPU batch-1): GBDT 2.33
/ 2.19 ms, T2S 3.20 (0.21 ms encoder interpreter overhead), DNN 0.226, raw MLP
0.0135. latency_elp.md should be deleted; benchmark_3way.py:226 reads a moved path.

## 10. GBDT as deployment candidate (M1)

Serialization mismatch: native file vs packed node arrays, about 15x apart.
DECISION PENDING: packed everywhere (recommended). T2S artifact: encoder tables
22.2%, student weights 76.1% (float32 container); true int8 serialization about
22,951 B. DECISION PENDING: re-serialize int8 (recommended). Envelope table:
every dataset has at least one fitting configuration at 256 KB; full ensemble
fits CTU-13 and ISCXIDS2012. Best fitting tree beats T2S by more than a point on
5 datasets, within a point on 4, T2S ahead on 2. CHECK that BestFit was selected
on validation, not test. CONSEQUENCE: the residency claims must be removed from
abstract, introduction, Section VI-E, and conclusion; the argument against the
tree is substrate-only. Positive: the student exceeds its seed-42 teacher on both
repaired datasets. Stale teacher comparisons must be regenerated.

## 11. Early exit (M4) and the proxy taxonomy

CICIoT2023 centerpiece: at theta 0.90, 99.1% exit at mean step 5.89, exit
accuracy 54.76 vs 98.85 (-44.13); 99.3% of wrong exits land on DoS; 99.3%
eventually correct, 74.6% after the exit step; wrong-exit step 2.25 vs recovery
11.16. Margin test: 0.8730 correct vs 0.8791 wrong, separation -0.0061; sweeping
a margin gate leaves a 24.5 to 45.9 point gap. Target-mismatch, not
miscalibration. t' conditioned on full-window-correct samples; report inclusion.
Gate decision predicted by wrong-exit fraction. Per-class t' bimodal; per-group
deferral is the only implementable remedy, not evaluated. Edge paragraph stale.
Counting inconsistency to fix. Unreported: theta_unc flag rate, escalation
energy, accuracy-energy curve, fraction exiting before their first input spike.
Proxy taxonomy: incomplete vs target-mismatched.

## 12. Encoder ablation (RQ-2)

Variants tree / quantile / no-KD / uniform / all. Threshold placement primary
(quantile loses on 9 of 11 and costs 6.6 to 79% more SOPs; caveat C11 level-shift
confound). Uniform second-order. No-KD hurts on 9 of 11 and exits earlier.
Production loses on UNSW-NB15 and slightly on CICIoT2023. The all variant is NOT
a dominating point under memory-inclusive accounting; M7 refuted.

## 13-15. Reviewer items, defects, claims

M5 (T >= max|E_j| + 1) OPEN; M6 closed; M7 refuted; M8 (raw-latency control arm)
not done; M9 must be redone for IoT N = 4 with pooled analysis in the appendix
and the CD diagram regenerated; M10 references, title, grouping. C3, C4, C6, C8,
C9, C10 untouched. Defect list: duplicated statistics subsection, stray heading
text, stale CSE rows, missing IoT tables (tab:results_iot,
tab:architecture_iot, tab:encoder_ablation_iot), six tables floating in Section
IV, orphan floats, MCC scale, Section V-B split contradiction, conclusion
residency claim.

CAN claim: IoT parity with the dense twin; exceeding both SNN baselines on every
IoT benchmark with a significant omnibus; lower operation-priced compute energy
on every dataset; threshold placement as primary mechanism; int8 nearly harmless
for the student but harmful for the dense twin and counterfactually for the
conventional SNN; the leakage audit and repair with pcap verification;
premature-commitment early exit and the proxy taxonomy; student exceeding teacher
on repaired datasets (pending 5-seed confirmation).

CANNOT claim: equivalence within one point; pairwise significance at N = 4;
state-of-the-art; unconditional total-energy advantage; measured energy on edge
or neuromorphic hardware; residency or efficiency advantage over the GBDT;
integer-only arithmetic; exact recovery of teacher bins from spike timing at
T = 20; beating the Vanilla SNN on the general-purpose benchmarks (it ties).

## 16-18. Computation status and next steps

Done: headline reruns, 44-cell fp32-vs-int8 capture, compression sweep, early-exit
chain, fc1 ratios, leakage audit and pcap verification, Phase-1 check.
Running: CSE encoder-variant cells, then ablation aggregation, deployment timing,
teacher tables. Decisions needed before the deployment stage: tree serialization
(packed vs native) and student serialization (int8 vs float32).

Next: serialization decisions; paste abstract_intro_iot.tex and decide the title;
fix self-contradictions; apply the reference mapping; rewrite VI-E and the VI-D
closing; redo statistics for IoT; merge the audit, quantization, and ISCX
paragraphs; regenerate IoT-scoped energy and the exit tables; then Section VII
and the conclusion.

---

## Repo-side reconciliation (written on the machine, 2026-09-21 19:40 UTC)

Points where the repository disagrees with, or has moved past, the document
above. The document is the authority on scope and claims; this section is the
authority on what has actually run.

### Already superseded by completed runs

* **Encoder ablation is nearly finished, not "14 cells".** 37 of 40 cells are
  done (all 20 Edge-IIoTset, and CSE-CIC-IDS2018 quantile, nokd, and all at 5
  seeds each plus uniform 42-43). Only uniform 44-46 remain, in flight.
* **The early-exit chain has been rerun on repaired data** (prefix extraction
  2026-09-21 04:42-06:21, chain to 06:50). The Section 11 note that the Edge
  paragraph is stale is correct, but the data to rewrite it now exists.
* **Two datasets swapped sides on the exit gate.** Edge-IIoTset is now REJECTED
  on all five seeds (it was admitted before the repair); CSE-CIC-IDS2018 is now
  ADMITTED on all five, thresholds 0.80-0.95, costing 0.14-0.97 macro-F1 at mean
  exit steps 10.4-17.7. The repaired anatomy table's Edge mean exit step of 9.31
  is a pre-repair artifact and must be regenerated.
* **The "5 of 11 datasets" count is wrong in both directions.** Current state:
  26 of 55 runs disabled; disabled on all five seeds for CICIoT2023,
  Edge-IIoTset, NSL-KDD and UNSW-NB15 (four datasets); partially disabled on
  ISCXIDS2012 (2 of 5), TON-IoT (3 of 5) and KDDCup99 (1 of 5). The honest
  statement is per run, not per dataset.
* **The sibling-attractor mechanism survives the repair**: per-class top-target
  share 90.1% against a dataset-level modal share of 50.5%. The 94.0% figure in
  older notes is pre-repair.

### Open decisions that are now time-critical

The deployment measurement runs automatically when the last three ablation cells
finish (estimated 01:00-03:00 UTC on 22 September) and it is the stage that
produces artifact sizes and latency. The two serialization decisions in Section
10 must be made BEFORE it runs, or it will measure the float32 container and the
native XGBoost file again:

1. Tree serialization: packed node arrays (recommended) vs XGBoost native.
2. Student serialization: true int8 (recommended, lossless) vs float32 container.

### Where the repo cannot help

The manuscript sources are not in this repository. `abstract_intro_iot.tex`,
`section_VI_complete.tex`, `m9_statistics.tex`, `section_replacements.tex` and
the review document exist only in the author's other session. Anything that
requires reading the current manuscript has to be pasted in.

### Artifacts that exist here and are current

`results/tables/`: tab_m3_quant, m3_quant_per_seed, tab_repaired_predictive,
tab_repaired_anatomy (Edge exit column stale), tab_fc1_ratio, tab_m6_per_seed,
tab_gbdt_envelope, tab_hyperparams, m10_merge_facts.md, sec5_audit_draft.tex,
sec6_quantization_draft.tex, sec4f_sec6_quant_draft.tex.
`results/diagnostics/`: per_seed_all_cells.csv (44 cells, repaired), m9_stats.txt,
fc1_ratio_repair.csv, leakage_audit/ (README, schema, features, verification).
`results/Journal_results/figures/cd_diagram.pdf` regenerated from the repaired
per-seed CSV. `results/early_exit/` regenerated post-repair.

---

## Repo-side reconciliation, part 2 (written on the machine, 2026-09-24)

Numbers are now frozen as LaTeX macros in `results/tables/numbers.tex`, generated
by `scripts/make_numbers.py` from a clean commit (tag `tnnls-submission`), with
per-macro provenance in `numbers_provenance.json`. Quote the macro, never a value
from this document. `scripts/check_numbers.py <file.tex>` flags typed numbers,
undefined macros and the superseded values below.

### Superseded numbers in sections 0 to 19 (do not reintroduce)

| Document says | Current | Macro or source | Why it changed |
|---|---|---|---|
| Compute energy vs dense twin about 3.3x; 3.32x (32-bit), 4.98x (int8) | 3.16x (32-bit), 4.74x (int8), ratio of IoT means | `\JNArithRatioIot`, `\JNArithRatioIotInt` | input layer counted for every system; Vanilla SNN double count removed (commit fce4506) |
| Memory is 95.7 to 99.9% of total energy | 92.3 to 99.7% (95.6 to 99.4% without TASNN); IoT four, 100 pJ SRAM, both memory models | `\JNMemShareMin`, `\JNMemShareMax` | recomputed on the band's systems with corrected counts |
| Band 0.94 to 2.06x; IoT-only band not computed | DNN-Tree / Tree-to-Spike total energy: deployed 0.60 to 1.35, full window 0.46 to 1.11; Tree-to-Spike wins only with reuse-aware accounting and cheap SRAM | `results/deployment/energy_crossover_iot.csv`, `\JNBand...` | IoT subset, corrected counts, both configurations |
| (not in document) | Vanilla SNN / Tree-to-Spike: total 0.36 to 0.56 (deployed), arithmetic 0.30 to 0.55; Tree-to-Spike uses 1.8 to 2.8x the Vanilla SNN's total | `\JNVanBandDep...`, `\JNVanArithRatioMin/Max`, `\JNTtsOverVanTotalMin/Max` | Vanilla SNN double count removed; the energy claim against it no longer holds |
| Tree-to-Spike artifact 53,398 B (fp32); encoder tables 22.2% | 21.4 KB IoT mean at int8; encoder tables 54.5% of it | `\JNKbTts`, `\JNEncShareTts` | int8 serialization, scales and biases float32 |
| Friedman chi2 9.6, p = 0.022 | exact p = 0.0069 (0.022 is the asymptotic value, unreliable at N = 4) | `\JNFriedmanPExact` | exact permutation test |
| Latency: GBDT 2.33 / 2.19 ms, Tree-to-Spike 3.20 ms | GBDT full 1.96, compact 1.72 (single-threaded); Tree-to-Spike 2.61 deployed, 2.89 full window; IoT means, seed 42, batch-1 CPU | `\JNLatMsGbdtFull`, `\JNLatMsGbdtCompact`, `\JNLatMsTts`, `\JNLatMsTtsFull` | GBDT had inherited four threads; the exit loop never broke |
| Means 75.72 / 84.64 / 84.08 / 75.50; margins +8.92, +9.14 | 75.71 / 84.64 / 84.08 / 75.50; +8.93, +9.14 | `\JNFone...IotMean`, `\JNDiffTtsVanIotMean` | unrounded |
| T2S minus DNN on CICIoT2023 -0.21 | -0.22 | `\JNDiffTtsDnnCicIoT` | unrounded |
| "On par" with the dense twin (unqualified) | Within one point at both 8-bit precisions, 1.05 apart in floating point; T2S - DNN = -1.05 fp32, +0.50 per-tensor, -0.64 per-channel (M3 reruns) | `\JNPcDiff...IotMean` | per-channel sensitivity check |
| No-KD hurts on 9 of 11 | on the IoT four, hurts on 3 of 4 (CICIoT2023 is +0.60 without KD) | `\JNAblNokd...Fone` | IoT scope |
| Quantile loses on 9 of 11 at 6.6 to 79% more SOPs | IoT: loses on 4 of 4 by 1.74 to 12.01 points at 20 to 37% more SOPs | `\JNAblQuantileLoses`, `\JNAblQuantileLossMin/Max`, `\JNAblQuantileExtraOpsMin/Max` | IoT scope, input-inclusive SOPs |
| Envelope: every dataset fits 256 KB | IoT at 128 KB: a configuration fits on 4 of 4, but within one point of the full ensemble only on Edge-IIoTset (1 of 4); seed 42, test-selected (disclosed) | `\JNEnvFitsIot`, `\JNEnvFitsNearFullIot` | leaf pricing fixed (one value per leaf) |
| Student exceeds its seed-42 teacher on both repaired datasets | five seeds: on the IoT four only Edge-IIoTset (+9.19); CSE-CIC-IDS2018 too (+5.77) but it is general-purpose | `\JNStudentMinusTeacher...` | five-seed paired comparison |
| GBDT arithmetic: compact 0.49x T2S, full 1.22x | per dataset in `tab_teacher_per_dataset.tex` (Rel. E column, deployed student, input-inclusive) | `results/tables/tab_teacher_per_dataset.csv` | input-inclusive student count, reselected compact prefix |
| Early-exit centerpiece figures (CICIoT2023 99.1%, -44.13, ...) | these are seed 42; five-seed wrong-exit share at theta 0.90 is 60.53% (37.89 to 76.69) | `\JNExitWrong...` | seed variability |
| "Gate decision predicted by wrong-exit fraction" | does NOT hold over five seeds: CICIoMT2024 (admitted on every seed) averages 19.25%, Edge-IIoTset (rejected on every seed) 18.83% | `results/tables/tab_exit_audit_iot.csv` | seed-42 pattern only |
| Edge-IIoTset 15-class, width 95; CSE width 82 | 14 classes, width 64; CSE width 78 | `table_datasets_overview.tex` | leakage repair |

### Resolved since 2026-09-21

* **Learning-rate conflict:** resolved from the run records. Tree-to-Spike and DNN-Tree 2e-3; Vanilla SNN and TASNN 1e-3 (their reference settings); all 10 epochs, batch 128.
* **Vanilla SNN parameter count:** trainable F*128 + 256 + 128K + K (IoT 6,790 to 16,266); the frozen beta*I recurrent matrix is folded to a scalar in storage and priced as a leak, not a matmul.
* **Compact GBDT:** one definition. The smallest boosting-round prefix within one point of the full teacher on validation, per seed, stored in packed node arrays (float32 thresholds, one 16-bit value per leaf). Reselected on the repaired corpora (Edge-IIoTset 1 to 4 rounds, 3.9 KB; CSE-CIC-IDS2018 93 to 109).
* **Serialization:** students int8 (bitwise-exact round trip), trees packed; decisions taken 2026-09-22.
* **Hyperparameters:** fixed 2026-06-11 as design defaults (commit 0fdeb96), unchanged since; no sweep chose them. The ladder (July to August) is an attribution study.
* **M5:** 45.9% of retained IoT features violate T >= |E_j| + 1 at T = 20; |E_j| is bimodal, so the rate is nearly flat in T (47.6% at 10, 43.2% at 40). Scoped in text. The ladder's T sweep (three IoT datasets) fails "within one point of T = 20": CICIoT2023 T = 10 -5.19, CICIoMT2024 T = 40 +5.06.
* **M8:** done. Encoder step +9.14 IoT mean, positive on all four, 2.8 to 7.4 seed SDs; distillation +0.40, mixed signs.
* **M9:** IoT exact Friedman in the main text; pooled eleven-dataset analysis for the appendix.
* **Quantization:** per-channel scaling recovers 73% of the dense twin's per-tensor loss (80% on Edge-IIoTset); TON-IoT seed 43 resists it (loss confined to fc1). Per-tensor remains the deployed configuration.
* **Retired to `results/superseded/`:** the pre-repair Edge-IIoTset ladder quarter, the 2026-08-12 reporting snapshot (sensitivity, crossed 2x2, loss ablations, non-inferiority with Edge-IIoTset -1.09), the pre-repair deployment audit, and five pre-repair LaTeX files. Not rerun, by decision.

---

## Reconciliation, part 3: manuscript state and guardrails (authoritative over earlier parts where they conflict)

Abbrev: T2S=Tree-to-Spike, DNN=DNN-Tree, Van=Vanilla SNN, TAS=TASNN, IoT4=CICIoT2023/CICIoMT2024/Edge-IIoTset/TON-IoT (order used below), GP=7 general-purpose. Data tag `23cf3ad`. IoT values five-seed unless noted; deployment = seed 42.

### 1. Claim verdict (write only the "survives" form)
| Claim | Verdict |
|---|---|
| Comparable to DNN on ML metrics | YES, descriptive: macro-F1 within 1.1 at every precision; ACC/MCC/DR within ~0.5. Exceptions: TON BACC -2.0; CICIoMT FAR 0.87 vs 0.10. Not equivalence (N=4). Macro precision NOT reported, do not claim. |
| Energy advantage vs DNN | Compute (arithmetic) YES: 3.16x, every IoT dataset, full window. Total (memory-incl.) CONDITIONAL. All estimated, none measured. |
| ...via fewer spikes | NO. T2S does 1.2-1.9x MORE ops than DNN MACs; saving = AC 0.9 vs MAC 4.6 pJ. |
| ...via early exit | NO for compute. Exit admitted only CICIoMT 5/5, TON 2/5; helps total energy only. |
| Beats Van and TAS | Accuracy YES every IoT dataset (+8.93, +9.14), encoder attributed (M8). Energy: beats TAS only; Van cheaper (2.7x less arithmetic, T2S 1.8-2.8x Van total). |
| Beats teacher | Only Edge-IIoTset (+9.19); weak teacher there. |
| Footprint/residency vs tree | NO claim. Substrate-only argument. |
| On GP benchmarks | T2S ties Van (76.07 vs 76.16), trails DNN (77.51). State plainly. |

### 2. Canonical numbers
- Macro-F1: T2S 72.22/89.42/83.94/92.98 mean 84.64; DNN 72.43/89.24/80.60/94.06 mean 84.08; Van mean 75.72; TAS 75.50. T2S-DNN -0.22/+0.18/+3.34/-1.08, median -0.02.
- Stats: Friedman chi2=9.60, exact p=0.0069; ranks 1.5/1.5/3.5/3.5; no pairwise test possible at N=4. Pooled 11 -> appendix.
- Precision: T2S-DNN fp32 -1.05 (DNN leads 4/4); per-tensor +0.50 reruns (+0.56 headline); per-channel -0.64 (DNN leads 3/4). Per-channel recovers 73% of DNN loss (Edge 80%: -5.97 -> -1.22). T2S 84.72/84.64/84.69. TON seed 43 resists (fc1 alone -> FAR 0.34 -> 14.33%). Van counterfactual Edge -14.91.
- M8 steps (mean): arch -0.21, precision -0.40, encoder +9.14 (every dataset, >=2.8 SD), distillation +0.40 (mixed sign); total +8.93. Raw arm 7-28% more params.
- Ablation: quantile loses 4/4 (1.7-12.0 pts, +20-37% ops); uniform -0.05 to -0.62, CICIoT -2.76; all 1.3-1.9x ops, +2.5 CICIoMT, +2.2 Edge; encoder +24-59% ops vs raw (no-KD basis).
- Energy: arithmetic nJ T2S 11.5/7.5/9.5/16.2, DNN 32.6/31.6/34.0/43.3, Van 3.5/2.8/5.2/5.3; DNN/T2S 2.82/4.20/3.58/2.66, 3.16 of means, int8 4.74. Memory share 92.3-99.7% (95.6-99.4 excl TAS) at 100 pJ. DNN/T2S memory-incl: deployed 0.60-1.35, full window 0.46-1.11; T2S cheaper only reuse-aware at 5-20 pJ deployed, 5 pJ full. Van/T2S 0.36-0.56.
- Deployment (seed 42, IoT mean): KB T2S 21.4 (encoder 54.5%, 41.0-67.2), DNN 20.8, Van 40.4, TAS 2092.0, GBDT full 748.0, compact 313.7; fits 128 KB 4/4/4/0/0/1; ms T2S 2.61 deployed (2.89 full), DNN 0.16, Van 1.34, TAS 1.95, GBDT 1.96/1.72.
- Teacher: full 82.95/90.54/74.75/96.67 (86.23), compact 85.67. Student-teacher -10.73/-1.12 n.s./+9.19/-3.69. Edge: 34/48 sweep configs beat full; best 79.78 vs 75.07 (seed 42); student +4.16 over best. Sweep <=128 KB: 4/4, within 1 pt of full only Edge; trails 1.08/2.03/5.86 (seed 42, test-selected).
- Early exit: admitted 0/5, 5/5, 0/5, 2/5; wrong@0.90 60.53/19.25/18.83/17.08% (seed-dependent); CICIoMT deployed -0.25 at 60.7% fewer ops; flagged 0.16/0.06/5.69/2.03%; margin 0.8730 correct vs 0.8791 wrong (seed 42). Old "wrong-exit predicts gate" claim DEAD.
- M5: 352/767 (45.9%) violate T>=|E_j|+1 at T=20; 54.1% exact; all-feature bound needs T=256 (12.8x); median |E_j| 117/90/1/1. Lemma 2 bound = 1/(2(T-1)).
- T sweep (CICIoT/CICIoMT/TON vs T=20): T10 -5.19/-0.06/+0.58; T40 -0.94/+5.06/+0.26; T80 +0.91/+3.48/-0.37 -> T=20 is a fixed cost-accuracy choice.
- Setup: hyperparams fixed 2026-06-11 (`0fdeb96`), not tuned; LR T2S/DNN 2e-3, Van/TAS 1e-3; timing EPYC 9555, 1 thread, batch 1, 500 warm-up, N=2000, median, load<2.0 x5 min; CICIoT 5% pre-split, CICIoMT 25% per provider partition, uniform, redrawn per seed, shared across models; CICIoMT split 73.4/8.2/18.4.

### 3. Manuscript state (drafts in Claude outputs)
- Structure: I Intro | II Background+Related (merged) | III Tree-to-Spike | IV Setup | V Results | VI Limitations | VII Conclusion | Appendix. Budget 15 pp incl. refs: 1.5/1.25/2/2/5.5/0.75/0.3/refs 1.5.
- Done: abstract/intro/conclusion (Raj edited); `section_II_background_related.tex`; `section_III_tree_to_spike.tex`; `section_IV_experimental_setup.tex`; `section_V_results.tex` (0 open cells, ~3.9 pp, 7 tables, 0 figures); `appendix_tsweep.tex`.
- Label map (removed -> use): lem:monotone->thm:sufficiency; prop:earlyexit, rem:calibration, tab:dep:artifact->sec:deploy; rem:approx->sec:formal; rem:pricing->sec:energymetrics; sec:related_work->sec:background; sec:crosscutting->sec:results; tab:notation, alg:training, all old Results figures -> delete refs.

### 4. Open items
1. Limitations (VI): rewrite to IoT scope; blocks drafted in `text_changes_ready_now.tex` [4],[5] (update [5] for per-channel).
2. Appendix: GP tables/paragraph (`section_V_datasets_iot.tex`), pooled stats + regenerated CD diagram, 11-dataset M3, CSE fc1/M6 tables, repriced envelope, GP ablation/exit (ISCX exception), T sweep.
3. Conclusion: "two to three times" -> "1.8 to 2.8 times" Van total energy.
4. First-person self-citation (double-blind): intro, II-A, II-C "our previous evaluation" -> third person; grep "our previous/prior/earlier", "we previously", "the group's".
5. `\cite{PRIOR-...}` keys (Raj). Subsample-fraction reason (cost vs paper; Raj).
6. Fig. 2 graphic: theta_exit validation-selected; drop "(common case)"; "int8-grid weights"; banner "only split thresholds remain"; distinct legend arrows.
7. Title: HiFi-SNN vs Tree-to-Spike.
8. Verify loss equations vs `tree_to_spike.py` (KD: no temperature on student logits, no tau^2).
9. Confirm manuscript cites no ladder analysis (pre-repair Edge).
10. `check_numbers.py` verification mode (typed values vs `numbers.tex`); run on manuscript before submission.
11. Record `subsample_fraction` in run configs. Push (`gh auth login`). Anonymized export from tag (exclude journal/notes, strip paths/history).

### 5. Anti-drift rules
- IoT4 primary; GP appendix only; justify by deployment setting, never by results.
- Manuscript: typed numbers, inline tables, no macros/`\input`; no em dashes; label every table five-seed vs seed 42; full-window vs deployed energy labelled.
- Never claim: fewer spikes as mechanism; early exit as reason for compute saving; energy win vs Van; unconditional total energy; equivalence; precision metric; student>teacher beyond Edge; footprint/residency vs tree; measured device energy; integer inference; state-of-the-art.
- New numbers only from `numbers.tex` at a clean tag; any new measurement path must reproduce the old path on unchanged cells first; pre-commit reporting before any new run.
