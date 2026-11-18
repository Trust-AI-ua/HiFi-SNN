# Pre-registration: neuromorphic portability and energy (track A) and stream-mode pilot (track B)

Written 2026-11-17 on branch `journal-r2` (after commit 960486da), before any of the runs
below. Thread variables stay unset. The GPU is held by revision-2 steps 5 and 6; track A and
B1 run on the CPU, B2 when the GPU frees.

Models are the reported Tree-to-Spike students (`results/Tree-to-Spike/<ds>/seed_<S>/`) on the
IoT four (CICIoT2023, CICIoMT2024, Edge-IIoTset, TON-IoT), seed 42 unless stated. "Integer
NumPy" is the headline integer path of step 2 (`scripts/r2_integer_inference.py`: Q = 8,
8-bit leak `(v * 230) >> 8`, delayed subtractive reset).

## Track A. Neuromorphic portability and energy

### Environment

A separate virtual environment `.venv-neuro`, Python 3.10, with exact versions pinned in
`requirements-neuro.txt`: lava-nc, lava-dl, snntorch, nir, nirtorch, sanafe. The training
environment (`.venv`) is not changed.

### A1. NIR round trip (descriptive)

For each dataset, export the seed-42 student with `snntorch.export_to_nir`, re-import it with
`snntorch.import_from_nir`, and run the full test split through both. Readout in both: argmax
of the summed output membrane. Report prediction agreement and the largest absolute
difference of the readout. Expected: 100% agreement. If it is not exact, the causes checked
first are the time step (snnTorch's NIR export uses dt = 1e-4 to map beta to tau) and the
reset mode; any fix is reported with its effect.

### A2. Lava fixed-point execution

**Sample.** Per dataset, a stratified sample of 2,000 test flows of seed 42: at least 50 per
class (all of a class if it has fewer), the remainder allocated in proportion to class size,
drawn with `numpy.random.default_rng(0)`. Integer NumPy already covers the full test sets;
Lava only has to confirm agreement with it.

**Network.** The three-layer student rebuilt in Lava (`Dense` -> LIF, three times) from the
int8 weights of the checkpoint (per-tensor scale s, as in step 2). LIF `du = 1` (the current
is the input of the step only) and `dv = 0.1` (beta = 0.9), in Lava's 12-bit fixed point;
bias and threshold converted to integers in the units of the layer's weight scale, as Lava's
fixed-point model requires. Input: the encoder's spike trains (computed on the host, exactly
as in step 2). Readout: the output layer's membrane, summed over the T steps, argmax. Run
configuration `Loihi2SimCfg(select_tag="fixed_pt")`.

**Reported.** Prediction agreement with integer NumPy and with snnTorch (float); macro-F1 on
the sample for Lava, integer NumPy and snnTorch, and the change Lava minus integer NumPy;
per-layer spike counts (layers 1, 2, 3) in Lava and in integer NumPy; any saturation of
Lava's 24-bit state. Diagnostic only: a NumPy emulation of Lava's fixed-point arithmetic, to
attribute disagreement to the reset, the leak rounding or saturation.

**Fidelity rule R7.** Lava is faithful on a dataset if its prediction agreement with integer
NumPy is at least 99% and its macro-F1 differs from integer NumPy's by at most 0.5 points on
the same sample. Portability is claimed only for datasets that pass.

**Reset rule R8.** Lava's standard LIF resets to zero, in the step of the spike; the students
were trained with snnTorch's subtractive reset, applied one step after the spike. In order:

1. Run with the standard Lava LIF. If R7 passes on all four datasets, stop.
2. Otherwise, write a custom Lava process model (fixed-point, `select_tag="fixed_pt"`) with
   snnTorch's delayed subtractive reset, and rerun R7. This counts as portable to Loihi 2 only
   if documentation confirms that Loihi 2's programmable neuron supports subtractive reset.
   INRC documentation is members-only; if no accessible documentation confirms it, the result
   is reported as "Lava fixed-point simulation, hardware support unconfirmed", and step 3
   runs as well.
3. Otherwise (R7 still fails, or support is unconfirmed), retrain zero-reset students
   (`reset_mechanism="zero"` in all three LIF layers, every other setting production) at
   seed 42 on the four datasets, 4 GPU runs queued after steps 5 and 6. They are reported as
   the "portable variant" with their own macro-F1, FAR and DR, beside the subtractive-reset
   students, and pass through A1 and A2 like them.

### A3. SANA-FE energy and latency (descriptive)

- **Installation check first.** Run a bundled Loihi example and confirm it reproduces the
  documented output; no network is simulated before this passes.
- **Network.** On SANA-FE's Loihi architecture description: input C -> 64 -> 64 -> K, one layer
  per core; the mapping is recorded in the output.
- **Input.** The recorded spike trains of the same 2,000 flows as A2.
- **Variants.** (i) Tree-to-Spike, full window; (ii) Tree-to-Spike with the calibrated early
  exit (C1-gate of step 4, the seed-42 thresholds in
  `results/journal_r2/exit_calibration/thresholds.json`), each flow stopped at its exit step;
  (iii) Vanilla SNN, the reported seed-42 model with its own encoder.
- **Reported.** Energy and latency per inference, mean and SD over flows, with SANA-FE's
  published calibration error for first-generation Loihi (about 10% for energy) as the stated
  uncertainty. Sensitivity: one rerun with a different core mapping (each hidden layer split
  across two cores). The encoder runs on the host and is counted separately: binary-search
  comparisons per flow, sum over kept features of ceil(log2(edges + 1)).
- **Output.** `results/neuro/energy_sanafe.csv` and a short method note.

## Track B. Stream-mode pilot (Edge-IIoTset packet captures)

### B1. Stream dataset

- **Source.** The Edge-IIoTset pcaps under `dataset/edge-iiotset/` (normal captures, and one
  attack class per attack capture). MITM is excluded, as in the repair. tshark 4.2.2.
- **Per packet.** Timestamp, 5-tuple, IP length, TCP flags, protocol.
- **Flows.** Bidirectional (unordered 5-tuple), 120 s idle timeout, first K = 32 packets
  kept. Packets without an IP layer (ARP and the like) are not assembled.
- **Labels.** A flow in a normal capture is Normal. A flow in an attack capture takes the
  capture's class if one endpoint is an attacker IP; otherwise it is ambiguous (background
  traffic in the attack capture) and dropped. Attacker IPs: the endpoints of the attack
  captures that never appear in any normal capture (this also covers spoofed sources).
  Normal-capture flows with an attacker-IP endpoint are dropped. A per-capture table of the
  top endpoints is reviewed before B2, and any exception is recorded.
- **Cap.** At most 100,000 flows per class, sampled uniformly (rng seed 0) after labeling,
  so the floods do not dominate the tensor; the uncapped counts are reported.
- **Features after each packet (19, all incrementally updatable).** Packet count; bytes;
  minimum, mean, maximum and SD of packet length; forward-packet share and backward bytes
  (forward = the flow's first sender); interarrival minimum, mean, maximum and SD; duration;
  SYN, ACK, FIN, RST and PSH counts; protocol (TCP, UDP, ICMP, other, as a code). IPs and
  ports are not features.
- **Storage.** A prefix tensor (flows x 32 x 19, float32; steps after the flow's last packet
  repeat its final state and are masked), labels, flow lengths, source capture.
- **Splits.** By flow, stratified by class, seeds 42 to 46: 20% test, then 10% of the rest
  for validation (the loader convention for Edge-IIoTset).
- **Checks, reported before B2.** Class counts (capped and uncapped); share of one-packet
  flows per class; the Layer B shortcut screen of `scripts/leakage_feature_audit.py`
  (indicator NMI, classes ruled out or in) on the prefix features at packets 1, 4 and the
  flow's last packet. Flags are reviewed by name. A feature judged a capture artifact is
  dropped, and the decision is recorded, before any B2 model is trained.

### B2. Pilot models (seeds 42 to 46)

- **Teacher.** XGBoost with the production Edge-IIoTset teacher settings (200 trees, depth 6,
  learning rate 0.1, inverse-frequency class weights) on the prefix vectors at every packet,
  each prefix weighted by 1 / (number of the flow's prefixes) so that each flow counts once.
  Its split thresholds define the stream encoder.
- **Tree-to-Spike-Stream.** Features kept and level edges exactly as the production encoder
  derives them from the teacher (importance pruning at the 0.20 quantile). At each packet,
  each kept feature's level index (number of edges at or below the value) is computed; the
  feature emits a spike on an "up" channel when the index rises and on a "down" channel when
  it falls, and nothing when it is unchanged. The student (64, 64 LIF, beta 0.9, subtractive
  reset) takes one step per packet; its running readout after packet k is the mean output
  membrane over packets 1..k. Training: the production loss (CE, KD at tau 2 against the
  teacher's posterior on the same prefix, ranking, sparsity, false-positive terms with the
  production weights) applied to the running readout at every packet up to the flow's
  length, 10 epochs, production optimizer settings; epoch selection on validation macro-F1
  at decision.
- **Decision.** The C1-gate of step 4 adapted to packets: per-packet temperatures fitted on
  validation by NLL, then the gate threshold chosen from the same grid by the same rule
  (eligible if validation macro-F1 is within 1.0 point of the decision at flow end; lowest
  cost; cost = executed SOPs). A flow is decided at the first packet whose calibrated
  confidence reaches the threshold, or at its last packet, or at packet 32.
- **Per-packet tree.** The same teacher evaluated after every packet, with the same decision
  rule (per-packet temperatures on its margins, same grid and eligibility; cost = packets
  evaluated).
- **Reference.** The compact GBDT within 128 KiB (step-3 selection procedure, on the prefix
  vectors), with the same decision rule. Reported, not in the go rule.
- **Metrics on test, at decision.** Macro-F1, FAR, DR. Packets-to-detection on attack flows,
  median and 90th percentile, over the attack flows that both the SNN and the per-packet
  tree detect (each model's own detected set as a secondary readout). Operations per packet:
  SOPs for the SNN (input-inclusive), node comparisons for the trees (path lengths summed
  over trees and classes), and the feature-update cost reported once, since both pay it.
  Share of packets that produce no input spikes.
- **Matched FAR.** Both models' benign thresholds (the rule of
  `2026-11-17_r2_matched_far.md`, applied to the posterior at decision) are tuned on
  validation to the same target, FAR <= 1%; the gate threshold is fixed first. Descriptive:
  the same at targets 0.1, 0.25, 0.5 and 2%.

### B3. Go/no-go rule R9 (five-seed means)

At the matched point (validation FAR <= 1% for both):

- **Go** if Tree-to-Spike-Stream's test macro-F1 is at most 2.0 points below the per-packet
  tree's, and at least one of: (a) its median packets-to-detection is lower than the tree's;
  (b) the tree's node comparisons per packet are at least 5 times its SOPs per packet.
- **No-go** otherwise: the paper ships with scope A only. Synaptic operations and node
  comparisons are different operations; criterion (b) is reported with that caveat.

### B4. If go

Add the CICIoMT2024 captures with the same B1 pipeline, run A2 (Lava check) on the stream
model, and compute SANA-FE energy per packet. Planned after the go decision; not
pre-registered in detail here.

## Amendment 1 (2026-11-17, after the endpoint review of B1, before any flow tensor was built)

The endpoint review (`results/neuro/b1_stream/endpoints.txt`) shows the attacker-IP rule
above is too wide. In every attack capture the victim is 192.168.0.128 (also present in the
normal captures). In the eleven non-flood captures 192.168.0.170 is the attacker (at least 95%
of packets) and the other novel endpoints are background internet hosts and multicast groups
(Google and Cloudflare addresses, 239.255.255.250), which the rule would label as attacks. In
the three floods (ICMP, TCP SYN, UDP) the sources are spoofed, novel, and talk only to the
victim. The labeling rule becomes:

- A flow in an attack capture takes the capture's class if one endpoint is the victim
  192.168.0.128 and the other is an attacker: 192.168.0.170 in the non-flood captures, any
  endpoint never seen in a normal capture in the three floods. Every other flow in an attack
  capture is dropped as ambiguous.
- Normal captures: unchanged (flows with an endpoint seen only in attack captures are dropped).

Also recorded: `Vulnerability scanner attack.pcap` is damaged at byte 111,714,300 of
140,377,652 (a record header claims a 136 MB packet); the 265,827 frames before it are used
and the remainder is not recovered (`*.truncated.json` beside the packet file).

## Amendment 2 (2026-11-17, PI instructions, before any A2 or A3 result was computed)

- **Labels.** In the paper the rules here are P5 (R7), P6 (R8) and P7 (R9); the full map is
  `results/preregistration/RULES.md`. Internal labels are unchanged in these files.
- **P5 is the A2 result.** The custom subtractive-reset process matching Lava's own arithmetic
  bit for bit (a toy check) is an implementation check, not the result. The result is
  agreement with integer NumPy on the 2,000-flow sample against P5.
- **P6, step 3 removed.** The zero-reset retrain is dropped; it is not queued. The paper
  describes the subtractive reset as a custom neuron model in Lava's fixed-point simulation,
  and claims Loihi 2 support only after INRC documentation confirms it.
- **A3, consistency gate.** Before any SANA-FE energy number is used, SANA-FE's per-layer spike
  counts are compared with Lava's (the P5-passing variant) for the same inputs, flow by flow.
  SANA-FE simulates its own neuron model; energy is reported only for a network whose spike
  counts match Lava's, and the match rate is reported with the energy.
- **B1 appendix.** The shortcut screen's flags are reported in an appendix table (state, feature,
  NMI, the classes ruled out or in with their P(active | class), and a one-line reason), with
  the class counts and one-packet shares.
- **Neuro-environment runs** go through `scripts/neuro_guard.py` (results/neuro/DEVIATIONS.md, D1).

## Amendment 3 (2026-11-17, PI instruction, before any B2 model was trained or scored)

Secondary analyses, descriptive. The primary rule R9 (paper P7) is unchanged.

**Attack-tool artifact classes**, fixed here from the B1 appendix evidence (seed-42 training
split, first-packet length): DDoS_ICMP (one length, 28 B), DDoS_TCP (one length, 160 B),
Port_Scanning (one length, 40 B), DDoS_UDP (two lengths, 28 B for 90% and 56 B; every flow has
one packet). These classes may be decided after one packet by any model.

1. **Per class.** For Tree-to-Spike-Stream, the per-packet tree and the compact GBDT at the
   matched point (validation FAR <= 1%): per attack class, detection rate at decision and
   packets-to-detection (median, 90th percentile) on that class's detected flows.
2. **Without the artifact classes.** The primary table recomputed on the test flows of the
   other ten classes: macro-F1 over those ten classes (a prediction of a removed class counts
   as an error), FAR, DR, packets-to-detection (median, 90th percentile, flows detected by both),
   and SOPs and node comparisons per packet. The thresholds are not refitted.
3. **Host-stack check.** The first packet's length also differs between the normal devices and
   the attacker host (98.9% of Normal flows start with a 52-byte packet; nearly every attack
   flow from 192.168.0.170 starts with 60 bytes, the TCP options of its SYN). To show how much
   of the benign/attack separation one packet size carries: a benign-versus-attack rule on the
   first packet's length alone (lengths ordered by their validation attack share, threshold
   tuned on validation to FAR <= 1%), its test FAR and DR, and the test DR of each model after
   packet 1 at the matched point.

Reported beside the primary results in `results/neuro/b2/SUMMARY.md`.

## Amendment 4 (2026-11-17, PI instruction, before any fixed-point Vanilla result)

Written after A2 (all four datasets) and A3 for Tree-to-Spike were in; no fixed-point Vanilla
SNN had been run on any path.

1. **Vanilla SNN pairing in A3.** A first A3 run simulated the Vanilla SNN with its float
   parameters on SANA-FE's 1/64 grid (82.8% to 87.8% of flows with the float model's hidden
   spike count); that pairs an energy with no measured accuracy and is superseded, not reported.
   Two pairings replace it:
   - **(b), primary: fixed-point Vanilla through the same chain as Tree-to-Spike.** Integer
     NumPy (`scripts/neuro_vanilla_int.py`: per-tensor int8 weights by the int8_grid rule,
     Q = 8, 8-bit leak round(256 beta), the ReLU floor of LeakyParallel, delayed subtractive
     reset at the output, readout by output spike count) -> Lava fixed point
     (`scripts/neuro_a2_vanilla.py`: hidden ReluLIFReset, output SubLIFReset, dv =
     round(4096 (1 - beta))) -> SANA-FE (`vanilla_fixed`) with the amendment-2 gate: energy only
     if every flow's per-layer spikes and output spike counts equal Lava's. Reported with the
     integer Vanilla's full-test macro-F1 next to the float record. Lava-versus-integer
     agreement is reported against the P5 thresholds, descriptively (P5 is the Tree-to-Spike rule).
   - **(a), secondary: float spike activity** (`vanilla_float`). The float parameters scaled by
     2^16 so SANA-FE's 1/64 grid is below float32 resolution; per-flow hidden and output spike
     counts compared with the GPU float replay (match rate reported, not a gate). Paired with
     the float record's macro-F1 and labelled "float spike activity".
     *Correction before any vanilla_float run:* SANA-FE casts v * 64 to a 32-bit int, and
     Edge-IIoTset's float hidden potentials reach about 1,356, so the scale is 2^16 or the
     largest power of two keeping |v| * 64 within 2^30 (2^13 for Edge-IIoTset), recorded per row.
   The Tree-to-Spike versus Vanilla energy comparison in the paper uses (b).
2. **P6, step 3 restored as a contingency.** The zero-reset retrain (dropped in amendment 2) is
   run only if INRC or the Loihi 2 neuron documentation shows that Loihi 2 cannot run the
   delayed subtractive reset. Otherwise it is not run.
3. **A2 per-class disagreements** (descriptive): for each dataset and reset variant, the true
   classes of the flows on which Lava and integer NumPy disagree, and whether each disagreement
   makes the prediction right or wrong (`results/neuro/a2_lava/disagreements_by_class.md`).
4. **B2 teacher curve** (descriptive): the per-packet teacher's test macro-F1 after k = 1..32
   packets on all five seeds (a flow that has ended keeps its last decision), as the baseline
   curve for the go/no-go figure (`results/neuro/b2/teacher_curve*.csv`).

*Note added after the A2 Vanilla and A3 results (descriptive, not a change to any rule).* Lava's
Vanilla agrees with the integer NumPy Vanilla on 89.9% to 98.9% of the sample (Lava's 12-bit
leak against the 8-bit leak of the integer rule, compounded by the hidden layer having no reset),
so the integer full-test macro-F1 is not exactly the accuracy of the network SANA-FE measured.
`scripts/neuro_lava_fulltest.py` runs the bit-exact emulation of Lava's arithmetic on the whole
test split for both models and reports that macro-F1 beside the pre-registered pairing.

## Amendment 5 (2026-11-17, PI instruction, before any frontier run): energy-accuracy frontier, exploratory

Written after A3 and after the variants' training records (macro-F1 of the float models) were
known; no variant had been through Lava or SANA-FE, and no variant's energy was known.

**Variants.** Trained seed-42 Tree-to-Spike models, no retraining, chosen as the settings that
spend less energy by construction:
- **T10:** the window sweep at T = 10 (`results/ladder/<ds>/T10`).
- **uniform:** uniform channel allocation (`results/encoder_ablation/<ds>/uniform`).
- **Reduced channels:** every one-factor setting with fewer encoder channels than the production
  encoder on all four datasets, namely prune quantile 0.40 (`q40`), pop_max 2 (`M2`), and
  pop_max 1 with the production pruning (`prune_only`), all under `results/ladder/<ds>/`.
  `tree_only` (pop_max 1, no pruning) is not included: it has as many channels as production
  or more.

CIC-IoT2023's variants were trained under the 10-epoch protocol, while its production model uses
the 20-epoch block; that dataset's frontier compares unequal training and is labelled so.

**Chain.** Unchanged from A2 and A3:
- `scripts/neuro_export_variants.py`: one teacher refit per dataset, checked against each
  variant's record; integer-encoder and float-student guards;
- Lava fixed point with the subtractive reset on the stratified 2,000-flow sample, equal bit for
  bit to the emulation;
- SANA-FE, full window, one layer per core, with the amendment-2 gate (energy only if every flow's
  per-layer spikes and readout equal Lava's);
- `scripts/neuro_frontier.py`.

No early exit for the variants: no exit gate was calibrated for them.

**Accuracy.** For every row, including Tree-to-Spike with the full window and with early exit and
the fixed-point Vanilla: the test macro-F1 of the simulated network, from the bit-exact emulation
of Lava's arithmetic on the whole test split (`scripts/neuro_lava_fulltest.py`, extended to the
C1-gate exit).

**Reported** (`scripts/neuro_frontier_summary.py`, `results/neuro/frontier/FRONTIER.md`), per
dataset:
- energy and latency per inference (mean, SD) and the simulated macro-F1;
- the host encoder's comparisons, separately;
- the Pareto set over (energy, macro-F1) among all rows;
- for each Tree-to-Spike row, whether it **dominates the fixed-point Vanilla**: higher simulated
  macro-F1 and mean energy no higher.

The analysis is descriptive. The frontier is reported whatever it shows, and a dominance claim
in the paper is made only where this check holds.

### Amendment 5b (2026-11-17, PI instruction, before any frontier result)

No variant had been through Lava or SANA-FE when this was written.

- **Window sweep completed.** T = 40 and T = 80 (`results/ladder/<ds>/T40`, `T80`) are added to
  the frontier beside T = 10. The T = 20 point is the production model, already in the table.
  Lava's per-flow slot is lengthened to max(32, T + 8) steps for the longer windows (the
  arithmetic is unchanged; the slot only has to hold the window and the synaptic delays).
- **Training protocol.** All ladder and encoder-ablation variants, the window sweep included,
  are 10-epoch runs. On TON-IoT, Edge-IIoTset and CICIoMT2024 this is the production schedule.
  On CIC-IoT2023 the production model uses the 20-epoch block, so that dataset's frontier
  compares unequal training.
- **Teacher reproduction.** Some ladder runs were made by `scripts/run_ladder.py` with
  OMP_NUM_THREADS = 16 // workers, which changes the XGBoost teacher. Where the default refit
  does not reproduce a variant's stored teacher, the export refits under the thread cap that
  does (`neuro_export_variants.py --omp N`, recorded per row). The teacher guard (test macro-F1
  equal to the record's) is unchanged, and a variant whose teacher cannot be reproduced is left
  out and reported as such.
