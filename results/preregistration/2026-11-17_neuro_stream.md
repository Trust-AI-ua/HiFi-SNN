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
