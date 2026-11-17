# Track A/B: wording for the paper (PI instructions, 2026-11-17)

- **Reset.** Describe the subtractive reset on Lava as a custom neuron model in Lava's
  fixed-point (Loihi 2) simulation. Claim it is supported on Loihi 2 hardware only after the
  INRC documentation confirms it.
- **SANA-FE.** The Tree-to-Spike networks are simulated with a custom soma model
  (`src/sanafe_plugin/loihi_lif_delayed.cpp`: SANA-FE's built-in Loihi LIF with the delayed
  subtractive reset) that uses the same per-event energy and latency prices as SANA-FE's
  calibrated Loihi neuron (`loihi_lif`). SANA-FE's published calibration error (about 10% for
  energy on first-generation Loihi) was measured with the built-in model, so it carries over
  only approximately. Energy is reported only for networks whose spikes per layer match Lava's
  on every flow; give the match rate with the numbers.
- **Decision rules** are P1 to P7 in the paper and appendix; map them to the internal labels
  once in the protocol appendix (`results/preregistration/RULES.md`). R1 to R6 stay the
  paper's requirements.
- **Stream dataset size:** 533,855 flows (cap 100,000 per class; 7,604,796 uncapped).
- **B2 secondary analyses** (amendment 3, pre-registered before any B2 result): per-class
  packets-to-detection, results without the attack-tool artifact classes (DDoS_ICMP, DDoS_TCP,
  Port_Scanning, DDoS_UDP), and the host-stack check (first-packet length alone). P7 is applied
  to the primary five-seed result only; a seed-42 read is exploratory.
- **B2 execution:** the stream student shared the GPU with the revision-2 training queue
  (operation counts, not timings, so sharing does not affect the results).
- The Lava model-discovery incident (results/neuro/DEVIATIONS.md, D1) is internal and is not
  reported in the paper.
- **Fidelity table (A2): report both reset rows.** One sentence: with Lava's standard LIF,
  which resets to zero in the spike step, agreement with the integer reference falls to 37% to
  93% (macro-F1 down 18 to 63 points); with the custom neuron model that applies snnTorch's
  delayed subtractive reset it is 99.4% to 99.85%. Reset semantics, not precision, decide
  whether the trained network survives the move to Lava. (Values from
  results/neuro/a2_lava/a2_lava.csv; type them from numbers.tex.)
- **CICIoMT2024 fails P5** on the F1 criterion (+0.69 > 0.5) with 99.85% agreement: 3 of 2,000
  flows disagree, two of them Spoofing flows (52 in the sample) that the integer reference
  calls Benign and Lava gets right. Report it as a fail, with the per-class breakdown.
- **Loihi 2 support for the subtractive reset** is not documented publicly. Loihi 2 neurons are
  microcode-programmable with programmable reset logic, and published work changes the reset
  in custom microcode (arXiv 2405.01305), but no public source shows a subtractive reset.
  Confirm with INRC (lava-loihi documentation is for INRC members) before claiming it.
- **Vanilla energy** (amendment 4): the paired comparison uses the fixed-point Vanilla through
  the same integer -> Lava -> SANA-FE chain, with its own (fixed-point) macro-F1 next to its
  energy. The float-spike-activity number, if shown, is labelled as such and paired with the
  float accuracy.
- **Stream teacher curve:** the flow-end macro-F1 of the per-packet teacher uses the stream
  feature set (19 incremental features from the pcaps). Do not compare it with the 83.94 from
  the CSV features.
- **A1 (NIR):** report the export workarounds (batch dimension, scalar LIF parameters, h5py
  compression) as interoperability findings for snnTorch 0.9.4 / NIR 1.0.8.
