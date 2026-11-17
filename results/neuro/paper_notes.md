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
