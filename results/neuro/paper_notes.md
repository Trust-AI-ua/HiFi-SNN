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
- **Energy claim (replaces "accuracy at similar energy").** Use the macro-F1 of the simulated
  networks (results/neuro/frontier/FRONTIER.md). With the integer-reference pairing the PI's
  draft reads: on a Loihi-calibrated model, Tree-to-Spike uses 1.4 to 2.1 times the energy per
  inference of the fixed-point conventional SNN over the full window, and is 2.4 to 28.1 macro-F1
  points more accurate. With the simulated-network macro-F1 (the network whose energy was
  measured) the gain is 2.2 to 22.0 points. Use that pairing, as amendment 5 does. Validated
  early exit brings Tree-to-Spike below the conventional SNN on CICIoMT2024 (246 against 423 nJ)
  and level on CIC-IoT2023 (437 against 446 nJ). Type every number from numbers.tex.
- **TON-IoT exception, stated plainly.** There the gain is about 2.2 to 2.4 points at about twice
  the energy, so the conventional SNN is a reasonable choice when energy matters most.
- **CICIoMT2024 and P5.** It fails P5 under the two-sided rule (|change in macro-F1| <= 0.5). The
  change is in Lava's favour: Lava is right on two Spoofing flows that the integer reference gets
  wrong. On the whole test split the simulated network and the integer reference differ by 0.02
  points.
- **Full-test scores of the simulated networks** come from the NumPy emulation of Lava's
  arithmetic, which matches Lava bit for bit on the 2,000-flow sample. Label them as such; Lava
  itself was run on the sample only.
- **NIR reset.** NIR 1.0's LIF, CubaLIF and IF nodes reset to v_reset (a reset to a potential);
  there is no subtractive reset in the specification (nir/ir/neuron.py, nir 1.0.8). An snnTorch
  model therefore comes back from NIR with a zero reset. Restoring the subtractive reset gives
  100% agreement on all four datasets; the imported model agrees on 39% to 97%. Pair this with the
  Lava zero-reset result as one porting lesson: reset semantics decide fidelity.
- **Step 5 wording** (wait for CICIoMT2024): training variance alone is about as large as
  data-split variance, and each about as large as the published SD. With five runs per arm the
  ordering is noise. Differences under about 3 points on CIC-IoT2023 cannot be resolved.
- **Pairing (PI, 2026-11-18): the simulated network's macro-F1 throughout**, wherever energy or the
  neuromorphic chain is discussed. The integer-reference scores are not used beside energy.
- **Fixed-point robustness, seed 42 on both sides** (float record to simulated network, full test):
  - Vanilla on Edge-IIoTset: 77.04 to 62.12 (-14.92).
  - Tree-to-Spike on Edge-IIoTset: 84.14 to 84.11 (-0.03). Do not use 83.94, which is the
    five-seed float mean.
  - Other datasets: Tree-to-Spike +0.05 / -0.23 / -0.10 (TON-IoT / CICIoMT2024 / CIC-IoT2023);
    Vanilla -0.42 / +0.49 / -0.38.
  - Drop the -20.92 (the integer reference's loss).
- **Fidelity section, one sentence:** Lava's fixed-point Vanilla agrees with the integer reference
  on only 88% to 99% of the full test (the 12-bit against the 8-bit leak, compounded because the
  hidden layer never resets), so the Vanilla energy is paired with the simulated network's own score.
- **Energy table caption:** "seed 42". Never compare its macro-F1 with the five-seed means
  (CIC-IoT2023: 79.18 here against 75.20).
- **Ratios:**
  - TON-IoT: 2.07 times Vanilla's energy over the full window and 1.64 with early exit (871
    against 532 nJ), for about 2.2 points.
  - Edge-IIoTset: 1.40 times, early exit disabled, for 22.0 points.
- **Latency sentence:** over the full window Tree-to-Spike is faster than Vanilla on the simulated
  chip on Edge-IIoTset (58.6 against 63.6 µs) and CICIoMT2024 (51.0 against 58.7). On CIC-IoT2023
  it is marginally faster (57.2 against 58.3). On TON-IoT it is slower (85.5 against 67.9), where
  its hidden layers spike most (147 / 502 spikes per flow). Steps (20 against 25) and spike
  traffic both set latency.
- **Step 5 sentence:** "Training-seed and data-sampling variation contribute comparably (SD 2.9 to
  3.4 macro-F1 points each), and rerunning the seed-42 configuration reproduces the stored result
  exactly." (TNNLS R2 #6.)
- **Framing (PI, 2026-11-18).** Vanilla's fixed-point collapse is one dataset (Edge-IIoTset) and is a
  supporting, worst-case-robustness result, not a general property. Wording: "In the fixed-point
  neuromorphic chain, Tree-to-Spike moves by at most 0.23 macro-F1 points on all four benchmarks,
  whereas the conventional SNN loses 14.92 points on Edge-IIoTset."
- **Headline results:**
  1. The encoder gain over conventional SNNs (\JNDiffTtsVanIotMean, IoT mean of five-seed float
     scores; the factorial ablation is pending).
  2. Energy with early exit on the simulated chip: better than Vanilla on all three measures on
     CICIoMT2024, and level on energy with 16.1 points more accuracy on CIC-IoT2023.
  3. Fidelity: the NIR round trip and Lava with the subtractive reset, plus the reset lesson.
- **Fidelity wording precision:**
  - The NIR round trip is exact in predictions (100% agreement, identical macro-F1 on all four
    datasets), not bit-exact in the readout: the largest readout difference is 0.45 (CIC-IoT2023),
    from float64 against float32 state.
  - The 99.4% to 99.85% Lava agreement is on the 2,000-flow sample. On the whole test split the
    bit-exact emulation agrees with the integer reference on 99.58% to 99.98%.
- **Energy: two models, one direction.** The manuscript's analytic model gives Tree-to-Spike
  \JNTtsOverVanTotalMin to \JNTtsOverVanTotalMax times Vanilla's total energy (five-seed means).
  SANA-FE gives 1.4 to 2.1 times over the full window (seed 42). Present them together: the
  simulated chip confirms the analytic model's direction at a somewhat lower ratio. Early exit is
  what changes the comparison on CICIoMT2024 and CIC-IoT2023.
