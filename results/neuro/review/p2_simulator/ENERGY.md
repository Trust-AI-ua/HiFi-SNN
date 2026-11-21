# Simulated energy: sample mean, prevalence-weighted, paired against the fixed-point Vanilla (amendment 8, part 3)

Five-seed means (SD over seeds). Paired difference = configuration minus the Vanilla SNN on the same sampled flows; negative means less energy than the Vanilla SNN. Prevalence = class means on the stratified 2,000-flow sample weighted by the full test split's class shares.

| Benchmark | Configuration | Seeds | Energy, sample (nJ) | Energy, prevalence (nJ) | Paired diff, sample (nJ) | Paired diff, prevalence (nJ) | Flows below Vanilla | Mean steps, sample / prevalence |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| edgeiiot | t2s_full | 5 | 737.9 (32.4) | 741.7 (29.7) | 190.1 (33.2) | 187.0 (29.8) | 0.0% | 20.00 / 20.00 |
| edgeiiot | t2s_exit | 5 | 698.4 (96.9) | 699.1 (101.6) | 150.6 (96.8) | 144.4 (100.7) | 8.9% | 19.02 / 18.93 |
| edgeiiot | T10 | 5 | 437.0 (10.7) | 435.8 (9.6) | -110.8 (26.2) | -119.0 (26.1) | 98.1% | 10.00 / 10.00 |
| edgeiiot | vanilla | 5 | 547.8 (23.1) | 554.7 (23.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0% | 25.00 / 25.00 |
| edgeiiot | vanilla_tuned | 5 | 545.9 (20.2) | 552.9 (20.1) | -1.9 (3.4) | -1.9 (3.4) | 46.0% | 25.00 / 25.00 |
| ciciomt2024 | t2s_full | 5 | 596.9 (12.4) | 591.8 (11.8) | 183.9 (14.1) | 186.4 (13.1) | 0.0% | 20.00 / 20.00 |
| ciciomt2024 | t2s_exit | 5 | 308.8 (149.6) | 298.0 (153.7) | -104.2 (149.1) | -107.4 (153.3) | 77.3% | 9.21 / 9.00 |
| ciciomt2024 | vanilla | 5 | 413.0 (7.2) | 405.4 (7.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0% | 25.00 / 25.00 |
| ciciomt2024 | vanilla_tuned | 5 | 413.0 (7.0) | 405.5 (6.8) | 0.0 (0.8) | 0.0 (0.8) | 43.8% | 25.00 / 25.00 |
| toniot | t2s_full | 5 | 1058.0 (42.6) | 1057.9 (42.6) | 517.3 (44.8) | 517.5 (44.8) | 0.0% | 20.00 / 20.00 |
| toniot | t2s_exit | 5 | 831.7 (66.2) | 825.4 (68.2) | 291.0 (65.7) | 284.9 (67.8) | 8.9% | 16.16 / 16.05 |
| toniot | T10 | 5 | 719.9 (23.3) | 720.1 (23.5) | 179.2 (27.5) | 179.6 (27.7) | 0.0% | 10.00 / 10.00 |
| toniot | vanilla | 5 | 540.7 (6.4) | 540.4 (6.4) | 0.0 (0.0) | 0.0 (0.0) | 0.0% | 25.00 / 25.00 |
| ciciot2023 | t2s_full | 5 | 782.8 (46.0) | 777.0 (47.0) | 327.4 (46.9) | 328.8 (46.4) | 0.0% | 20.00 / 20.00 |
| ciciot2023 | t2s_exit | 5 | 595.5 (180.3) | 581.1 (187.7) | 140.1 (179.4) | 132.9 (185.7) | 27.2% | 14.75 / 14.57 |
| ciciot2023 | vanilla | 5 | 455.4 (6.6) | 448.2 (7.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0% | 25.00 / 25.00 |

## The two five-seed energy claims

- **edgeiiot T10**: paired difference -110.8 nJ on the sample, -119.0 nJ prevalence-weighted; traffic-mix share of the sample difference 8.2 nJ; per seed (sample / prevalence): 42: -91.8 / -99.7, 43: -93.1 / -101.2, 44: -134.9 / -142.0, 45: -143.6 / -152.5, 46: -90.5 / -99.4
- **ciciomt2024 t2s_exit**: paired difference -104.2 nJ on the sample, -107.4 nJ prevalence-weighted; traffic-mix share of the sample difference 3.2 nJ; per seed (sample / prevalence): 42: -176.6 / -179.1, 43: -164.2 / -171.3, 44: -192.2 / -195.8, 45: 161.0 / 165.8, 46: -148.8 / -156.4
  - gate admitted on 4 of 5 seeds; admitted seeds only: -170.4 nJ (sample), -175.6 nJ (prevalence); gate-disabled seeds: 45 (161.0 nJ)

Where the gate is evaluated and what is charged: the C1 gate (per-step temperature, then the maximum class probability against `gate_c1`) is evaluated on the host from the float student's per-step prefix outputs (`results/prefix_outputs/<ds>/treespike/seed_<S>/test_prefix.npz`); the chip then runs that flow for the gate's window. SANA-FE (`src/sanafe_plugin/build/loihi_delayed.yaml`, the bundled Loihi prices) charges only the chip, for the executed steps plus one fill/drain step per layer: every mapped neuron's access (51.2 pJ) and update (21.6 pJ) each step (force_update), each output spike (69.3 pJ) and outgoing message (111 pJ), each synaptic event (35.5 pJ per dense connection of a spiking input; zero weights are mapped and charged), and network-on-chip hops (3.0 to 4.2 pJ). The gate's computation, the host encoder and input transfer are not charged.
