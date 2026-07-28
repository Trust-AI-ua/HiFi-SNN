### 4-way model anatomy - classes, architecture, parameters, spikes, timing, energy

Mean +/- std over 5 seeds. DNN-Tree is the dense iso-architecture twin: it emits no spikes, so its cell reports **activated neurons** per sample (nonzero hidden ReLU outputs of the int8 model, marked `(act.)`) - the dense analog, but not interchangeable with spikes (spikes accumulate over T steps). **TTFS** = mean encoder time-to-first-spike (steps/T; spiking channels only). **Exit** = mean early-exit timestep (steps/T; DNN-Tree is one-shot, TASNN has no early exit). **Rel.E** = compute energy relative to Tree-to-Spike on the same dataset (n x = n times ours). Architecture uses the seed-42 encoder width; params are the across-seed mean (the tree-guided encoder width shifts by a few channels per seed).

| Dataset | Classes | Model | Architecture (in - hidden - out) | Params | Storage KB (fp32/int8) | Spikes/sample | TTFS | Exit | Rel.E |
|---|--:|---|---|--:|--:|--:|--:|--:|--:|
| CICIoT2023 | 8 | Vanilla SNN | 46 - 128 - 8 (LIF, T=25) | 23.6K | 92.0 / 23.0 | 217 +/- 14.7 | 11.2 +/- 0.0 /25 | 15.6 +/- 0.4 /25 | 0.76x |
| CICIoT2023 | 8 | Tree-to-Spike | 37 - 64 - 64 - 8 (LIF, T=20) | 7.2K | 42.0 / 20.9 (incl. enc.) | 352 +/- 46.2 | 14.8 +/- 0.2 /20 | 4.7 +/- 1.6 /20 | 1.00x |
| CICIoT2023 | 8 | DNN-Tree | 37 - 64 - 64 - 8 (ReLU) | 7.2K | 42.0 / 20.9 (incl. enc.) | 31 +/- 3.1 (act.) | -- | -- | 2.82x |
| CICIoT2023 | 8 | TASNN | 46 - Tr(d=128x3) - STC64 - 128 - 64 - 8 (T=16) | 491.7K | 1,920.7 / 480.2 | 13 +/- 1.0 | -- | -- | 8kx (analytic) |
| CICIoMT2024 | 6 | Vanilla SNN | 45 - 128 - 6 (LIF, T=25) | 23.2K | 90.5 / 22.6 | 171 +/- 13.8 | 10.2 +/- 0.0 /25 | 21.3 +/- 2.4 /25 | 1.06x |
| CICIoMT2024 | 6 | Tree-to-Spike | 37 - 64 - 64 - 6 (LIF, T=20) | 7.0K | 40.7 / 20.2 (incl. enc.) | 208 +/- 14.6 | 14.5 +/- 0.2 /20 | 4.8 +/- 1.4 /20 | 1.00x |
| CICIoMT2024 | 6 | DNN-Tree | 37 - 64 - 64 - 6 (ReLU) | 7.0K | 40.7 / 20.2 (incl. enc.) | 33 +/- 3.2 (act.) | -- | -- | 4.20x |
| CICIoMT2024 | 6 | TASNN | 45 - Tr(d=128x3) - STC64 - 128 - 64 - 6 (T=16) | 490.3K | 1,915.2 / 478.8 | 12 +/- 2.8 | -- | -- | 12kx (analytic) |
| Edge-IIoTset | 15 | Vanilla SNN | 95 - 128 - 15 (LIF, T=25) | 30.7K | 120.1 / 30.0 | 235 +/- 28.4 | 3.2 +/- 0.0 /25 | 7.8 +/- 0.2 /25 | 1.75x |
| Edge-IIoTset | 15 | Tree-to-Spike | 39 - 64 - 64 - 15 (LIF, T=20) | 7.8K | 34.0 / 11.1 (incl. enc.) | 227 +/- 18.9 | 14.6 +/- 0.1 /20 | 7.9 +/- 1.2 /20 | 1.00x |
| Edge-IIoTset | 15 | DNN-Tree | 39 - 64 - 64 - 15 (ReLU) | 7.8K | 34.0 / 11.1 (incl. enc.) | 29 +/- 3.8 (act.) | -- | -- | 4.03x |
| Edge-IIoTset | 15 | TASNN | 95 - Tr(d=128x3) - STC64 - 128 - 64 - 15 (T=16) | 543.3K | 2,122.2 / 530.6 | 12 +/- 0.4 | -- | -- | 24kx (analytic) |
| TON-IoT | 10 | Vanilla SNN | 115 - 128 - 10 (LIF, T=25) | 32.6K | 127.5 / 31.9 | 336 +/- 10.8 | 3.6 +/- 0.0 /25 | 14.0 +/- 0.6 /25 | 1.14x |
| TON-IoT | 10 | Tree-to-Spike | 70 - 64 - 64 - 10 (LIF, T=20) | 9.5K | 42.7 / 14.8 (incl. enc.) | 597 +/- 36.9 | 14.7 +/- 0.2 /20 | 9.0 +/- 1.2 /20 | 1.00x |
| TON-IoT | 10 | DNN-Tree | 70 - 64 - 64 - 10 (ReLU) | 9.5K | 42.7 / 14.8 (incl. enc.) | 62 +/- 2.6 (act.) | -- | -- | 2.66x |
| TON-IoT | 10 | TASNN | 115 - Tr(d=128x3) - STC64 - 128 - 64 - 10 (T=16) | 562.8K | 2,198.5 / 549.6 | 38 +/- 2.1 | -- | -- | 16kx (analytic) |
