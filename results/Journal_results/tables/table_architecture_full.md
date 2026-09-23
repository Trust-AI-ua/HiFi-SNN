### 4-way model anatomy - classes, architecture, parameters, spikes, timing, energy

Mean +/- std over 5 seeds. DNN-Tree is the dense iso-architecture twin: it emits no spikes, so its cell reports **activated neurons** per sample (nonzero hidden ReLU outputs of the int8 model, marked `(act.)`) - the dense analog, but not interchangeable with spikes (spikes accumulate over T steps). **TTFS** = mean encoder time-to-first-spike (steps/T; spiking channels only). **Exit** = mean early-exit timestep (steps/T; DNN-Tree is one-shot, TASNN has no early exit). **Rel.E** = compute energy relative to Tree-to-Spike on the same dataset (n x = n times ours). Architecture uses the seed-42 encoder width; params are the across-seed mean (the tree-guided encoder width shifts by a few channels per seed).

| Dataset | Classes | Model | Architecture (in - hidden - out) | Params | Storage KB (fp32/int8) | Spikes/sample | TTFS | Exit | Rel.E |
|---|--:|---|---|--:|--:|--:|--:|--:|--:|
| KDDCup99 | 5 | Vanilla SNN | 119 - 128 - 5 (LIF, T=25) | 32.5K | 127.0 / 31.8 | 237 +/- 51.9 | 4.7 +/- 0.0 /25 | 4.9 +/- 0.3 /25 | 0.31x |
| KDDCup99 | 5 | Tree-to-Spike | 64 - 64 - 64 - 5 (LIF, T=20) | 9.0K | 40.4 / 13.9 (incl. enc.) | 169 +/- 14.8 | 16.1 +/- 0.1 /20 | 1.3 +/- 0.1 /20 | 1.00x |
| KDDCup99 | 5 | DNN-Tree | 64 - 64 - 64 - 5 (ReLU) | 9.0K | 40.4 / 13.9 (incl. enc.) | 36 +/- 3.8 (act.) | -- | -- | 5.42x |
| KDDCup99 | 5 | TASNN | 119 - Tr(d=128x3) - STC64 - 128 - 64 - 5 (T=16) | 566.0K | 2,210.8 / 552.7 | 9 +/- 1.1 | -- | -- | 36kx (analytic) |
| NSL-KDD | 5 | Vanilla SNN | 122 - 128 - 5 (LIF, T=25) | 32.9K | 128.5 / 32.1 | 359 +/- 11.1 | 7.7 +/- 0.0 /25 | 8.3 +/- 0.6 /25 | 0.32x |
| NSL-KDD | 5 | Tree-to-Spike | 65 - 64 - 64 - 5 (LIF, T=20) | 8.9K | 40.8 / 14.6 (incl. enc.) | 296 +/- 28.1 | 16.4 +/- 0.1 /20 | 2.5 +/- 0.4 /20 | 1.00x |
| NSL-KDD | 5 | DNN-Tree | 65 - 64 - 64 - 5 (ReLU) | 8.9K | 40.8 / 14.6 (incl. enc.) | 58 +/- 3.6 (act.) | -- | -- | 4.20x |
| NSL-KDD | 5 | TASNN | 122 - Tr(d=128x3) - STC64 - 128 - 64 - 5 (T=16) | 569.0K | 2,222.8 / 555.7 | 19 +/- 1.6 | -- | -- | 29kx (analytic) |
| CTU-13 | 2 | Vanilla SNN | 37 - 128 - 2 (LIF, T=25) | 21.6K | 84.5 / 21.1 | 368 +/- 72.6 | 10.2 +/- 0.0 /25 | 4.8 +/- 0.2 /25 | 1.27x |
| CTU-13 | 2 | Tree-to-Spike | 24 - 64 - 64 - 2 (LIF, T=20) | 5.9K | 24.2 / 7.0 (incl. enc.) | 72 +/- 13.8 | 8.5 +/- 0.3 /20 | 1.1 +/- 0.1 /20 | 1.00x |
| CTU-13 | 2 | DNN-Tree | 24 - 64 - 64 - 2 (ReLU) | 5.9K | 24.2 / 7.0 (incl. enc.) | 40 +/- 4.2 (act.) | -- | -- | 10.1x |
| CTU-13 | 2 | TASNN | 37 - Tr(d=128x3) - STC64 - 128 - 64 - 2 (T=16) | 481.3K | 1,880.2 / 470.0 | 11 +/- 1.8 | -- | -- | 29kx (analytic) |
| ISCX2012 | 2 | Vanilla SNN | 175 - 128 - 2 (LIF, T=25) | 39.3K | 153.5 / 38.4 | 107 +/- 34.6 | 7.8 +/- 0.1 /25 | 3.7 +/- 0.6 /25 | 0.20x |
| ISCX2012 | 2 | Tree-to-Spike | 51 - 64 - 64 - 2 (LIF, T=20) | 7.7K | 32.7 / 10.1 (incl. enc.) | 127 +/- 21.0 | 16.0 +/- 0.1 /20 | 1.8 +/- 0.2 /20 | 1.00x |
| ISCX2012 | 2 | DNN-Tree | 51 - 64 - 64 - 2 (ReLU) | 7.7K | 32.7 / 10.1 (incl. enc.) | 19 +/- 1.6 (act.) | -- | -- | 4.96x |
| ISCX2012 | 2 | TASNN | 175 - Tr(d=128x3) - STC64 - 128 - 64 - 2 (T=16) | 622.8K | 2,432.7 / 608.2 | 2 +/- 0.3 | -- | -- | 61kx (analytic) |
| CIC-IDS2017 | 5 | Vanilla SNN | 78 - 128 - 5 (LIF, T=25) | 27.3K | 106.5 / 26.6 | 353 +/- 16.9 | 13.4 +/- 0.0 /25 | 10.6 +/- 0.5 /25 | 0.57x |
| CIC-IDS2017 | 5 | Tree-to-Spike | 68 - 64 - 64 - 5 (LIF, T=20) | 8.5K | 43.4 / 18.5 (incl. enc.) | 301 +/- 42.8 | 14.4 +/- 0.2 /20 | 7.5 +/- 0.9 /20 | 1.00x |
| CIC-IDS2017 | 5 | DNN-Tree | 68 - 64 - 64 - 5 (ReLU) | 8.5K | 43.4 / 18.5 (incl. enc.) | 26 +/- 1.1 (act.) | -- | -- | 3.56x |
| CIC-IDS2017 | 5 | TASNN | 78 - Tr(d=128x3) - STC64 - 128 - 64 - 5 (T=16) | 523.9K | 2,046.6 / 511.6 | 7 +/- 1.1 | -- | -- | 15kx (analytic) |
| CSE-CIC-2018 | 5 | Vanilla SNN | 78 - 128 - 5 (LIF, T=25) | 27.3K | 106.5 / 26.6 | 147 +/- 22.0 | -- | 12.6 +/- 1.1 /25 | 0.38x |
| CSE-CIC-2018 | 5 | Tree-to-Spike | 63 - 64 - 64 - 5 (LIF, T=20) | 8.6K | 50.0 / 24.8 (incl. enc.) | 319 +/- 51.6 | -- | 16.0 +/- 1.1 /20 | 1.00x |
| CSE-CIC-2018 | 5 | DNN-Tree | 63 - 64 - 64 - 5 (ReLU) | 8.6K | 50.0 / 24.8 (incl. enc.) | --- | -- | -- | 3.16x |
| CSE-CIC-2018 | 5 | TASNN | 78 - Tr(d=128x3) - STC64 - 128 - 64 - 5 (T=16) | 523.9K | 2,046.6 / 511.6 | 9 +/- 1.1 | -- | -- | 14kx (analytic) |
| UNSW-NB15 | 10 | Vanilla SNN | 196 - 128 - 10 (LIF, T=25) | 43.0K | 168.0 / 42.0 | 197 +/- 20.8 | 11.0 +/- 0.0 /25 | 16.5 +/- 0.9 /25 | 0.19x |
| UNSW-NB15 | 10 | Tree-to-Spike | 157 - 64 - 64 - 10 (LIF, T=20) | 14.9K | 73.5 / 30.0 (incl. enc.) | 538 +/- 44.3 | 17.4 +/- 0.1 /20 | 9.1 +/- 1.7 /20 | 1.00x |
| UNSW-NB15 | 10 | DNN-Tree | 157 - 64 - 64 - 10 (ReLU) | 14.9K | 73.5 / 30.0 (incl. enc.) | 51 +/- 2.6 (act.) | -- | -- | 3.07x |
| UNSW-NB15 | 10 | TASNN | 196 - Tr(d=128x3) - STC64 - 128 - 64 - 10 (T=16) | 645.8K | 2,522.8 / 630.7 | 37 +/- 2.9 | -- | -- | 23kx (analytic) |
| CICIoT2023 | 8 | Vanilla SNN | 46 - 128 - 8 (LIF, T=25) | 23.6K | 92.0 / 23.0 | 217 +/- 16.4 | 11.2 +/- 0.0 /25 | 15.6 +/- 0.5 /25 | 0.30x |
| CICIoT2023 | 8 | Tree-to-Spike | 37 - 64 - 64 - 8 (LIF, T=20) | 7.2K | 42.0 / 20.9 (incl. enc.) | 352 +/- 51.7 | 14.8 +/- 0.2 /20 | 4.7 +/- 1.8 /20 | 1.00x |
| CICIoT2023 | 8 | DNN-Tree | 37 - 64 - 64 - 8 (ReLU) | 7.2K | 42.0 / 20.9 (incl. enc.) | 31 +/- 3.5 (act.) | -- | -- | 2.82x |
| CICIoT2023 | 8 | TASNN | 46 - Tr(d=128x3) - STC64 - 128 - 64 - 8 (T=16) | 491.7K | 1,920.7 / 480.2 | 13 +/- 1.1 | -- | -- | 8kx (analytic) |
| CICIoMT2024 | 6 | Vanilla SNN | 45 - 128 - 6 (LIF, T=25) | 23.2K | 90.5 / 22.6 | 171 +/- 15.4 | 10.2 +/- 0.0 /25 | 21.3 +/- 2.7 /25 | 0.37x |
| CICIoMT2024 | 6 | Tree-to-Spike | 37 - 64 - 64 - 6 (LIF, T=20) | 7.0K | 40.7 / 20.2 (incl. enc.) | 208 +/- 16.3 | 14.5 +/- 0.3 /20 | 4.8 +/- 1.6 /20 | 1.00x |
| CICIoMT2024 | 6 | DNN-Tree | 37 - 64 - 64 - 6 (ReLU) | 7.0K | 40.7 / 20.2 (incl. enc.) | 33 +/- 3.5 (act.) | -- | -- | 4.20x |
| CICIoMT2024 | 6 | TASNN | 45 - Tr(d=128x3) - STC64 - 128 - 64 - 6 (T=16) | 490.3K | 1,915.2 / 478.8 | 12 +/- 3.1 | -- | -- | 12kx (analytic) |
| Edge-IIoTset | 14 | Vanilla SNN | 64 - 128 - 14 (LIF, T=25) | 26.6K | 104.1 / 26.0 | 345 +/- 33.9 | -- | 15.4 +/- 1.2 /25 | 0.55x |
| Edge-IIoTset | 14 | Tree-to-Spike | 38 - 64 - 64 - 14 (LIF, T=20) | 7.5K | 32.9 / 10.9 (incl. enc.) | 269 +/- 7.1 | -- | 9.3 +/- 2.6 /20 | 1.00x |
| Edge-IIoTset | 14 | DNN-Tree | 38 - 64 - 64 - 14 (ReLU) | 7.5K | 32.9 / 10.9 (incl. enc.) | --- | -- | -- | 3.58x |
| Edge-IIoTset | 14 | TASNN | 64 - Tr(d=128x3) - STC64 - 128 - 64 - 14 (T=16) | 511.3K | 1,997.4 / 499.3 | 15 +/- 0.5 | -- | -- | 14kx (analytic) |
| TON-IoT | 10 | Vanilla SNN | 115 - 128 - 10 (LIF, T=25) | 32.6K | 127.5 / 31.9 | 336 +/- 12.0 | 3.6 +/- 0.0 /25 | 14.0 +/- 0.7 /25 | 0.33x |
| TON-IoT | 10 | Tree-to-Spike | 70 - 64 - 64 - 10 (LIF, T=20) | 9.5K | 42.7 / 14.8 (incl. enc.) | 597 +/- 41.2 | 14.7 +/- 0.3 /20 | 9.0 +/- 1.3 /20 | 1.00x |
| TON-IoT | 10 | DNN-Tree | 70 - 64 - 64 - 10 (ReLU) | 9.5K | 42.7 / 14.8 (incl. enc.) | 62 +/- 3.0 (act.) | -- | -- | 2.66x |
| TON-IoT | 10 | TASNN | 115 - Tr(d=128x3) - STC64 - 128 - 64 - 10 (T=16) | 562.8K | 2,198.5 / 549.6 | 38 +/- 2.4 | -- | -- | 16kx (analytic) |
