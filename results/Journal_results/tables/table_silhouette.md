### Cluster cleanliness - silhouette of each model's penultimate representation (UMAP)

Higher = better-separated classes. Best per row in **bold** (`n/a` = degenerate embedding). Companion to `figures/embeddings_<dataset>.pdf`.

| Dataset | Vanilla SNN | Tree-to-Spike | DNN-Tree | TASNN |
|---|--:|--:|--:|--:|
| KDDCup99 | 0.334 | 0.505 | 0.549 | **0.762** |
| NSL-KDD | 0.129 | 0.216 | 0.288 | **0.402** |
| CTU-13 | 0.326 | 0.436 | 0.780 | **0.963** |
| ISCX2012 | 0.527 | 0.750 | 0.727 | **0.950** |
| CIC-IDS2017 | 0.032 | 0.431 | 0.308 | **0.777** |
| CSE-CIC-2018 | 0.072 | 0.136 | -0.037 | **0.361** |
| UNSW-NB15 | -0.020 | 0.093 | 0.089 | **0.261** |
| CICIoT2023 | 0.008 | 0.183 | **0.378** | 0.069 |
| CICIoMT2024 | 0.080 | 0.374 | **0.564** | 0.128 |
| Edge-IIoTset | 0.429 | 0.646 | 0.440 | **0.868** |
| TON-IoT | 0.162 | 0.205 | 0.266 | **0.668** |
| **Mean** | 0.189 | 0.362 | 0.396 | **0.564** |
