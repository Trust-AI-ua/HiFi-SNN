| step | ciciot2023 | ciciomt2024 | edgeiiot | toniot | IoT mean | sign consistent |
|---|---|---|---|---|---|---|
| 1 architecture+training (and T 25->20) | -1.53 | 3.18 | 1.07 | -2.21 | 0.13 | no |
| 2 weight precision (fp32 -> int8) | -0.24 | 0.25 | -1.44 | -0.22 | -0.41 | no |
| 3 encoder (raw latency -> tree-guided TTFS) | 13.56 | 12.0 | 8.39 | 3.11 | 9.26 | yes, all + |
| 4 teacher-posterior supervision | 0.69 | 1.39 | 0.54 | 0.27 | 0.72 | yes, all + |
