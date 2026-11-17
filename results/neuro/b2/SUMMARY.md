# B2 stream pilot: results and rule R9 (paper P7)

Means over seeds [42, 43, 44, 45, 46], test split, percent. Matched point: benign threshold tuned on validation to FAR <= 1% for each model, after its gate.

| Model | Target FAR | Macro-F1 | FAR | DR | P2D median (both) | P2D p90 (both) | Ops per packet |
|---|--:|--:|--:|--:|--:|--:|--:|
| compact_gbdt | 0.1% | 81.30 | 0.03 | 100.00 | nan | nan | 2394 |
| compact_gbdt | 0.25% | 81.28 | 0.14 | 100.00 | nan | nan | 2394 |
| compact_gbdt | 0.5% | 81.14 | 0.47 | 100.00 | nan | nan | 2394 |
| compact_gbdt | 1% | 80.95 | 0.98 | 100.00 | nan | nan | 2394 |
| compact_gbdt | 2% | 80.77 | 1.33 | 100.00 | nan | nan | 2394 |
| t2s_stream | 0.1% | 55.16 | 0.01 | 74.19 | 2.0 | 6.0 | 968 |
| t2s_stream | 0.25% | 59.47 | 0.12 | 92.27 | 2.0 | 5.8 | 968 |
| t2s_stream | 0.5% | 61.26 | 0.25 | 96.29 | 2.0 | 5.6 | 968 |
| t2s_stream | 1% | 61.46 | 0.64 | 96.75 | 2.0 | 5.6 | 968 |
| t2s_stream | 2% | 62.56 | 1.30 | 100.00 | 2.0 | 5.4 | 968 |
| tree_per_packet | 0.1% | 81.38 | 0.03 | 100.00 | 1.4 | 4.0 | 10012 |
| tree_per_packet | 0.25% | 81.33 | 0.25 | 100.00 | 1.0 | 4.0 | 10012 |
| tree_per_packet | 0.5% | 81.20 | 0.50 | 100.00 | 1.0 | 4.0 | 10012 |
| tree_per_packet | 1% | 81.00 | 0.99 | 100.00 | 1.0 | 4.0 | 10012 |
| tree_per_packet | 2% | 80.39 | 1.61 | 100.00 | 1.0 | 4.0 | 10012 |

## Rule R9

- Macro-F1, Tree-to-Spike-Stream minus per-packet tree: -19.54 (limit -2.00): outside.
- (a) median packets-to-detection: 2.00 vs 1.00: not lower.
- (b) tree comparisons per packet / SNN SOPs per packet: 10.35 (needs >= 5): met. Synaptic operations and node comparisons are different operations.
- Share of executed packets with no input spikes (SNN): 0.0%.

**Decision: NO-GO.** The paper ships with scope A only.

## Secondary analyses (amendment 3, descriptive; P7 unchanged)

Artifact classes: DDoS_ICMP, DDoS_TCP, Port_Scanning, DDoS_UDP.

### Without the artifact classes (matched point)

| Model | Macro-F1 | FAR | DR | P2D median (both) | P2D p90 (both) | Ops per packet |
|---|--:|--:|--:|--:|--:|--:|
| t2s_stream | 59.31 | 0.64 | 99.37 | 5.0 | 8.2 | 1115 |
| tree_per_packet | 76.47 | 0.99 | 100.00 | 3.4 | 4.0 | 10874 |
| compact_gbdt | 78.88 | 0.98 | 100.00 | 4.2 | 5.4 | 2400 |

### Per attack class: detection rate / median packets-to-detection (matched point)

| Class | t2s_stream | tree_per_packet | compact_gbdt |
|---|--:|--:|--:|
| Backdoor | 99.9 / 2.0 | 100.0 / 2.0 | 100.0 / 2.0 |
| DDoS_HTTP | 99.8 / 2.0 | 100.0 / 2.0 | 100.0 / 2.0 |
| DDoS_ICMP | 100.0 / 1.0 | 100.0 / 1.0 | 100.0 / 1.0 |
| DDoS_TCP | 88.7 / 2.0 | 100.0 / 1.0 | 100.0 / 2.0 |
| DDoS_UDP | 98.0 / 1.0 | 100.0 / 1.0 | 100.0 / 1.0 |
| Fingerprinting | 98.5 / 2.0 | 100.0 / 2.0 | 100.0 / 2.0 |
| Password | 99.2 / 5.0 | 100.0 / 3.8 | 100.0 / 4.4 |
| Port_Scanning | 100.0 / 2.0 | 100.0 / 2.0 | 100.0 / 2.0 |
| Ransomware | 100.0 / 2.0 | 100.0 / 2.0 | 100.0 / 2.0 |
| SQL_injection | 100.0 / 8.6 | 100.0 / 4.0 | 100.0 / 4.0 |
| Uploading | 100.0 / 2.0 | 100.0 / 2.0 | 100.0 / 2.0 |
| Vulnerability_scanner | 99.9 / 12.0 | 100.0 / 4.4 | 100.0 / 5.0 |
| XSS | 99.7 / 12.6 | 100.0 / 3.0 | 100.0 / 3.4 |

### Host-stack check

- First-packet length alone (validation FAR <= 1%): test FAR 0.35, DR 99.99.
- Each model's DR after packet 1 at the matched threshold: t2s_stream 78.04, tree_per_packet 100.00, compact_gbdt 100.00
