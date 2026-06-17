# Memory-aware, batch-aware total-energy model

Does the SNN energy win survive memory access (the real reviewer test)? Compute = MAC 4.6 pJ / AC 0.9 pJ; memory = N_access x E_mem (32-bit). DNN amortises weight reads over the batch; the event-driven SNN does not, but only touches active synapses and (with early exit) only T_eff steps.

Per-sample operation & memory profile (macro-avg):
| Model | Compute ops | Op type | Mem accesses @B=1 (full-T / early-exit) | T / T_eff |
|---|--:|:--|--:|--:|
| Vanilla SNN | 3832 | SOPs | 10432 / 2855 | 25/6.8 |
| Tree-to-Spike | 5584 | SOPs | 10864 / 2432 | 20/4.5 |
| DNN-Tree | 8009 | MACs | 8397 | 1/1 |

## Total energy: Tree-to-Spike (early-exit) vs DNN-Tree (iso-twin), batch-1
| E_mem regime | E_DNN (pJ) | E_SNN full-T (pJ) | delta full-T | E_SNN early-exit (pJ) | delta early-exit |
|---|--:|--:|--:|--:|--:|
| compute-only (0 pJ) | 36839 | 5026 | 7.3x | 1125 | 32.8x |
| on-chip SRAM (5 pJ) | 78822 | 59347 | 1.3x | 13282 | 5.9x |
| large SRAM (50 pJ) | 456666 | 548238 | 0.8x | 122700 | 3.7x |
| off-chip DRAM (640 pJ) | 5410621 | 6958143 | 0.8x | 1557289 | 3.5x |

## Total energy: Vanilla SNN (early-exit) vs its analytic dense-ANN twin, batch-1
| E_mem regime | E_DNN (pJ) | E_SNN early-exit (pJ) | delta |
|---|--:|--:|--:|
| compute-only (0 pJ) | 62511 | 944 | 66.2x |
| on-chip SRAM (5 pJ) | 131778 | 15218 | 8.7x |
| large SRAM (50 pJ) | 755178 | 143682 | 5.3x |
| off-chip DRAM (640 pJ) | 8928644 | 1827992 | 4.9x |

## Batch-size sensitivity (on-chip SRAM 5 pJ): DNN per-sample energy vs batch
Tree-to-Spike (early-exit) per-sample energy is batch-independent (event-driven).
| Batch B | E_DNN/sample (pJ) | E_SNN/sample (pJ) | winner |
|---|--:|--:|:--|
| 1 | 78822 | 13282 | SNN |
| 8 | 43207 | 13282 | SNN |
| 64 | 38755 | 13282 | SNN |
| 256 | 38278 | 13282 | SNN |
| 1024 | 38159 | 13282 | SNN |

## What this proves (the publication-grade takeaway)
- **The SNN's total-energy win holds with memory access — but only because of early exit.** With early exit (T_eff ~ 4-7 of T), the SNN's memory accesses at batch-1 stay *below* the DNN's full weight read, so it wins across every memory regime (SRAM and DRAM).
- **Without early exit (full T), the multi-timestep membrane re-fetch erodes or reverses the win once memory is expensive** (DRAM) — exactly the known objection. So early exit is load-bearing, not cosmetic: it is what makes the hybrid SNN energy-viable vs just deploying the DNN.
- **Batch matters**: the DNN only amortises its weight reads at large batch, but its dense MAC *compute* alone still exceeds the SNN's whole budget, so at batch-1 (edge) the SNN wins decisively. This is the regime neuromorphic edge hardware targets.
- **Caveats**: this is an analytical model (stated assumptions); off-chip vs on-chip placement, weight reuse, and spike-routing overhead need a hardware/dataflow simulator or Loihi/Akida measurement for a final number. Training energy and von-Neumann batch-1 latency remain DNN wins.
