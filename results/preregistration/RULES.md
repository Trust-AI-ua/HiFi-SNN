# Pre-registered decision rules: paper labels

The paper's requirements are R1 to R6. The pre-registered decision rules are labelled P1 to P7
in the paper and its appendix. The pre-registration files and scripts keep their internal
labels (R2, R3, R5 to R9); this table is the only place the two are mapped, and it is
reproduced once in the protocol appendix.

| Paper | Internal | Pre-registration | Decides |
|---|---|---|---|
| P1 | R2 | `2026-11-16_r2_integer_gbdt.md` | whether integer-only execution hurts the spiking model |
| P2 | R3 | `2026-11-16_r2_integer_gbdt.md` | whether the validation-selected compact GBDT wins on more benchmarks |
| P3 | R5 | `2026-11-17_r2_matched_far.md` | whether the false-alarm advantage survives a matched FAR |
| P4 | R6 | `2026-11-17_r2_steps4_6.md` | whether a calibrated early exit is kept |
| P5 | R7 | `2026-11-17_neuro_stream.md` | Lava fixed-point fidelity (agreement >= 99%, macro-F1 change <= 0.5) |
| P6 | R8 | `2026-11-17_neuro_stream.md` | what to do if the reset mismatch breaks P5 |
| P7 | R9 | `2026-11-17_neuro_stream.md` | stream pilot go/no-go (scope of the paper) |

Internal labels R1 and R4 were not used for decision rules.
