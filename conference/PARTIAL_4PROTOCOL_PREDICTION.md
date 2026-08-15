# Pre-committed prediction: four-protocol rehearsal of `protocol_shift.py`

**Written and committed before the run. No four-protocol statistic existed at the
time this file was frozen.**

## Status of the run

`PARTIAL_4PROTOCOL_REHEARSAL`. This is a **pipeline rehearsal**, not a result.

- It exists to find bugs in `scripts/protocol_shift.py` now rather than on the
  afternoon the complete data lands.
- **Nothing from it enters the manuscript.**
- **Kendall's tau is not computed on it.** Tau is registered in
  `PREREGISTRATION_ADDENDUM.md` §4 at 25 blocks (5 protocols x 5 seeds). A
  four-protocol value is not a weaker version of the registered quantity; it is
  a different quantity with no registered status. Ranking and encoding
  aggregation only.

## Why the missing protocol biases the answer, and in which direction

CIC-IDS2017 is not a random fifth protocol. On the two quantities that matter
here it is the extreme member of the set:

| protocol      | timing premium (pp) | latency − rate (LP, pp) |
|---------------|--------------------:|------------------------:|
| CTU-13 f0     |               +54.5 |                   +1.09 |
| CIC-IDS2017   |               +30.0 |                   +5.65 |
| NSL-KDD       |                +1.2 |                   +3.75 |
| KDDCup99      |                −0.1 |                   +2.20 |
| CTU-13 f1     |                −9.0 |                   −0.48 |

CIC has the second-largest timing premium and **the largest latency-over-rate
margin of the five**. Excluding it drops the mean timing premium from 15.32 to
11.65 and removes latency's best protocol from the aggregation.

## The prediction

**Adding CIC-IDS2017 should move latency up in the encoding aggregation relative
to the four-protocol figure.**

Consequences fixed in advance, so neither outcome can be read as confirmation
after the fact:

1. **If the four-protocol result puts rate ahead of latency**, an unknown
   fraction of that ordering is CIC's absence. The four-protocol number does not
   license drafting the "rate leads" branch as settled. The five-protocol number
   is the one that counts.
2. **If adding CIC does not move latency up**, the prediction is wrong. That is
   informative in its own right — it would mean the encoding aggregation is not
   being carried by the protocol with the largest per-protocol margin — and it
   earns a sentence.
3. **If the four-protocol result already puts latency ahead**, adding CIC is
   expected to widen rather than reverse it, and the outcome is uninformative
   about the prediction.

## Registered-quantity firewall

The rehearsal writes to `results/analysis/protocol_shift_partial4/`, which is
not the path `finalize.py` reads. The registered five-protocol run writes to
`results/analysis/protocol_shift/` and overwrites nothing produced here.

Frozen 2026-08-15, before `scripts/protocol_shift.py --partial` was executed.
