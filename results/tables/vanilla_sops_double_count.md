# The Vanilla SNN's input layer is counted twice in the published tables

Found 2026-09-22 while reconciling `src/opcount.py` against the published
figures. Approved and applied 2026-09-23 (this commit): `scripts/benchmark_4way.py`
and `scripts/make_results_table.py` no longer add an input term to the Vanilla SNN;
every table and figure they and `scripts/make_arch_table.py` produce is regenerated.

## What happens

Two model classes in this repo use different internal conventions for
`spike_stats.synops_per_sample`:

| Model | Stored value counts |
|---|---|
| Tree-to-Spike (`src/models/tree_to_spike.py` `forward`) | hidden layers only: `n_spk1·hidden[1] + n_spk2·K` |
| Vanilla SNN (`src/models/snn.py` `predict_early_exit`) | **both** layers: `input_spikes·hidden + hidden_spikes·K` |

The table generators (`scripts/make_results_table.py`, `scripts/benchmark_4way.py`)
add an input-layer term to **both**, documented as "the input->layer-1 term, which
the stored synops_per_sample omits". That is true for Tree-to-Spike and false for
the Vanilla SNN.

The added term is also `num_features · hidden_size`, which assumes every feature
fires. Under latency coding with the epsilon of Eq. 3, most do not.

## Magnitude, seed 42

| Dataset | Features | Firing | Recorded (complete) | Generator adds | Published total | Overstated |
|---|---|---|---|---|---|---|
| CICIoT2023 | 46 | 18.9 (41%) | 3,740 | 5,888 | 9,628 | 2.57x |
| CICIoMT2024 | 45 | -- | 3,196 | 5,760 | 8,956 | 2.80x |
| TON-IoT | 115 | 25.2 (22%) | 5,755 | 14,720 | 20,475 | 3.56x |

Confirmed against the published Rel.E column of the 4-way anatomy table, which is
only reproducible with the extra term: CICIoMT2024 published 1.06x, reproduced
1.06x; TON-IoT published 1.14x, reproduced 1.08x (seed 42 against a five-seed
mean). See `scripts/guard_sops.py`, Guard C.

## Proof: the published value exceeds the single-counting ceiling

Under single counting a LeakyParallel SNN can perform at most
`F * H + hidden_spikes * K` synaptic operations per sample: `F * H` is the input
layer with EVERY feature firing (latency coding fires each at most once), plus one
accumulate per measured hidden spike per class. Five-seed means,
`scripts/guard_vanilla_ceiling.py`:

| Dataset | Ceiling (single count) | Recorded | Published | Over ceiling by |
|---|---|---|---|---|
| CICIoT2023 | 7,355 | 3,889 | 9,777 (manuscript, 8,799 pJ) | 2,422 |
| CICIoMT2024 | 6,632 | 3,077 | 8,837 | 2,205 |
| TON-IoT | 17,376 | 5,895 | 20,615 | 3,239 |

No single-counting path can produce the published values. The recorded values sit
below the ceiling, as they must. Independently, the Phase-1 audit records the
LeakyParallel charging rule as input spikes x hidden width + hidden spikes x output
width, i.e. input-inclusive.

TASNN checked for the same pattern: its analytic formula (`src/tasnn_energy.py`)
adds nothing to a stored value, so there is no double count. Its spiking terms
price every synapse at every step rather than per spike, a dense upper bound worth
about 1% of its total (0.92 of 94.4 M pJ on CICIoT2023). Left as is and disclosed.

## Consequence

The comparison reverses. Tree-to-Spike's own input-inclusive counts are 8,465
(CICIoMT2024) and 18,879 (TON-IoT), so against a correctly counted Vanilla SNN:

Regenerated Rel.E (Vanilla SNN energy / Tree-to-Spike energy, five seeds), before
-> after:

| Dataset | Before | After |
|---|---|---|
| KDDCup99 | 2.12x | 0.31x |
| NSL-KDD | 1.78x | 0.32x |
| CTU-13 | 2.89x | 1.27x |
| ISCX2012 | 3.07x | 0.20x |
| CIC-IDS2017 | 1.40x | 0.57x |
| CSE-CIC-2018 | 1.11x | 0.38x |
| UNSW-NB15 | 1.21x | 0.19x |
| CICIoT2023 | 0.76x | 0.30x |
| CICIoMT2024 | 1.06x | 0.37x |
| Edge-IIoTset | 1.75x | 0.55x |
| TON-IoT | 1.14x | 0.33x |

The Vanilla SNN is now the lowest-arithmetic model on ten of eleven datasets
(CTU-13 is the exception). Mean Vanilla SNN energy across datasets falls from
15,083 to 3,825 pJ at 32 bits.

The conventional SNN performs 2.6x to 3.3x FEWER synaptic operations than
Tree-to-Spike. That is not surprising once stated plainly: it has one hidden
layer against two, and its latency code fires only the features above epsilon,
while tree-guided TTFS fires every channel exactly once.

The claim against the DNN-Tree is unaffected, being MAC against AC at
iso-architecture. The claim against the conventional SNN has to become an
accuracy claim, not an arithmetic one.

## Fix (applied)

One line in each generator: the input term is added only for Tree-to-Spike, whose
stored value omits it. `snn.py` and `tree_to_spike.py` are deliberately NOT changed
to agree, because every stored `results.json` already uses the current conventions
and would then need regenerating. Tree-to-Spike cells in the regenerated tables
change only in best-cell highlighting; no Tree-to-Spike value moved.
