# TASNN — competitor baseline (Transformer-Augmented Spiking Neural Network)

This directory holds the per-dataset run configs for **TASNN**, the strongest
recent hybrid SNN-NIDS competitor, implemented here as a **fair baseline** to
position our **Tree-to-Spike** model against.

- **Model:** [`src/models/tasnn.py`](../../src/models/tasnn.py)
- **Trainer:** [`src/training_tasnn.py`](../../src/training_tasnn.py)
- **Runner:** `scripts/run_tasnn.py` · `scripts/run_tasnn_tmux.sh`
- **Results:** `results/TASNN/<dataset>/seed_<S>/` → deduped into `results/summary_tasnn.csv`

## Source

> *Energy-efficient intrusion detection with a protocol-aware transformer-spiking
> hybrid model* (**TASNN**), **Scientific Reports** (2026) **16**:7095.
> DOI: [10.1038/s41598-026-37367-4](https://doi.org/10.1038/s41598-026-37367-4)

Our implementation ports the **learned core** of the authors' reference release
(`tasnn.py`), re-expressed against this repo's `DatasetBundle` / metrics / energy /
early-exit machinery. The equation numbers below refer to the paper.

## Architecture we implement (the learned core)

Two parallel paths over each record, fused by a cross-modal gate and read out by
two heads whose softmaxes are ensembled at inference:

```
 x ∈ [0,1]^F  (our shared features)
   ├─ transformer path ─ FlowToken Transformer Encoder (FTE) ──────────► h_TR ─► head_tr ─┐
   │                        per-feature tokens + [CLS] + MHSA (Eq.13)        │            │
   │                                                                          ▼            ▼
   └─ spike path ─ MASE (TTFS+population, Eq.10-11) ─► STC (causal conv, Eq.15)        ensemble
                      ─► × XMG gate g=σ(W_g·h_TR) (Eq.19) ─► LIF×2 w/ lateral          (softmax
                         inhibition + surrogate (Eq.16-17) ─► readout ─► head_snn ──►  mean)
```

| Eq. | Component | Status |
|-----|-----------|--------|
| 10–11 | **MASE** — TTFS (learnable latency α) + population coding | ✅ faithful (RF range adapted to [0,1] MinMax inputs) |
| 13 | **FlowToken Transformer Encoder** + `[CLS]` | ✅ adapted — per-feature tokens (see omissions) |
| 15 | **STC** — causal 1-D temporal convolution | ✅ faithful |
| 16–17 | **LIF** + lateral inhibition + sigmoid-surrogate gradient | ✅ faithful |
| 19 | **XMG** cross-modal gating `g = σ(W_g·h_TR)` | ✅ faithful |
| 24–25 | **Dual heads** (transformer + spiking) | ✅ faithful + softmax-ensemble inference |
| 26 | **Focal loss** | ✅ (γ-focusing on; α class-rebalancing gated — see fairness) |
| 27 | **Cross-modal KD** (KL between heads) | ✅ faithful |
| 29 | **Homeostatic** firing-rate regularizer | ✅ |
| 30 | **Energy** (spike-count) penalty | ✅ |
| 31 | **Gate-entropy** regularizer | ✅ |
| 32 | **Weighted total objective** | ✅ faithful |

Paper defaults are used: `d_model=128`, `heads=4`, `layers=3`, `FFN=256`, LIF
`128→64`, `T=16`, `β=0.9`; loss weights `λ_TR=λ_SNN=1, λ_KD=0.5, λ_homeo=0.1,
λ_energy=1e-3, λ_gate=1e-3`. TASNN is the heavyweight competitor, deliberately *not*
size-matched: about 0.5M parameters against a few thousand for Tree-to-Spike and DNN-Tree
(per-dataset counts in `results/Journal_results/tables/table_architecture_full.md`). The
question is whether a much smaller tree-guided SNN can match it at far lower energy.

## What we omitted — and why (fairness to Tree-to-Spike)

Every model in this repo (Vanilla SNN, **Tree-to-Spike**, DNN-Tree, TASNN) is
fed the **same features** from `src/data` (one-hot + log1p + MinMax→[0,1]), with
**no resampling**, the **same splits**, and the same per-dataset `subsample_fraction`
/ `label_mode`. To keep TASNN on that identical footing we drop the reference's
preprocessing stage and the model pieces that are inseparable from it:

| Ref (Eq.) | Omitted | Reason |
|-----------|---------|--------|
| 1–2 | **PAAN** protocol-aware normalization | we use our shared log1p+MinMax (required by the spike encoders, applied to *all* models) |
| 3–4 | **PRE** port-role binning + protocol embeddings | needs raw ports / protocol semantics our shared feature set doesn't expose |
| 5 | **CPSMOTE-Tomek** rebalancing | **the key one** — class resampling would inflate TASNN's minority-class F1; no other model here resamples |
| 6–9 | **PFR** pseudo-flow reconstruction | needs timestamps / 5-tuples; our records are independent rows |
| 12 | **EDC** eventified delta coding | consumes PFR delta channels |
| — | neighbour flow-tokens, protocol-positional emb., rank-order categorical coding | need PFR micro-flows / PRE embeddings; FTE instead tokenises the record's own features |
| 20–23, 28 | **SAIF / SPS** feature selection + saliency-alignment loss | operate on the privileged feature set, not the forward model |
| 18 | **STDP warm-up** | the reference itself disables it for its reported results |

Net effect: the **learned model is faithful**; the only omissions are the parts that
ride on the privileged preprocessing we exclude — so TASNN competes on **architecture**,
not on a data pipeline our own models never get.

## Two fairness knobs held to our footing

1. **Focal `alpha` (class rebalancing) is OFF** (`class_weighted: false`, the default).
   The paper leaves it on; turning it on would give TASNN a class-balance edge our
   models don't have. The γ hard-example focusing (instance-level, not class-level)
   stays on. Set `class_weighted: true` to match the paper exactly.
2. **`T = 16`** (TASNN's native window) rather than our latency models' `T = 25` —
   faithful to the paper; each model reports energy at its own native `T`.

## Datasets & running

Configs mirror the **Tree-to-Spike** dataset set 1-for-1, with identical
`subsample_fraction` (and `label_mode` where the loader supports it):

| Dataset | subsample | label_mode | Dataset | subsample | label_mode |
|---|--:|---|---|--:|---|
| kddcup99 | full | — | ciciomt2024 | 0.25 | category |
| nslkdd | full | — | ciciot2023 | 0.05 | category |
| cicids2017 | 1.0 | — | unswnb15 | 1.0 | (native) |
| iscxids2012 | 1.0 | — | edgeiiot | 1.0 | full |
| ctu13 | 1.0 | — | toniot | 1.0 | full |
| cse-cic-ids2018 | 0.5 | — | | | |

```bash
# one dataset, multi-seed, in a detached tmux session
./scripts/run_tasnn_tmux.sh nslkdd --seeds 42 43 44 45 46
# or directly
python scripts/run_tasnn.py unswnb15 --seeds 42 43 44 45 46
```

> Note: the transformer makes TASNN markedly slower per epoch than the MLP-based
> models on the large datasets (cse-cic-ids2018 at `0.5` ≈ 8M rows; ciciot2023,
> ciciomt2024). Budget accordingly or lower `subsample_fraction` for a quick pass.
