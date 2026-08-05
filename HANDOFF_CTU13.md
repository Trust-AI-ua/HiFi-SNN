# Handoff — Tree-to-Spike CTU-13 leakage experiment

**Written:** 2026-08-05 · **Repo:** `/home/student3/HiSNN` · **Branch:** `main`
**Git HEAD:** `f00112b99dfc7e3de7a338a7935fc767e03b056e` (31 uncommitted entries per
`git status --porcelain`: 5 modified + 26 untracked, this file included — the work below
is uncommitted; commit or stash before any risky operation)

---

## 0. TL;DR for the next session

Do **one** thing first: fix the `causal_random` branch of `diagnostic()` in
`scripts/ctu13_stage1.py`, then resume the remaining 33 of 36 diagnostic runs.
**Do not launch the 300-run CTU-13 model matrix** until those 36 results are reviewed
against the go/no-go rule in §6.

> **The root cause is NOT what was assumed.** The handoff brief anticipated a pandas
> index-alignment bug and proposed a positional-assignment fix. The actual traceback is
> `KeyError: 'SrcAddr'` — `_three_condition.parquet` **does not contain the raw
> identifier columns** that `causal_host_features()` needs. Positional realignment would
> not fix it. See §4 for the real fix.

---

## 1. State of play — what is frozen and must not be touched

| Item | State | Location |
|---|---|---|
| TON-IoT package | **FROZEN, COMPLETE** — 200/200 runs, 0 failures | `results/leakage/` |
| TON-IoT fold manifests | **FROZEN** — 8 files (2 protocols × 4 folds) | `results/leakage/manifests/` |
| CTU-13 timestamp audit | **COMPLETE, PASSES** | `results/ctu13/timestamp_audit.json` |
| CTU-13 scenario folds | **FROZEN, all 4 pass gate** | `results/ctu13/scenario_folds.json` |
| CTU-13 three-condition features | **BUILT** (82.9 MB) | `dataset/ctu13/_three_condition.parquet` |
| 36-run diagnostic | **3 of 36 done, then crashed** | `results/ctu13/stage1.log` |
| CTU-13 model matrix (300 runs) | **NEVER LAUNCHED** — 0 result dirs | — |
| ⚠ `results/ctu13/SNN_LeakyParallel/` | stale phase-1 artifact (Jun 25), **not** matrix output | ignore |
| 700-run sensitivity sweep | **PAUSED at 270/700** — resumable | `results/ladder/` |

Do not modify: frozen scenario folds, timestamp processing, TON-IoT manifests, or any
completed TON-IoT results.

**Sweep resume command** (only after CTU-13 + CICIoMT + early-exit are done):
```bash
cd ~/HiSNN && nohup .venv/bin/python scripts/run_ladder.py --workers 5 \
  >> results/ladder_sens.log 2>&1 &
```
The skip-guard makes it idempotent; the 270 completed runs are reused.

---

## 2. Environment

```
python 3.12.3       torch 2.12.0+cu130 (cuda 13.0)   numpy 2.4.4
pandas 3.0.2        scikit-learn 1.8.0                scipy 1.17.1
xgboost 3.2.0       snntorch 0.9.4
```
Hardware: 1× NVIDIA H200 NVL (143 GB), 16 CPU cores, 125 GB RAM.
Venv: `/home/student3/HiSNN/.venv`. **CPU is the bottleneck** (XGBoost `n_jobs=4` per
run) — 5 concurrent workers is the practical ceiling.

---

## 3. Essential files

| # | File | Role |
|---|---|---|
| 1 | `scripts/ctu13_stage1.py` | audit + scenario MIP + 36-run diagnostic (**has the bug**) |
| 2 | `src/data/ctu13_causal.py` | `causal_host_features()`, `build_conditions()`, `HOST_COLS` |
| 3 | `src/data/ctu13.py` | original loader; `_aggregate_per_source()` = the legacy leak |
| 4 | `results/ctu13/timestamp_audit.json` | audit output |
| 5 | `results/ctu13/scenario_folds.json` | frozen 4-fold scenario assignment |
| 6 | `results/ctu13/stage1.log` | 3 completed diagnostics + full traceback |
| 7 | `dataset/ctu13/_three_condition.parquet` | feature table (schema in §5) |

**Launch command for the 36 diagnostics** (no config file; all settings are in-script):
```bash
cd ~/HiSNN && nohup .venv/bin/python scripts/ctu13_stage1.py \
  > results/ctu13/stage1.log 2>&1 &
```
Diagnostic model: `RandomForestClassifier(n_estimators=40, n_jobs=8, random_state=42)`.

---

## 4. THE BUG — real cause and correct fix

### Traceback (verbatim)
```
Traceback (most recent call last):
  File ".../pandas/core/indexes/base.py", line 3641, in get_loc
    return self._engine.get_loc(casted_key)
  ...
KeyError: 'SrcAddr'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/student3/HiSNN/scripts/ctu13_stage1.py", line 204, in <module>
    r = diagnostic(d, fo["folds"])
  File "/home/student3/HiSNN/scripts/ctu13_stage1.py", line 164, in diagnostic
    ch.append(causal_host_features(g))
  File "/home/student3/HiSNN/src/data/ctu13_causal.py", line 76, in causal_host_features
    src = df["SrcAddr"].to_numpy()
  File ".../pandas/core/frame.py", line 4378, in __getitem__
    indexer = self.columns.get_loc(key)
  File ".../pandas/core/indexes/base.py", line 3648, in get_loc
    raise KeyError(key) from err
KeyError: 'SrcAddr'
```

### Diagnosis
`causal_host_features(df)` requires these raw columns:
`SrcAddr, DstAddr, Dport, Sport, Proto, State, Dur, TotPkts, TotBytes, SrcBytes`.

`_three_condition.parquet` retains only the **feature** columns. Missing:
`SrcAddr`, `DstAddr`, `StartTime`, `_t`.

`build_conditions()` in `src/data/ctu13_causal.py` selects:
```python
base = df[["Dur","TotPkts","TotBytes","SrcBytes","sTos","dTos",
           "Sport","Dport","Proto","Dir","State","y","scenario"]]
```
— dropping the grouping key and the timestamp needed for chronological ordering.

Conditions A (`legacy_random`) and C (`causal_scenario`) read precomputed
`legacy__*` / `causal__*` columns and are unaffected. Only condition B
(`causal_random`), which must **recompute** aggregates inside each partition, needs them.

### Correct fix (choose one)

**Option 1 — rebuild the parquet with the raw columns (recommended).**
In `src/data/ctu13_causal.py::build_conditions`, add `SrcAddr`, `DstAddr` and `_t` to
`base`, delete `dataset/ctu13/_three_condition.parquet`, and re-run
`python -m src.data.ctu13_causal` (~3 min; it prints per-scenario progress).
Adds ~2 string columns × 801k rows.

**Option 2 — carry the raw columns in a side table** keyed by `_row_pos`, and join them
in `diagnostic()` before calling `causal_host_features`.

### Then also fix the positional assignment
The original brief's concern is still valid as a *latent* second bug — the current code
does `ch.reindex(d.index[part])` after a `groupby`, which is fragile. Apply the
positional pattern once the KeyError is resolved:

```python
frame = frame.reset_index(drop=True)
frame["_row_pos"] = np.arange(len(frame), dtype=np.int64)
# ... split into partitions carrying _row_pos ...
H = np.full((len(frame), len(HOST_COLS)), np.nan, dtype=np.float64)
seen = np.zeros(len(frame), dtype=np.int64)
for part_idx in (tr, va, te):
    sub = frame.iloc[part_idx].sort_values(["scenario", "_t"], kind="stable")
    for _, g in sub.groupby("scenario", sort=True):
        f = causal_host_features(g)              # index == g.index
        H[g["_row_pos"].to_numpy()] = f.to_numpy()
        seen[g["_row_pos"].to_numpy()] += 1
assert seen.min() == 1 and seen.max() == 1, "every row written exactly once"
assert not np.isnan(H).any(), "no unwritten positions"
```

**Ordering requirement:** `causal_host_features` assumes rows arrive in chronological
order. `build_conditions` sorted by `_t` before writing, so parquet row order is
already chronological *within scenario*; if you re-sort inside a partition you must
sort by `_t`, not by index.

### Smoke test before resuming
```bash
cd ~/HiSNN && .venv/bin/python -c "
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from src.data.ctu13_causal import causal_host_features, HOST_COLS
d = pd.read_parquet('dataset/ctu13/_three_condition.parquet')
g = d[d.scenario==6].head(5000).copy()          # smallest scenario
f = causal_host_features(g)
assert len(f)==len(g) and f.index.equals(g.index)
assert not f.isna().any().any()
print('OK', f.shape, 'first row all-zero (no history):', bool((f.iloc[0]==0).all()))
"
```

---

## 5. Parquet schema — `dataset/ctu13/_three_condition.parquet`

* **rows:** 801,132  · **cols:** 47 · **size:** 82.9 MB
* **index:** `RangeIndex`, unique, contiguous (0..801131)
* **row order:** grouped by scenario ascending; chronological (`_t`) within scenario
* **identifier columns for restoring order:** `scenario` + positional index.
  There is **no** `_row_pos` column yet — add one after `reset_index(drop=True)`.
* **class balance:** `y=1` (Botnet) 444,699 = 55.5%

| Group | Columns | dtype |
|---|---|---|
| flow numerics | `Dur` f64, `TotPkts` i64, `TotBytes` i64, `SrcBytes` i64, `sTos` f64, `dTos` f64, `Sport` f64, `Dport` f64 | |
| flow categoricals | `Proto`, `Dir`, `State` | `str` |
| label / group | `y` i64, `scenario` i64 | |
| legacy aggregates | `legacy__<17 names>` | i64/f64 |
| causal aggregates | `causal__<17 names>` | f64 |

The 17 `HOST_COLS` names (suffixes after `legacy__` / `causal__`):
`src_flow_count, src_unique_dsts, src_unique_dports, src_unique_sports,
src_unique_protos, src_unique_states, src_total_dur, src_mean_dur, src_std_dur,
src_total_pkts, src_mean_pkts, src_total_bytes, src_mean_bytes,
src_total_src_bytes, src_mean_src_bytes, src_port_entropy, src_proto_entropy`

**MISSING (cause of the bug):** `SrcAddr`, `DstAddr`, `StartTime`, `_t`.

---

## 6. Completed results

### Timestamp audit — PASSES
```
total rows 801,132 | parsed 801,132 | invalid 0
all 13 scenarios sort to valid chronological sequences
no scenario pair overlaps in time; resolution 1 µs
acceptance_online_causal: True
```

### Scenario folds — feasible, all 4 pass the gate
Gate: ≥1000 Normal and ≥1000 Botnet in test; ≤60% single-scenario share; 3–4 scenarios
per test fold; ≥6 training scenarios; ≥500 of each class in validation.

| fold | test scenarios | val | train rows | val rows | test rows | test botnet% | max share | pass |
|---|---|---|--:|--:|--:|--:|--:|:--|
| 0 | 2, 4, 8 | 1, 3 | 378,980 | 57,909 | 364,243 | 58.4 | 0.590 | ✅ |
| 1 | 7, 9, 11 | 1, 3 | 532,279 | 57,909 | 210,944 | 54.4 | 0.579 | ✅ |
| 2 | 1, 3, 5, 10 | 0 | 648,869 | 71,348 | 80,915 | 44.9 | 0.372 | ✅ |
| 3 | 0, 6, 12 | 1, 3 | 598,193 | 57,909 | 145,030 | 55.9 | 0.496 | ✅ |

Scenario overlap between train/val/test is **0** in every fold. Folds are deliberately
unequal: scenario 8 holds 26.8% of all flows, so any fold containing it needs ≥44.7% of
the data to satisfy the 60% domination limit — the 24% test target and the domination
limit cannot both bind, and the domination limit was kept (TON-IoT precedent).

### Diagnostic — 3 of 36 complete
| fold | condition | features | macro-F1 | FAR |
|---|---|---|--:|--:|
| 0 | legacy_random | base | 99.99 | 0.01 |
| 0 | legacy_random | host | 100.00 | 0.00 |
| 0 | legacy_random | combined | 100.00 | 0.00 |

**Failed at:** fold 0, condition `causal_random`, feature subset `base` (the first run of
that condition). Folds 1–3 and conditions B/C never executed.

### Prior standalone finding (from an earlier ad-hoc check, not in the 36)
Host aggregates **alone** on a random split: legacy → 100.00 macro-F1,
causal → 99.99. Making aggregation causal does **not** remove separability, which
suggests the ceiling is a **host-associated shortcut** (per-source aggregates encode host
identity, and label is a property of the host) rather than a future-window leak.
Use the wording *host-associated shortcut*, not "label leakage through host identity",
until the feature-subset and host-overlap evidence in §7 supports the stronger claim.

### Go/no-go rule for the full matrix
Launch the 300-run matrix **only if** scenario-disjoint macro-F1 drops ≥1 pp vs
causal-random, **or** scenario-disjoint FAR rises ≥1 pp. If every feature set stays
>99.5 macro-F1 with negligible FAR change, **stop** and report CTU-13 as a null
scenario-shift result.

---

## 7. Queue after the diagnostic (do not reorder)

1. **Review the 36 diagnostic results** against the go/no-go rule.
2. CTU-13 host-identity audit: unique hosts per scenario; hosts in multiple scenarios;
   train/test host overlap per fold; per-host label purity; % of hosts >90/95/99%
   single-class; host-only diagnostic by fold.
3. CICIoMT2024 duplicate-free sensitivity — **already known:** 195 duplicate clusters,
   357 overlapping test rows (0.022%), **100% label agreement**. Remaining: re-evaluate
   existing checkpoints on the duplicate-free test subset (no retraining) and report
   ΔM for macro-F1/MCC/DR/FAR. If |ΔM| < 0.1 pp, state the overlap is immaterial.
4. Validation-selected early-exit sweep, θ ∈ {0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.99};
   pick earliest θ within 1.0 pp of full-window **validation** macro-F1; disable if none
   qualifies; apply once to test.
5. Deployment baselines.
6. Resume the 700-run sweep (§1).

---

## 8. Context the numbers won't convey

* **TON-IoT final claim is protocol-wide degradation, not an architecture ranking.**
  A pilot with one seed per cell showed Tree-to-Spike degrading least; with 5 seeds per
  fold that ordering dissolved. All five models lose 8–10 macro-F1 and gain 12–19 pp FAR
  under source-IP-disjoint evaluation. Do not revive the ranking claim.
* **Only 1 of 4 target-domain datasets supports a new group-disjoint protocol**
  (TON-IoT). CICIoMT2024 has provider capture separation; CICIoT2023 and Edge-IIoTset are
  infeasible (documented in `rebuttal2_response.tex`, `tab:audit`).
* **The paper's central evidence is the encoder attribution**, not generalization or
  efficiency: encoder main effect **+8.32 macro-F1** (95% CI [+3.56, +13.00]), objective
  main effect −0.33 (null), interaction −0.11 (null), from a complete 2×2.
* **Response letters:** `rebuttal_response.tex` (round 1, superseded on affected points),
  `rebuttal2_response.tex` (round 2, governs where they disagree — see its
  `tab:withdrawn`).
* Several assumptions have failed audit in this project (Edge-IIoT chronology, TON-IoT
  host semantics, CICIoMT2024 "zero overlap", and this parquet schema). **Verify before
  asserting.**
