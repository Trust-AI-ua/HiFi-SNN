# Handoff — Tree-to-Spike CTU-13 leakage experiment

**Written:** 2026-08-05 · **Repo:** `/home/student3/HiSNN` · **Branch:** `main`
**Git HEAD:** `f00112b99dfc7e3de7a338a7935fc767e03b056e` (31 uncommitted entries per
`git status --porcelain`: 5 modified + 26 untracked, this file included — the work below
is uncommitted; commit or stash before any risky operation)

---

## 0. TL;DR for the next session

The blocking bug is **fixed and the parquet rebuilt** (2026-08-05, commit `926abed`
checkpointed the pre-fix state). The 36-run diagnostic has been relaunched.
**Do not launch the 300-run CTU-13 model matrix** until those 36 results are reviewed
against the go/no-go rule in §6.

> **The root cause was not what the original brief assumed.** That brief anticipated a
> pandas index-alignment bug and prescribed a positional-assignment fix. The actual
> traceback was `KeyError: 'SrcAddr'` — `_three_condition.parquet` did not retain the raw
> identifier columns `causal_host_features()` needs. Positional realignment would not have
> fixed it. Verified: the `reindex()` call was in fact *correct*, because the frame carries
> a unique contiguous `RangeIndex` (0..801131). It was fragile, not broken.
>
> Two further bugs were found while fixing this, both of which would have bitten on
> resume. See §4.

---

## 1. State of play — what is frozen and must not be touched

| Item | State | Location |
|---|---|---|
| TON-IoT package | **FROZEN, COMPLETE** — 200/200 runs, 0 failures | `results/leakage/` |
| TON-IoT fold manifests | **FROZEN** — 8 files (2 protocols × 4 folds) | `results/leakage/manifests/` |
| CTU-13 timestamp audit | **COMPLETE, PASSES** | `results/ctu13/timestamp_audit.json` |
| CTU-13 scenario folds | **FROZEN, all 4 pass gate** | `results/ctu13/scenario_folds.json` |
| CTU-13 three-condition features | **REBUILT** with raw cols, 52 cols, 95 MB | `dataset/ctu13/_three_condition.parquet` |
| 36-run diagnostic | **36 of 36 COMPLETE** | `results/ctu13/diagnostic36.json` |
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

**Launch command for the 36 diagnostics** (no config file; all settings are in-script;
~4 min total). Note the log filename — do **not** redirect over `stage1.log`, which holds
the original traceback, or `stage1_resume.log`, which holds the completed 36:
```bash
cd ~/HiSNN && nohup .venv/bin/python scripts/ctu13_stage1.py \
  > results/ctu13/stage1_$(date +%Y%m%d_%H%M).log 2>&1 &
```
The frozen audit and fold manifest are reused automatically, so re-running is safe.
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

### Fix as applied (2026-08-05) — all three bugs

**Bug 1 (live, fixed): missing raw columns.** `build_conditions` now retains
`SrcAddr`, `DstAddr`, `StartTime`, `_t`, and assigns `_row_pos = arange(n)` once after
the concat. `METADATA_COLS` and `require_raw_columns()` were added to
`src/data/ctu13_causal.py`. The parquet was deleted and rebuilt
(`python -m src.data.ctu13_causal`, ~2 min): **52 columns, 801,132 rows,
botnet 444,699 — row count and label balance identical to the pre-fix artifact**, so
the rebuild changed no data, only the retained schema.

**Bug 2 (NOT live — hardening only): the positional assignment.** Verified: the
frame carries a unique contiguous `RangeIndex` (0..801131), so the old
`ch.reindex(d.index[part])` was *correct* by label. It was fragile, not wrong. The
positional pattern is now in `_partition_local_causal()` in `scripts/ctu13_stage1.py`
with `seen`/NaN assertions. **Do not expect it to change any metric** — it did not.

**Bug 3 (live, found while fixing, would have broken the resume): stringified
scenario ids.** `scenario_folds()` wrote its manifest with `json.dump(..., default=str)`,
which turned the numpy int64 scenario ids into **strings**: `"test_scenarios": ["2","4","8"]`.
Loading that manifest and calling `d.scenario.isin(["2","4","8"])` on an int64 column
matches **zero rows** — an empty condition-C partition, not an error. It never surfaced
because the original run held the folds in memory as int64 and crashed at fold 0
condition B, before condition C ran. Fixed in two places: `scenario_folds()` now casts
to `int`, and `_load_frozen_folds()` coerces on read. A partition-size assertion against
the frozen manifest now catches any recurrence.

**Also added: frozen-artifact protection.** `__main__` previously recomputed the
timestamp audit and re-solved the fold MIP on every invocation, overwriting both frozen
files. The MIP has a 120 s time limit and no unique optimum, so re-solving could silently
return a different-but-feasible assignment and invalidate every result computed against
it. Both are now reused when present; delete the file deliberately to regenerate.

**Metadata is never a model input.** `SrcAddr` is host identity and the CTU-13 label is
a property of the host, so feeding it to a classifier would manufacture the very shortcut
this experiment measures. `diagnostic()` asserts `METADATA_COLS` is disjoint from every
feature list on every run.

**Ordering requirement:** `causal_host_features` assumes rows arrive in chronological
order. `build_conditions` sorts by `_t` before writing, so parquet row order is already
chronological *within scenario*. A random partition destroys that order, so
`_partition_local_causal` re-sorts by `["scenario", "_t"]` — by time, never by index.
This is why `_t` had to survive into the artifact.

### Smoke test (passes against the rebuilt parquet)
`/tmp/.../scratchpad/smoke_causal_random.py` — schema check, `require_raw_columns`,
`_row_pos == 0..n-1`, metadata/feature disjointness, the original single-scenario check,
and the full `causal_random` path on a real 3-way random partition (3.6 s for 801k rows).

It also asserts condition B is **not a no-op**: a partition-local recompute matches the
whole-scenario `causal__*` columns on only 578/801,132 rows (0.1%), and mean
`src_flow_count` halves (16,070 → 8,459). If those ever converge, condition B is
measuring nothing and the comparison is void.

---

## 5. Parquet schema — `dataset/ctu13/_three_condition.parquet`

* **rows:** 801,132  · **cols:** 52 (was 47 before the fix) · **size:** 95 MB
* **index:** `RangeIndex`, unique, contiguous (0..801131)
* **row order:** grouped by scenario ascending; chronological (`_t`) within scenario
* **`_row_pos`** is present and verified equal to `0..n-1`; scatter partition results
  with it rather than relying on index alignment surviving a groupby.
* **class balance:** `y=1` (Botnet) 444,699 = 55.5% (unchanged by the rebuild)

| Group | Columns | dtype |
|---|---|---|
| flow numerics | `Dur` f64, `TotPkts` i64, `TotBytes` i64, `SrcBytes` i64, `sTos` f64, `dTos` f64, `Sport` f64, `Dport` f64 | |
| flow categoricals | `Proto`, `Dir`, `State` | `str` |
| label / group | `y` i64, `scenario` i64 | |
| legacy aggregates | `legacy__<17 names>` | i64/f64 |
| causal aggregates | `causal__<17 names>` | f64 |
| **metadata — never a model input** | `SrcAddr` str, `DstAddr` str, `StartTime` str, `_t` datetime64[us], `_row_pos` i64 | |

The 17 `HOST_COLS` names (suffixes after `legacy__` / `causal__`):
`src_flow_count, src_unique_dsts, src_unique_dports, src_unique_sports,
src_unique_protos, src_unique_states, src_total_dur, src_mean_dur, src_std_dur,
src_total_pkts, src_mean_pkts, src_total_bytes, src_mean_bytes,
src_total_src_bytes, src_mean_src_bytes, src_port_entropy, src_proto_entropy`

All five metadata columns are now present (`require_raw_columns()` enforces this before
any causal recomputation). Their absence was the original cause of the bug.

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

### Diagnostic — 36 of 36 COMPLETE (2026-08-05)
Raw records: `results/ctu13/diagnostic36.json`. Log: `results/ctu13/stage1_resume.log`
(the pre-fix `stage1.log` is preserved for its traceback). Mean over 4 folds:

| features | legacy_random | causal_random | causal_scenario | Δ_aggregation | Δ_scenario |
|---|--:|--:|--:|--:|--:|
| base F1 | 99.99 | 99.99 | 99.23 | +0.00 | **−0.75** |
| base FAR | 0.02 | 0.02 | 1.12 | +0.00 | **+1.10** |
| host F1 | 100.00 | 95.81 | 70.69 | **−4.19** | **−25.12** |
| host FAR | 0.00 | 4.64 | 12.12 | **+4.64** | **+7.47** |
| combined F1 | 100.00 | 99.31 | 77.77 | −0.69 | **−21.54** |
| combined FAR | 0.00 | 0.04 | 9.50 | +0.04 | **+9.47** |

**The 3 pre-crash runs reproduced exactly** (fold 0 legacy_random 99.99 / 100.00 / 100.00),
confirming the rebuild altered no data.

### ⚠ Report per fold, never the mean — the spread is the finding
`causal_scenario` macro-F1 by fold:

| fold | test scenarios | base | host | combined |
|---|---|--:|--:|--:|
| 0 | 2, 4, 8 | 99.98 | 87.11 | 99.50 |
| 1 | 7, 9, 11 | 99.27 | **25.72** | **27.40** |
| 2 | 1, 3, 5, 10 | 97.70 | 89.61 | 97.63 |
| 3 | 0, 6, 12 | 99.99 | 80.33 | 86.54 |

`combined` spans **27.40 → 99.50**. The 77.77 mean summarises nothing and must not be
quoted as a point estimate. Fold 1 (test = 7, 9, 11) is a near-total collapse and needs
explanation before any claim is written; it is the single most informative cell here.

### Corrects the earlier standalone finding — do not reuse it
An earlier ad-hoc check reported causal host aggregates alone at 99.99 on a random split,
and concluded that *making aggregation causal does not remove separability*. **That check
used the precomputed whole-scenario `causal__*` columns, not a partition-local recompute.**
Condition B, done properly, gives host-only **95.81** (Δ = −4.19, FAR 0.00 → 4.64). The two
differ on 99.9% of rows. So there *is* a measurable aggregation-leak component; the old
"no effect" wording is an artifact of the shortcut and is retired.

The sharper reading, which the base column supports: under a **random** split the ceiling
is carried by the *base flow features* (99.99 in every condition) — no aggregation fix can
move it. Host aggregates only look decisive because they are redundant with an already
saturated signal. Under scenario shift they become an active liability: adding them to base
**costs** 21.5 pp mean macro-F1 (99.23 → 77.77) and raises FAR 1.12 → 9.50. That is
shortcut reliance inverting under distribution shift, and it is a stronger result than the
leak framing.

### Go/no-go rule — TRIGGERED on all three feature sets
Rule: launch **only if** scenario-disjoint macro-F1 drops ≥1 pp vs causal-random **or**
scenario-disjoint FAR rises ≥1 pp; stop if every feature set stays >99.5 macro-F1 with
negligible FAR change.

* `combined`: −21.54 pp F1, +9.47 pp FAR → **GO** (both criteria, by >20×)
* `host`: −25.12 pp F1, +7.47 pp FAR → **GO** (both criteria)
* `base`: −0.75 pp F1 (below threshold) but **+1.10 pp FAR → GO** (FAR criterion)

CTU-13 is **not** a null scenario-shift result. The 300-run matrix is warranted.
**It has still not been launched** — that is a deliberate human decision point, and
fold 1 should be understood first.

---

## 7. Queue after the diagnostic (do not reorder)

1. ~~Review the 36 diagnostic results against the go/no-go rule.~~ **DONE** — triggered
   on all three feature sets (§6). Matrix is warranted but deliberately not launched.
   **Next actual step: explain fold 1** (test = 7, 9, 11), where combined collapses to
   27.40 macro-F1 against 99.50 on fold 0. Until that is understood, no CTU-13 claim
   should be written, and the host-identity audit in step 2 is the natural instrument
   for it.
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
