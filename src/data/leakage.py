"""Leakage-resistant splits for the IoT/IIoT/IoMT primary datasets.

Round-2 reviewer item 1. The production loaders in this package build features by
one-hot encoding over the *whole* table and then take a random stratified split,
so the categorical vocabulary and the split are both computed with knowledge of
the test rows. This module rebuilds the same datasets under disjoint-partition
protocols with strictly train-only fitting, and reports the diagnostics the
reviewer asked for (duplicate counts, per-class split counts, class coverage).

What it does NOT do is pretend a split is possible when the released data cannot
support it. Feasibility, established by inspecting the distributed files:

  Edge-IIoTset  host-disjoint  YES  (`ip.src_host` intact)
                chronological  NO   `frame.time` is corrupted in the released
                                    ML CSV: the Wireshark format
                                    "Dec  1, 2021 11:44:10.081753000 EAT"
                                    contains a comma after the day, which broke
                                    the CSV export. Only " 2021 11:44:10.0817 "
                                    survives -- no month, no day. 94% of values
                                    "parse" but to a nonsense 1982-2075 range
                                    spanning 42 spurious dates. Unusable.
  TON-IoT       host-disjoint  YES  (`src_ip` intact, 51 sources)
                chronological  NO   no timestamp column in the released file
  CICIoMT2024   capture-disj.  YES  already honoured (author train/test dirs)
  CICIoT2023    none           NO   169 Spark part-files of extracted features
                                    only; no IP, device, session or time key

Edge-IIoTset caveat: the testbed has ~4 active sources carrying 94% of rows, so a
host-disjoint split is coarse (effectively hold-one-device-out) and will shift the
class distribution hard. That is reported, not hidden.

Usage
-----
    from src.data.leakage import load_leakage_bundle
    b = load_leakage_bundle("toniot", split_mode="host", seed=42)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.nslkdd import DatasetBundle

REPO = Path(__file__).resolve().parents[2]
UNK = "__unk__"
MIN_VAL_GROUPS = 3   # validation must span several source endpoints

# ----------------------------------------------------------------------------
# raw table builders: features + label + the split key, WITHOUT encoding
# ----------------------------------------------------------------------------
_EDGE_CSV = "Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
_EDGE_DROP = [  # identifier / payload columns dropped as features (see loader)
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
]
_TON_DROP = ["src_ip", "dst_ip", "dns_query", "http_uri", "http_user_agent",
             "ssl_subject", "ssl_issuer"]


def _raw_edgeiiot(data_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str], np.ndarray, Dict]:
    cache = data_dir / "_leakage_raw.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = pd.read_csv(data_dir / _EDGE_CSV, low_memory=False)
        df.to_parquet(cache)
    stats = {"rows_original": int(len(df))}

    groups = df["ip.src_host"].astype(str).to_numpy()
    y_raw = df["Attack_type"].astype(str).str.strip().to_numpy()
    feat = df.drop(columns=[c for c in _EDGE_DROP + ["Attack_label", "Attack_type"]
                            if c in df.columns])

    keep = feat.notna().all(axis=1).to_numpy()
    feat, y_raw, groups = feat[keep], y_raw[keep], groups[keep]
    stats["rows_after_dropna"] = int(len(feat))

    # duplicate census on the FEATURE rows (label included, as the loader does)
    key = pd.util.hash_pandas_object(feat.assign(_y=y_raw), index=False)
    stats["exact_duplicate_rows"] = int(len(key) - key.nunique())
    dedup = ~key.duplicated().to_numpy()
    feat, y_raw, groups = feat[dedup], y_raw[dedup], groups[dedup]
    stats["rows_after_dedup"] = int(len(feat))

    classes = ["Normal"] + sorted(set(y_raw) - {"Normal"})
    y = np.array([classes.index(v) for v in y_raw], dtype=np.int64)
    return feat.reset_index(drop=True), y, classes, groups, stats


def _raw_toniot(data_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str], np.ndarray, Dict]:
    df = pd.read_csv(data_dir / "train_test_network.csv", encoding="utf-8-sig",
                     low_memory=False)
    stats = {"rows_original": int(len(df))}
    groups = df["src_ip"].astype(str).to_numpy()
    y_raw = df["type"].astype(str).str.strip().to_numpy()
    feat = df.drop(columns=[c for c in _TON_DROP + ["type", "label"] if c in df.columns])

    key = pd.util.hash_pandas_object(feat.assign(_y=y_raw), index=False)
    stats["exact_duplicate_rows"] = int(len(key) - key.nunique())
    stats["rows_after_dropna"] = int(len(feat))
    # TON-IoT's production loader does NOT de-duplicate; we keep that behaviour so
    # the comparison isolates the split, and report the count instead.
    stats["rows_after_dedup"] = int(len(feat))

    classes = ["normal"] + sorted(set(y_raw) - {"normal"})
    y = np.array([classes.index(v) for v in y_raw], dtype=np.int64)
    return feat.reset_index(drop=True), y, classes, groups, stats


_BUILDERS = {"edgeiiot": ("dataset/edge-iiotset", _raw_edgeiiot),
             "toniot": ("dataset/ton-iot", _raw_toniot)}


# ----------------------------------------------------------------------------
# disjoint splitting
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# FROZEN fold-validity gate.  These thresholds are fixed here, before any model
# is trained, and no model metric may enter the decision.  Every candidate fold
# is evaluated against them and the pass/fail result with the failing condition
# is reported for all folds, retained or not.
# ----------------------------------------------------------------------------
GATE = {
    "min_test_benign_rows": 500,
    "min_test_attack_rows": 500,
    "max_single_group_share_of_test": 0.90,
    "min_test_attack_subclasses": 2,
    "require_both_labels_in": ("train", "val", "test"),
    "require_zero_cross_partition_exact_duplicates": True,
}


def label_aware_group_bins(groups: np.ndarray, y_bin: np.ndarray, n_folds: int,
                           max_share: float = GATE["max_single_group_share_of_test"]
                           ) -> List[List[str]]:
    """Pack groups into bins balancing BOTH benign and attack support.

    Size-only packing satisfies the domination limit but can still produce a bin
    with almost no benign traffic: on TON-IoT the two largest source IPs are both
    attack-heavy, so pairing them (which fixes max-share) yields 91,940 attack
    rows against 48 benign. Each group is therefore placed in the bin whose
    *worst* filled dimension would grow least, which keeps benign and attack
    support simultaneously viable in every fold.
    """
    uniq = np.unique(groups)
    ben = np.array([int(((groups == g) & (y_bin == 0)).sum()) for g in uniq])
    att = np.array([int(((groups == g) & (y_bin == 1)).sum()) for g in uniq])
    tot = ben + att
    order = np.argsort(-tot, kind="stable")
    bt, at = max(ben.sum() / n_folds, 1), max(att.sum() / n_folds, 1)

    bins: List[List[int]] = [[] for _ in range(n_folds)]
    bl = np.zeros(n_folds); al = np.zeros(n_folds); tl = np.zeros(n_folds)
    for i in order:
        cost = [max((bl[b] + ben[i]) / bt, (al[b] + att[i]) / at) for b in range(n_folds)]
        b = int(np.argmin(cost))
        bins[b].append(i); bl[b] += ben[i]; al[b] += att[i]; tl[b] += tot[i]

    def share(b):
        return max(tot[j] for j in bins[b]) / tl[b] if tl[b] else 1.0

    for _ in range(4 * len(uniq)):                       # max-share repair
        bad = [b for b in range(n_folds) if share(b) > max_share]
        if not bad:
            break
        b = min(bad, key=lambda x: tl[x])
        donors = sorted((d for d in range(n_folds) if d != b and len(bins[d]) > 1),
                        key=lambda d: -tl[d])
        if not donors:
            break
        d = donors[0]
        # prefer donating a group that helps b's weakest dimension
        weak_ben = bl[b] / bt < al[b] / at
        g = max(bins[d], key=lambda i: (ben[i] if weak_ben else att[i], tot[i]))
        bins[d].remove(g); bl[d] -= ben[g]; al[d] -= att[g]; tl[d] -= tot[g]
        bins[b].append(g); bl[b] += ben[g]; al[b] += att[g]; tl[b] += tot[g]
    return [[uniq[i] for i in b] for b in bins]


def constrained_group_bins(groups: np.ndarray, n_folds: int,
                           max_share: float = GATE["max_single_group_share_of_test"]
                           ) -> List[List[str]]:
    """Pack groups into n_folds bins such that no bin is dominated by one group.

    Bins are deliberately allowed to be UNEQUAL in size. On TON-IoT the largest
    source IP holds 29.2% of all rows, so a bin containing it must hold at least
    size/max_share = 32.4% of the dataset to satisfy the domination limit --
    which no equal 4- or 5-way partition can provide. Equal-size packing is
    therefore infeasible by arithmetic, not by bad luck, and the honest fix is
    unequal bins rather than discarding a fold after the fact.
    """
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    bins: List[List[int]] = [[] for _ in range(n_folds)]
    load = np.zeros(n_folds, dtype=np.int64)
    for i in order:
        b = int(np.argmin(load))
        bins[b].append(i)
        load[b] += counts[i]

    def share(b):
        return counts[bins[b]].max() / load[b] if load[b] else 1.0

    for _ in range(4 * len(uniq)):                     # repair pass
        bad = [b for b in range(n_folds) if share(b) > max_share]
        if not bad:
            break
        b = min(bad, key=lambda x: load[x])
        donors = sorted((d for d in range(n_folds) if d != b and len(bins[d]) > 1),
                        key=lambda d: -load[d])
        if not donors:
            break
        d = donors[0]
        g = max(bins[d], key=lambda i: counts[i])       # move donor's largest group
        bins[d].remove(g); load[d] -= counts[g]
        bins[b].append(g); load[b] += counts[g]
    return [[uniq[i] for i in b] for b in bins]


def grouped_kfold_split(groups: np.ndarray, fold: int, n_folds: int = 4,
                        y_bin: np.ndarray | None = None,
                        y_fine: np.ndarray | None = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-disjoint K-fold rotation: test = bin[fold], val = bin[fold+1], rest train.

    Groups are packed largest-first into the currently-smallest bin, so the bins
    have comparable row counts despite highly skewed group sizes. Rotating the
    fold gives genuinely different partitions -- a size-driven greedy assignment
    is deterministic, so varying a random seed alone reproduces the same split
    and would give a false impression of repeated sampling.
    """
    # Folds come from the exact constrained solver (src/data/fold_solver.py) when
    # label information is available: every source group lands in exactly one test
    # fold and every fold satisfies the frozen gate, or the solver certifies
    # infeasibility. The greedy packers are retained only as a fallback.
    if y_bin is not None and y_fine is not None:
        from src.data.fold_solver import solve_folds
        r = solve_folds(groups, y_bin, y_fine, n_folds,
                        GATE["min_test_benign_rows"], GATE["min_test_attack_rows"],
                        GATE["max_single_group_share_of_test"],
                        GATE["min_test_attack_subclasses"])
        if not r["feasible"]:
            raise ValueError(
                f"no valid {n_folds}-fold source-disjoint assignment exists under the "
                f"frozen gate (solver certificate: {r['status']}). Try a different K.")
        bins = r["bins"]
    else:
        bins = (label_aware_group_bins(groups, y_bin, n_folds) if y_bin is not None
                else constrained_group_bins(groups, n_folds))
    te_g = set(bins[fold % n_folds])
    pool = ~np.isin(groups, list(te_g))                 # outer-training pool
    va_g = (inner_val_hosts(groups[pool], y_bin[pool]) if y_bin is not None
            else set(bins[(fold + 1) % n_folds]))
    part = np.array(["train"] * len(groups), dtype=object)
    part[np.isin(groups, list(te_g))] = "test"
    part[np.isin(groups, list(va_g))] = "val"
    return (np.where(part == "train")[0], np.where(part == "val")[0],
            np.where(part == "test")[0])


def greedy_group_split(groups: np.ndarray, test_frac: float, val_frac: float,
                       seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign whole groups to train/val/test, never splitting a group.

    Groups are placed largest-first into whichever partition is furthest below
    its target share (longest-processing-time scheduling). With a handful of
    dominant hosts a purely random assignment produces wildly unbalanced splits;
    this keeps the sizes near target while preserving strict disjointness.
    """
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = np.argsort(-counts, kind="stable")
    # break size ties randomly so different seeds give different partitions
    jitter = rng.permutation(len(uniq))
    order = order[np.argsort(jitter[order], kind="stable")]
    order = order[np.argsort(-counts[order], kind="stable")]

    targets = {"train": 1.0 - test_frac - val_frac, "val": val_frac, "test": test_frac}
    filled = {k: 0.0 for k in targets}
    total = counts.sum()
    assign: Dict[str, str] = {}
    for i in order:
        deficit = {k: targets[k] - filled[k] / total for k in targets}
        part = max(deficit, key=deficit.get)
        assign[uniq[i]] = part
        filled[part] += counts[i]

    part_of = np.array([assign[g] for g in groups])
    return (np.where(part_of == "train")[0], np.where(part_of == "val")[0],
            np.where(part_of == "test")[0])


def inner_val_hosts(groups_pool: np.ndarray, y_bin_pool: np.ndarray,
                    target_frac: float = 0.10) -> set:
    """Choose a source-IP-disjoint validation subset of the outer-training pool.

    Using a whole outer fold as validation is unusable here: the solver's bins are
    deliberately unequal (that is what makes the constraints satisfiable), so a
    bin-as-validation scheme produced budgets ranging from 18% to 39% and one fold
    whose validation set nearly matched its training set. Instead we take roughly
    `target_frac` of the outer-training ROWS, selected as whole source groups so
    that train and validation remain source-disjoint, and require both labels.
    """
    uniq, counts = np.unique(groups_pool, return_counts=True)
    target = target_frac * counts.sum()
    ben = np.array([int(((groups_pool == g) & (y_bin_pool == 0)).sum()) for g in uniq])
    att = np.array([int(((groups_pool == g) & (y_bin_pool == 1)).sum()) for g in uniq])

    # Accumulate SMALLEST-first so validation is spread over several endpoints:
    # taking large groups first lets a single host reach the row target alone, and
    # a one-endpoint validation set is a weak basis for selecting exit thresholds,
    # checkpoints and calibration.
    # Pick a few MID-SIZED groups: largest-first among those small enough that no
    # single one dominates validation. Smallest-first sweeps the long tail into
    # validation and leaves training with only a handful of endpoints; largest-first
    # lets one endpoint satisfy the target alone. This keeps validation to a few
    # groups (~target rows) and leaves the endpoint diversity in training.
    order = np.argsort(-counts, kind="stable")
    cap = target * 0.7
    chosen, tot = [], 0
    for i in order:
        if tot >= target and len(chosen) >= MIN_VAL_GROUPS:
            break
        if counts[i] <= cap:
            chosen.append(i); tot += counts[i]
    if not chosen:
        chosen = [int(np.argmin(counts))]
    # guarantee both labels are represented in validation
    for lab, arr in ((0, ben), (1, att)):
        if sum(arr[i] for i in chosen) == 0:
            cand = [i for i in range(len(uniq)) if arr[i] > 0 and i not in chosen]
            if cand:
                chosen.append(min(cand, key=lambda i: counts[i]))
    return {uniq[i] for i in chosen}


def ensure_train_coverage(groups: np.ndarray, y: np.ndarray, tr: np.ndarray,
                          va: np.ndarray, te: np.ndarray):
    """Move whole groups into train until every class is represented there.

    A purely size-driven group split can leave a class entirely outside train --
    on TON-IoT, `backdoor` and `ransomware` are each launched from a single host.
    Those classes then become unlearnable by construction and macro-F1 is no
    longer comparable with the flow-level protocol, which would confound "harder
    split" with "impossible task". We therefore move the *smallest* group that
    carries each missing class into train, keeping the partition strictly
    group-disjoint, and record which groups were moved.
    """
    tr, va, te = list(tr), list(va), list(te)
    moved: List[str] = []
    for _ in range(len(np.unique(groups))):
        have = set(y[np.array(tr, dtype=int)].tolist()) if tr else set()
        missing = [c for c in np.unique(y) if c not in have]
        if not missing:
            break
        outside = np.array(va + te, dtype=int)
        cand = [g for g in np.unique(groups[outside])
                if set(y[outside][groups[outside] == g].tolist()) & set(missing)]
        if not cand:
            break
        g = min(cand, key=lambda gg: int((groups == gg).sum()))
        moved.append(str(g))
        idx = np.where(groups == g)[0]
        s = set(idx.tolist())
        tr += list(idx)
        va = [i for i in va if i not in s]
        te = [i for i in te if i not in s]
    return (np.array(sorted(tr), dtype=int), np.array(sorted(va), dtype=int),
            np.array(sorted(te), dtype=int), moved)


# ----------------------------------------------------------------------------
# train-only encoding
# ----------------------------------------------------------------------------
def encode_train_only(feat: pd.DataFrame, tr: np.ndarray, va: np.ndarray,
                      te: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """One-hot categoricals with a TRAINING-ONLY vocabulary plus an explicit
    <unk> column per categorical, log1p the numerics, MinMax on train only."""
    obj_cols = feat.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in feat.columns if c not in obj_cols]

    Xnum = feat[num_cols].apply(pd.to_numeric, errors="coerce")
    Xnum = Xnum.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xnum = np.log1p(np.clip(Xnum.to_numpy(), 0.0, None)).astype(np.float32)

    blocks, vocab_sizes, n_unk = [Xnum], {}, {}
    for c in obj_cols:
        col = feat[c].astype(str)
        vocab = sorted(col.iloc[tr].unique())          # TRAIN ONLY
        vocab_sizes[c] = len(vocab)
        index = {v: i for i, v in enumerate(vocab)}
        oh = np.zeros((len(col), len(vocab) + 1), dtype=np.float32)   # +1 = <unk>
        codes = col.map(index)
        known = codes.notna().to_numpy()
        oh[np.arange(len(col))[known], codes[known].astype(int).to_numpy()] = 1.0
        oh[~known, len(vocab)] = 1.0
        n_unk[c] = {"val": int((~known)[va].sum()), "test": int((~known)[te].sum())}
        blocks.append(oh)

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    scaler = MinMaxScaler().fit(X[tr])                  # TRAIN ONLY
    Xtr = scaler.transform(X[tr]).astype(np.float32)
    Xva = np.clip(scaler.transform(X[va]), 0.0, 1.0).astype(np.float32)
    Xte = np.clip(scaler.transform(X[te]), 0.0, 1.0).astype(np.float32)
    meta = {"n_numeric": len(num_cols), "n_categorical": len(obj_cols),
            "train_vocab_sizes": vocab_sizes, "unknown_category_rows": n_unk,
            "n_features": int(X.shape[1])}
    return Xtr, Xva, Xte, meta


# ----------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------
def check_split_validity(y_bin: np.ndarray, y_fine: np.ndarray, groups: np.ndarray,
                         tr, va, te, row_hash: np.ndarray) -> Dict[str, object]:
    """Evaluate one candidate fold against the frozen GATE.

    Uses only partition composition -- never a model metric -- so the decision
    cannot be contaminated by performance. Every condition is reported
    individually so that retained and rejected folds are equally auditable.
    """
    gtr, gva, gte = set(groups[tr]), set(groups[va]), set(groups[te])
    _, gc = np.unique(groups[te], return_counts=True)
    biggest = float(gc.max() / gc.sum()) if len(te) else 1.0

    # exact feature-duplicate rows shared across partitions: a group-disjoint
    # split does not prevent identical records appearing on two different hosts
    htr, hva, hte = set(row_hash[tr]), set(row_hash[va]), set(row_hash[te])
    xdup = {"train_test": len(htr & hte), "train_val": len(htr & hva),
            "val_test": len(hva & hte)}

    c = {
        "no_group_overlap": not (gtr & gva or gtr & gte or gva & gte),
        "both_labels_in_train": len(set(y_bin[tr].tolist())) >= 2,
        "both_labels_in_val": len(set(y_bin[va].tolist())) >= 2,
        "both_labels_in_test": len(set(y_bin[te].tolist())) >= 2,
        "test_benign_rows": int((y_bin[te] == 0).sum()),
        "test_attack_rows": int((y_bin[te] == 1).sum()),
        "test_attack_subclasses": int(len(set(y_fine[te].tolist()) - {0})),
        "largest_test_group_share": round(biggest, 4),
        "cross_partition_exact_duplicate_hashes": xdup,
    }
    failed = []
    if not c["no_group_overlap"]:                                failed.append("group_overlap")
    for k in ("train", "val", "test"):
        if not c[f"both_labels_in_{k}"]:                         failed.append(f"single_label_in_{k}")
    if c["test_benign_rows"] < GATE["min_test_benign_rows"]:     failed.append("test_benign_support")
    if c["test_attack_rows"] < GATE["min_test_attack_rows"]:     failed.append("test_attack_support")
    if c["test_attack_subclasses"] < GATE["min_test_attack_subclasses"]:
        failed.append("test_attack_subclass_diversity")
    if biggest > GATE["max_single_group_share_of_test"]:         failed.append("single_group_dominates_test")
    if GATE["require_zero_cross_partition_exact_duplicates"] and sum(xdup.values()):
        failed.append("cross_partition_exact_duplicates")
    c["failed_conditions"] = failed
    c["passes"] = not failed
    return c


def audit_folds(dataset: str, n_folds: int, label_mode: str = "binary") -> Dict:
    """Evaluate EVERY candidate fold against the gate, before any training."""
    rel, builder = _BUILDERS[dataset]
    feat, y_fine, fine_names, groups, _ = builder(REPO / rel)
    y_bin = (y_fine != 0).astype(np.int64)
    y = y_bin if label_mode == "binary" else y_fine
    row_hash = pd.util.hash_pandas_object(
        feat.assign(_y=y_fine), index=False).to_numpy()
    keep = ~pd.Series(row_hash).duplicated().to_numpy()   # dedup BEFORE assignment
    feat, y_fine, y_bin, groups = (feat[keep].reset_index(drop=True), y_fine[keep],
                                   y_bin[keep], groups[keep])
    y = y_bin if label_mode == "binary" else y_fine
    row_hash = row_hash[keep]
    out_dedup = {"rows_before_dedup": int(len(keep)), "rows_after_dedup": int(keep.sum()),
                 "exact_duplicate_rows_removed": int((~keep).sum())}
    out = {"dataset": dataset, "n_folds": n_folds, "label_mode": label_mode,
           "dedup": out_dedup, "gate": {k: v for k, v in GATE.items()}, "folds": {}}
    for f in range(n_folds):
        tr, va, te = grouped_kfold_split(groups, f, n_folds=n_folds, y_bin=y_bin, y_fine=y_fine)
        tr, va, te, _ = ensure_train_coverage(groups, y, tr, va, te)
        out["folds"][f] = check_split_validity(y_bin, y_fine, groups, tr, va, te, row_hash)
    out["all_folds_pass"] = all(v["passes"] for v in out["folds"].values())
    return out


def load_leakage_bundle(dataset: str, split_mode: str = "host", seed: int = 42,
                        batch_size: int = 128, val_fraction: float = 0.1,
                        test_fraction: float = 0.2, num_workers: int = 0,
                        label_mode: str = "binary", strict: bool = True, n_folds: int = 4,
                        fold_override: int | None = None,
                        report_path: str | None = None, **_ignored) -> DatasetBundle:
    if dataset not in _BUILDERS:
        raise ValueError(f"no leakage-resistant protocol implemented for {dataset!r}; "
                         f"available: {sorted(_BUILDERS)}")
    if split_mode == "flow":
        pass  # binary flow-level control: same dedup and encoding, random split
    elif split_mode != "host":
        raise ValueError(
            f"split_mode={split_mode!r} is not available for {dataset!r}. "
            "Chronological splits are impossible for both Edge-IIoTset (corrupted "
            "frame.time) and TON-IoT (no timestamp column) -- see module docstring.")

    rel, builder = _BUILDERS[dataset]
    feat, y_fine, fine_names, groups, stats = builder(REPO / rel)
    # global exact-duplicate removal BEFORE any partitioning, in every protocol,
    # so the flow-level control and the host-disjoint protocol see the same rows
    _h = pd.util.hash_pandas_object(feat.assign(_y=y_fine), index=False).to_numpy()
    _k = ~pd.Series(_h).duplicated().to_numpy()
    stats["exact_duplicate_rows_removed_before_split"] = int((~_k).sum())
    feat = feat[_k].reset_index(drop=True); y_fine = y_fine[_k]; groups = groups[_k]

    # Binary is the PRIMARY host-disjoint protocol. On TON-IoT the source host is
    # near-collinear with the attack subclass (47/51 hosts >90% single-class,
    # median purity 100%), so a 10-class host-disjoint split is substantially
    # class-disjoint and its macro-F1 is not comparable with the standard 10-class
    # benchmark. Benign-vs-attack has benign traffic on 40 of 51 hosts and is a
    # genuine unseen-host generalization test.
    if label_mode == "binary":
        y = (y_fine != 0).astype(np.int64)
        classes = ["benign", "attack"]
    else:
        y, classes = y_fine, fine_names

    # seeds 42..46 map to folds 0..4 of a group-disjoint 5-fold rotation, so the
    # five runs are genuinely different partitions rather than the same one
    # fold_override decouples the outer partition from the optimisation seed:
    # without it, seed and fold are the same parameter and within-fold seed
    # variability cannot be separated from between-fold effects.
    fold = fold_override if fold_override is not None else (seed - 42) % n_folds

    # Frozen manifests are authoritative. If one exists for this (dataset,
    # protocol, fold) the partition is READ, never recomputed -- five trainers
    # reconstructing folds independently is exactly how silent drift enters, and
    # the manifest and bundle code paths were observed to disagree on the
    # train/validation boundary before this was enforced.
    _mp = MANIFEST_DIR / f"{dataset}_{'flow' if split_mode == 'flow' else 'host'}_fold{fold}.json"
    if _mp.exists():
        _m = json.loads(_mp.read_text())
        tr = np.array(_m["row_indices"]["train"], dtype=int)
        va = np.array(_m["row_indices"]["val"], dtype=int)
        te = np.array(_m["row_indices"]["test"], dtype=int)
        missing_before, moved = [], []
        _rh2 = pd.util.hash_pandas_object(feat.assign(_y=y_fine), index=False).to_numpy()
        validity = check_split_validity((y_fine != 0).astype(np.int64), y_fine,
                                        groups, tr, va, te, _rh2)
        Xtr, Xva, Xte, meta = encode_train_only(feat, tr, va, te)
        ytr, yva, yte = y[tr], y[va], y[te]
        _from_manifest = True
    elif split_mode == "flow":
        # CONTROL: identical rows, dedup and encoding; only the partition differs.
        # Comparing binary-flow against binary-host isolates the partition effect
        # from the change in task granularity.
        # Budget-matched to the host-disjoint protocol: 4 stratified outer folds
        # (25% test) with 10% of the remainder held out for validation, so the two
        # protocols differ in PARTITION ONLY, not in data budget.
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        trp, te = list(skf.split(np.zeros(len(y)), y))[fold]
        from sklearn.model_selection import train_test_split as _tts
        tr, va = _tts(trp, test_size=val_fraction, random_state=seed, stratify=y[trp])
        tr, va, te = np.sort(tr), np.sort(va), np.sort(te)
    else:
        tr, va, te = grouped_kfold_split(groups, fold, n_folds=n_folds,
                                         y_bin=(y_fine != 0).astype(np.int64), y_fine=y_fine)
        _from_manifest = False
        missing_before = [classes[i] for i in np.unique(y) if i not in set(y[tr].tolist())]
        tr, va, te, moved = ensure_train_coverage(groups, y, tr, va, te)
        _rh = pd.util.hash_pandas_object(feat.assign(_y=y_fine), index=False).to_numpy()
        validity = check_split_validity((y_fine != 0).astype(np.int64), y_fine, groups,
                                        tr, va, te, _rh)
        Xtr, Xva, Xte, meta = encode_train_only(feat, tr, va, te)
        ytr, yva, yte = y[tr], y[va], y[te]

    def counts(ys):
        u, c = np.unique(ys, return_counts=True)
        return {classes[int(k)]: int(v) for k, v in zip(u, c)}

    # subclass composition is reported even in binary mode, so the appendix
    # diagnostic can state exactly which attack types each partition contains
    def fine_counts(idx):
        u, c = np.unique(y_fine[idx], return_counts=True)
        return {fine_names[int(k)]: int(v) for k, v in zip(u, c)}

    report = {
        "dataset": dataset, "split_mode": split_mode, "seed": seed,
        "fold": fold, "n_folds": n_folds, "label_mode": label_mode,
        "split_validity": validity,
        "subclass_counts": {"train": fine_counts(tr), "val": fine_counts(va),
                            "test": fine_counts(te)},
        "subclasses_absent_from_test": [fine_names[i] for i in range(len(fine_names))
                                        if i not in set(y_fine[te].tolist())],
        "n_test_subclasses_supported": int(len(set(y_fine[te].tolist()))),
        "n_subclasses_total": len(fine_names),
        "cleaning": stats,
        "n_groups": int(len(np.unique(groups))),
        "groups_per_partition": {
            "train": int(len(np.unique(groups[tr]))),
            "val": int(len(np.unique(groups[va]))),
            "test": int(len(np.unique(groups[te])))},
        "rows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "class_counts": {"train": counts(ytr), "val": counts(yva), "test": counts(yte)},
        "classes_missing_from_train_before_coverage_fix": missing_before,
        "groups_moved_to_train_for_coverage": moved,
        "classes_missing_from_train": [classes[i] for i in range(len(classes))
                                       if i not in set(ytr.tolist())],
        "classes_missing_from_test": [classes[i] for i in range(len(classes))
                                      if i not in set(yte.tolist())],
        "encoding": meta,
        "split_hash": hashlib.sha256(
            np.concatenate([tr, va, te]).tobytes()).hexdigest()[:16],
    }
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True))
    # The gate governs the host-disjoint protocol. The flow-level control is
    # deliberately NOT source-disjoint -- group overlap is its defining property --
    # so it is exempt from the disjointness condition by construction.
    if strict and split_mode == "host" and not validity["passes"]:
        # Fold 0 of TON-IoT puts the single largest host (29% of all rows) alone in
        # the test bin: 100% single-group domination with 48 benign rows. Such a
        # fold cannot support a benign-vs-attack generalisation claim, so it is
        # refused rather than silently reported.
        raise ValueError(
            f"{dataset} fold {fold} fails the split-validity conditions and is not "
            f"reportable: {validity}. Use a different fold, or pass strict=False to "
            f"inspect it deliberately.")
    print(f"  [leakage/{dataset}/{split_mode}] rows {report['rows']} | "
          f"groups {report['groups_per_partition']} | "
          f"missing-from-train {report['classes_missing_from_train']} | "
          f"features {meta['n_features']}", flush=True)

    def _loader(Xs, ys, shuffle):
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys.astype(np.int64)))
        g = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, generator=g, drop_last=False)

    return DatasetBundle(
        train_loader=_loader(Xtr, ytr, True),
        val_loader=_loader(Xva, yva, False),
        test_loader=_loader(Xte, yte, False),
        test21_loader=None,
        num_features=Xtr.shape[1],
        num_classes=len(classes),
        class_names=classes,
        class_counts_train=counts(ytr),
    )


# ----------------------------------------------------------------------------
# Frozen fold manifests
# ----------------------------------------------------------------------------
# Every model consumes these files rather than reconstructing folds itself. The
# trainers keep their own config schemas, but they may not reinterpret the
# partition: the manifest is the single source of truth for which rows and which
# source endpoints belong to train, validation and test.
#
# The inner-validation rule (mid-sized endpoint groups, MIN_VAL_GROUPS>=3, ~10%
# of outer-training rows) was fixed to satisfy STRUCTURAL constraints -- avoiding
# both single-endpoint validation sets and validation partitions holding most of
# the available endpoints -- and was selected before any classifier was evaluated
# on these protocols. No model metric entered its design.
MANIFEST_DIR = REPO / "results" / "leakage" / "manifests"


def freeze_manifest(dataset: str, protocol: str, fold: int, n_folds: int = 4) -> Path:
    """Write (once) the immutable manifest for one outer fold."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / f"{dataset}_{protocol}_fold{fold}.json"
    if out.exists():
        return out

    rel, builder = _BUILDERS[dataset]
    feat, y_fine, fine_names, groups, stats = builder(REPO / rel)
    rh = pd.util.hash_pandas_object(feat.assign(_y=y_fine), index=False).to_numpy()
    keep = ~pd.Series(rh).duplicated().to_numpy()
    feat = feat[keep].reset_index(drop=True)
    y_fine, groups, rh = y_fine[keep], groups[keep], rh[keep]
    y_bin = (y_fine != 0).astype(np.int64)

    if protocol == "flow":
        from sklearn.model_selection import StratifiedKFold, train_test_split
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        trp, te = list(skf.split(np.zeros(len(y_bin)), y_bin))[fold]
        tr, va = train_test_split(trp, test_size=0.10, random_state=42,
                                  stratify=y_bin[trp])
        tr, va, te = np.sort(tr), np.sort(va), np.sort(te)
    else:
        tr, va, te = grouped_kfold_split(groups, fold, n_folds=n_folds,
                                         y_bin=y_bin, y_fine=y_fine)
        tr, va, te, _ = ensure_train_coverage(groups, y_bin, tr, va, te)

    Xtr, Xva, Xte, meta = encode_train_only(feat, tr, va, te)

    def counts(idx, arr, names):
        u, c = np.unique(arr[idx], return_counts=True)
        return {names[int(k)]: int(v) for k, v in zip(u, c)}

    def sha(a):
        return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()[:16]

    man = {
        "dataset": dataset, "protocol": protocol, "fold": fold, "n_folds": n_folds,
        "frozen_utc": pd.Timestamp.utcnow().isoformat(),
        "row_indices": {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()},
        "source_ips": {k: sorted(set(groups[v].tolist()))
                       for k, v in (("train", tr), ("val", va), ("test", te))},
        "n_endpoints": {k: int(len(set(groups[v].tolist())))
                        for k, v in (("train", tr), ("val", va), ("test", te))},
        "partition_hashes": {"train": sha(tr), "val": sha(va), "test": sha(te)},
        "binary_counts": {k: counts(v, y_bin, ["benign", "attack"])
                          for k, v in (("train", tr), ("val", va), ("test", te))},
        "subclass_counts": {k: counts(v, y_fine, fine_names)
                            for k, v in (("train", tr), ("val", va), ("test", te))},
        "feature_width": int(meta["n_features"]),
        "vocabulary_hash": hashlib.sha256(
            json.dumps(meta["train_vocab_sizes"], sort_keys=True).encode()).hexdigest()[:16],
        "unk_activation": meta["unknown_category_rows"],
        "endpoint_overlap": {
            "train_val": len(set(groups[tr]) & set(groups[va])),
            "train_test": len(set(groups[tr]) & set(groups[te])),
            "val_test": len(set(groups[va]) & set(groups[te]))},
        "cross_partition_exact_duplicates": {
            "train_test": len(set(rh[tr]) & set(rh[te])),
            "train_val": len(set(rh[tr]) & set(rh[va])),
            "val_test": len(set(rh[va]) & set(rh[te]))},
        "dedup": {"rows_removed_before_split": int((~keep).sum()),
                  "rows_retained": int(keep.sum())},
        "inner_validation_rule": (
            "mid-sized endpoint groups, >=%d groups, ~10%% of outer-training rows; "
            "fixed on structural grounds before any classifier evaluation"
            % MIN_VAL_GROUPS),
    }
    out.write_text(json.dumps(man, indent=2, sort_keys=True))
    return out


def load_manifest(dataset: str, protocol: str, fold: int) -> Dict:
    p = MANIFEST_DIR / f"{dataset}_{protocol}_fold{fold}.json"
    if not p.exists():
        raise FileNotFoundError(f"fold manifest missing: {p}. Run freeze_manifest first.")
    return json.loads(p.read_text())
