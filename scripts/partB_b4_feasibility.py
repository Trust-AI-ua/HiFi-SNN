#!/usr/bin/env python3
"""B4 feasibility (no training), pre-registered in Part B.

(b) TON-IoT host-disjoint split. Groups are source IPs of train_test_network.csv (dropped
as a feature, kept here only to group). Per seed, a seeded permutation of hosts assigns
hosts to test until the test set holds at least 20% of rows (the paper's test fraction);
validation is 10% of the remaining rows, stratified, as in the loader. Feasible only if
every class keeps training support on every seed; otherwise the classes that lose it are
reported per seed and nothing runs on that split.

(c) CICIoMT2024 provider split: the documentary evidence is recorded in summary.json with
its sources and limits; the exact-duplicate rate between the provider partitions comes
from the B5 run (results/journal_partB/B5/).

    python scripts/partB_b4_feasibility.py
Output: results/journal_partB/B4/<run_id>/{ton_host_split.csv,ton_hosts_by_class.csv,summary.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

TEST_FRACTION, VAL_FRACTION = 0.2, 0.1


def main():
    from sklearn.model_selection import train_test_split
    st = C.stamp("B4")
    out = C.run_dir(st)
    raw = pd.read_csv(C.REPO / "dataset/ton-iot/train_test_network.csv", encoding="utf-8-sig",
                      usecols=["src_ip", "type"], low_memory=False)
    y = raw["type"].astype(str).str.strip().to_numpy()
    host = raw["src_ip"].astype(str).to_numpy()
    # guard: the rows line up with the loader's cache (same count, same label sequence)
    z = np.load(C.REPO / "dataset/ton-iot/_preprocessed_cache.npz", allow_pickle=True)
    names = [str(c) for c in z["classes"]]
    if len(z["y"]) != len(y) or not np.array_equal(np.array(names)[z["y"]], y):
        sys.exit("TON-IoT CSV rows do not line up with the loader cache")
    classes = names
    hosts, counts = np.unique(host, return_counts=True)
    byc = pd.crosstab(host, y)
    byc.to_csv(out / "ton_hosts_by_class.csv")
    cls_hosts = {c: int((byc[c] > 0).sum()) for c in classes}
    cls_top_share = {c: float(byc[c].max() / byc[c].sum()) for c in classes}
    rows = []
    for seed in C.SEEDS:
        order = np.random.RandomState(seed).permutation(len(hosts))
        test_hosts, n = set(), 0
        for i in order:
            if n >= TEST_FRACTION * len(y):
                break
            test_hosts.add(hosts[i]); n += counts[i]
        te = np.isin(host, list(test_hosts))
        idx_pool = np.where(~te)[0]
        y_pool = y[idx_pool]
        # classes with too few pool rows cannot be stratified; they are reported, not dropped
        vc = pd.Series(y_pool).value_counts()
        strat = y_pool if (vc >= 2).all() and len(vc) > 1 else None
        tr_idx, va_idx = train_test_split(idx_pool, test_size=VAL_FRACTION, random_state=seed, stratify=strat)
        for c in classes:
            rows.append(dict(seed=seed, cls=c, n_train=int((y[tr_idx] == c).sum()), n_val=int((y[va_idx] == c).sum()),
                             n_test=int((y[te] == c).sum()), test_rows_pct=100 * te.mean(), test_hosts=len(test_hosts)))
    d = pd.DataFrame(rows)
    d.to_csv(out / "ton_host_split.csv", index=False)
    lost = d[d.n_train == 0][["seed", "cls", "n_test"]].to_dict("records")
    no_test = d[d.n_test == 0][["seed", "cls", "n_train"]].to_dict("records")
    summary = dict(
        stamp=st,
        ton_host_disjoint=dict(n_rows=int(len(y)), n_hosts=int(len(hosts)),
                               largest_host_row_share=float(counts.max() / len(y)),
                               hosts_per_class=cls_hosts, top_host_share_per_class=cls_top_share,
                               classes_losing_training_support=lost, classes_absent_from_test=no_test,
                               feasible=not lost,
                               outcome=("feasible: every class keeps training support on every seed" if not lost else
                                        "not feasible: some classes lose all training support; nothing runs on this split")),
        ciciomt2024_provider_split=dict(
            established="The dataset authors split at the level of pcap files: 80% of the pcap files form 'train' and "
                        "20% form 'test' (Dadkhah et al., CICIoMT2024, Internet of Things 2024; the sentence was obtained "
                        "through search results for the paper, which is paywalled and could not be opened here). Test "
                        "flows therefore come from pcap files never used in training.",
            not_established="Whether the pcap files are separate recording sessions or chunks of one recording per "
                            "attack. Several attacks have many numbered train files and two numbered test files "
                            "(for example TCP_IP-DDoS-ICMP 1-8 train, 1-2 test), consistent with either reading. The "
                            "CSVs carry no timestamps.",
            sources=["https://www.unb.ca/cic/datasets/iomt-dataset-2024.html",
                     "https://www.sciencedirect.com/science/article/pii/S2542660524002920",
                     "https://ieee-dataport.org/documents/ciciomt2024wifimqtt"],
            duplicate_rate="see results/journal_partB/B5 (exact feature-vector duplicates between the provider partitions)"))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary["ton_host_disjoint"], indent=1, default=str))
    print(d.pivot_table(index="cls", columns="seed", values="n_train").to_string())


if __name__ == "__main__":
    main()
