#!/usr/bin/env python3
"""Input-layer operations actually executed by the M8 raw-latency arm.

The arm uses the conventional latency code (encoder_mode raw_vanilla): one channel
per raw feature, and NO spike at or below epsilon = 0.01. Unlike the tree-guided
code, not every channel fires, so the table generators' input term
``channels * hidden[0]`` would overstate this arm, the same error as the Vanilla SNN
double count. The stored synops (hidden layers only) is completed here with the
spikes actually emitted: input spikes per sample = features with level > epsilon,
level = the loader's [0, 1] value clipped, exactly as TreeGuidedSpikeEncoder.encode.

Guard: the encoder's own encode() on a 2,000-row slice must emit exactly this count.

    python scripts/m8_input_spikes.py
Output: results/diagnostics/m8_input_spikes.csv
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
EPS = 0.01


def main():
    from src.data import get_dataset, DATASET_LOADERS
    from src.training_treespike import _arrays_from_bundle
    from src.models import TreeGuidedSpikeEncoder
    rows = []
    for ds in IOT:
        cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
        lp = inspect.signature(DATASET_LOADERS[ds]).parameters
        for s in range(42, 47):
            kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=s)
            for o in ("subsample_fraction", "test_fraction", "label_mode"):
                if o in cfg and o in lp:
                    kw[o] = cfg[o]
            b = get_dataset(ds, **kw)
            Xte = _arrays_from_bundle(b)[4]
            n_in = (np.clip(Xte, 0.0, 1.0) > EPS).sum(1)
            # guard against the encoder itself
            enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
                                         prune_quantile=cfg["prune_quantile"], jitter=cfg["jitter"],
                                         encoder_mode="raw_vanilla").fit(None, Xte.shape[1])
            sp = enc.encode(Xte[:2000])
            if not np.array_equal(sp.sum((0, 2)).numpy(), n_in[:2000]):
                raise SystemExit(f"{ds}/{s}: encode() disagrees with the count; refused")
            r = json.loads((REPO / f"results/encoder_ablation/{ds}/m8_raw/seed_{s}/results.json").read_text())
            if len(Xte) != r["test_plus"].get("n_samples", len(Xte)):
                raise SystemExit(f"{ds}/{s}: test size differs from the M8 run; refused")
            h0 = r["config"]["hidden"][0]
            hid = r["test_plus"]["spike_stats"]["synops_per_sample"]
            rows.append(dict(dataset=ds, seed=s, features=Xte.shape[1],
                             input_spikes=float(n_in.mean()), hidden_sops=hid,
                             total_sops=hid + float(n_in.mean()) * h0,
                             naive_total_sops=hid + Xte.shape[1] * h0,
                             macro_f1=100 * r["test_plus"]["metrics"]["f1_macro"]))
            print(f"{ds}/s{s}: {n_in.mean():.1f} of {Xte.shape[1]} features fire; "
                  f"SOPs {rows[-1]['total_sops']:,.0f} (channels x h0 would give "
                  f"{rows[-1]['naive_total_sops']:,.0f})", flush=True)
    pd.DataFrame(rows).to_csv(REPO / "results/diagnostics/m8_input_spikes.csv", index=False)
    print("[wrote] results/diagnostics/m8_input_spikes.csv")


if __name__ == "__main__":
    main()
