#!/usr/bin/env python3
"""Proof that the published Vanilla SNN operation counts double-count the input layer.

Under SINGLE counting, a LeakyParallel SNN's synaptic operations per sample can be
at most
        ceiling = F * H  +  hidden_spikes * K
because latency coding fires each of the F input features at most once (F * H is
the input layer with EVERY feature firing), and the output layer costs one
accumulate per hidden spike per class. A published value above this ceiling cannot
have come from single counting.

Hidden spikes are measured on the test set (first 20,000 rows) with each seed's
own checkpoint. Also reports the recorded value, which must sit BELOW the ceiling.

    python scripts/guard_vanilla_ceiling.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.models import SNN  # noqa: E402
from src.encoding import get_encoder  # noqa: E402
from src.data import get_dataset, DATASET_LOADERS  # noqa: E402
from src.training_treespike import _arrays_from_bundle  # noqa: E402

PAPER = {"ciciot2023": 8799 / 0.9}      # the manuscript's 8,799 pJ at 0.9 pJ per AC


def main():
    ok = True
    for ds in ["ciciot2023", "ciciomt2024", "toniot"]:
        cfg = yaml.safe_load(open(REPO / f"configs/{ds}/SNN_LeakyParallel/latency/config.yaml"))
        tcfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
        H, T = cfg["hidden_size"], int(cfg["num_steps"])
        ceil, rec, pub = [], [], []
        for s in [42, 43, 44, 45, 46]:
            base = REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_{s}"
            if not (base / "checkpoint.pt").exists():
                continue
            params = inspect.signature(DATASET_LOADERS[ds]).parameters
            kw = dict(batch_size=128, val_fraction=0.1, seed=s)
            for o in ("subsample_fraction", "test_fraction", "label_mode"):
                if o in tcfg and o in params:
                    kw[o] = tcfg[o]
            Xte = _arrays_from_bundle(get_dataset(ds, **kw))[4][:20000].astype("float32")
            sd = torch.load(base / "checkpoint.pt", map_location="cpu")
            F, K = sd["lp.rnn.weight_ih_l0"].shape[1], sd["fc.weight"].shape[0]
            m = SNN(num_features=F, hidden_size=H, num_classes=K,
                    neuron_name=cfg["neuron"], num_steps=T)
            m.load_state_dict(sd)
            m.eval()
            with torch.no_grad():
                _, _, hidden, _ = m._output_spike_train(
                    get_encoder(cfg["encoding"])(torch.from_numpy(Xte), T))
            hid = float(hidden) / len(Xte)
            r = json.loads((base / "results.json").read_text())["test_plus"]["spike_stats"]
            ceil.append(F * H + hid * K)
            rec.append(r["synops_per_sample"])
            pub.append(r["synops_per_sample"] + F * H)          # the generators' convention
        c, rc, p = np.mean(ceil), np.mean(rec), PAPER.get(ds, np.mean(pub))
        src = "manuscript" if ds in PAPER else "published convention"
        above = p > c
        ok &= above and rc <= c
        print(f"{ds:12s} ceiling(single count)={c:8.0f}  recorded={rc:7.0f} "
              f"(<= ceiling: {rc <= c})  published={p:7.0f} [{src}]  "
              f"exceeds ceiling by {p - c:7.0f} -> {'DOUBLE COUNT PROVEN' if above else 'not proven'}")
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
