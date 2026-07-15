#!/usr/bin/env python3
"""Backfill DNN-Tree activated-neuron counts (the dense analog of hidden spikes).

The DNN-Tree runs recorded MACs but not activation statistics. This script
replays each run's ``checkpoint.pt`` (eval only, no retraining): it rebuilds the
deterministic split + per-seed XGBoost teacher + tree-guided encoder exactly as
``src/training_dnn.py`` did, applies the same int8 fake-quant when the run used
it, and counts the mean number of NONZERO ReLU outputs in the two hidden layers
per sample over the full test set. That count is the one-shot dense analog of
the SNN's hidden-layer spikes/sample (an active ReLU neuron does work exactly
once per input; a spiking neuron does work once per emitted spike).

Writes, idempotently, into each results.json:
    test_plus["activation_stats"] = {
        "active_neurons_per_sample": float,   # mean nonzero hidden ReLU outputs
        "activation_density": float,          # / (hidden0 + hidden1)
        "hidden_neurons": int,
    }

Usage:  python scripts/backfill_dnn_activations.py [--only nslkdd ...] [--force]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import get_dataset, DATASET_LOADERS
from src.models import DNNStudent, TreeGuidedSpikeEncoder, quantize_int8_
from src.training_treespike import _fit_teacher, _arrays_from_bundle

DATASETS = ["kddcup99", "nslkdd", "ctu13", "iscxids2012", "cicids2017",
            "cse-cic-ids2018", "unswnb15", "ciciot2023", "ciciomt2024",
            "edgeiiot", "toniot"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def active_neurons(model: DNNStudent, X: np.ndarray, batch: int = 4096):
    """Mean count of nonzero hidden ReLU outputs per sample."""
    total = 0.0
    n = 0
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
        a1 = model._act(model.fc1(xb))
        a2 = model._act(model.fc2(a1))
        total += float((a1 > 0).sum() + (a2 > 0).sum())
        n += xb.shape[0]
    return total / max(n, 1)


def backfill_dataset(ds: str, force: bool):
    rjs = sorted(glob.glob(str(REPO / f"results/DNN-Tree/{ds}/seed_*/results.json")))
    if not rjs:
        print(f"[{ds}] no runs found"); return 0
    # loader kwargs from the Tree-to-Spike config (the DNN twin mirrors it)
    tcfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    lp = __import__("inspect").signature(DATASET_LOADERS[ds]).parameters
    done = 0
    for rj in rjs:
        r = json.load(open(rj))
        if (not force) and "activation_stats" in r.get("test_plus", {}):
            print(f"[{ds}] {Path(rj).parent.name}: skip (present)"); done += 1; continue
        cfg = r["config"]; seed = cfg["seed"]
        dk = dict(seed=seed)
        for opt in ("subsample_fraction", "test_fraction", "label_mode"):
            if opt in tcfg and opt in lp:
                dk[opt] = tcfg[opt]
        bundle = get_dataset(ds, **dk)
        Xtr, ytr, _, _, Xte, _ = _arrays_from_bundle(bundle)
        nc = bundle.num_classes

        t0 = time.time()
        fit_cfg = dict(tcfg); fit_cfg["seed"] = seed
        teacher, _, _ = _fit_teacher(Xtr, ytr, nc, fit_cfg)
        enc = TreeGuidedSpikeEncoder(
            T=tcfg["T"], pop_max=tcfg["pop_max"], prune_quantile=tcfg["prune_quantile"],
            jitter=tcfg["jitter"], encoder_mode=tcfg.get("encoder_mode", "tree"),
        ).fit(teacher, Xtr.shape[1])
        if enc.out_dim != r["data"]["encoder_channels"]:
            print(f"[{ds}] seed_{seed}: encoder mismatch ({enc.out_dim} vs "
                  f"{r['data']['encoder_channels']}) - SKIPPING"); continue

        model = DNNStudent(enc.out_dim, nc, hidden=tuple(cfg["hidden"]),
                           clip_relu=cfg.get("clip_relu")).to(DEVICE)
        model.load_state_dict(torch.load(Path(rj).parent / "checkpoint.pt",
                                         map_location=DEVICE))
        model.eval()
        if cfg.get("quantize"):
            quantize_int8_(model)                      # mirror the eval-time model

        Xd = enc.encode_dense(Xte).numpy()
        act = active_neurons(model, Xd)
        h = sum(cfg["hidden"])
        r["test_plus"]["activation_stats"] = {
            "active_neurons_per_sample": act,
            "activation_density": act / h,
            "hidden_neurons": int(h),
        }
        with open(rj, "w") as f:
            json.dump(r, f, indent=2)
        done += 1
        print(f"[{ds}] seed_{seed}: active={act:.1f}/{h} "
              f"(density {act/h:.2f})  [{time.time()-t0:.0f}s]", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=DATASETS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    total = sum(backfill_dataset(ds, args.force) for ds in args.only)
    print(f"\ndone: {total} runs in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
