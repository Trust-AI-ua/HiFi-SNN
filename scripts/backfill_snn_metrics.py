#!/usr/bin/env python3
"""Backfill early-exit + SynOps metrics for completed SNN ablation runs.

The SNN sweep recorded energy / sparsity / params / timing but not the
early-exit or SynOps metrics that the hybrids report. This script replays each
completed run's saved ``checkpoint.pt`` (NO retraining — eval only), rebuilds the
deterministic test split, runs ``SNN.predict_early_exit`` (added additively to
the model), and:

  * augments that run's ``results.json`` in place with
      test_plus["early_exit"]            = {accuracy, f1_macro, mean_exit_step,
                                            max_steps, uncertain_frac}
      test_plus["spike_stats"]["synops_per_sample"]
    (and the same for test_minus_21 where present), and
  * writes a companion ``results/summary_snn_efficiency.csv`` keyed by
    (dataset, neuron, encoding, seed).

It deliberately does NOT touch ``results/summary.csv`` or ``src/training.py`` so
it cannot race the still-running cse-cic-ids2018 sweep. Idempotent and
resumable: a run whose results.json already has an ``early_exit`` block is
skipped unless ``--force``.

Usage:
    python scripts/backfill_snn_metrics.py                  # all datasets
    python scripts/backfill_snn_metrics.py --only nslkdd kddcup99
    python scripts/backfill_snn_metrics.py --force
    python scripts/backfill_snn_metrics.py --eval-batch 2048
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import get_dataset, DATASET_LOADERS
from src.encoding import get_encoder
from src.metrics import compute_metrics
from src.models import SNN

DATASETS = ["nslkdd", "kddcup99", "cicids2017", "ctu13", "cse-cic-ids2018"]
SUMMARY_HEADER = [
    "dataset", "neuron", "encoding", "seed",
    "early_exit_accuracy", "early_exit_f1_macro", "mean_exit_step",
    "uncertain_frac", "synops_per_sample",
    "test21_early_exit_accuracy", "test21_mean_exit_step",
]


def _dataset_kwargs_for(ds, neuron, enc, cfg):
    """Reconstruct the loader kwargs exactly as src/training.py did: pull the
    knobs from the run's config.yaml, override seed with the run's actual seed."""
    cfg_path = REPO_ROOT / "configs" / ds / f"SNN_{neuron}" / enc / "config.yaml"
    file_cfg = {}
    if cfg_path.exists():
        file_cfg = yaml.safe_load(open(cfg_path)) or {}
    loader_params = inspect.signature(DATASET_LOADERS[ds]).parameters
    kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"],
              seed=cfg["seed"])
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in file_cfg and opt in loader_params:
            kw[opt] = file_cfg[opt]
    return kw


@torch.no_grad()
def _early_exit_over_loader(model, loader, encoder, num_steps, device):
    """Run early-exit over a dataset; returns (y_true, ee_pred, exit_steps,
    n_uncertain, synops_sum, n)."""
    all_true, all_pred, all_exit = [], [], []
    n_uncertain = 0; synops_sum = 0.0; n = 0
    for X, y in loader:
        X = X.to(device)
        spikes = encoder(X, num_steps)
        pr, ex, unc, _total, _sites, syn = model.predict_early_exit(spikes)
        all_pred.append(pr.cpu().numpy())
        all_exit.append(ex.cpu().numpy())
        all_true.append(y.numpy())
        n_uncertain += int(unc.sum().item())
        synops_sum += float(syn) * X.shape[0]
        n += X.shape[0]
    return (np.concatenate(all_true), np.concatenate(all_pred),
            np.concatenate(all_exit), n_uncertain, synops_sum, n)


def _ee_block(model, loader, encoder, num_steps, device, class_names):
    yt, ee_pred, exits, n_unc, syn_sum, n = _early_exit_over_loader(
        model, loader, encoder, num_steps, device)
    m = compute_metrics(yt, ee_pred, y_score=None, class_names=class_names)
    return {
        "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
        "mean_exit_step": float(np.mean(exits)), "max_steps": int(num_steps),
        "uncertain_frac": float(n_unc) / max(n, 1),
    }, (syn_sum / max(n, 1))


class _Bundle:
    """Lightweight holder for the reusable per-(dataset, seed) test data."""
    __slots__ = ("num_features", "num_classes", "class_names",
                 "test_loader", "test21_loader")


def _build_bundle(ds, rep_record, eval_batch):
    """Build the test/test21 loaders once for a (dataset, seed). The split is
    identical across all neuron/encoding combos of that seed, so this is reused.
    Uses the run's config + the matching config.yaml for loader kwargs."""
    cfg = rep_record["config"]
    kw = _dataset_kwargs_for(ds, rep_record["neuron"], rep_record["encoding"], cfg)
    bundle = get_dataset(ds, **kw)
    b = _Bundle()
    b.num_features = bundle.num_features
    b.num_classes = bundle.num_classes
    b.class_names = bundle.class_names
    b.test_loader = DataLoader(bundle.test_loader.dataset, batch_size=eval_batch, shuffle=False)
    b.test21_loader = (DataLoader(bundle.test21_loader.dataset, batch_size=eval_batch,
                                  shuffle=False) if bundle.test21_loader is not None else None)
    return b


def backfill_one(rj_path: Path, r: dict, b: "_Bundle", device, force: bool):
    """Backfill a single run using a prebuilt (dataset, seed) bundle."""
    if (not force) and "early_exit" in r.get("test_plus", {}):
        return "skip"
    neuron = r["neuron"]; enc = r["encoding"]; cfg = r["config"]
    num_steps = cfg["num_steps"]; hidden = cfg["hidden_size"]
    ckpt = rj_path.with_name("checkpoint.pt")
    if not ckpt.exists():
        return "no-ckpt"

    model = SNN(num_features=b.num_features, hidden_size=hidden,
                num_classes=b.num_classes, neuron_name=neuron,
                num_steps=num_steps).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    encoder = get_encoder(enc)

    ee, synops = _ee_block(model, b.test_loader, encoder, num_steps, device, b.class_names)
    r["test_plus"]["early_exit"] = ee
    r["test_plus"].setdefault("spike_stats", {})["synops_per_sample"] = synops

    t21 = r.get("test_minus_21")
    if t21 is not None and b.test21_loader is not None:
        ee21, synops21 = _ee_block(model, b.test21_loader, encoder, num_steps, device, b.class_names)
        r["test_minus_21"]["early_exit"] = ee21
        r["test_minus_21"].setdefault("spike_stats", {})["synops_per_sample"] = synops21

    with open(rj_path, "w") as f:
        json.dump(r, f, indent=2)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=DATASETS, help="datasets to backfill")
    ap.add_argument("--force", action="store_true", help="recompute even if early_exit present")
    ap.add_argument("--eval-batch", type=int, default=1024)
    ap.add_argument("--neuron", default=None, help="restrict to one neuron, e.g. LeakyParallel")
    ap.add_argument("--encoding", default=None, help="restrict to one encoding, e.g. latency")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  eval_batch={args.eval_batch}  force={args.force}")

    from collections import defaultdict
    rows = []
    counts = {"ok": 0, "skip": 0, "no-ckpt": 0, "fail": 0}
    t0 = time.time()
    for ds in args.only:
        rjs = sorted((REPO_ROOT / "results" / ds).glob("SNN_*/*/seed_*/results.json"))
        if args.neuron and args.encoding:
            want = f"SNN_{args.neuron}/{args.encoding}/"
            rjs = [rj for rj in rjs if want in str(rj)]
        if not rjs:
            continue
        # group by seed so the (dataset, seed) test bundle is built once and
        # reused across all 27 neuron/encoding combos of that seed.
        by_seed = defaultdict(list)
        for rj in rjs:
            try:
                r = json.load(open(rj))
                by_seed[r["config"]["seed"]].append((rj, r))
            except Exception as e:
                counts["fail"] += 1
                print(f"  FAIL parse {rj.relative_to(REPO_ROOT)}: {e}")
        print(f"\n=== {ds}: {len(rjs)} runs across seeds {sorted(by_seed)} ===")
        for seed, items in sorted(by_seed.items()):
            bundle = None
            for rj, r in items:
                try:
                    if (not args.force) and "early_exit" in r.get("test_plus", {}):
                        counts["skip"] += 1
                        continue
                    if bundle is None:                       # build once per (ds, seed)
                        bundle = _build_bundle(ds, r, args.eval_batch)
                    status = backfill_one(rj, r, bundle, device, args.force)
                except Exception as e:
                    status = "fail"
                    print(f"  FAIL {rj.relative_to(REPO_ROOT)}: {type(e).__name__}: {e}")
                else:
                    counts[status] = counts.get(status, 0) + 1
                    if status != "ok":
                        continue
                if status == "ok":
                    tp_ee = r["test_plus"]["early_exit"]
                    syn = r["test_plus"]["spike_stats"].get("synops_per_sample")
                    t21 = r.get("test_minus_21") or {}
                    t21_ee = t21.get("early_exit", {}) if t21 else {}
                    rows.append({
                        "dataset": r["dataset"], "neuron": r["neuron"],
                        "encoding": r["encoding"], "seed": r["config"]["seed"],
                        "early_exit_accuracy": tp_ee["accuracy"],
                        "early_exit_f1_macro": tp_ee["f1_macro"],
                        "mean_exit_step": tp_ee["mean_exit_step"],
                        "uncertain_frac": tp_ee["uncertain_frac"],
                        "synops_per_sample": syn,
                        "test21_early_exit_accuracy": t21_ee.get("accuracy"),
                        "test21_mean_exit_step": t21_ee.get("mean_exit_step"),
                    })
                    print(f"  ok  {r['neuron']:14}/{r['encoding']:8} seed={seed}"
                          f"  exit={tp_ee['mean_exit_step']:.1f}/{tp_ee['max_steps']}"
                          f"  ee_f1={tp_ee['f1_macro']:.3f}  synops={syn:.0f}")

    # companion summary: merge with any prior rows, dedup on (dataset,neuron,encoding,seed)
    out_csv = REPO_ROOT / "results" / "summary_snn_efficiency.csv"
    prior = []
    if out_csv.exists() and out_csv.stat().st_size > 0:
        with open(out_csv, newline="") as f:
            rd = csv.DictReader(f)
            if set(rd.fieldnames or []) == set(SUMMARY_HEADER):
                key = {(r["dataset"], r["neuron"], r["encoding"], str(r["seed"])) for r in rows}
                prior = [r for r in rd if (r["dataset"], r["neuron"], r["encoding"],
                                           str(r["seed"])) not in key]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_HEADER)
        w.writeheader()
        for r in prior + rows:
            w.writerow(r)

    print(f"\n{'='*60}\nbackfill done in {(time.time()-t0)/60:.1f} min — "
          f"ok={counts['ok']} skip={counts['skip']} no-ckpt={counts['no-ckpt']} fail={counts['fail']}")
    print(f"companion summary: {out_csv}  ({len(prior)+len(rows)} rows)")


if __name__ == "__main__":
    main()
