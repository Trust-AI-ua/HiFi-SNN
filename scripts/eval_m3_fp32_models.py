#!/usr/bin/env python3
"""M3 for the models whose reported numbers are fp32: Vanilla SNN
(SNN_LeakyParallel/latency) and TASNN. Inference only, nothing retrained.

Their stored checkpoints are floating point (config has no ``quantize``), so the
fp32 side is re-evaluated from the checkpoint (and checked against the stored
test macro-F1) and the int8 side applies the manuscript's projection
``src.models.quantize_int8_`` (per-tensor symmetric, s = max|W|/127, nn.Linear
weights only) to a copy. For TASNN, non-Linear weights (conv, attention
in_proj stored as a raw Parameter, embeddings) are NOT projected by that
function; the per-row ``linear_frac`` records the share of weight parameters
the projection actually covers.

``int8_all_*`` applies the same per-tensor rule to EVERY weight tensor with
dim >= 2 (nn.RNN, conv, attention included; biases untouched as in the
manuscript). For Vanilla SNN the Linear-only projection covers ~2% of weights,
so int8_all is the meaningful column there. For DNN-Tree and Tree-to-Spike all
weights are Linear, so the two projections coincide.

TASNN samples Bernoulli spikes at inference (src/models/tasnn.py), so a single
pass is stochastic (~0.05 macro-F1 run to run). Every pass is therefore preceded
by the same torch.manual_seed(seed), so the fp32 and int8 passes see identical
draws and the delta is paired; ``repro_abs_diff`` against the stored value
still contains that sampling noise. Vanilla SNN is deterministic.

The int8 columns are a counterfactual for these two models: the paper reports them
in fp32.

    python scripts/eval_m3_fp32_models.py            # all 110 cells
    python scripts/eval_m3_fp32_models.py --models vanilla --datasets nslkdd
Output: results/diagnostics/m3_fp32_models.csv (append-merge on model,dataset,seed)
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import get_dataset, DATASET_LOADERS  # noqa: E402
from src.metrics import compute_metrics, compute_binary_ids_metrics  # noqa: E402
from src.models import quantize_int8_  # noqa: E402

DATASETS = ["kddcup99", "nslkdd", "ctu13", "iscxids2012", "cicids2017", "unswnb15",
            "ciciot2023", "ciciomt2024", "edgeiiot", "toniot", "cse-cic-ids2018"]
SEEDS = [42, 43, 44, 45, 46]
OUT = REPO / "results/diagnostics/m3_fp32_models.csv"
KEY = ["model", "dataset", "seed"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bundle(dataset, cfg, seed):
    params = inspect.signature(DATASET_LOADERS[dataset]).parameters
    kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg.get("val_fraction", 0.1), seed=seed)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in params and cfg[opt] is not None:
            kw[opt] = cfg[opt]
    return get_dataset(dataset, **kw)


def _vanilla(dataset, seed):
    import src.training as tr
    from src.encoding import get_encoder
    from src.models import SNN
    rd = REPO / f"results/{dataset}/SNN_LeakyParallel/latency/seed_{seed}"
    cfg = yaml.safe_load(open(REPO / f"configs/{dataset}/SNN_LeakyParallel/latency/config.yaml"))
    cfg.update(json.load(open(rd / "results.json"))["config"])
    cfg = tr._resolve(cfg)
    tr._set_seed(seed)
    b = _bundle(dataset, cfg, seed)
    enc = get_encoder(cfg["encoding"])
    m = SNN(num_features=b.num_features, hidden_size=cfg["hidden_size"],
            num_classes=b.num_classes, neuron_name=cfg["neuron"],
            num_steps=cfg["num_steps"]).to(DEV)
    m.load_state_dict(torch.load(rd / "checkpoint.pt", map_location=DEV))

    def run(mod):
        o = tr._eval(mod, b.test_loader, enc, cfg["num_steps"], DEV)
        return o["y_true"], o["y_pred"], o["y_score"]
    return m, run, b.class_names, rd


def _tasnn(dataset, seed):
    import src.training_tasnn as tt
    from src.models import TASNN
    rd = REPO / f"results/TASNN/{dataset}/seed_{seed}"
    cfg = yaml.safe_load(open(REPO / f"experiments/TASNN/{dataset}/config.yaml"))
    cfg.update(json.load(open(rd / "results.json"))["config"])
    cfg = tt._resolve(cfg)
    tt._set_seed(seed)
    b = _bundle(dataset, cfg, seed)
    m = TASNN(num_features=b.num_features, num_classes=b.num_classes,
              num_steps=cfg["num_steps"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
              n_layers=cfg["n_layers"], ff=cfg["ff"], dropout=cfg["dropout"],
              pop_M=cfg["pop_M"], stc_channels=cfg["stc_channels"],
              stc_kernel=cfg["stc_kernel"], lif_hidden=tuple(cfg["lif_hidden"]),
              lif_beta=cfg["lif_beta"], lif_v_th=cfg["lif_v_th"], lif_gamma=cfg["lif_gamma"],
              surrogate_beta=cfg["surrogate_beta"], target_rate=cfg["target_rate"]).to(DEV)
    m.load_state_dict(torch.load(rd / "checkpoint.pt", map_location=DEV))

    def run(mod):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        o = tt._eval(mod, b.test_loader, DEV)
        return o["y_true"], o["y_pred"], o["y_score"]
    return m, run, b.class_names, rd


def quantize_all_(model):
    with torch.no_grad():
        for _, w in model.named_parameters():
            if w.dim() >= 2:
                sc = w.abs().max() / 127.0 + 1e-12
                w.copy_(torch.round(w / sc).clamp(-127, 127) * sc)
    return model


def _linear_frac(m):
    lin = sum(x.weight.numel() for x in m.modules() if isinstance(x, nn.Linear))
    allw = sum(p.numel() for n, p in m.named_parameters() if p.dim() >= 2)
    return lin / max(allw, 1)


def _metrics(y, p, s, names):
    mm = compute_metrics(y, p, y_score=s, class_names=names)
    bb = compute_binary_ids_metrics(y, p, normal_class_index=0)
    return (100 * mm["f1_macro"], 100 * mm["matthews_corrcoef"],
            100 * bb["false_alarm_rate"], 100 * bb["detection_rate"])


def _flush(rows):
    new = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT)
        old = old[~old.set_index(KEY).index.isin(new.set_index(KEY).index)]
        new = pd.concat([old, new], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(KEY).to_csv(OUT, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vanilla", "tasnn"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    a = ap.parse_args()
    done = set()
    if OUT.exists():
        done = set(map(tuple, pd.read_csv(OUT)[KEY].astype(str).values.tolist()))
    for model in a.models:
        for d in a.datasets:
            for s in a.seeds:
                if (model, d, str(s)) in done:
                    continue
                m, run, names, rd = (_vanilla if model == "vanilla" else _tasnn)(d, s)
                y, p, sc = run(m)
                fp = _metrics(y, p, sc, names)
                q = quantize_int8_(copy.deepcopy(m))
                y, p, sc = run(q)
                q8 = _metrics(y, p, sc, names)
                qa = quantize_all_(copy.deepcopy(m))
                if hasattr(qa, "flatten_parameters"):
                    qa.flatten_parameters()
                y, p, sc = run(qa)
                qall = _metrics(y, p, sc, names)
                stored = 100 * json.load(open(rd / "results.json"))["test_plus"]["metrics"]["f1_macro"]
                row = dict(model=model, dataset=d, seed=s,
                           fp_macro_f1=fp[0], int8_macro_f1=q8[0], d_macro_f1=q8[0] - fp[0],
                           fp_mcc=fp[1], int8_mcc=q8[1], fp_FAR=fp[2], int8_FAR=q8[2],
                           fp_DR=fp[3], int8_DR=q8[3],
                           int8_all_macro_f1=qall[0], d_all_macro_f1=qall[0] - fp[0],
                           int8_all_mcc=qall[1], int8_all_FAR=qall[2], int8_all_DR=qall[3],
                           stored_macro_f1=stored, repro_abs_diff=abs(fp[0] - stored),
                           linear_frac=_linear_frac(m))
                print(f"[m3] {model} {d} {s} fp={fp[0]:.2f} int8={q8[0]:.2f} int8_all={qall[0]:.2f} "
                      f"stored={stored:.2f} lin={row['linear_frac']:.2f}", flush=True)
                _flush([row])
                del m, q, qa
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
