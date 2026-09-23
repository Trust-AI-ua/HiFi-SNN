#!/usr/bin/env python3
"""Would per-channel int8 scaling restore the dense twin? No training.

For each (model, IoT dataset, seed) the fp32 weights saved by the M3 reruns
(results/diagnostics/val_curves/<model>/<ds>/runs/seed_<S>/checkpoint_fp32.pt) are
projected to int8 two ways and scored on test:

  per-tensor   the deployed quantizer, src.models.quantize_int8_:
               one scale max|W|/127 per weight matrix
  per-channel  one scale per OUTPUT row, s_i = max_j |W_ij| / 127, the standard
               remedy for a single outlier setting a whole layer's scale

Both apply to every nn.Linear, the deployed quantizer's scope. Biases stay fp32.

Guards, each fatal for the cell: the rebuilt encoder must equal the channel table
saved by that run; the recomputed fp32 score must equal the recorded fp_macro_f1;
the recomputed per-tensor score must equal the recorded int8_macro_f1. Only then
is the per-channel score reported.

Pre-committed reporting (decision 2026-09-23): if per-channel scaling restores the
dense twin, parity becomes "within about one point, ordering set by the
quantizer"; if not, the current framing stands.

    python scripts/perchannel_check.py <DNN-Tree|Tree-to-Spike> <dataset> <seed> [cpu|cuda]
    python scripts/perchannel_check.py --aggregate
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/diagnostics/perchannel"
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
TOL = 1e-6


def quantize_per_channel_(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                w = m.weight.data
                s = w.abs().amax(dim=1, keepdim=True) / 127.0 + 1e-12
                m.weight.data = torch.round(w / s).clamp(-127, 127) * s
    return model


def cell(model_name, ds, seed, device="cpu"):
    import src.training_treespike as tts
    import src.training_dnn as tdnn
    from src.models import TreeGuidedSpikeEncoder, SNNStudent, DNNStudent, quantize_int8_
    from src.data import get_dataset, DATASET_LOADERS
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    import inspect

    run = REPO / f"results/diagnostics/val_curves/{model_name}/{ds}/runs/seed_{seed}"
    rec = json.loads((run / "test_fp32_int8.json").read_text())
    saved_ch = json.loads((run / "encoder_channels.json").read_text())["channels"]
    mod = tdnn if model_name == "DNN-Tree" else tts
    cfg = yaml.safe_load(open(REPO / f"experiments/{model_name}/{ds}/config.yaml"))
    cfg.update(seed=seed, device=device)
    dev = torch.device(device)
    cfg = mod._resolve(cfg)

    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    dk = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=seed)
    for o in ("subsample_fraction", "test_fraction", "label_mode"):
        if o in cfg and o in lp:
            dk[o] = cfg[o]
    bundle = get_dataset(ds, **dk)
    Xtr, ytr, _, _, Xte, yte = mod._arrays_from_bundle(bundle)
    teacher, _, _ = mod._fit_teacher(Xtr, ytr, bundle.num_classes, cfg)
    if model_name == "DNN-Tree":
        enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
                                     prune_quantile=cfg["prune_quantile"],
                                     jitter=cfg["jitter"]).fit(teacher, Xtr.shape[1])
        build = lambda: DNNStudent(enc.out_dim, bundle.num_classes, hidden=tuple(cfg["hidden"]),
                                   clip_relu=cfg["clip_relu"])
        score_fn = lambda m: tdnn._forward_batches(m.to(dev), enc, Xte, cfg["batch_size"], dev)[:2]
    else:
        enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
                                     prune_quantile=cfg["prune_quantile"], jitter=cfg["jitter"],
                                     encoder_mode=cfg["encoder_mode"]).fit(teacher, Xtr.shape[1], X=Xtr)
        build = lambda: SNNStudent(enc.out_dim, bundle.num_classes, hidden=tuple(cfg["hidden"]),
                                   beta=cfg["beta"], leaky=cfg["leaky"])
        score_fn = lambda m: tts._encode_forward_in_batches(m.to(dev), enc, Xte, cfg["batch_size"], dev)[:2]

    # guard 1: encoder
    mine = [(int(j), float(o)) for j, o in enc.channels_]
    ref = [(int(c["feature"]), float(c["offset"])) for c in saved_ch]
    if mine != ref:
        raise SystemExit(f"{model_name}/{ds}/{seed}: encoder differs from the saved table; refused")

    fp = build()
    fp.load_state_dict(torch.load(run / "checkpoint_fp32.pt", map_location="cpu"))
    fp.eval()

    def evaluate(m):
        m.eval()
        with torch.no_grad():
            pred, score = score_fn(m)
        mm = compute_metrics(yte, pred, y_score=score, class_names=bundle.class_names)
        bb = compute_binary_ids_metrics(yte, pred, normal_class_index=0)
        return dict(f1=100 * mm["f1_macro"], mcc=100 * mm["matthews_corrcoef"],
                    FAR=100 * bb["false_alarm_rate"], DR=100 * bb["detection_rate"])

    # The M3 reruns scored on the device training defaulted to (CUDA when present).
    # On datasets whose macro-F1 averages over very small classes, near-tie logits
    # can flip between CPU and GPU arithmetic, so a cell is evaluated on whichever
    # device reproduces its recorded scores; the guards below decide.
    r_fp = evaluate(fp)
    r_pt = evaluate(quantize_int8_(copy.deepcopy(fp)))
    # guards 2 and 3: the recorded numbers must be reproduced before a new one counts
    if abs(r_fp["f1"] - rec["fp_macro_f1"]) > TOL or abs(r_pt["f1"] - rec["int8_macro_f1"]) > TOL:
        raise SystemExit(f"{model_name}/{ds}/{seed}: recorded scores not reproduced "
                         f"(fp {r_fp['f1']:.4f} vs {rec['fp_macro_f1']:.4f}, "
                         f"int8 {r_pt['f1']:.4f} vs {rec['int8_macro_f1']:.4f}); refused")
    r_pc = evaluate(quantize_per_channel_(copy.deepcopy(fp)))
    out = dict(model=model_name, dataset=ds, seed=seed,
               **{f"fp32_{k}": v for k, v in r_fp.items()},
               **{f"per_tensor_{k}": v for k, v in r_pt.items()},
               **{f"per_channel_{k}": v for k, v in r_pc.items()},
               device=device, guards="encoder, fp32 and per-tensor reproduced")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{model_name}_{ds}_seed{seed}.json").write_text(json.dumps(out, indent=1))
    print(f"{model_name:13s} {ds:12s} {seed}: fp32 {r_fp['f1']:.2f} | per-tensor {r_pt['f1']:.2f} "
          f"| per-channel {r_pc['f1']:.2f}", flush=True)


def aggregate():
    import pandas as pd
    d = pd.DataFrame([json.loads(f.read_text()) for f in sorted(OUT.glob("*.json"))])
    n = d.groupby("model").size().to_dict()
    cols = ["fp32_f1", "per_tensor_f1", "per_channel_f1"]
    g = d.groupby(["dataset", "model"])[cols].agg(["mean", "std"])
    print(f"cells: {n}\n")
    print(g.round(2).to_string())
    m = d.groupby(["model", "dataset"])[cols].mean().groupby("model").mean()
    print("\nIoT mean macro-F1\n" + m.round(2).to_string())
    diff = (d[d.model == "Tree-to-Spike"].set_index(["dataset", "seed"])[cols]
            - d[d.model == "DNN-Tree"].set_index(["dataset", "seed"])[cols])
    per_ds = diff.groupby("dataset").mean().reindex(IOT)
    print("\nTree-to-Spike - DNN-Tree, per dataset\n" + per_ds.round(2).to_string())
    print("\nIoT mean difference:", per_ds.mean().round(2).to_dict())
    d.to_csv(REPO / "results/tables/tab_perchannel.csv", index=False)
    return g, per_ds


if __name__ == "__main__":
    if sys.argv[1] == "--aggregate":
        aggregate()
    else:
        cell(sys.argv[1], sys.argv[2], int(sys.argv[3]),
             sys.argv[4] if len(sys.argv) > 4 else "cpu")
