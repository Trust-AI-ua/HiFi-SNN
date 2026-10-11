#!/usr/bin/env python3
"""Verify the deployment serialization before it is used for measurement.

Checks, per dataset at seed 42:

  1. int8 round trip. For Tree-to-Spike and DNN-Tree, reconstructing q * s from
     the stored int8 codes and per-tensor scales must give BIT-IDENTICAL float32
     weights, and test predictions must be identical to the current artifact.
  2. Packed trees. Predictions of the packed artifact (8-bit thresholds,
     leaf_bits leaves) are compared with the native model. Where they differ,
     macro-F1 is reported ON THE PACKED ARTIFACT, on the same principle as the
     student: the reported accuracy is the accuracy of what ships.
  3. Per-component byte breakdown for every artifact.

    python scripts/verify_serialization.py [--datasets ...] [--cap 20000]
Output: results/deployment/serialization_verification.{json,csv}
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src import serialize as S  # noqa: E402
from src.data import get_dataset, DATASET_LOADERS  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402

DATASETS = ["kddcup99", "nslkdd", "unswnb15", "cicids2017", "cse-cic-ids2018",
            "iscxids2012", "ctu13", "toniot", "edgeiiot", "ciciomt2024", "ciciot2023"]
OUT = REPO / "results/deployment"


def bundle_for(ds, cfg, seed=42):
    params = inspect.signature(DATASET_LOADERS[ds]).parameters
    kw = dict(batch_size=cfg.get("batch_size", 128), val_fraction=cfg.get("val_fraction", 0.1),
              seed=seed)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in cfg and opt in params:
            kw[opt] = cfg[opt]
    return get_dataset(ds, **kw)


def check_int8(ds, seed, cap):
    from src.models import TreeGuidedSpikeEncoder, SNNStudent, DNNStudent
    from src.training_treespike import _fit_teacher, _arrays_from_bundle
    rows = []
    cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    b = bundle_for(ds, cfg, seed)
    Xtr, ytr, _, _, Xte, yte = _arrays_from_bundle(b)
    teacher, _, _ = _fit_teacher(Xtr, ytr, b.num_classes, {**cfg, "seed": seed})
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
                                 prune_quantile=cfg["prune_quantile"],
                                 jitter=cfg["jitter"]).fit(teacher, Xtr.shape[1])
    Xc, yc = Xte[:cap], yte[:cap]
    for tag, path, build in (
            ("Tree-to-Spike", REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/checkpoint.pt",
             lambda: SNNStudent(enc.out_dim, b.num_classes, hidden=tuple(cfg["hidden"]),
                                beta=cfg["beta"], leaky=cfg["leaky"])),
            ("DNN-Tree", REPO / f"results/DNN-Tree/{ds}/seed_{seed}/checkpoint.pt",
             lambda: DNNStudent(enc.out_dim, b.num_classes, hidden=tuple(cfg["hidden"])))):
        if not path.exists():
            rows.append(dict(dataset=ds, system=tag, status="missing checkpoint"))
            continue
        sd = torch.load(path, map_location="cpu")
        pay = S.int8_student_payload(sd)["payload"]
        exact = True
        for k, v in sd.items():
            a = v.numpy()
            if a.ndim == 2:
                rec = pay[k]["q"].astype(np.float32) * np.float32(pay[k]["scale"])
                exact &= bool(np.array_equal(rec, a.astype(np.float32)))
        m = build(); m.load_state_dict(sd); m.eval()
        m2 = build()
        sd2 = {k: (torch.from_numpy(pay[k]["q"].astype(np.float32) * np.float32(pay[k]["scale"]))
                   if (hasattr(v, "ndim") and v.ndim == 2) else v) for k, v in sd.items()}
        m2.load_state_dict(sd2); m2.eval()
        with torch.no_grad():
            if tag == "Tree-to-Spike":
                sp = enc.encode(Xc)
                p1 = m(sp)[0].argmax(1).numpy(); p2 = m2(sp)[0].argmax(1).numpy()
            else:
                dz = enc.encode_dense(Xc)
                p1 = m(dz).argmax(1).numpy(); p2 = m2(dz).argmax(1).numpy()
        bb = S.int8_student_bytes(sd)
        rows.append(dict(dataset=ds, system=tag, status="ok", bitwise_identical=exact,
                         predictions_identical=bool(np.array_equal(p1, p2)),
                         n_compared=int(len(p1)), **bb))
    return rows, (teacher, b, Xte, yte)


def check_packed(ds, teacher, b, Xte, yte, cap, leaf_bits=16):
    Xc, yc = Xte[:cap], yte[:cap]
    packed = S.pack_booster(teacher, None, leaf_bits=leaf_bits, n_classes=b.num_classes)
    import pandas as pd
    Xdf = pd.DataFrame(Xc, columns=[f"f{i}" for i in range(Xc.shape[1])])
    native = teacher.predict(Xdf)
    pk = S.predict_packed(packed, Xc, b.num_classes, leaf_bits=leaf_bits)
    agree = float(np.mean(native == pk))
    f_native = 100 * compute_metrics(yc, native, class_names=b.class_names)["f1_macro"]
    f_packed = 100 * compute_metrics(yc, pk, class_names=b.class_names)["f1_macro"]
    return dict(dataset=ds, system="GBDT full (packed)", status="ok",
                prediction_agreement=agree, macro_f1_native=f_native,
                macro_f1_packed=f_packed, delta_macro_f1=f_packed - f_native,
                deploy_bytes=packed["deploy_bytes"], n_internal=packed["n_internal"],
                n_leaves=packed["n_leaves"], leaf_bits=leaf_bits,
                threshold_max_abs_error=packed["threshold_max_abs_error"],
                thresholds_outside_unit_interval=packed["thresholds_outside_unit_interval"],
                serialization_format=S.SERIALIZATION_FORMAT, n_compared=int(len(pk)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--skip-packed", action="store_true")
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        r, ctx = check_int8(ds, a.seed, a.cap)
        rows += r
        for x in r:
            print(f"  {ds:16s} {x['system']:20s} " +
                  (f"bitwise={x.get('bitwise_identical')} preds={x.get('predictions_identical')} "
                   f"bytes={x.get('model_bytes')}" if x["status"] == "ok" else x["status"]), flush=True)
        if not a.skip_packed:
            teacher, b, Xte, yte = ctx
            p = check_packed(ds, teacher, b, Xte, yte, a.cap)
            rows.append(p)
            print(f"  {ds:16s} {'GBDT packed':20s} agree={p['prediction_agreement']:.4f} "
                  f"F1 native={p['macro_f1_native']:.2f} packed={p['macro_f1_packed']:.2f} "
                  f"bytes={p['deploy_bytes']}", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "serialization_verification.csv", index=False)
    (OUT / "serialization_verification.json").write_text(json.dumps(rows, indent=1, default=str))
    ok = df[df.status == "ok"]
    bad = ok[(ok.get("bitwise_identical") == False) | (ok.get("predictions_identical") == False)]
    print(f"\n[wrote] {OUT/'serialization_verification.csv'}")
    print("int8 round trip:", "ALL EXACT" if len(bad) == 0 else f"{len(bad)} FAILURES")


if __name__ == "__main__":
    main()
