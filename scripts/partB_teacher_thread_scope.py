#!/usr/bin/env python3
"""Scope of the deployment_measure.py thread defect (found 2026-10-07 by guard Gb).

deployment_measure.measure() calls torch.set_num_threads(1) BEFORE fitting the XGBoost
teacher; that caps the OpenMP threads XGBoost trains with, and the fitted teacher can differ
from the production teacher (CICIoT2023 seed 42: test macro-F1 0.83136 vs 0.83392). For one
dataset at seed 42, this fits the teacher in a fresh process either as production does
("default") or after torch.set_num_threads(1) ("torch1"), and records what the deployment
rows depend on: teacher test macro-F1, tree nodes, packed bytes (full ensemble), encoder edge
thresholds, channels and encoder bytes.

    python scripts/partB_teacher_thread_scope.py <dataset> <default|torch1>   # one JSON line
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

ds, mode = sys.argv[1], sys.argv[2]
if mode == "torch1":
    torch.set_num_threads(1)
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from src.training_treespike import _fit_teacher  # noqa: E402
from src.models import TreeGuidedSpikeEncoder  # noqa: E402
from src import serialize as SER  # noqa: E402
import deployment_measure as DM  # noqa: E402

C.no_thread_caps()
b, arr = C.load_bundle(ds, 42)
Xtr, ytr, _, _, Xte, yte = arr
cfg = C.t2s_config(ds, 42)
t, _, present = _fit_teacher(Xtr, ytr, b.num_classes, cfg)
f1 = float(f1_score(yte, present[t.predict(pd.DataFrame(Xte, columns=[f"f{i}" for i in range(Xte.shape[1])]))],
                    average="macro", zero_division=0))
enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                             jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")).fit(t, Xtr.shape[1], X=Xtr)
eb = DM.encoder_bytes(enc)
print(json.dumps(dict(dataset=ds, mode=mode, teacher_f1=f1,
                      stored_teacher_f1=C.stored_record(ds, 42)["test_plus"]["teacher_upper_bound"]["f1_macro"],
                      n_nodes=int(len(t.get_booster().trees_to_dataframe())),
                      packed_bytes=int(SER.pack_booster(t, None, leaf_bits=16, n_classes=len(present))["deploy_bytes"]),
                      n_edges=eb["n_edge_thresholds"], n_channels=eb["n_channels"], encoder_bytes=eb["encoder_bytes"])))
