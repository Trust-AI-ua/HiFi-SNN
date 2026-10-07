#!/usr/bin/env python3
"""B8 child: peak resident memory of one deployed system over the latency workload.

Runs in a fresh process. Imports the same libraries for every system (numpy, pandas,
torch, xgboost, the repository's models and encoders), records the resident set size
(VmRSS) after imports as the baseline, loads one artifact written by
scripts/deployment_measure.py (artifact_dir), runs the deployment latency workload on one
CPU thread (500 warm-up + 2,000 batch-1 calls on the same input subset), and reports the
process's peak resident set size (ru_maxrss) and the increment over the baseline.

    python scripts/partB_b8_memchild.py <artifact.pkl>      # prints one JSON line
"""
from __future__ import annotations

import json
import pickle
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401  (common import set)
import torch
import xgboost  # noqa: F401  (common import set)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from src.models import SNNStudent, DNNStudent, SNN, TASNN  # noqa: E402
from src.encoding import get_encoder  # noqa: E402

N_WARMUP, N_TIMED = 500, 2000


def rss_kb() -> int:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return -1


def build(a):
    k = a["kind"]
    if k == "gbdt":
        t, ir, F = a["teacher"], a["iteration_range"], a["F"]
        cols = [f"f{i}" for i in range(F)]
        return lambda x: t.predict(pd.DataFrame(x, columns=cols), iteration_range=ir)
    if k == "vanilla":
        v = a["vcfg"]
        m = SNN(num_features=a["F"], hidden_size=v["hidden_size"], num_classes=a["K"],
                neuron_name=v["neuron"], num_steps=int(v["num_steps"]))
        m.load_state_dict(a["sd"]); m.eval()
        enc, T = get_encoder(v["encoding"]), int(v["num_steps"])
        return lambda x: m(enc(torch.from_numpy(x), T))
    if k == "tasnn":
        c = a["tcfg"]
        m = TASNN(num_features=a["F"], num_classes=a["K"], num_steps=c["num_steps"], d_model=c["d_model"],
                  n_heads=c["n_heads"], n_layers=c["n_layers"], ff=c["ff"], dropout=c["dropout"],
                  pop_M=c["pop_M"], stc_channels=c["stc_channels"], stc_kernel=c["stc_kernel"],
                  lif_hidden=tuple(c["lif_hidden"]), lif_beta=c["lif_beta"], lif_v_th=c["lif_v_th"],
                  lif_gamma=c["lif_gamma"], surrogate_beta=c["surrogate_beta"], target_rate=c["target_rate"])
        m.load_state_dict(a["sd"]); m.eval()
        return lambda x: m(torch.from_numpy(x))
    if k == "dnn":
        enc = a["enc"]
        m = DNNStudent(enc.out_dim, a["K"], hidden=a["hidden"], clip_relu=a["clip_relu"])
        m.load_state_dict(a["sd"]); m.eval()
        return lambda x: m(enc.encode_dense(x))
    if k == "t2s":
        enc, th = a["enc"], a["theta"]
        m = SNNStudent(enc.out_dim, a["K"], hidden=a["hidden"], beta=a["beta"], leaky=a["leaky"])
        m.load_state_dict(a["sd"]); m.eval()
        if th is None:
            return lambda x: m(enc.encode(x))
        return lambda x: m.predict_early_exit(enc.encode(x), theta_exit=th, stop_when_done=True)
    if k == "ple":
        from partB_b3_ple import PLEMLP
        m = PLEMLP(a["bins"], a["enc_cols"], a["raw_cols"], a["width"], a["K"])
        m.load_state_dict(a["sd"]); m.eval()
        return lambda x: m(torch.from_numpy(x))
    raise SystemExit(f"unknown artifact kind {k}")


def main():
    torch.set_num_threads(1)
    base = rss_kb()
    a = pickle.load(open(sys.argv[1], "rb"))
    X = a.pop("Xsub")
    fn = build(a)
    loaded = rss_kb()
    with torch.no_grad():
        for i in range(min(N_WARMUP + N_TIMED, len(X))):
            fn(X[i:i + 1])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(dataset=a["dataset"], seed=a["seed"], system=a["system"], baseline_rss_kb=base,
                          after_load_rss_kb=loaded, peak_rss_kb=peak, peak_minus_baseline_kb=peak - base,
                          calls=int(min(N_WARMUP + N_TIMED, len(X))))))


if __name__ == "__main__":
    main()
