#!/usr/bin/env python3
"""Energy-accuracy frontier (exploratory; results/preregistration/2026-11-17_neuro_stream.md, amendment 5):
export trained seed-42 Tree-to-Spike variants for the Lava and SANA-FE chain, as scripts/neuro_export.py
exports the reported model.

Runs in .venv. The variants share the dataset's production teacher (they change only the encoder or
the window), so the teacher is refitted once per dataset and checked against every variant's record
(teacher test macro-F1 equal to the stored one). Per variant: the encoder with the variant's
configuration, the student from the variant's checkpoint, guards that the integer encoder equals the
float encoder and that the float student reproduces the record; the integer path (Q = 8, 8-bit leak)
has no step-2 record for variants, so its macro-F1 is stored rather than checked.

    .venv/bin/python scripts/neuro_export_variants.py <dataset> <tag>=<results subdir> [...]
Output: results/neuro/inputs/<ds>_<tag>_seed42.npz (the keys of neuro_export.py, plus run_dir,
float_f1, int_f1)
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/neuro/inputs"
SEED = 42


def main():
    if any(os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise SystemExit("thread variables are set; the teacher would not be the production teacher")
    import r2_integer_inference as RI
    from neuro_export import stratified_sample, int_counts
    from src.data import get_dataset, DATASET_LOADERS
    from src.metrics import compute_metrics
    from src.models.tree_to_spike import TreeGuidedSpikeEncoder, SNNStudent
    from src.training_treespike import _arrays_from_bundle, _fit_teacher
    ds = sys.argv[1]
    variants = dict(a.split("=", 1) for a in sys.argv[2:])
    base = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    recs = {t: json.loads((REPO / d / f"seed_{SEED}/results.json").read_text()) for t, d in variants.items()}
    cfgs = {}
    for t, rec in recs.items():
        c = dict(base); c.update(rec["config"]); c["dataset"], c["seed"] = ds, SEED
        cfgs[t] = c
    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    loader = lambda c: tuple(sorted({"batch_size": c["batch_size"], "val_fraction": c["val_fraction"], "seed": SEED,
                                     **{o: c[o] for o in ("subsample_fraction", "test_fraction", "label_mode")
                                        if o in c and o in lp and c[o] is not None}}.items()))
    teacher_keys = lambda c: tuple(sorted((k, v) for k, v in c.items() if k.startswith("teacher") or k == "class_weighted"))
    if len({loader(c) for c in cfgs.values()}) != 1 or len({teacher_keys(c) for c in cfgs.values()}) != 1:
        raise SystemExit("variants differ in loader or teacher settings; export them separately")
    c0 = next(iter(cfgs.values()))
    b = get_dataset(ds, **dict(loader(c0)))
    Xtr, ytr, Xva, yva, Xte, yte = _arrays_from_bundle(b)
    K, F = b.num_classes, Xtr.shape[1]
    score = lambda y, p: compute_metrics(y, p, class_names=b.class_names)["f1_macro"]
    teacher, _, present = _fit_teacher(Xtr, ytr, K, c0)
    t_f1 = score(yte, present[teacher.predict(pd.DataFrame(Xte, columns=[f"f{i}" for i in range(F)]))])
    samp = stratified_sample(yte)
    for t, cfg in cfgs.items():
        rec = recs[t]
        if t_f1 != rec["test_plus"]["teacher_upper_bound"]["f1_macro"]:
            raise SystemExit(f"{ds}/{t}: refit teacher is not the variant's stored teacher")
        enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                     jitter=cfg["jitter"], encoder_mode=cfg.get("encoder_mode", "tree")).fit(teacher, F, X=Xtr)
        ienc, T = RI.IntEncoder(enc), enc.T
        st = SNNStudent(enc.out_dim, K, hidden=tuple(cfg["hidden"]), beta=cfg["beta"], leaky=cfg["leaky"])
        rdir = REPO / variants[t] / f"seed_{SEED}"
        st.load_state_dict(torch.load(rdir / "checkpoint.pt", map_location="cpu"))
        st.eval()
        run, _, layers = RI.int_snn(st, 8, "8bit")
        stimes, logits, ipred = [], [], []
        for i in range(0, len(Xte), RI.CHUNK):
            x = Xte[i:i + RI.CHUNK]
            sp = enc.encode(x)
            s = ienc.spike_times(x)
            if not torch.equal(sp, (torch.arange(T).view(T, 1, 1) == torch.from_numpy(s).unsqueeze(0)).float()):
                raise SystemExit(f"{ds}/{t}: integer encoder spikes differ from the float encoder")
            with torch.no_grad():
                logits.append(st(sp)[0].numpy())
            stimes.append(s.astype(np.int8)); ipred.append(run(s, T))
        stimes, logits, ipred = np.concatenate(stimes), np.concatenate(logits), np.concatenate(ipred)
        if score(yte, logits.argmax(1)) != rec["test_plus"]["metrics"]["f1_macro"]:
            raise SystemExit(f"{ds}/{t}: float student does not reproduce the record")
        p2, cnt = int_counts(layers, stimes[samp].astype(np.int64), T)
        if not np.array_equal(p2, ipred[samp]):
            raise SystemExit(f"{ds}/{t}: counting integer run disagrees with the integer run")
        sd = {k: v.detach().numpy() for k, v in st.state_dict().items()}
        extra = dict(n_edges=np.array([len(enc.edges_[j]) for j in ienc.feats], np.int64))
        for i, (n, L) in enumerate(zip(("fc1", "fc2", "fc3"), layers), 1):
            extra.update({f"W{i}": L["W"].T.astype(np.int8), f"b{i}": sd[f"{n}.bias"], f"s{i}": np.float64(L["s"]),
                          f"theta{i}": np.int64(L["theta"]), f"bq{i}": L["b"]})
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"{ds}_{t}_seed{SEED}.npz", spike_times=stimes, labels=yte.astype(np.int64),
                            float_logits=logits, int_pred=ipred, sample=samp, int_counts=cnt,
                            class_names=np.array(b.class_names), T=np.int64(T), beta=np.float64(cfg["beta"]),
                            run_dir=variants[t], float_f1=100 * score(yte, logits.argmax(1)),
                            int_f1=100 * score(yte, ipred), **extra)
        print(f"[done] {ds}/{t}: C={stimes.shape[1]} T={T} float F1 {100 * score(yte, logits.argmax(1)):.2f} "
              f"int F1 {100 * score(yte, ipred):.2f}", flush=True)


if __name__ == "__main__":
    main()
