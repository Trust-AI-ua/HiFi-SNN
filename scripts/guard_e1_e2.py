#!/usr/bin/env python3
"""Guards G1, G3, G4 for the E1/E2 experiments (see
results/preregistration/2026-09-24_matched_info_and_entropy_edges.md). Exits non-zero
on any failure; the launch script runs no training unless this passes.

  G4  entropy_cuts recovers known cut points on synthetic data, deterministically
  G1  on every IoT dataset at seed 42, the modified encoder in "tree" and "quantile"
      modes is bitwise identical to the COMMITTED encoder (edges, channels, encode(),
      encode_dense()); "entropy" keeps the tree channels exactly, never exceeds the
      tree's per-feature edge count, and its filler and shortfall counts are recorded
  G3  the time-quantized dense output equals 1 - (first-spike step)/(T - 1) of encode()

    python scripts/guard_e1_e2.py [--ref HEAD]
Output: results/diagnostics/guard_e1_e2.json
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
IOT = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]


def g4():
    from src.models.tree_to_spike import entropy_cuts
    rng = np.random.default_rng(0)
    x = rng.random(20000).astype(np.float32)
    y = (x > 0.37).astype(np.int64)
    c, f = entropy_cuts(x, y, 1)
    ok1 = len(c) == 1 and abs(float(c[0]) - 0.37) < 1e-3 and f == 0
    y3 = np.digitize(x, [0.2, 0.5, 0.8]).astype(np.int64)
    c3, f3 = entropy_cuts(x, y3, 3)
    ok3 = len(c3) == 3 and np.allclose(c3, [0.2, 0.5, 0.8], atol=1e-3) and f3 == 0
    c5, f5 = entropy_cuts(x, y3, 5)                        # 2 more than the information supports
    ok5 = len(c5) == 5 and f5 == 2
    again = entropy_cuts(x, y3, 5)[0]
    det = np.array_equal(c5, again)
    print(f"G4 single cut {c} {ok1} | three cuts {c3} {ok3} | five cuts, fillers {f5} {ok5} | deterministic {det}")
    return ok1 and ok3 and ok5 and det


def load_committed(ref):
    src = subprocess.check_output(["git", "show", f"{ref}:src/models/tree_to_spike.py"], cwd=REPO, text=True)
    tmp = Path(tempfile.mkdtemp()) / "tts_committed.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("tts_committed", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def g1_g3(ref):
    from src.data import get_dataset, DATASET_LOADERS
    from src.training_treespike import _fit_teacher, _arrays_from_bundle
    import src.models.tree_to_spike as NEW
    OLD = load_committed(ref)
    out, ok = {}, True
    for ds in IOT:
        cfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
        lp = inspect.signature(DATASET_LOADERS[ds]).parameters
        kw = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=42)
        for o in ("subsample_fraction", "test_fraction", "label_mode"):
            if o in cfg and o in lp:
                kw[o] = cfg[o]
        b = get_dataset(ds, **kw)
        Xtr, ytr, _, _, Xte, _ = _arrays_from_bundle(b)
        teacher, _, _ = _fit_teacher(Xtr, ytr, b.num_classes, {**cfg, "seed": 42})
        Xs = Xte[:20000]
        rec = {}
        mk = lambda M, mode: M.TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
                                                      prune_quantile=cfg["prune_quantile"],
                                                      jitter=cfg["jitter"], encoder_mode=mode)
        for mode in ("tree", "quantile"):
            eo = mk(OLD, mode).fit(teacher, Xtr.shape[1], X=Xtr)
            en = mk(NEW, mode).fit(teacher, Xtr.shape[1], X=Xtr, y=ytr)
            same_edges = eo.edges_.keys() == en.edges_.keys() and all(np.array_equal(eo.edges_[j], en.edges_[j]) for j in eo.edges_)
            same = (same_edges and eo.channels_ == en.channels_
                    and np.array_equal(eo.encode(Xs).numpy(), en.encode(Xs).numpy())
                    and np.array_equal(eo.encode_dense(Xs).numpy(), en.encode_dense(Xs).numpy()))
            rec[f"{mode}_bitwise_identical"] = bool(same)
            ok &= same
            if mode == "tree":
                tree_enc = en
        ent = mk(NEW, "entropy").fit(teacher, Xtr.shape[1], X=Xtr, y=ytr)
        kept = sorted({j for j, _ in tree_enc.channels_})
        counts_ok = all(len(ent.edges_[j]) <= len(tree_enc.edges_[j]) for j in kept)
        rec["entropy_channels_identical"] = ent.channels_ == tree_enc.channels_
        rec["entropy_count_never_exceeds_tree"] = bool(counts_ok)
        rec["entropy_edges_total"] = int(sum(len(ent.edges_[j]) for j in kept))
        rec["tree_edges_total"] = int(sum(len(tree_enc.edges_[j]) for j in kept))
        rec["entropy_filler_cuts"] = int(sum(ent.entropy_filler_.values()))
        rec["entropy_differs_from_tree"] = any(not np.array_equal(ent.edges_[j], tree_enc.edges_[j]) for j in kept)
        ok &= rec["entropy_channels_identical"] and counts_ok and rec["entropy_differs_from_tree"]
        # G3: time-quantized dense == 1 - first-spike step / (T - 1)
        tree_enc.dense_time_quantized = True
        lhat = tree_enc.encode_dense(Xs).numpy()
        sp = tree_enc.encode(Xs).numpy()                     # [T, B, C]
        fires_once = bool((sp.sum(0) == 1).all())
        t_first = sp.argmax(0)
        g3 = fires_once and np.array_equal(lhat, (1.0 - t_first / (cfg["T"] - 1)).astype(np.float32))
        tree_enc.dense_time_quantized = False
        rec["G3_every_channel_fires_once"] = fires_once
        rec["G3_lhat_matches_spike_time"] = bool(g3)
        rec["G3_levels_changed_by_quantization_share"] = float((lhat != tree_enc.encode_dense(Xs).numpy()).mean())
        ok &= g3
        out[ds] = rec
        print(f"G1/G3 {ds}: {rec}", flush=True)
    return ok, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD")
    a = ap.parse_args()
    ok4 = g4()
    ok13, rec = g1_g3(a.ref)
    (REPO / "results/diagnostics/guard_e1_e2.json").write_text(json.dumps(
        {"ref": a.ref, "G4": ok4, "G1_G3": ok13, "per_dataset": rec}, indent=1, default=str))
    print(f"\nG4 {'PASS' if ok4 else 'FAIL'} | G1+G3 {'PASS' if ok13 else 'FAIL'}")
    sys.exit(0 if (ok4 and ok13) else 1)
