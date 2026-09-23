#!/usr/bin/env python3
"""Guard: every new operation-count path must reproduce the published one.

Four defects in this round came from a new measurement path diverging from the
one that produced the paper's numbers. This checks the two paths that feed the
deployment work against the published convention BEFORE the expensive quiet-box
run, on the datasets the repair did not touch.

Guard A (Tree-to-Spike prefixes). The paper's SOPs are input-inclusive, assembled
by the table generators as stored ``synops_per_sample`` (hidden layers only) plus
``encoder_channels * hidden[0]``. The regenerated prefix artifact computes the
same quantity per timestep. At the FULL window the two must agree, because every
channel has fired by then.

Guard B (Vanilla SNN). src/opcount must reproduce the convention the SNN's own
run recorded: input spikes x hidden + hidden spikes x classes, with the frozen
beta * I recurrent matrix counted as a leak rather than a matmul.

    python scripts/guard_sops.py
Exit status is non-zero if any guard fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# the two corpora the repair changed: their stored runs predate it
UNCHANGED = ["kddcup99", "nslkdd", "unswnb15", "cicids2017", "iscxids2012",
             "ctu13", "toniot", "ciciomt2024", "ciciot2023"]
TOL = 1e-6


def guard_a(seed=42):
    print("Guard A: prefix full-window SOPs vs the published convention")
    ok = True
    for ds in UNCHANGED:
        rj = REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/results.json"
        pf = REPO / f"results/prefix_outputs/{ds}/treespike/seed_{seed}/test_prefix.npz"
        if not (rj.exists() and pf.exists()):
            print(f"  {ds:16s} SKIP (missing artifact)")
            continue
        r = json.loads(rj.read_text())
        stored = r["test_plus"]["spike_stats"]["synops_per_sample"]
        ch = r["data"]["encoder_channels"]
        h = r["config"]["hidden"]
        h0 = h[0] if isinstance(h, (list, tuple)) else 64
        published = stored + ch * h0
        z = np.load(pf, allow_pickle=True)
        prefix_full = float(z["cum_sops"][:, -1].mean())
        rel = abs(prefix_full - published) / max(published, 1)
        status = "OK" if rel < 1e-3 else "FAIL"
        ok &= rel < 1e-3
        print(f"  {ds:16s} published={published:10.1f} prefix={prefix_full:10.1f} "
              f"rel={rel:.2e} {status}")
    return ok


def guard_b(seed=42):
    print("\nGuard B: src/opcount vs the Vanilla SNN's own recorded convention")
    import inspect

    import torch
    import yaml

    from src.opcount import count_ops
    from src.models import SNN
    from src.encoding import get_encoder
    from src.data import get_dataset, DATASET_LOADERS
    from src.training_treespike import _arrays_from_bundle
    ok = True
    for ds in ["toniot", "ciciot2023", "ciciomt2024"]:
        rj = REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_{seed}/results.json"
        ck = REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_{seed}/checkpoint.pt"
        if not (rj.exists() and ck.exists()):
            print(f"  {ds:16s} SKIP")
            continue
        rec = json.loads(rj.read_text())["test_plus"]["spike_stats"].get("synops_per_sample")
        if rec is None:
            print(f"  {ds:16s} SKIP (run recorded none)")
            continue
        cfg = yaml.safe_load(open(REPO / f"configs/{ds}/SNN_LeakyParallel/latency/config.yaml"))
        tcfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
        params = inspect.signature(DATASET_LOADERS[ds]).parameters
        kw = dict(batch_size=128, val_fraction=0.1, seed=seed)
        for o in ("subsample_fraction", "test_fraction", "label_mode"):
            if o in tcfg and o in params:
                kw[o] = tcfg[o]
        Xte = _arrays_from_bundle(get_dataset(ds, **kw))[4]
        sd = torch.load(ck, map_location="cpu")
        F = sd["lp.rnn.weight_ih_l0"].shape[1]
        K = sd["fc.weight"].shape[0]
        T = int(cfg["num_steps"])
        m = SNN(num_features=F, hidden_size=cfg["hidden_size"], num_classes=K,
                neuron_name=cfg["neuron"], num_steps=T)
        m.load_state_dict(sd)
        m.eval()
        enc = get_encoder(cfg["encoding"])
        X = Xte[:20000].astype("float32")
        with torch.no_grad(), count_ops(m, batch_size=len(X)) as c:
            m(enc(torch.from_numpy(X), T))
        rel = abs(c.sops - rec) / max(rec, 1)
        status = "OK" if rel < 0.02 else "FAIL"          # subset vs full test set
        ok &= rel < 0.02
        print(f"  {ds:16s} recorded={rec:10.1f} opcount={c.sops:10.1f} "
              f"leak(separate)={c.leak_ops:8.0f} macs={c.macs:.0f} rel={rel:.3f} {status}")
    return ok


def guard_c(seed=42):
    """Reproduce the published Rel.E column of the 4-way anatomy table.

    Rel.E is energy relative to Tree-to-Spike on the same dataset. Reproducing it
    pins down exactly which convention each published number used, which is how
    the Vanilla SNN double count was found: its Rel.E is only reproducible if the
    generator's ``num_features * hidden_size`` term is added ON TOP of a recorded
    value that already contains the input layer.
    """
    import yaml
    print("\nGuard C: published Rel.E (4-way anatomy table), seed 42")
    E_MAC, E_AC = 4.6, 0.9
    from src.tasnn_energy import tasnn_energy_pj
    published = {"ciciomt2024": {"vanilla": 1.06, "dnn": 4.20, "tasnn": 12000},
                 "toniot": {"vanilla": 1.14, "dnn": 2.66},
                 "ciciot2023": {"dnn": 2.82, "tasnn": 8000}}
    ok = True
    for ds, want in published.items():
        rt = json.loads((REPO / f"results/Tree-to-Spike/{ds}/seed_{seed}/results.json").read_text())
        h = rt["config"]["hidden"]
        h0 = h[0] if isinstance(h, (list, tuple)) else 64
        t2s = (rt["test_plus"]["spike_stats"]["synops_per_sample"]
               + rt["data"]["encoder_channels"] * h0)
        e_t2s = t2s * E_AC
        if "vanilla" in want:
            rv = json.loads((REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_{seed}"
                             / "results.json").read_text())
            vcfg = yaml.safe_load(open(REPO / f"configs/{ds}/SNN_LeakyParallel/latency/config.yaml"))
            rec = rv["test_plus"]["spike_stats"]["synops_per_sample"]
            paper = rec + rv["data"]["num_features"] * vcfg["hidden_size"]
            got, corrected = paper * E_AC / e_t2s, rec * E_AC / e_t2s
            hit = abs(got - want["vanilla"]) / want["vanilla"] < 0.12
            ok &= hit
            print(f"  {ds:14s} vanilla published={want['vanilla']:.2f}x "
                  f"paper-convention={got:.2f}x {'OK' if hit else 'FAIL'} | "
                  f"WITHOUT the double count it is {corrected:.2f}x")
        if "tasnn" in want:
            r2 = json.loads((REPO / f"results/TASNN/{ds}/seed_{seed}/results.json").read_text())
            e, _, _ = tasnn_energy_pj(r2)
            got = e / e_t2s
            hit = 0.5 < got / want["tasnn"] < 2.0
            ok &= hit
            print(f"  {ds:14s} tasnn   published~{want['tasnn']:,}x analytic={got:,.0f}x "
                  f"{'OK' if hit else 'FAIL'}")
    return ok


def guard_d():
    """src/tasnn_energy.py must stay character-identical to benchmark_4way's copy."""
    import re
    print("\nGuard D: TASNN formula copy vs the published original")
    orig = (REPO / "scripts/benchmark_4way.py").read_text()
    mine = (REPO / "src/tasnn_energy.py").read_text()
    grab = lambda t: re.search(r"def tasnn_energy_pj\(rec\):.*?return mac \* E_MAC \+ ac \* E_AC.*?$",
                               t, re.S | re.M)
    a, b = grab(orig), grab(mine)
    same = bool(a and b and a.group(0).strip() == b.group(0).strip())
    print(f"  identical: {same} {'OK' if same else 'FAIL'}")
    return same


if __name__ == "__main__":
    a = guard_a()
    b = guard_b()
    c = guard_c()
    d = guard_d()
    print(f"\nGuard A {'PASS' if a else 'FAIL'} | Guard B {'PASS' if b else 'FAIL'} "
          f"| Guard C {'PASS' if c else 'FAIL'} | Guard D {'PASS' if d else 'FAIL'}")
    sys.exit(0 if (a and b and c and d) else 1)
