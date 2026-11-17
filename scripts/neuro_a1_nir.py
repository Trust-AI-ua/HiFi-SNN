#!/usr/bin/env python3
"""A1: NIR round trip of the seed-42 Tree-to-Spike students (results/preregistration/2026-11-17_neuro_stream.md).

Runs in .venv-neuro. Inputs from scripts/neuro_export.py (results/neuro/inputs/<ds>_seed42.npz).

  reference  the checkpoint loaded into an nn.Sequential of snnTorch layers (init_hidden), run
             step by step; must reproduce the production student's float readout
  exported   snntorch.export_to_nir(reference) -> NIR graph (saved as .nir)
  imported   snntorch.import_from_nir(graph), run step by step, as imported
  restored   the same imported graph with each LIF's reset set back to snnTorch's delayed
             subtractive reset (NIR's LIF node has no subtractive reset; see the note)

Readout in all: argmax of the mean output membrane over the T steps. Full test split.

    .venv-neuro/bin/python scripts/neuro_a1_nir.py [datasets...]
Output: results/neuro/a1_nir/{<ds>.json, <ds>.nir, nir_roundtrip.csv}
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import snntorch as snn
import snntorch.utils  # snn.utils is a submodule: import it explicitly
from snntorch.export_nir import export_to_nir
from snntorch.import_nir import import_from_nir
import nir

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
INP = REPO / "results/neuro/inputs"
OUT = REPO / "results/neuro/a1_nir"
DS = ["ciciot2023", "ciciomt2024", "edgeiiot", "toniot"]
CHUNK = 65536


def reference(ckpt: dict) -> torch.nn.Sequential:
    lin = lambda w, b: (lambda m: (m.weight.data.copy_(w), m.bias.data.copy_(b), m)[-1])(torch.nn.Linear(w.shape[1], w.shape[0]))
    lif = lambda i, out=False: snn.Leaky(beta=ckpt[f"lif{i}.beta"].clone(), threshold=ckpt[f"lif{i}.threshold"].clone(),
                                         init_hidden=True, reset_mechanism="subtract", output=out)
    seq = torch.nn.Sequential(
        lin(ckpt["fc1.weight"], ckpt["fc1.bias"]), lif(1),
        lin(ckpt["fc2.weight"], ckpt["fc2.bias"]), lif(2),
        lin(ckpt["fc3.weight"], ckpt["fc3.bias"]), lif(3, out=True))
    return seq.eval()


@torch.no_grad()
def run_reference(seq, st, T):
    out = []
    for i in range(0, len(st), CHUNK):
        s = torch.from_numpy(st[i:i + CHUNK].astype(np.int64))
        snn.utils.reset(seq)
        acc = 0.0
        for t in range(T):
            _, mem = seq((s == t).float())
            acc = acc + mem
        out.append((acc / T).numpy())
    return np.concatenate(out)


@torch.no_grad()
def run_graph(net, st, T, out_key):
    out = []
    for i in range(0, len(st), CHUNK):
        s = torch.from_numpy(st[i:i + CHUNK].astype(np.int64))
        state, acc = None, 0.0
        for t in range(T):
            _, state = net((s == t).float(), state)
            mem = state.state[out_key]
            acc = acc + (mem[0] if isinstance(mem, tuple) else mem)
        out.append((acc / T).numpy())
    return np.concatenate(out)


def restore_subtract(net):
    """Replace every imported snn.Leaky (reset 'zero', no delay) by snnTorch's default subtractive delayed reset."""
    n = 0
    for name, m in list(net.named_children()):
        if isinstance(m, snn.Leaky):
            setattr(net, name, snn.Leaky(beta=m.beta.detach().clone(), threshold=m.threshold.detach().clone(),
                                         reset_mechanism="subtract", init_hidden=False))
            n += 1
    return n


def lif_desc(m):
    return dict(beta=float(np.asarray(m.beta.detach()).ravel()[0]), threshold=float(np.asarray(m.threshold.detach()).ravel()[0]),
                reset=m.reset_mechanism, reset_delay=bool(getattr(m, "reset_delay", True)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in (sys.argv[1:] or DS):
        z = np.load(INP / f"{ds}_seed42.npz")
        st, y, T, flog = z["spike_times"], z["labels"], int(z["T"]), z["float_logits"]
        ckpt = torch.load(REPO / f"results/Tree-to-Spike/{ds}/seed_42/checkpoint.pt", map_location="cpu")
        seq = reference(ckpt)
        ref = run_reference(seq, st, T)
        g = export_to_nir(seq, torch.zeros(1, st.shape[1]))
        nir.write(str(OUT / f"{ds}.nir"), g)
        g2 = nir.read(str(OUT / f"{ds}.nir"))
        lifs = [k for k, v in g2.nodes.items() if isinstance(v, nir.LIF)]
        out_key = [a for a, b in g2.edges if b == "output"][0]
        net = import_from_nir(g2)
        imported_lifs = {k: lif_desc(m) for k, m in net.named_children() if isinstance(m, snn.Leaky)}
        imp = run_graph(net, st, T, out_key)
        n_restored = restore_subtract(net)
        res = run_graph(net, st, T, out_key)
        params_equal = all(np.array_equal(np.asarray(g2.nodes[k].weight), ckpt[f"fc{i}.weight"].numpy())
                           and np.array_equal(np.asarray(g2.nodes[k].bias), ckpt[f"fc{i}.bias"].numpy())
                           for i, k in enumerate([k for k, v in g2.nodes.items() if isinstance(v, nir.Affine)], 1))
        pr = ref.argmax(1)
        r = dict(dataset=ds, n_test=int(len(y)), T=T, channels=int(st.shape[1]),
                 reference_vs_production=dict(agreement=float(np.mean(pr == flog.argmax(1))),
                                              max_abs_readout=float(np.abs(ref - flog).max())),
                 nir_nodes={k: type(v).__name__ for k, v in g2.nodes.items()},
                 nir_lif={k: dict(tau=np.asarray(g2.nodes[k].tau).tolist(), r=np.asarray(g2.nodes[k].r).tolist(),
                                  v_threshold=np.asarray(g2.nodes[k].v_threshold).tolist(),
                                  v_reset=np.asarray(g2.nodes[k].v_reset).tolist()) for k in lifs},
                 affine_parameters_bit_exact=bool(params_equal), imported_lif=imported_lifs,
                 imported=dict(agreement=float(np.mean(imp.argmax(1) == pr)), max_abs_readout=float(np.abs(imp - ref).max())),
                 restored=dict(n_lif_restored=n_restored, agreement=float(np.mean(res.argmax(1) == pr)),
                               max_abs_readout=float(np.abs(res - ref).max())))
        from sklearn.metrics import f1_score
        for k, p in (("reference", pr), ("imported", imp.argmax(1)), ("restored", res.argmax(1))):
            r.setdefault("macro_f1", {})[k] = 100 * f1_score(y, p, average="macro")
        (OUT / f"{ds}.json").write_text(json.dumps(r, indent=1))
        rows.append(dict(dataset=ds, n_test=r["n_test"], ref_vs_production=100 * r["reference_vs_production"]["agreement"],
                         imported_agreement=100 * r["imported"]["agreement"], imported_max_abs=r["imported"]["max_abs_readout"],
                         restored_agreement=100 * r["restored"]["agreement"], restored_max_abs=r["restored"]["max_abs_readout"],
                         f1_reference=r["macro_f1"]["reference"], f1_imported=r["macro_f1"]["imported"],
                         f1_restored=r["macro_f1"]["restored"]))
        print(rows[-1], flush=True)
    p = OUT / "nir_roundtrip.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        new = pd.concat([old[~old.dataset.isin(new.dataset)], new])
    new.sort_values("dataset").to_csv(p, index=False)


if __name__ == "__main__":
    main()
