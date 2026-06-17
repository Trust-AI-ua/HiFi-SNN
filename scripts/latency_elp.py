#!/usr/bin/env python3
"""Clean batch-1 latency + Energy-Latency-Product (ELP) re-measure from checkpoints,
for Vanilla SNN (LeakyParallel/latency), Tree-to-Spike, and DNN-Tree.

Why CPU single-thread: edge deployment is batch-1 on modest hardware, and the
shared GPU is busy (contended timings are not publication-grade). Single-thread
CPU is reproducible, uncontended, and edge-realistic; it also honestly exposes the
SNN's T-step sequential cost (which neuromorphic hardware would remove).

Latency is input-*value*-independent, so we time the loaded models on correctly
shaped random inputs (no need to refit teachers/encoders). Energy per sample is
taken from each run's recorded ops (SNN SOPs x 0.9 pJ; DNN MACs x 4.6 pJ).
ELP = energy_pj * batch1_latency_ms.

    python scripts/latency_elp.py            # seeds 42 43 44, 200 reps
"""
from __future__ import annotations
import glob, json, time, sys, statistics as st
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import SNN, SNNStudent, DNNStudent

E_MAC, E_AC = 4.6, 0.9
DS = ["kddcup99", "nslkdd", "cicids2017", "ctu13", "iscxids2012", "cse-cic-ids2018"]
SEEDS = [42, 43, 44]
WARMUP, REPS = 20, 200
DEV = torch.device("cpu")
torch.set_num_threads(1)


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def vanilla_dir(ds, seed):
    for base in (f"results/SNN_Model_compare_Neuron_vs_Encoding/{ds}",
                 f"results/{ds}"):
        p = Path(base) / "SNN_LeakyParallel" / "latency" / f"seed_{seed}"
        if (p / "results.json").exists():
            return p
    return None


def _time(model, make_input):
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(make_input())
        t0 = time.perf_counter()
        for _ in range(REPS):
            model(make_input())
        return (time.perf_counter() - t0) * 1000.0 / REPS   # ms / forward


def measure_vanilla(p):
    r = json.load(open(p / "results.json"))
    nf, nc = r["data"]["num_features"], r["data"]["num_classes"]
    hs, ns = r["config"]["hidden_size"], r["config"]["num_steps"]
    m = SNN(nf, hs, nc, "LeakyParallel", ns).to(DEV)
    m.load_state_dict(torch.load(p / "checkpoint.pt", map_location=DEV)); m.eval()
    sops = r["test_plus"]["spike_stats"].get("synops_per_sample")
    spk1 = lambda: (torch.rand(ns, 1, nf, device=DEV) > 0.5).float()
    return _time(m, spk1), (sops * E_AC if sops else None)


def measure_tree(p):
    r = json.load(open(p / "results.json"))
    ind, nc = r["data"]["encoder_channels"], r["data"]["num_classes"]
    hidden = tuple(r["config"]["hidden"]); T = r["config"]["T"]
    beta, leaky = r["config"]["beta"], r["config"]["leaky"]
    m = SNNStudent(ind, nc, hidden=hidden, beta=beta, leaky=leaky).to(DEV)
    m.load_state_dict(torch.load(p / "checkpoint.pt", map_location=DEV)); m.eval()
    sops = r["test_plus"]["spike_stats"].get("synops_per_sample")
    spk1 = lambda: (torch.rand(T, 1, ind, device=DEV) > 0.5).float()
    return _time(m, spk1), (sops * E_AC if sops else None)


def measure_dnn(p):
    r = json.load(open(p / "results.json"))
    ind, nc = r["data"]["encoder_channels"], r["data"]["num_classes"]
    hidden = tuple(r["config"]["hidden"]); clip = r["config"].get("clip_relu")
    m = DNNStudent(ind, nc, hidden=hidden, clip_relu=clip).to(DEV)
    m.load_state_dict(torch.load(p / "checkpoint.pt", map_location=DEV)); m.eval()
    macs = r["test_plus"]["compute"]["macs_per_sample"]
    x1 = lambda: torch.rand(1, ind, device=DEV)
    return _time(m, x1), macs * E_MAC


MODELS = [("Vanilla SNN", lambda ds, s: vanilla_dir(ds, s), measure_vanilla),
          ("Tree-to-Spike", lambda ds, s: Path(f"results/Tree-to-Spike/{ds}/seed_{s}"), measure_tree),
          ("DNN-Tree", lambda ds, s: Path(f"results/DNN-Tree/{ds}/seed_{s}"), measure_dnn)]


def main():
    print(f"device=cpu (1 thread)  warmup={WARMUP} reps={REPS}  seeds={SEEDS}\n")
    cells = {}  # (model, ds) -> (lat_ms, e_pj)
    for name, locate, measure in MODELS:
        for ds in DS:
            lats, es = [], []
            for s in SEEDS:
                p = locate(ds, s)
                if p is None or not (p / "checkpoint.pt").exists():
                    continue
                try:
                    lat, e = measure(p)
                    lats.append(lat); es.append(e)
                except Exception as ex:
                    print(f"  !! {name}/{ds}/seed{s}: {type(ex).__name__}: {ex}")
            if lats:
                cells[(name, ds)] = (mean(lats), mean(es))
                print(f"  {name:14}/{ds:16} lat_bs1={mean(lats):.3f} ms  E={mean(es) if mean(es) else float('nan'):.0f} pJ")

    L = ["# Batch-1 latency & Energy-Latency Product (ELP) — publication re-measure",
         "",
         f"CPU single-thread, batch-1, {REPS} reps after {WARMUP} warmup, seeds {SEEDS} averaged. "
         "Energy = recorded ops x Horowitz (SOP 0.9 pJ / MAC 4.6 pJ). ELP = energy_pj x latency_ms.",
         "",
         "## Latency (ms/sample, batch-1)",
         "| Model | " + " | ".join(d[:8] for d in DS) + " | mean |",
         "|---|" + "--:|" * (len(DS) + 1)]
    for name, _, _ in MODELS:
        vals = [cells.get((name, ds), (None, None))[0] for ds in DS]
        L.append(f"| {name} | " + " | ".join(f"{v:.3f}" if v else "-" for v in vals) +
                 f" | {mean(vals):.3f} |" if mean(vals) else " | - |")

    L += ["", "## ELP (pJ·ms/sample, batch-1) — lower is better",
          "| Model | " + " | ".join(d[:8] for d in DS) + " | mean |",
          "|---|" + "--:|" * (len(DS) + 1)]
    for name, _, _ in MODELS:
        elps = []
        for ds in DS:
            lat, e = cells.get((name, ds), (None, None))
            elps.append(lat * e if (lat and e) else None)
        L.append(f"| {name} | " + " | ".join(f"{v:.0f}" if v else "-" for v in elps) +
                 f" | {mean(elps):.0f} |" if mean(elps) else " | - |")

    # headline means
    def mmean(name, idx):
        return mean([cells[(name, ds)][idx] for ds in DS if (name, ds) in cells])
    L += ["", "## Summary (macro-avg)",
          "| Model | Energy (pJ) | Batch-1 latency (ms) | ELP (pJ·ms) |", "|---|--:|--:|--:|"]
    for name, _, _ in MODELS:
        e = mmean(name, 1); lat = mmean(name, 0)
        L.append(f"| {name} | {e:.0f} | {lat:.3f} | {e*lat:.0f} |" if e and lat else f"| {name} | - | - | - |")

    L += ["", "## How to read this (honest framing)",
          "- **Energy** is the hardware-agnostic SNN win (SOPs x 0.9 pJ vs MACs x 4.6 pJ).",
          "- **Batch-1 latency on a CPU favours the one-shot DNN**: the SNNs run T sequential "
          "timesteps (Vanilla T=25, Tree-to-Spike T=20), so wall-clock is ~T x higher. This is a "
          "von-Neumann artifact — on event-driven neuromorphic hardware the SNN timesteps run "
          "concurrently and this latency collapses.",
          "- **ELP therefore reflects the conventional-hardware trade-off**, not the neuromorphic "
          "target. Report energy (delta) as the headline efficiency claim and ELP with this caveat.",
          ]
    Path("results/latency_elp.md").write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print("\n[wrote results/latency_elp.md]")

    import csv
    with open("results/latency_elp.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "dataset", "batch1_latency_ms", "energy_pj", "elp_pj_ms"])
        for (name, ds), (lat, e) in cells.items():
            w.writerow([name, ds, lat, e, (lat * e if (lat and e) else "")])
    print("[wrote results/latency_elp.csv]")


if __name__ == "__main__":
    main()
