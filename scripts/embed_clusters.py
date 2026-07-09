#!/usr/bin/env python3
"""Intermediate-representation cluster viz for the 4-way comparison.

For one dataset, extracts each model's **penultimate representation** (the input to
its final classification layer), projects all four to 2D with the *same* UMAP
settings, and plots a shared 2x2 figure coloured by class - showing how each
paradigm separates the attack classes in its learned feature space.

Models: Vanilla SNN (LeakyParallel/latency), Tree-to-Spike, DNN-Tree, TASNN.
Loads the seed-42 checkpoints; for Tree-to-Spike / DNN-Tree it refits the
(deterministic) XGBoost teacher to rebuild the encoder the student was trained with.

Run:  python scripts/embed_clusters.py --dataset nslkdd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import get_dataset, DATASET_LOADERS
from src.encoding import get_encoder
from src.models import SNN, SNNStudent, DNNStudent, TreeGuidedSpikeEncoder, TASNN
from src.training_treespike import _fit_teacher, _arrays_from_bundle

SEED = 42
N_VIZ = 2500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = REPO / "results" / "Journal_results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = REPO / "results" / "Journal_results" / "embeddings_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Per-class qualitative palette matching the 4-way figures (Paul Tol bright +
# Okabe-Ito + Tol muted/light extensions, colorblind-aware, ordered for contrast).
# Replaces matplotlib tab10/tab20 so every figure shares the IEEE-bright scheme.
PALETTE = ["#4477AA", "#EE6677", "#228833", "#E69F00", "#66CCEE", "#AA3377",
           "#CCBB44", "#332288", "#009988", "#882255", "#999933", "#FFAABB",
           "#000000", "#BBBBBB", "#EE8866"]


class Penult:
    """Forward hook capturing a module's input; averages over calls (T for SNN, 1 else)."""
    def __init__(self):
        self.buf = []

    def __call__(self, m, inp, out):
        self.buf.append(inp[0].detach().float().cpu())

    def mean(self):
        return torch.stack(self.buf, 0).mean(0).numpy()


def _ckpt(p):
    return torch.load(REPO / p, map_location="cpu")


def _write_silhouette_csv(ds, models, embs):
    """Append/update this dataset's silhouettes in the canonical CSV (dedup on dataset)."""
    import csv
    p = REPO / "results" / "Journal_results" / "embeddings_silhouette.csv"
    prior = [r for r in csv.reader(open(p))] if p.exists() else []
    prior = [r for r in prior if r and r[0] not in (ds, "dataset")]
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "model", "silhouette"])
        for r in prior:
            w.writerow(r)
        for name in models:
            if name in embs:
                s = embs[name][1]
                w.writerow([ds, name, f"{s:.4f}" if np.isfinite(s) else "nan"])


@torch.no_grad()
def emb_vanilla(ds, Xte, nf, nc):
    base = f"results/{ds}/SNN_LeakyParallel/latency/seed_{SEED}"
    cfg = json.load(open(REPO / base / "results.json"))["config"]
    model = SNN(num_features=nf, hidden_size=cfg["hidden_size"], num_classes=nc,
                neuron_name="LeakyParallel", num_steps=cfg["num_steps"]).to(DEVICE)
    model.load_state_dict(_ckpt(base + "/checkpoint.pt")); model.eval()
    h = Penult(); model.fc.register_forward_hook(h)
    spikes = get_encoder("latency")(torch.tensor(Xte, dtype=torch.float32), cfg["num_steps"]).to(DEVICE)
    model(spikes)
    return h.mean()


def _teacher_encoder(ds, Xtr, ytr, nc):
    tcfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    tcfg["seed"] = SEED
    teacher, _, _ = _fit_teacher(Xtr, ytr, nc, tcfg)
    enc = TreeGuidedSpikeEncoder(T=tcfg["T"], pop_max=tcfg["pop_max"],
                                 prune_quantile=tcfg["prune_quantile"], jitter=tcfg["jitter"],
                                 encoder_mode=tcfg.get("encoder_mode", "tree")).fit(teacher, Xtr.shape[1])
    return enc


@torch.no_grad()
def emb_tree(ds, enc, Xte, nc):
    base = f"results/Tree-to-Spike/{ds}/seed_{SEED}"
    cfg = json.load(open(REPO / base / "results.json"))["config"]
    student = SNNStudent(enc.out_dim, nc, hidden=tuple(cfg["hidden"]),
                         beta=cfg["beta"], leaky=cfg["leaky"]).to(DEVICE)
    student.load_state_dict(_ckpt(base + "/checkpoint.pt")); student.eval()
    h = Penult(); student.fc3.register_forward_hook(h)
    student(enc.encode(Xte).to(DEVICE))
    return h.mean()


@torch.no_grad()
def emb_dnn(ds, enc, Xte, nc):
    base = f"results/DNN-Tree/{ds}/seed_{SEED}"
    cfg = json.load(open(REPO / base / "results.json"))["config"]
    student = DNNStudent(enc.out_dim, nc, hidden=tuple(cfg["hidden"]),
                         clip_relu=cfg.get("clip_relu")).to(DEVICE)
    student.load_state_dict(_ckpt(base + "/checkpoint.pt")); student.eval()
    h = Penult(); student.fc3.register_forward_hook(h)
    student(enc.encode_dense(Xte).to(DEVICE))
    return h.mean()


@torch.no_grad()
def emb_tasnn(ds, Xte, nf, nc):
    base = f"results/TASNN/{ds}/seed_{SEED}"
    cfg = json.load(open(REPO / base / "results.json"))["config"]
    model = TASNN(num_features=nf, num_classes=nc, num_steps=cfg["num_steps"],
                  d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                  ff=cfg["ff"], dropout=cfg["dropout"], pop_M=cfg["pop_M"],
                  stc_channels=cfg["stc_channels"], stc_kernel=cfg["stc_kernel"],
                  lif_hidden=tuple(cfg["lif_hidden"]), lif_beta=cfg["lif_beta"],
                  lif_v_th=cfg["lif_v_th"], lif_gamma=cfg["lif_gamma"],
                  surrogate_beta=cfg["surrogate_beta"], target_rate=cfg["target_rate"]).to(DEVICE)
    model.load_state_dict(_ckpt(base + "/checkpoint.pt")); model.eval()
    h = Penult(); model.head_tr.register_forward_hook(h)   # transformer CLS representation
    model(torch.tensor(Xte, dtype=torch.float32).to(DEVICE))
    return h.mean()


def project(Z):
    import umap
    Z = StandardScaler().fit_transform(Z)
    return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED,
                     metric="euclidean").fit_transform(Z)


MODEL_ORDER = ["Vanilla SNN", "Tree-to-Spike", "DNN-Tree", "TASNN"]


def _save_cache(ds, embs, yv, names):
    """Persist the 2D projections so restyling doesn't need GPU recompute."""
    arrs = {"yv": yv, "names": np.asarray(names, dtype=object)}
    for name, (P, sil) in embs.items():
        arrs[f"P::{name}"] = P
        arrs[f"sil::{name}"] = np.float64(sil)
    np.savez(CACHE / f"{ds}.npz", **arrs)


def _load_cache(ds):
    p = CACHE / f"{ds}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    embs = {k[3:]: (z[k], float(z[f"sil::{k[3:]}"])) for k in z.files if k.startswith("P::")}
    return embs, z["yv"], list(z["names"])


def compute(ds):
    tcfg = yaml.safe_load(open(REPO / f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    lp = __import__("inspect").signature(DATASET_LOADERS[ds]).parameters
    dk = dict(seed=SEED)
    for opt in ("subsample_fraction", "test_fraction", "label_mode"):
        if opt in tcfg and opt in lp:
            dk[opt] = tcfg[opt]
    bundle = get_dataset(ds, **dk)
    nf, nc, names = bundle.num_features, bundle.num_classes, bundle.class_names
    Xtr, ytr, Xval, yval, Xte, yte = _arrays_from_bundle(bundle)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xte), min(N_VIZ, len(Xte)), replace=False)
    Xv, yv = Xte[idx].astype(np.float32), yte[idx]

    print(f"[embed] {ds}: {len(Xv)} test points, {nc} classes, {nf} features  (device={DEVICE})")
    enc = _teacher_encoder(ds, Xtr, ytr, nc)
    models = {
        "Vanilla SNN":   lambda: emb_vanilla(ds, Xv, nf, nc),
        "Tree-to-Spike": lambda: emb_tree(ds, enc, Xv, nc),
        "DNN-Tree":      lambda: emb_dnn(ds, enc, Xv, nc),
        "TASNN":         lambda: emb_tasnn(ds, Xv, nf, nc),
    }
    embs = {}
    for name, fn in models.items():
        try:
            Z = np.nan_to_num(fn(), nan=0.0, posinf=0.0, neginf=0.0)
            try:
                sil = float(silhouette_score(StandardScaler().fit_transform(Z), yv)) \
                    if len(np.unique(yv)) > 1 else float("nan")
            except Exception:
                sil = float("nan")
            if not np.isfinite(sil):
                sil = float("nan")
            embs[name] = (project(Z), sil)
            print(f"  {name:14} penultimate dim={Z.shape[1]}  "
                  f"silhouette={f'{sil:.3f}' if np.isfinite(sil) else 'n/a'}")
        except Exception as e:
            print(f"  {name:14} FAILED: {e}")
    _write_silhouette_csv(ds, list(models), embs)
    _save_cache(ds, embs, yv, names)
    return embs, yv, names


def plot(ds, embs, yv, names):
    """Consistent 2x2 figure, per-class colors from the shared IEEE palette."""
    nc = len(names)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150,
                         "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.6))
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        if name not in embs:
            ax.set_visible(False); continue
        P, sil = embs[name]
        for c in range(nc):
            m = yv == c
            ax.scatter(P[m, 0], P[m, 1], s=5, color=PALETTE[c % len(PALETTE)],
                       label=names[c] if len(names) <= 12 else None, alpha=0.6, linewidths=0)
        sstr = f"{sil:.2f}" if np.isfinite(sil) else "n/a"
        ax.set_title(f"{name}  (silhouette={sstr})"); ax.set_xticks([]); ax.set_yticks([])
    if nc <= 12:
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=min(nc, 6), frameon=False,
                   markerscale=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Penultimate-layer representation (UMAP) - {ds}  "
                 f"[same projection settings; higher silhouette = cleaner clusters]", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"embeddings_{ds}.{ext}")
    print(f"[wrote] {OUT}/embeddings_{ds}.pdf + .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nslkdd")
    ap.add_argument("--force", action="store_true",
                    help="recompute embeddings even if a cache exists")
    args = ap.parse_args()
    ds = args.dataset

    cached = None if args.force else _load_cache(ds)
    if cached is not None:
        print(f"[embed] {ds}: replotting from cache (use --force to recompute)")
        embs, yv, names = cached
    else:
        embs, yv, names = compute(ds)
    plot(ds, embs, yv, names)


if __name__ == "__main__":
    main()
