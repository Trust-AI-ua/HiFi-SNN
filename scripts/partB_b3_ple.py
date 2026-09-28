#!/usr/bin/env python3
"""B3: target-aware piecewise-linear encoding with a dense MLP (pre-registered, Part B).

Reference package rtdl_num_embeddings 0.0.12 (Gorishniy et al.), installed with --no-deps.
Encoding: compute_bins(X_train, n_bins, y=y_train, regression=False,
tree_kwargs={"min_samples_leaf": 64, "min_impurity_decrease": 1e-4}) on training rows only,
then PiecewiseLinearEncoding (within-bin coordinates intact). Columns with fewer than two
distinct training values pass through unencoded.
Model: PLE -> Linear(d, W) -> ReLU -> Linear(W, W) -> ReLU -> Linear(W, K); no dropout.
Training: Adam, weight decay 0, batch 128, unweighted cross-entropy from labels (no
distillation). Validation macro-F1 after every epoch; the best epoch's checkpoint is kept
(ties: earliest). Same loaders, loader arguments, splits and seeds as the stored runs
(split guard in partB_common.load_bundle).
Precision: fp32 (selected checkpoint), per-tensor int8 and per-channel int8 of every
nn.Linear weight (the production quantizers); bin edges stay float32.
Accounting: MACs = sum of Linear in x out + encoded features x bins (encoding);
energy at 4.6 pJ per 32-bit MAC and 0.23 pJ per int8 MAC; encoder storage = bin edges x 4 B.

    python scripts/partB_b3_ple.py search <dataset> <config_id 0-15>
    python scripts/partB_b3_ple.py select                  # per dataset, on validation macro-F1
    python scripts/partB_b3_ple.py final <dataset> <seed>  # selected configuration
Output: results/journal_partB/B3/{search,final}/<dataset>/<cfg|seed>/results.json
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

GRID = [dict(n_bins=b, width=w, lr=lr, epochs=e)
        for b, w, lr, e in itertools.product((16, 64), (128, 256), (1e-3, 3e-4), (10, 30))]
TREE_KW = {"min_samples_leaf": 64, "min_impurity_decrease": 1e-4}
BATCH = 128
E_MAC32, E_MAC8 = 4.6, 0.23
OUT = C.REPO / "results/journal_partB/B3"


class PLEMLP(nn.Module):
    def __init__(self, bins, enc_cols, raw_cols, width, n_classes):
        super().__init__()
        from rtdl_num_embeddings import PiecewiseLinearEncoding
        self.enc_cols, self.raw_cols = enc_cols, raw_cols
        self.ple = PiecewiseLinearEncoding(bins)
        d = int(self.ple.get_output_shape()[0]) + len(raw_cols)
        self.fc1, self.fc2, self.fc3 = nn.Linear(d, width), nn.Linear(width, width), nn.Linear(width, n_classes)
        self.d_in = d

    def forward(self, x):
        z = self.ple(x[:, self.enc_cols])
        if self.raw_cols:
            z = torch.cat([z, x[:, self.raw_cols]], dim=1)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(z)))))


def macs(m: PLEMLP):
    lin = sum(l.in_features * l.out_features for l in (m.fc1, m.fc2, m.fc3))
    return lin + int(m.ple.get_output_shape()[0]), lin


@torch.no_grad()
def predict(m, X, bs=65536):
    m.eval()
    out = [m(X[i:i + bs]) for i in range(0, len(X), bs)]
    logits = torch.cat(out)
    return logits.argmax(1).cpu().numpy(), torch.softmax(logits, 1).cpu().numpy()


def run(ds, seed, cfg, out_dir):
    from sklearn.metrics import f1_score
    from rtdl_num_embeddings import compute_bins
    from src.metrics import compute_metrics, compute_binary_ids_metrics
    from src.models.tree_to_spike import quantize_int8_, quantize_int8_per_channel_
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = C.stamp("B3")
    b, (Xtr, ytr, Xva, yva, Xte, yte) = C.load_bundle(ds, seed)
    nuniq = np.array([np.unique(Xtr[:, j]).size for j in range(Xtr.shape[1])])
    enc_cols = [int(j) for j in np.where(nuniq >= 2)[0]]
    raw_cols = [int(j) for j in np.where(nuniq < 2)[0]]
    t0 = time.time()
    bins = compute_bins(torch.from_numpy(Xtr[:, enc_cols]), n_bins=cfg["n_bins"], y=torch.from_numpy(ytr),
                        regression=False, tree_kwargs=TREE_KW)
    bin_time = time.time() - t0
    m = PLEMLP(bins, enc_cols, raw_cols, cfg["width"], b.num_classes).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=0.0)
    lossf = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    Xva_t, Xte_t = torch.from_numpy(Xva).to(dev), torch.from_numpy(Xte).to(dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    curve, best, best_state = [], -1.0, None
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        m.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(dev)
        run_loss, nb = 0.0, 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = lossf(m(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            run_loss += loss.item(); nb += 1
        pv, _ = predict(m, Xva_t)
        vf1 = 100 * f1_score(yva, pv, average="macro", zero_division=0)
        curve.append(dict(epoch=ep, train_loss=run_loss / nb, val_macro_f1=vf1, val_accuracy=100 * float((pv == yva).mean())))
        if vf1 > best:                      # strict: ties keep the earlier epoch
            best, best_state, best_ep = vf1, copy.deepcopy(m.state_dict()), ep
        print(f"[{ds} s{seed} {cfg}] epoch {ep} loss {run_loss/nb:.4f} val macro-F1 {vf1:.2f}", flush=True)
    train_time = time.time() - t0
    m.load_state_dict(best_state)
    res = {}
    for name, fn in (("fp32", None), ("per_tensor_int8", quantize_int8_), ("per_channel_int8", quantize_int8_per_channel_)):
        mm = m if fn is None else fn(copy.deepcopy(m))
        p, s = predict(mm, Xte_t)
        res[name] = dict(metrics=compute_metrics(yte, p, y_score=s, class_names=b.class_names),
                         binary_ids_metrics=compute_binary_ids_metrics(yte, p, normal_class_index=0))
    total_macs, lin_macs = macs(m)
    rec = dict(stamp=st, dataset=ds, seed=seed, config=cfg, tree_kwargs=TREE_KW, batch_size=BATCH,
               loader_kwargs=C.loader_kwargs(ds, seed), package="rtdl_num_embeddings==0.0.12",
               data=dict(n_test=int(len(yte)), class_counts_train=b.class_counts_train, num_classes=b.num_classes,
                         encoded_features=len(enc_cols), passthrough_features=len(raw_cols),
                         bins_per_feature=[int(len(x) - 1) for x in bins]),
               model=dict(d_in=m.d_in, parameters=int(sum(p.numel() for p in m.parameters())),
                          encoder_storage_bytes=int(sum(len(x) for x in bins) * 4),
                          macs_total=int(total_macs), macs_linear=int(lin_macs),
                          energy_pj_32bit=total_macs * E_MAC32, energy_pj_int8=total_macs * E_MAC8),
               training=dict(curve=curve, best_epoch=best_ep, best_val_macro_f1=best, train_time_s=train_time,
                             bin_time_s=bin_time, device=str(dev)),
               test=res)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "checkpoint_fp32.pt")
    torch.save(dict(bins=[x.cpu() for x in bins], enc_cols=enc_cols, raw_cols=raw_cols), out_dir / "bins.pt")
    (out_dir / "results.json").write_text(json.dumps(rec, indent=1, default=str))
    f = lambda k: 100 * res[k]["metrics"]["f1_macro"]
    print(f"[done] {ds} s{seed} {cfg}: best epoch {best_ep} val {best:.2f} | test fp32 {f('fp32'):.2f} "
          f"pt {f('per_tensor_int8'):.2f} pc {f('per_channel_int8'):.2f}", flush=True)


def select():
    sel = {}
    for ds in C.IOT:
        rs = []
        for i in range(len(GRID)):
            p = OUT / f"search/{ds}/cfg_{i:02d}/results.json"
            if not p.exists():
                sys.exit(f"{ds}: search cell {i} missing")
            r = json.loads(p.read_text())
            rs.append((r["training"]["best_val_macro_f1"], -r["model"]["macs_total"], i, r["training"]["best_epoch"]))
        best = max(rs)                         # highest validation macro-F1; ties: fewer MACs
        sel[ds] = dict(config_id=best[2], config=GRID[best[2]], val_macro_f1=best[0], best_epoch=best[3])
    (OUT / "selection.json").write_text(json.dumps(dict(stamp=C.stamp("B3"), selection=sel), indent=1, default=str))
    print(json.dumps(sel, indent=1))


def main():
    C.no_thread_caps()
    mode = sys.argv[1]
    if mode == "search":
        ds, i = sys.argv[2], int(sys.argv[3])
        d = OUT / f"search/{ds}/cfg_{i:02d}"
        if (d / "results.json").exists():
            print(f"[skip] {d}"); return
        run(ds, 42, GRID[i], d)
    elif mode == "select":
        select()
    elif mode == "final":
        ds, seed = sys.argv[2], int(sys.argv[3])
        sel = json.loads((OUT / "selection.json").read_text())["selection"][ds]
        d = OUT / f"final/{ds}/seed_{seed}"
        if (d / "results.json").exists():
            print(f"[skip] {d}"); return
        if seed == 42:                         # seed 42 reuses its search run (pre-registered)
            src = OUT / f"search/{ds}/cfg_{sel['config_id']:02d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "REUSED_FROM.txt").write_text(str(src.relative_to(C.REPO)) + "\n")
            return
        run(ds, seed, sel["config"], d)


if __name__ == "__main__":
    main()
