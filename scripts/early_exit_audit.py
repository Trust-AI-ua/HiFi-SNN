#!/usr/bin/env python3
"""Why does the validation gate disable early exit on 5 of 11 datasets?

Two stories with very different consequences:

  A  exit is well calibrated but almost nothing reaches theta before T, so the gate
     correctly declines to save anything. Benign: early exit simply does not help.
  B  exit fires early on the WRONG class -- prior-driven saturation of the prefix
     posterior -- the accuracy gap blows the bound, and the gate disables it as a
     safety measure. That is a calibration failure, and it means the datasets where
     the gate PASSED need auditing too, since the same saturation may be present
     just under the accuracy threshold.

Distinguishing signature: the exited-sample accuracy and the class composition of
early exits. Under A the exit fraction is near zero. Under B the exit fraction is
high and exits concentrate on the majority class.

No training: checkpoints are reloaded and the prefix posterior is replayed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
GRID = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]

def replay(ds, seed=42):
    from src.data import get_dataset
    from src.models import TreeGuidedSpikeEncoder, SNNStudent, quantize_int8_
    from src.training_treespike import _arrays_from_bundle, _fit_teacher
    import yaml
    cfg = yaml.safe_load(open(REPO/f"experiments/Tree-to-Spike/{ds}/config.yaml"))
    dk = dict(seed=seed)
    for o in ("subsample_fraction","test_fraction","label_mode"):
        if o in cfg: dk[o] = cfg[o]
    b = get_dataset(ds, **dk)
    Xtr, ytr, _, _, Xte, yte = _arrays_from_bundle(b)
    teacher, _, _ = _fit_teacher(Xtr, ytr, b.num_classes, {**cfg, "seed": seed})
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
            prune_quantile=cfg["prune_quantile"], jitter=cfg["jitter"]).fit(
            teacher, Xtr.shape[1], X=Xtr)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SNNStudent(enc.out_dim, b.num_classes, hidden=tuple(cfg["hidden"]),
                   beta=cfg["beta"], leaky=cfg["leaky"]).to(dev)
    m.load_state_dict(torch.load(REPO/f"results/Tree-to-Spike/{ds}/seed_{seed}/checkpoint.pt",
                                 map_location=dev))
    if cfg.get("quantize"): quantize_int8_(m)
    m.eval()

    T = cfg["T"]; conf_t, pred_t = [], []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            sp = enc.encode(Xte[i:i+512]).to(dev)
            m1, m2, m3 = m.lif1.init_leaky(), m.lif2.init_leaky(), m.lif3.init_leaky()
            acc = torch.zeros(sp.shape[1], m.fc3.out_features, device=dev)
            cs, ps = [], []
            for t in range(T):
                s1, m1 = m.lif1(m.fc1(sp[t]), m1)
                s2, m2 = m.lif2(m.fc2(s1), m2)
                s3, m3 = m.lif3(m.fc3(s2), m3)
                acc = acc + m3
                p = F.softmax(acc/(t+1), -1); c, k = p.max(-1)
                cs.append(c.cpu().numpy()); ps.append(k.cpu().numpy())
            conf_t.append(np.stack(cs, 1)); pred_t.append(np.stack(ps, 1))
    return np.concatenate(conf_t), np.concatenate(pred_t), yte, b.class_names

def audit(conf, pred, y, names, ds):
    T = conf.shape[1]; full_acc = (pred[:, -1] == y).mean()*100
    maj = int(np.bincount(y).argmax())
    print(f"\n{'='*72}\n{ds}: full-window acc {full_acc:.2f} | T={T} | "
          f"majority class '{names[maj]}' = {(y==maj).mean():.1%} of test\n{'='*72}")
    print(f"{'theta':>6} {'exit%':>7} {'mean_exit':>10} {'acc_exited':>11} "
          f"{'acc_final':>10} {'gap':>7} {'exits on maj%':>14}")
    for th in GRID:
        fired = conf >= th
        first = np.where(fired.any(1), fired.argmax(1), T-1)
        exited = fired.any(1)
        pe = pred[np.arange(len(y)), first]
        acc_ex = (pe[exited] == y[exited]).mean()*100 if exited.any() else float("nan")
        acc_all = (pe == y).mean()*100
        onmaj = (pe[exited] == maj).mean()*100 if exited.any() else float("nan")
        print(f"{th:6.2f} {exited.mean()*100:6.1f}% {first.mean()+1:10.2f} "
              f"{acc_ex:11.2f} {acc_all:10.2f} {acc_all-full_acc:+7.2f} {onmaj:13.1f}%")
    # per-class behaviour at the deployed threshold
    th = 0.90; fired = conf >= th
    first = np.where(fired.any(1), fired.argmax(1), T-1); exited = fired.any(1)
    pe = pred[np.arange(len(y)), first]
    print(f"\n  per-class at theta=0.90:")
    print(f"  {'class':22s} {'n':>7} {'exit%':>7} {'mean_exit':>10} {'recall_exit':>12} {'recall_full':>12}")
    for c in np.unique(y):
        m_ = y == c
        rf = (pred[m_, -1] == c).mean()*100
        re = (pe[m_] == c).mean()*100
        print(f"  {names[c][:22]:22s} {m_.sum():7d} {exited[m_].mean()*100:6.1f}% "
              f"{first[m_].mean()+1:10.2f} {re:12.2f} {rf:12.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ciciot2023", "edgeiiot"])
    a = ap.parse_args()
    for ds in a.datasets:
        try:
            audit(*replay(ds), ds)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[fail] {ds}: {e}")
