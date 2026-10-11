#!/usr/bin/env python3
"""Is max-posterior the wrong exit statistic? Two diagnostics plus a candidate remedy.

D1  argmax trajectory. For samples that exit WRONG under the max-posterior gate, does
    the prediction later flip to the correct class if the window is allowed to run?
    The mean flip step t' is the honest lower bound on a usable exit step.

D2  margin at exit. Is top1-top2 systematically smaller for wrong exits? If so the
    failure is that max-posterior ignores how close the runner-up is, and a margin
    gate is a remedy that needs no architectural change.

Then simulates a margin gate over a grid and reports what fraction of the timestep
saving survives and at what accuracy cost -- turning a limitation into a proposal,
or showing that it cannot be rescued that way.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))

def replay_top2(ds, seed=42):
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
    T = cfg["T"]; C1, C2, P = [], [], []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            sp = enc.encode(Xte[i:i+512]).to(dev)
            m1, m2, m3 = m.lif1.init_leaky(), m.lif2.init_leaky(), m.lif3.init_leaky()
            acc = torch.zeros(sp.shape[1], m.fc3.out_features, device=dev)
            c1, c2, pp = [], [], []
            for t in range(T):
                s1, m1 = m.lif1(m.fc1(sp[t]), m1)
                s2, m2 = m.lif2(m.fc2(s1), m2)
                s3, m3 = m.lif3(m.fc3(s2), m3)
                acc = acc + m3
                p = F.softmax(acc/(t+1), -1)
                v, k = p.topk(2, -1)
                c1.append(v[:,0].cpu().numpy()); c2.append(v[:,1].cpu().numpy())
                pp.append(k[:,0].cpu().numpy())
            C1.append(np.stack(c1,1)); C2.append(np.stack(c2,1)); P.append(np.stack(pp,1))
    return (np.concatenate(C1), np.concatenate(C2), np.concatenate(P), yte, b.class_names)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="ciciot2023")
    a = ap.parse_args()
    c1, c2, pred, y, names = replay_top2(a.dataset)
    T = c1.shape[1]; N = len(y)
    full = pred[:, -1]; full_acc = (full == y).mean()*100
    TH = 0.90
    fired = c1 >= TH; ex = fired.any(1)
    t_star = np.where(ex, fired.argmax(1), T-1)
    pe = pred[np.arange(N), t_star]
    wrong = ex & (pe != y)
    print(f"{a.dataset}: full acc {full_acc:.2f} | theta={TH} exit {ex.mean()*100:.1f}% "
          f"| wrong exits {wrong.sum()} ({wrong.mean()*100:.1f}% of all)")

    print("\n--- D1: does the prediction recover if the window runs? ---")
    correct_later = (pred == y[:, None])
    has = correct_later[wrong].any(1)
    firstc = np.where(has, correct_later[wrong].argmax(1), -1)
    after = has & (firstc > t_star[wrong])
    print(f"  wrong-exit samples whose argmax reaches the true class later: "
          f"{has.mean()*100:.1f}%")
    print(f"  ...and does so AFTER the exit step:                          "
          f"{after.mean()*100:.1f}%")
    if after.any():
        print(f"  mean exit step t*        : {t_star[wrong][after].mean()+1:.2f}")
        print(f"  mean recovery step t'    : {firstc[after].mean()+1:.2f}")
        print(f"  -> usable exit lower bound for this dataset: ~{firstc[after].mean()+1:.1f}/{T}")

    print("\n--- D2: margin (top1-top2) at exit, right vs wrong ---")
    marg = c1 - c2
    me = marg[np.arange(N), t_star]
    r = ex & (pe == y)
    print(f"  right exits: margin mean {me[r].mean():.4f}  median {np.median(me[r]):.4f}")
    print(f"  wrong exits: margin mean {me[wrong].mean():.4f}  median {np.median(me[wrong]):.4f}")
    print(f"  separation (right-wrong): {me[r].mean()-me[wrong].mean():+.4f}")

    print("\n--- remedy: margin gate (exit when top1-top2 >= m) ---")
    print(f"  {'m':>6} {'exit%':>7} {'mean_exit':>10} {'acc':>8} {'gap':>8} {'saving_kept':>12}")
    base_save = T - (t_star.mean()+1)
    for m_ in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        f2 = marg >= m_; e2 = f2.any(1)
        t2 = np.where(e2, f2.argmax(1), T-1)
        p2 = pred[np.arange(N), t2]
        acc = (p2 == y).mean()*100
        save = T - (t2.mean()+1)
        print(f"  {m_:6.2f} {e2.mean()*100:6.1f}% {t2.mean()+1:10.2f} {acc:8.2f} "
              f"{acc-full_acc:+8.2f} {save/base_save*100:11.1f}%")
