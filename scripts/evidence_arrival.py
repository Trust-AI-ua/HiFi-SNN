#!/usr/bin/env python3
"""Evidence-arrival step t': when does the correct class first become the argmax?

Measured under the FULL window, independent of any confidence rule. For each sample
correct at T, t' is the first timestep whose argmax equals the true label. It is a
property of the taxonomy and the representation, not of the exit gate, so it gives a
principled floor on any exit threshold: no rule reading the prefix can be right before
the evidence arrives.

Reported as a distribution, not a mean -- if t' is bimodal (some siblings separating
early, others late) then a PER-CLASS exit threshold remains viable even where a global
one is not, and closing that door on the mean alone would be wrong.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from scripts.early_exit_audit import replay

DATASETS = ["kddcup99","nslkdd","ctu13","iscxids2012","cicids2017","cse-cic-ids2018",
            "unswnb15","ciciot2023","ciciomt2024","edgeiiot","toniot"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=DATASETS)
    a = ap.parse_args(); out = {}
    print(f"{'dataset':17s} {'T':>3} {'mean':>6} {'med':>5} {'p90':>5} {'%>T/2':>7} "
          f"{'undef':>6} {'wrong%':>7} {'modal target':>20} {'share':>6} {'bimod':>6}")
    for ds in a.datasets:
        try:
            conf, pred, y, names = replay(ds)
        except Exception as e:
            print(f"{ds:17s} FAILED: {str(e)[:40]}"); continue
        T = conf.shape[1]
        ok = pred[:, -1] == y                       # correct at full window
        hit = (pred == y[:, None]) & ok[:, None]
        has = hit.any(1)
        tp = hit.argmax(1)[has] + 1                 # 1-indexed
        yy = y[has]
        per_class = {names[c]: float(tp[yy == c].mean()) for c in np.unique(yy)}
        # Defined-t' fraction per class. A class never correct at T has no t' at all;
        # if many classes are in that state the aggregate averages over whichever
        # classes happen to be recoverable -- a taxonomy-vs-sample-budget property,
        # NOT an early-exit failure mode, and it belongs in a different paragraph.
        defined = {}
        for c in np.unique(y):
            n_c = int((y == c).sum())
            defined[names[c]] = dict(n=n_c,
                frac_defined=float((has & (y == c)).sum() / max(n_c, 1)),
                recall_full=float((pred[y == c, -1] == c).mean()))
        # Where do wrong exits LAND? Concentrated on one or two siblings implies an
        # attractor; diffuse implies late separation across the taxonomy.
        TH = 0.90
        fired = conf >= TH; ex = fired.any(1)
        t_star = np.where(ex, fired.argmax(1), T - 1)
        pe = pred[np.arange(len(y)), t_star]
        wrong = ex & (pe != y)
        if wrong.sum():
            tgt, cnt = np.unique(pe[wrong], return_counts=True)
            k = int(np.argmax(cnt))
            modal_target, modal_share = names[int(tgt[k])], float(cnt[k] / cnt.sum())
            top2 = float(np.sort(cnt)[::-1][:2].sum() / cnt.sum())
        else:
            modal_target, modal_share, top2 = "-", 0.0, 0.0
        vals = np.array(sorted(per_class.values()))
        # crude bimodality: gap between consecutive sorted class means vs spread
        gap = float(np.max(np.diff(vals))) if len(vals) > 1 else 0.0
        bimodal = gap > 0.35 * (vals.max() - vals.min() + 1e-9) and len(vals) > 2
        out[ds] = dict(T=int(T), mean=float(tp.mean()), median=float(np.median(tp)),
                       p90=float(np.percentile(tp, 90)),
                       frac_t1=float((tp == 1).mean()),
                       frac_late=float((tp > T/2).mean()),
                       per_class=per_class, bimodal=bool(bimodal), class_gap=gap,
                       defined=defined, modal_wrong_target=modal_target,
                       modal_wrong_share=modal_share, top2_wrong_share=top2,
                       wrong_exit_frac=float(wrong.mean()),
                       n_classes_undefined=int(sum(1 for v in defined.values()
                                                   if v["frac_defined"] < 0.05)))
        r = out[ds]
        print(f"{ds:17s} {T:3d} {r['mean']:6.2f} {r['median']:5.1f} {r['p90']:5.1f} "
              f"{r['frac_late']*100:6.1f}% {r['n_classes_undefined']:6d} "
              f"{r['wrong_exit_frac']*100:6.1f}% {r['modal_wrong_target'][:20]:>20} "
              f"{r['modal_wrong_share']*100:5.1f}% {str(r['bimodal']):>6}")
        json.dump(out, open(REPO/"results/evidence_arrival.json","w"), indent=2)
    print("\nper-class t' where bimodal (per-class thresholds may still be viable):")
    for ds, r in out.items():
        if r["bimodal"]:
            s = sorted(r["per_class"].items(), key=lambda kv: kv[1])
            print(f"  {ds}: " + ", ".join(f"{k}={v:.1f}" for k, v in s))
