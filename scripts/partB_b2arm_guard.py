#!/usr/bin/env python3
"""Guards for the B2 fourth arm (encoder_mode "entropy_tree", amendment A2), before any run.

  U   median_fill reproduces the entropy cutter's own fillers on a feature with no
      information (single-class labels), where every cut the cutter makes is a filler.
  G1  On every IoT dataset at seed 42, the encoder modes tree, quantile, uniform, all
      and entropy are bitwise identical to the encoder at 5d9baff (edges, channels,
      encode, encode_dense).
  G2  entropy_tree keeps the tree's channels exactly and never exceeds |E_j| cuts.
  G3  Each non-filler entropy_tree cut splits the training values exactly as the sklearn
      threshold it came from (x >= edge  <=>  x > threshold).
  G5  The per-sample weights given to the trees equal the teacher's (captured at
      XGBClassifier.fit).

    python scripts/partB_b2arm_guard.py
Output: results/journal_partB/B2arm_guard/<run_id>/guard.json (exit 1 on any failure)
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

REF = "5d9baff"


def committed_encoder():
    src = subprocess.check_output(["git", "show", f"{REF}:src/models/tree_to_spike.py"], cwd=C.REPO, text=True)
    tmp = Path(tempfile.mkdtemp()) / "tts_ref.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("tts_ref", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unit_fill():
    import src.models.tree_to_spike as NEW
    rng = np.random.default_rng(0)
    x = np.round(rng.random(5000), 3).astype(np.float32)
    y = np.zeros(5000, dtype=np.int64)
    ref, n_fill = NEW.entropy_cuts(x, y, 7)
    mine = NEW.median_fill(x, [], 7)
    return bool(n_fill == 7 and np.array_equal(np.sort(ref), np.sort(mine)))


def main():
    import xgboost
    import src.models.tree_to_spike as NEW
    from sklearn.tree import DecisionTreeClassifier
    C.no_thread_caps()
    st = C.stamp("B2arm_guard")
    out = C.run_dir(st)
    OLD = committed_encoder()
    ok_u = unit_fill()
    print(f"U median_fill reproduces the cutter's fillers: {ok_u}", flush=True)
    ok, rec = ok_u, {"U": ok_u}
    for ds in C.IOT:
        seed = 42
        cfg = C.t2s_config(ds, seed)
        b, arrays = C.load_bundle(ds, seed)
        Xtr, ytr, Xte = arrays[0], arrays[1], arrays[4]
        seen, orig_fit = {}, xgboost.XGBClassifier.fit

        def spy(self, X, y, *a, sample_weight=None, **kw):
            seen["w"] = np.asarray(sample_weight).copy()
            return orig_fit(self, X, y, *a, sample_weight=sample_weight, **kw)
        xgboost.XGBClassifier.fit = spy
        try:
            teacher, _ = C.fit_teacher_guarded(ds, seed, b, arrays)
        finally:
            xgboost.XGBClassifier.fit = orig_fit
        power = float(cfg.get("teacher_cost_power", 1.0))
        w = NEW.teacher_class_weights(ytr, power)
        r = {"G5": bool(np.array_equal(seen["w"], w[ytr]))}
        mk = lambda M, mode: M.TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"],
                                                      prune_quantile=cfg["prune_quantile"], jitter=cfg["jitter"],
                                                      encoder_mode=mode, cost_power=power).fit(teacher, Xtr.shape[1], X=Xtr, y=ytr)
        Xs = Xte[:20000]
        for mode in ("tree", "quantile", "uniform", "all", "entropy"):
            eo, en = mk(OLD, mode), mk(NEW, mode)
            same = (eo.edges_.keys() == en.edges_.keys()
                    and all(np.array_equal(eo.edges_[j], en.edges_[j]) for j in eo.edges_)
                    and eo.channels_ == en.channels_
                    and np.array_equal(eo.encode(Xs).numpy(), en.encode(Xs).numpy())
                    and np.array_equal(eo.encode_dense(Xs).numpy(), en.encode_dense(Xs).numpy()))
            r[f"G1_{mode}"] = bool(same)
        tree = mk(NEW, "tree")
        et = mk(NEW, "entropy_tree")
        kept = sorted({j for j, _ in tree.channels_})
        r["G2_channels"] = et.channels_ == tree.channels_
        r["G2_counts"] = all(len(et.edges_[j]) <= len(tree.edges_[j]) for j in kept)
        g3 = True
        sw = w[ytr].astype(np.float64)
        for j in kept:
            x = Xtr[:, j]
            u = np.unique(x)
            if u.size < 2:
                continue
            clf = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=len(tree.edges_[j]) + 1, random_state=0)
            clf.fit(x.reshape(-1, 1), ytr, sample_weight=sw)
            thr = clf.tree_.threshold[clf.tree_.children_left != -1]
            e = et.edges_[j]
            for t in thr:
                edge = u[np.searchsorted(u.astype(np.float64), t, side="right")]
                if edge not in e or not np.array_equal(u >= edge, u.astype(np.float64) > t):
                    g3 = False
        r["G3"] = g3
        d = et.diagnostics_
        r["filler_share"] = d["filler_total"] / max(d["edges_entropy_total"], 1)
        r["edges"] = [d["edges_entropy_total"], d["edges_tree_total"]]
        rec[ds] = r
        ok &= all(v for k, v in r.items() if k.startswith("G"))
        print(f"{ds}: {r}", flush=True)
    (out / "guard.json").write_text(json.dumps(dict(stamp=st, ref=REF, ok=bool(ok), per_dataset=rec), indent=1, default=str))
    print("ALL GUARDS PASS" if ok else "GUARD FAILURE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
