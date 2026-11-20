#!/usr/bin/env python3
"""Amendment 8, part 1 summary: stagewise predictions on Edge-IIoTset (results/preregistration/2026-11-17_neuro_stream.md).
Runs in .venv after scripts/neuro_stagewise.py (val and test, seeds 42-46).

  predictions_<split>_seed<S>.csv.gz  flow_id, true class, and the predicted class of every model and stage
  stage_f1.csv                         per split, seed and stage: macro-F1, agreement with each model's first stage
  per_class_f1.csv                     per split, seed, stage and class: F1
  STAGEWISE.md                         five-seed tables: the stage at which each model loses accuracy, the classes
                                       affected, and the Tree-to-Spike minus Vanilla gap at each matched stage

    .venv/bin/python scripts/neuro_review_stagewise_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results/neuro/review/p1_stagewise"
SEEDS = (42, 43, 44, 45, 46)
V_STAGES = ["float_gpu", "float64", "proj_w", "proj_w_leak12", "integer_q8", "lava", "lava_nosat"]
T_STAGES = ["fp32", "float", "integer_q8", "lava"]
LABEL = {"float_gpu": "float (GPU, record)", "float64": "float64 (CPU)", "proj_w": "int8-projected weights",
         "proj_w_leak12": "projected + 12-bit leak", "integer_q8": "integer Q = 8", "lava": "Lava fixed point",
         "lava_nosat": "Lava, no saturation", "fp32": "fp32 (before projection)", "float": "int8-projected weights (deployed)"}
# matched stages: (label, vanilla stage, student stage)
MATCHED = [("float, before projection", "float_gpu", "fp32"), ("int8-projected weights, float dynamics", "proj_w", "float"),
           ("integer Q = 8", "integer_q8", "integer_q8"), ("Lava fixed point", "lava", "lava")]


def main():
    from sklearn.metrics import f1_score
    rows, pc = [], []
    for split in ("val", "test"):
        for s in SEEDS:
            f = D / f"edgeiiot_seed{s}_{split}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            y, names = z["labels"], [str(c) for c in z["class_names"]]
            preds = {k[5:]: z[k] for k in z.files if k.startswith("pred_")}
            cols = dict(flow_id=z["flow_id"], true_class=y, true_class_name=np.array(names)[y])
            cols.update({k: v for k, v in preds.items()})
            pd.DataFrame(cols).to_csv(D / f"predictions_{split}_seed{s}.csv.gz", index=False)
            for k, p in preds.items():
                model, stage = ("vanilla", k[8:]) if k.startswith("vanilla_") else (("t2s_T10", k[8:]) if k.startswith("t2s_T10_") else ("t2s", k[4:]))
                first = preds["vanilla_float_gpu"] if model == "vanilla" else preds.get(f"{model}_fp32", preds.get(f"{model}_float"))
                rows.append(dict(split=split, seed=s, model=model, stage=stage, macro_f1=100 * f1_score(y, p, average="macro"),
                                 agreement_with_first_stage=float((p == first).mean())))
                for c, v in enumerate(100 * f1_score(y, p, average=None, labels=range(len(names)))):
                    pc.append(dict(split=split, seed=s, model=model, stage=stage, class_id=c, class_name=names[c], f1=v,
                                   support=int((y == c).sum())))
    R, P = pd.DataFrame(rows), pd.DataFrame(pc)
    R.to_csv(D / "stage_f1.csv", index=False); P.to_csv(D / "per_class_f1.csv", index=False)
    ms = lambda x: f"{x.mean():.2f} ({x.std(ddof=1):.2f})"
    L = ["# Stagewise predictions, Edge-IIoTset, seeds 42-46 (amendment 8, part 1)", "",
         "Macro-F1, five-seed mean (SD). Every flow of the split. Checks (test): the float stages reproduce the records, "
         "integer Q = 8 equals the saved integer predictions, Lava equals the confirmation runs (full-test macro-F1 and "
         "Lava/SANA-FE's predictions on the 2,000-flow sample); the student's fp32 checkpoint projects onto the deployed one.", ""]
    for split in ("test", "val"):
        r = R[R.split == split]
        if r.empty:
            continue
        L += [f"## {split} split", "", "| Model | Stage | Seeds | Macro-F1 | Change from previous stage | Agreement with first stage |",
              "|---|---|--:|--:|--:|--:|"]
        for model, stages in (("vanilla", V_STAGES), ("t2s", T_STAGES), ("t2s_T10", ["float", "integer_q8", "lava"])):
            prev = None
            for st in stages:
                g = r[(r.model == model) & (r.stage == st)].set_index("seed")
                if g.empty:
                    continue
                d = "" if prev is None or st == "lava_nosat" else ms(g.macro_f1 - prev.reindex(g.index))
                L.append(f"| {model} | {LABEL.get(st, st)} | {len(g)} | {ms(g.macro_f1)} | {d} | {100 * g.agreement_with_first_stage.mean():.1f}% |")
                if st != "lava_nosat":
                    prev = g.macro_f1
        if split == "test":
            L += ["", "### Tree-to-Spike minus Vanilla SNN at matched stages (test)", "",
                  "| Stage | Vanilla | Tree-to-Spike | Gap | Gap per seed |", "|---|--:|--:|--:|---|"]
            for lab, vs, ts in MATCHED:
                a = r[(r.model == "vanilla") & (r.stage == vs)].set_index("seed").macro_f1
                b = r[(r.model == "t2s") & (r.stage == ts)].set_index("seed").macro_f1
                if a.empty or b.empty:
                    continue
                gap = (b - a).dropna()
                L.append(f"| {lab} | {ms(a)} | {ms(b)} | {ms(gap)} | " + ", ".join(f"{k}: {v:.2f}" for k, v in gap.items()) + " |")
            p = P[(P.split == "test")]
            a = p[(p.model == "vanilla") & (p.stage == "float_gpu")].groupby("class_name").f1.mean()
            b = p[(p.model == "vanilla") & (p.stage == "proj_w")].groupby("class_name").f1.mean()
            c = p[(p.model == "vanilla") & (p.stage == "lava")].groupby("class_name").f1.mean()
            t0 = p[(p.model == "t2s") & (p.stage == "fp32")].groupby("class_name").f1.mean()
            t1 = p[(p.model == "t2s") & (p.stage == "lava")].groupby("class_name").f1.mean()
            sup = p[(p.model == "vanilla") & (p.stage == "float_gpu")].groupby("class_name").support.mean()
            t = pd.DataFrame(dict(support=sup, v_float=a, v_proj=b, v_lava=c, t_fp32=t0, t_lava=t1))
            t["v_drop_at_projection"] = t.v_proj - t.v_float
            t = t.sort_values("v_drop_at_projection")
            L += ["", "### Per-class F1 (test, five-seed mean): where the Vanilla SNN loses", "",
                  "| Class | Test support (mean) | Vanilla float | Vanilla projected | Change at projection | Vanilla Lava | Student fp32 | Student Lava |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|"]
            for k, q in t.iterrows():
                L.append(f"| {k} | {q.support:.0f} | {q.v_float:.2f} | {q.v_proj:.2f} | {q.v_drop_at_projection:+.2f} | {q.v_lava:.2f} | "
                         f"{q.t_fp32:.2f} | {q.t_lava:.2f} |")
        L.append("")
    for s in SEEDS:
        f = D / f"edgeiiot_seed{s}_test_summary.json"
        if f.exists():
            j = json.loads(f.read_text())
            L.append(f"- seed {s}: Lava hidden-membrane saturation on {100 * j['lava_hidden_saturated_frac']:.3f}% of test flows; "
                     f"output spike-count ties in the float record on {100 * j['float_gpu_ties_frac']:.2f}%; "
                     f"weight scales s_h {j['weight_scale']['s_h']:.4f}, s_o {j['weight_scale']['s_o']:.4f}")
    (D / "STAGEWISE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
