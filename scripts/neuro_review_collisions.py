#!/usr/bin/env python3
"""Amendment 8, part 4 summary: multichannel collisions of the deployed encoders, pooled
(results/preregistration/2026-11-17_neuro_stream.md). Runs after scripts/neuro_encoder_tables.py.

    python3 scripts/neuro_review_collisions.py
Output: results/neuro/review/p3_encoders/{collisions.csv, COLLISIONS.md}
"""
import glob
import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "results/neuro/review/p3_encoders"


def main():
    rows, feats = [], []
    for f in sorted(glob.glob(str(D / "*_seed*_*.json"))):
        j = json.loads(Path(f).read_text())
        rows.append(dict(dataset=j["dataset"], seed=j["seed"], model=j["model"], T=j["T"], n_features_retained=j["n_features_retained"],
                         n_channels=j["n_channels"], teacher_omp=j["teacher_omp_num_threads"], **j["summary"]))
        feats += [dict(dataset=j["dataset"], seed=j["seed"], model=j["model"], **{k: v for k, v in x.items()
                       if k not in ("thresholds", "channels")}, n_channels=len(x["channels"])) for x in j["features"]]
    R, F = pd.DataFrame(rows), pd.DataFrame(feats)
    R.to_csv(D / "collisions.csv", index=False)
    L = ["# Multichannel collisions of the deployed Tree-to-Spike encoders (amendment 8, part 4)", "",
         "A retained feature's level k (0 to |E_j|) emits one spike time per channel; two levels collide when every channel "
         "emits the same time. The single-channel bound |E_j| >= T is the criterion behind the paper's 45.9% "
         "(computed there on the diagnostic encoder tables, pooled IoT); here it is recomputed on the production encoders. "
         "Every table was checked to reproduce the deployed spike times on the whole test split.", "",
         "| Encoder | Dataset | Seeds | Retained features | Bound violated | Features with a collision | Level distinctions lost | "
         "... with the centre channel alone | Test values in a merged tuple (mean over features) |", "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for (m, ds), g in R.groupby(["model", "dataset"]):
        fg = F[(F.model == m) & (F.dataset == ds)]
        L.append(f"| {m} (T = {g['T'].iloc[0]}) | {ds} | {len(g)} | {g.n_features_retained.mean():.1f} | "
                 f"{100 * fg.single_channel_bound_violated.mean():.1f}% | {100 * fg.collides.mean():.1f}% | "
                 f"{100 * fg.levels_lost.sum() / fg.n_levels.sum():.1f}% | {100 * fg.levels_lost_centre_channel_alone.sum() / fg.n_levels.sum():.1f}% | "
                 f"{100 * fg.test_values_in_merged_tuple.mean():.1f}% |")
    p = F[F.model == "production"]
    multi = p[p.n_channels > 1]
    L += ["", f"Pooled over the four IoT datasets and five seeds (production, T = 20): {len(p)} retained features; "
          f"{100 * p.single_channel_bound_violated.mean():.1f}% violate the single-channel bound; "
          f"{100 * p.collides.mean():.1f}% have at least one collision among their emitted tuples; "
          f"{100 * p.levels_lost.sum() / p.n_levels.sum():.1f}% of all level distinctions are lost "
          f"({100 * p.levels_lost_centre_channel_alone.sum() / p.n_levels.sum():.1f}% with the centre channel alone); "
          f"a mean {100 * p.test_values_in_merged_tuple.mean():.1f}% of a feature's test values fall in a merged tuple.",
          f"Features with more than one channel: {len(multi)} ({100 * len(multi) / len(p):.1f}%); among them "
          f"{100 * multi.collides.mean():.1f}% collide, and the extra channels recover "
          f"{int((multi.levels_lost_centre_channel_alone - multi.levels_lost).sum())} of "
          f"{int(multi.levels_lost_centre_channel_alone.sum())} levels the centre channel alone would merge."]
    (D / "COLLISIONS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
