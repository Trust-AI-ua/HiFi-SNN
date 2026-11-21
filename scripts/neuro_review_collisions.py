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
    L += ["", "## Definitions", "",
          "For a retained feature j with |E_j| thresholds and m_j channels (offsets o_c):",
          "",
          "- **Level** k in 0 .. |E_j|: the number of thresholds at or below the value; the feature has |E_j| + 1 levels.",
          "- **Emitted tuple** of level k: (t_1(k), ..., t_m(k)), t_c(k) = round((1 - clip(k/|E_j| + o_c, 0, 1)) (T - 1)) "
          "(float32 level, NumPy round half to even).",
          "- **Distinct tuples** D_j: the number of different tuples over all |E_j| + 1 levels.",
          "- **Bound violated**: |E_j| >= T (a single channel has T spike times, so |E_j| + 1 > T levels must share a time). "
          "Share = violating features / retained features, pooled over features (the 45.9% definition).",
          "- **Collides**: D_j < |E_j| + 1, i.e. at least two levels emit the same tuple on every channel. "
          "Share = colliding features / retained features, pooled over features.",
          "- **Levels lost** for feature j: |E_j| + 1 - D_j. **Level distinctions lost** = sum over features of levels lost / "
          "sum over features of |E_j| + 1, pooled over features (so features with many thresholds dominate).",
          "- **... with the centre channel alone**: the same count using only the channel with the smallest |o_c|, "
          "i.e. what one channel would lose; the difference is what the extra channels recover.",
          "- **Test values in a merged tuple**: for feature j, the share of test flows whose level shares its tuple with at "
          "least one other level; the table reports the mean over features. The stricter share, where the other level also "
          f"occurs in the test split, is in the JSONs: pooled production mean {100 * p.test_values_merged_with_seen_level.mean():.2f}% "
          f"against {100 * p.test_values_in_merged_tuple.mean():.2f}%, largest per-feature difference "
          f"{100 * (p.test_values_in_merged_tuple - p.test_values_merged_with_seen_level).abs().max():.2f} points.",
          "- Counts are of the encoder function, not of classification loss: two merged levels cost accuracy only if they "
          "should be told apart."]
    (D / "COLLISIONS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
