"""Readers for the summary CSVs that respect row provenance.

After the leakage repair, results/tables/summary*.csv keep every row and mark it
with `row_provenance` (see scripts/tag_summary_provenance.py). Anything that
aggregates must go through read_summary so contaminated rows cannot leak back in.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CURRENT = ("unaffected", "post_repair")


def read_summary(path, include=CURRENT) -> pd.DataFrame:
    """Rows whose provenance is in ``include`` (default: the current numbers).

    Files written before tagging have no provenance column; they are refused
    rather than silently read, because they may mix pre- and post-repair rows.
    """
    df = pd.read_csv(Path(path))
    if "row_provenance" not in df.columns:
        raise ValueError(f"{path} has no row_provenance column; run "
                         "scripts/tag_summary_provenance.py first")
    return df[df.row_provenance.isin(include)].drop(columns="row_provenance")
