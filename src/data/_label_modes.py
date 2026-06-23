"""Shared label-granularity switch for the SOTA-comparison datasets.

Each of CICIoT2023 / CICIoMT2024 / Edge-IIoTset is benchmarked in the literature at
three label granularities. We cache the *finest* labels and remap to the requested
mode at load time, so one cache serves all three:

  * ``"full"``     — the finest grained labels (34 / 19 / 15), native names kept.
  * ``"category"`` — the paper's coarse attack-category grouping (8 / 6 / 6).
  * ``"binary"``   — benign vs attack.

In every mode the benign class is index 0, so
``compute_binary_ids_metrics(normal_class_index=0)`` is valid.
"""
from __future__ import annotations

import numpy as np


def apply_label_mode(label_mode, y_fine, fine_names, category_fn, benign_fine):
    """Map fine label codes -> (y, class_names) for the requested granularity.

    y_fine: int64 codes into ``fine_names`` (benign must be at index 0).
    category_fn: maps a fine attack name -> its coarse category name.
    benign_fine: the fine name of the benign class (e.g. 'BenignTraffic' / 'Normal').
    """
    if label_mode == "full":
        return y_fine, list(fine_names)
    if label_mode == "binary":
        bi = fine_names.index(benign_fine)
        return (y_fine != bi).astype(np.int64), ["Benign", "Attack"]
    if label_mode == "category":
        cats = ["Benign" if n == benign_fine else category_fn(n) for n in fine_names]
        cat_names = ["Benign"] + sorted(set(cats) - {"Benign"})
        cidx = {c: i for i, c in enumerate(cat_names)}
        remap = np.array([cidx[c] for c in cats], dtype=np.int64)
        return remap[y_fine], cat_names
    raise ValueError(f"label_mode must be 'binary'|'category'|'full', got {label_mode!r}")
