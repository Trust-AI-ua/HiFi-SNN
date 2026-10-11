"""Deployment serialization: store each artifact in the format its deployed
values actually require.

Rule (agreed 2026-09-22):

  Tree-to-Spike, DNN-Tree   int8 weights + per-tensor float32 scale; biases
                            float32 (they are not on the grid); encoder
                            threshold tables unchanged at float32, because the
                            thresholds are not on a grid and requantizing them
                            would move bin boundaries and change the model.
  Vanilla SNN, TASNN,       float32 (reported and executed in floating point).
  raw MLP (fp arm)
  GBDT full and compact     packed node arrays, the same format the compression
                            sweep prices: per internal node 2 B feature index +
                                    4 B float32 threshold + 4 B two uint16 children; one
                            value per leaf at ``leaf_bits`` (a per-class ensemble
                            stores one margin per leaf, not a K-vector).

SERIALIZATION_FORMAT is stamped into every artifact that uses this module so a
measurement made under the previous convention (float32 container for the
students, XGBoost native JSON for the trees) can never be confused with one made
under this rule.

Packed thresholds default to float32. One-byte thresholds were measured and
rejected: they cost 16.9 macro-F1 points on UNSW-NB15 and 46.4 on CICIoT2023, and
neither per-feature scaling nor 16-bit fields recovered it (CICIoT2023 still lost
21.5 points at 99.1% prediction agreement, because the changed predictions land in
rare classes). Leaf quantization is nearly free by comparison (16-bit costs at most
0.14 points), so leaves stay at 16 bits. ``predict_packed`` measures what the
packed bytes compute rather than assuming they match the native model.
"""
from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np

SERIALIZATION_FORMAT = "v2-int8-student-packed-tree"
INT8_SCALE_BYTES = 4      # one float32 scale per weight tensor
FLOAT_BYTES = 4


def _pickle_bytes(obj) -> int:
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp = f.name
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh, protocol=4)
        return os.path.getsize(tmp)
    finally:
        os.unlink(tmp)


# ------------------------------------------------------------------ students
def int8_student_payload(state_dict) -> dict:
    """Serializable payload for a student whose weights sit on the int8 grid.

    Each 2-D weight becomes int8 codes plus one float32 scale; everything else
    (biases, 1-D buffers) stays float32. Raises if a weight is not on its grid,
    since storing int8 codes would then be lossy.
    """
    payload, n_int8, n_scale, n_float = {}, 0, 0, 0
    for k, v in state_dict.items():
        a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        if a.ndim == 2:
            s0 = float(np.abs(a).max()) / 127.0
            if s0 == 0.0:
                q, s = np.zeros_like(a, dtype=np.int8), np.float32(0.0)
            else:
                q = np.round(a / s0).clip(-127, 127)
                # The scale that BUILT the artifact is not max|W|/127 recomputed after
                # the fact: quantization moves max|W| by an ULP, and quantize_int8_ adds
                # 1e-12. Recovering it from the values themselves (w / q on non-zero
                # codes) makes q * s reproduce the stored float32 exactly.
                nz = q != 0
                s = np.float32(np.median((a[nz] / q[nz]).astype(np.float64))) if nz.any() \
                    else np.float32(s0)
                if not np.array_equal(q.astype(np.int8).astype(np.float32) * s,
                                      a.astype(np.float32)):
                    raise ValueError(f"{k} is not on the int8 grid; refusing to store int8")
                q = q.astype(np.int8)
            payload[k] = {"q": q, "scale": np.float32(s)}
            n_int8 += q.size
            n_scale += 1
        else:
            payload[k] = a.astype(np.float32)
            n_float += a.size
    return {"payload": payload, "n_int8_weights": n_int8, "n_scales": n_scale,
            "n_float_values": n_float}


def int8_student_bytes(state_dict) -> dict:
    """Byte breakdown for the int8 form, plus the real serialized size."""
    p = int8_student_payload(state_dict)
    return {"model_bytes": _pickle_bytes(p["payload"]),
            "model_int8_bytes": p["n_int8_weights"],
            "model_scale_bytes": p["n_scales"] * INT8_SCALE_BYTES,
            "model_float_bytes": p["n_float_values"] * FLOAT_BYTES,
            "n_int8_weights": p["n_int8_weights"], "n_scales": p["n_scales"],
            "n_float_values": p["n_float_values"],
            "serialization_format": SERIALIZATION_FORMAT}


def _is_scaled_identity(a) -> bool:
    """Exactly c * I: constant diagonal, every off-diagonal exactly zero."""
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        return False
    d = np.diag(a)
    return bool((d == d[0]).all() and np.array_equal(a - np.diag(d), np.zeros_like(a)))


def float32_model_bytes(state_dict) -> dict:
    """Float32 artifacts (Vanilla SNN, TASNN, raw MLP fp arm).

    A weight that is exactly ``c * I`` is stored as the scalar ``c``, by the same
    rule applied to the trees: an artifact is stored in the form its deployed
    values require. The Vanilla SNN keeps its leak inside ``LeakyParallel``'s
    ``nn.RNN`` as a frozen 128x128 hidden-to-hidden matrix equal to beta * I, 64 KB
    that inference does not need and the trainable-parameter count already
    excludes. The test is exact, so a matrix that is merely close to diagonal is
    stored in full. This shrinks a baseline, not Tree-to-Spike.
    """
    d, folded, folded_values = {}, [], 0
    for k, v in state_dict.items():
        a = (v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)).astype(np.float32)
        if _is_scaled_identity(a):
            folded.append(k)
            folded_values += a.size - 1
            d[k] = np.float32(np.diag(a)[0])          # the scalar is all that ships
        else:
            d[k] = a
    n = int(sum(np.asarray(a).size for a in d.values()))
    return {"model_bytes": _pickle_bytes(d), "model_float_bytes": n * FLOAT_BYTES,
            "n_float_values": n, "identity_tensors_folded": ";".join(folded),
            "identity_values_folded": folded_values,
            "serialization_format": SERIALIZATION_FORMAT}


# --------------------------------------------------------------------- trees
def pack_booster(teacher, n_rounds=None, leaf_bits: int = 16, n_classes: int = 2,
                 threshold_bits: int = 32) -> dict:
    """Packed node arrays for an XGBoost ensemble, with a decodable payload.

    Returns the arrays, the byte count under the sweep's format, and the
    quantization error introduced by the 8-bit thresholds.
    """
    bst = teacher.get_booster()
    df = bst.trees_to_dataframe()
    if n_rounds is not None:
        keep = sorted(df.Tree.unique())[:n_rounds * max(1, n_classes if n_classes > 2 else 1)]
        df = df[df.Tree.isin(keep)]
    internal = df[df.Feature != "Leaf"]
    leaves = df[df.Feature == "Leaf"]
    thr = internal.Split.to_numpy(dtype=np.float64)
    levels = float(2 ** threshold_bits - 1)
    thr_q = np.round(np.clip(thr, 0.0, 1.0) * levels)
    thr_deq = thr_q / levels
    n_int, n_leaf = len(internal), len(leaves)
    # ONE value per leaf, for every K. A multiclass XGBoost ensemble is K trees
    # per round, each tree scoring its own class, so a leaf holds one margin, not
    # a K-vector; ``trees_to_dataframe`` shows one Gain per leaf row. Pricing K
    # values at every leaf inflated multiclass teachers by 1.7x to 3.2x.
    deploy_bytes = (n_int * (2 + threshold_bits // 8 + 4)
                    + n_leaf * (leaf_bits // 8))
    return {"n_internal": int(n_int), "n_leaves": int(n_leaf),
            "deploy_bytes": int(deploy_bytes), "leaf_bits": leaf_bits,
            "threshold_bits": threshold_bits,
            "threshold_max_abs_error": float(np.abs(thr - thr_deq).max()) if n_int else 0.0,
            "thresholds_outside_unit_interval": int(((thr < 0) | (thr > 1)).sum()),
            "frame": df, "thr_dequantized": thr_deq,
            "serialization_format": SERIALIZATION_FORMAT}


def predict_packed(packed: dict, X: np.ndarray, n_classes: int, base_score: float = 0.5,
                   leaf_bits: int = 16) -> np.ndarray:
    """Predictions of the packed artifact: 8-bit thresholds and quantized leaves.

    Traverses the dequantized tree frame directly, so what is measured is what
    the packed bytes would compute, not what the native model computes.
    """
    df = packed["frame"].copy()
    df["thr_q"] = np.nan
    mask = df.Feature != "Leaf"
    df.loc[mask, "thr_q"] = packed["thr_dequantized"]
    leaf_vals = df.loc[~mask, "Gain"].to_numpy(dtype=np.float64)
    if leaf_bits < 32 and len(leaf_vals):
        lo, hi = leaf_vals.min(), leaf_vals.max()
        step = (hi - lo) / (2 ** leaf_bits - 1) if hi > lo else 1.0
        df.loc[~mask, "Gain"] = lo + np.round((leaf_vals - lo) / step) * step
    nodes = {r.ID: r for r in df.itertuples()}
    trees = sorted(df.Tree.unique())
    roots = {t: df[(df.Tree == t) & (df.Node == 0)].ID.iloc[0] for t in trees}
    k = n_classes if n_classes > 2 else 1
    out = np.zeros((len(X), k), dtype=np.float64)
    for ti, t in enumerate(trees):
        node = np.array([roots[t]] * len(X), dtype=object)
        active = np.ones(len(X), dtype=bool)
        while active.any():
            for nid in np.unique(node[active]):
                r = nodes[nid]
                sel = active & (node == nid)
                if r.Feature == "Leaf":
                    out[sel, ti % k] += r.Gain
                    active[sel] = False
                    continue
                j = int(str(r.Feature)[1:])
                go_left = X[sel, j] < r.thr_q
                nxt = np.where(go_left, r.Yes, r.No)
                node[sel] = nxt
    if k == 1:
        p = 1.0 / (1.0 + np.exp(-(out[:, 0] + np.log(base_score / (1 - base_score)))))
        return (p >= 0.5).astype(np.int64)
    return out.argmax(1)
