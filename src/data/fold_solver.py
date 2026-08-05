"""Constrained source-group fold assignment, solved exactly rather than greedily.

Round-2 follow-up. A greedy packer with a repair loop failed to place TON-IoT's
dominant endpoint (192.168.1.30: 55,344 rows, 48 benign) into any valid test
fold, but that is evidence about the heuristic, not about the problem. The
validity conditions constrain the *fold*, not the individual host, so the
dominant endpoint may legitimately share a fold with benign-rich endpoints.

This module states fold construction as a mixed-integer feasibility problem and
solves it with HiGHS via ``scipy.optimize.milp``. Either it returns an
assignment in which every source group appears in exactly one test fold and
every fold satisfies the frozen gate, or it returns a certificate of
infeasibility -- which is a defensible result, unlike a heuristic that gave up.

Model
-----
Binary z[h,k] = 1 iff source group h is assigned to fold k.

    (1)  sum_k z[h,k] = 1                              for every group h
    (2)  sum_h benign[h] z[h,k] >= min_benign          for every fold k
    (3)  sum_h attack[h] z[h,k] >= min_attack          for every fold k
    (4)  rows[h] z[h,k] - s * sum_j rows[j] z[j,k] <= 0    for every h, k
         (linearises "no group exceeds share s of its fold"; vacuous when z=0)
    (5)  lo <= sum_h rows[h] z[h,k] <= hi              row-balance band
    (6)  u[c,k] <= sum_{h: group h carries subclass c} z[h,k]    (u binary)
         sum_c u[c,k] >= min_subclasses                for every fold k

Objective is feasibility only (zero cost); HiGHS reports infeasibility exactly.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import lil_matrix


def solve_folds(groups: np.ndarray, y_bin: np.ndarray, y_fine: np.ndarray,
                n_folds: int, min_benign: int, min_attack: int,
                max_share: float, min_subclasses: int,
                balance_tol: float = 0.60, time_limit: float = 120.0) -> Dict:
    """Return {'feasible': bool, 'bins': [[group,...], ...], 'status': str}."""
    uniq = np.unique(groups)
    H, K = len(uniq), n_folds
    ben = np.array([int(((groups == g) & (y_bin == 0)).sum()) for g in uniq], float)
    att = np.array([int(((groups == g) & (y_bin == 1)).sum()) for g in uniq], float)
    tot = ben + att
    subs = sorted(set(y_fine.tolist()) - {0})
    carries = np.array([[float(((groups == g) & (y_fine == c)).any()) for c in subs]
                        for g in uniq])                      # [H, S]
    S = len(subs)

    nz = H * K                       # z[h,k] -> h*K + k
    nu = S * K                       # u[c,k] -> nz + c*K + k
    n = nz + nu

    rows, lb, ub = [], [], []

    def add(row, lo, hi):
        rows.append(row); lb.append(lo); ub.append(hi)

    # (1) each group in exactly one fold
    for h in range(H):
        r = lil_matrix((1, n))
        for k in range(K):
            r[0, h * K + k] = 1.0
        add(r, 1.0, 1.0)

    per = tot.sum() / K
    for k in range(K):
        # (2) benign support   (3) attack support
        rb = lil_matrix((1, n)); ra = lil_matrix((1, n)); rt = lil_matrix((1, n))
        for h in range(H):
            rb[0, h * K + k] = ben[h]
            ra[0, h * K + k] = att[h]
            rt[0, h * K + k] = tot[h]
        add(rb, float(min_benign), np.inf)
        add(ra, float(min_attack), np.inf)
        # (5) row balance band around the equal share
        add(rt, per * (1.0 - balance_tol), per * (1.0 + balance_tol))
        # (4) domination limit, linearised
        for h in range(H):
            r = lil_matrix((1, n))
            for j in range(H):
                r[0, j * K + k] = -max_share * tot[j]
            r[0, h * K + k] += tot[h]
            add(r, -np.inf, 0.0)
        # (6) subclass diversity
        for ci in range(S):
            r = lil_matrix((1, n))
            r[0, nz + ci * K + k] = 1.0
            for h in range(H):
                if carries[h, ci]:
                    r[0, h * K + k] = -1.0
            add(r, -np.inf, 0.0)
        r = lil_matrix((1, n))
        for ci in range(S):
            r[0, nz + ci * K + k] = 1.0
        add(r, float(min_subclasses), np.inf)

    from scipy.sparse import vstack
    A = vstack(rows).tocsr()
    cons = LinearConstraint(A, np.array(lb), np.array(ub))
    res = milp(c=np.zeros(n), constraints=cons,
               integrality=np.ones(n), bounds=(0, 1),
               options={"time_limit": time_limit, "presolve": True})

    if res.x is None:
        return {"feasible": False, "status": res.message, "bins": None}
    z = np.asarray(res.x[:nz]).reshape(H, K)
    bins = [[uniq[h] for h in range(H) if z[h, k] > 0.5] for k in range(K)]
    return {"feasible": True, "status": res.message, "bins": bins}
