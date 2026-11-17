"""Per-packet decisions for track B (results/preregistration/2026-11-17_neuro_stream.md, B2).

The C1-gate of revision-2 step 4, adapted to packets, for any model that emits logits after
every packet (the stream SNN's running readout, a tree's margins):
  * per-packet temperature tau_k fitted on validation by NLL (bounded search on log tau in
    [-3, 3]) over the flows that have a k-th packet;
  * a flow is decided at the first packet whose calibrated confidence reaches theta, or at its
    last packet;
  * theta from THETAS, eligible if validation macro-F1 is within MARGIN of the decision at
    flow end; lowest mean cost; ties to higher macro-F1, lower FAR, higher theta.
Then the benign threshold of the matched-FAR check, on the posterior at decision.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

THETAS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
MARGIN = 1.0


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def nll(z, y):
    z = z - z.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())


def fit_taus(Z, y, L):
    taus = np.ones(Z.shape[1])
    for k in range(Z.shape[1]):
        rows = L > k
        if rows.sum() < 10:
            taus[k] = taus[k - 1] if k else 1.0
            continue
        zk = Z[rows, k].astype(np.float64)
        r = minimize_scalar(lambda lt: nll(zk / np.exp(lt), y[rows]), bounds=(-3, 3), method="bounded",
                            options=dict(xatol=1e-4))
        taus[k] = float(np.exp(r.x))
    return taus


def calibrated(Z, taus):
    return softmax(Z.astype(np.float64) / taus[None, :, None])


def decide(prob, L, theta):
    """Decision step (0-based) and posterior at decision. theta None: decide at flow end."""
    n, P, _ = prob.shape
    end = np.minimum(L, P) - 1
    if theta is None:
        first = end
    else:
        valid = np.arange(P)[None, :] < L[:, None]
        fired = (prob.max(2) >= theta) & valid
        first = np.where(fired.any(1), fired.argmax(1), end)
    return first, prob[np.arange(n), first]


def scores(y, pred, benign=0):
    b = y == benign
    return dict(macro_f1=100 * f1_score(y, pred, average="macro"),
                far=100 * float((pred[b] != benign).mean()) if b.any() else float("nan"),
                dr=100 * float((pred[~b] != benign).mean()) if (~b).any() else float("nan"))


def gate(prob_v, y_v, L_v, cum_cost_v):
    """cum_cost_v [n, P]: cost of executing packets 1..k+1. Returns theta (None if none eligible) and the table."""
    first, Pd = decide(prob_v, L_v, None)
    f_end = scores(y_v, Pd.argmax(1))["macro_f1"]
    cand = []
    for th in THETAS:
        first, Pd = decide(prob_v, L_v, th)
        s = scores(y_v, Pd.argmax(1))
        cost = cum_cost_v[np.arange(len(first)), first].mean()
        cand.append(dict(theta=th, **s, mean_cost=float(cost), eligible=bool(s["macro_f1"] >= f_end - MARGIN)))
    elig = [c for c in cand if c["eligible"]]
    best = sorted(elig, key=lambda c: (c["mean_cost"], -c["macro_f1"], c["far"], -c["theta"]))[0] if elig else None
    return (best["theta"] if best else None), dict(flow_end_macro_f1=f_end, candidates=cand)


def threshold_for(pb_benign, f):
    """Largest t with mean(pb_benign < t) <= f (the matched-FAR rule)."""
    v = np.sort(pb_benign)
    k = int(np.floor(f * len(v) + 1e-9))
    return float(v[k]) if k < len(v) else float(np.nextafter(v[-1], np.inf))


def thresholded(P, t):
    att = 1 + P[:, 1:].argmax(1)
    return np.where(P[:, 0] >= t, 0, att)
