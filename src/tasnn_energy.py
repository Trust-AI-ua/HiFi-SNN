"""TASNN's analytic per-sample energy, the formula the published figures use.

This is a VERBATIM copy of ``tasnn_energy_pj`` from scripts/benchmark_4way.py.
It lives here because that script has no ``__main__`` guard: importing it to reach
the function re-runs the whole analysis and rewrites every published table and
figure as a side effect. scripts/guard_sops.py checks this copy against the
original character for character, so the two cannot drift apart.

Hook-based counting (src/opcount.py) cannot replace this: MultiheadAttention holds
its input projection in a raw parameter and computes QK^T and AV as bare matmuls,
so a hook-based count lands at roughly half.
"""
from __future__ import annotations

E_MAC, E_AC = 4.6, 0.9


def tasnn_energy_pj(rec):
    """Analytic per-sample energy for TASNN (its forward isn't SOP-instrumented).

    Transformer/dense MACs @ E_MAC + spiking-core ACs (operations on binary spike
    inputs) @ E_AC. Counts, per sample over T steps:
      DENSE (MAC): feature tokeniser F*d; transformer n_l*(4Ld^2 + 2L^2 d + 2Ld*ff)
                   [in/out proj + QK^T + AV + FFN]; XMG d*Cstc; LIF1.fc Cstc*h1*T
                   (real current input); heads d*C + h2*C.
      SPIKE (AC):  STC conv Cstc*c_spike*k*T (binary spike input); LIF1 lateral h1^2*T;
                   LIF2.fc h1*h2*T (spike input); LIF2 lateral h2^2*T.
    L=F+1 (CLS+features); c_spike=F*(1+pop_M). LayerNorm/softmax negligible (excluded).
    """
    c = rec["config"]; F = rec["data"]["num_features"]; C = rec["data"]["num_classes"]
    d, nl, ff, T = c["d_model"], c["n_layers"], c["ff"], c["num_steps"]
    popM, Cstc, k = c["pop_M"], c["stc_channels"], c["stc_kernel"]
    h1, h2 = c["lif_hidden"]
    L = F + 1
    c_spike = F * (1 + popM)
    mac = (F * d                                              # tokeniser
           + nl * (4 * L * d * d + 2 * L * L * d + 2 * L * d * ff)  # transformer
           + d * Cstc                                        # XMG gate
           + Cstc * h1 * T                                   # LIF1.fc (real input)
           + d * C + h2 * C)                                 # dual heads
    ac = (Cstc * c_spike * k * T                             # STC conv on spikes
          + h1 * h1 * T                                      # LIF1 lateral
          + h1 * h2 * T                                      # LIF2.fc (spike input)
          + h2 * h2 * T)                                     # LIF2 lateral
    return mac * E_MAC + ac * E_AC, mac, ac                  # (32b energy, MAC count, AC count)
