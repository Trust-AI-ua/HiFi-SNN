#!/usr/bin/env python3
"""Regenerate the complete evidence package in one deterministic pass.

The experimental record is closed once B + C1 + C2 finish. From that point the
only thing that should change is the manuscript, and every number in it must
come from re-running this script rather than from a value someone remembers.
Running it twice on the same result files must produce identical output.

The five final outputs, in the order the manuscript uses them:

  1. Screening statistics   full 27-configuration mean ranks, Friedman,
                            Nemenyi CD diagram, focused LP/latency vs LP/rate
  2. Strict confirmation    Axis A (encoding) and Axis B (neuron), each labelled
                            with its held-fixed factor so neither can be read as
                            a v2 27-way ranking
  3. Mechanism             early-readout curves, T95/T99, terminal F1, spike and
                            SOP savings
  4. Protocol robustness   CTU-13 100->92, KDDCup99 deduplication sensitivity,
                            split audit, protocol-audit table
  5. Boundary conditions   realised-FAR drift, known vs unseen detection

It also prints a coverage report --- which protocols, variants and seeds are
actually present --- because a table silently computed over three seeds instead
of five is the failure mode that survives review and should not.

    python scripts/finalize.py            # run everything available
    python scripts/finalize.py --coverage # report coverage only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.readout import discover  # noqa: E402

PY = sys.executable

#: (output group, label, argv). Order is the manuscript's order.
STEPS = [
    (1, "screening statistics (v1, 27 configurations)",
     ["scripts/rank_analysis.py", "--input",
      "results/v1_submitted/sweep_540.csv", "--tag", "v1"]),
    (2, "strict confirmation — Axis A, encoding (neuron fixed)",
     ["scripts/axis_analysis.py", "--axis", "encoding"]),
    (2, "strict confirmation — Axis B, neuron (encoding fixed)",
     ["scripts/axis_analysis.py", "--axis", "neuron"]),
    (3, "mechanism — early readout, T95/T99, spike saving",
     ["scripts/early_readout.py"]),
    (3, "mechanism — spike decomposition and SOPs",
     ["scripts/efficiency.py"]),
    (4, "protocol robustness — split audit",
     ["scripts/split_audit.py", "--from-cache"]),
    (4, "protocol robustness — bidirectional class-support audit",
     ["scripts/support_audit.py"]),
    (4, "protocol robustness — dual macro-F1 (fixed universe vs test-present)",
     ["scripts/dual_macro_f1.py"]),
    (4, "protocol robustness — dedup sensitivity, ENCODING axis",
     ["scripts/dedup_analysis.py", "--population", "encoding_axis_lp"]),
    (4, "protocol robustness — dedup sensitivity, NEURON axis",
     ["scripts/dedup_analysis.py", "--population", "neuron_axis_latency",
      "--a", "Leaky/latency", "--b", "LeakyParallel/latency"]),
    (5, "boundary — fixed FAR, per class, duplicate-free",
     ["scripts/security_analysis.py"]),
    (5, "boundary — NSL-KDD known vs unseen attack types",
     ["scripts/nsl_known_unseen.py"]),
    (0, "LaTeX tables", ["scripts/make_tables.py", "--tag", "v1"]),
]

EXPECTED_SEEDS = 5


def coverage() -> str:
    """What is actually on disk, so a thin cell cannot pass unnoticed."""
    runs = discover(require_artifacts=False)
    by = defaultdict(set)
    arts = defaultdict(int)
    for r in runs:
        by[(r.protocol, r.variant)].add(r.seed)
        arts[(r.protocol, r.variant)] += r.has_artifacts()

    lines = [f"{len(runs)} runs on disk across {len(by)} (protocol, variant) cells",
             "", f"{'protocol':24s} {'variant':28s} seeds  artifacts"]
    incomplete = 0
    for (proto, var), seeds in sorted(by.items()):
        flag = "" if len(seeds) >= EXPECTED_SEEDS else "  <- INCOMPLETE"
        incomplete += bool(flag)
        lines.append(f"{proto:24s} {var:28s} {len(seeds):5d}  "
                     f"{arts[(proto, var)]:9d}{flag}")
    lines.append("")
    lines.append(f"{incomplete} cell(s) below {EXPECTED_SEEDS} seeds"
                 if incomplete else
                 f"all cells at {EXPECTED_SEEDS} seeds")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true",
                    help="report coverage and exit without running analyses")
    ap.add_argument("--groups", nargs="+", type=int, default=None,
                    help="restrict to output groups 1-5 (0 = tables)")
    args = ap.parse_args()

    cov = coverage()
    print("=" * 78)
    print("COVERAGE")
    print("=" * 78)
    print(cov)
    if args.coverage:
        return 0

    print("\n" + "=" * 78)
    print("REGENERATING EVIDENCE PACKAGE")
    print("=" * 78)
    failed, skipped, ok = [], [], []
    t0 = time.time()
    for group, label, argv in STEPS:
        if args.groups is not None and group not in args.groups:
            continue
        print(f"\n[{group}] {label}\n{'-'*78}", flush=True)
        p = subprocess.run([PY] + argv, cwd=ROOT, capture_output=True, text=True)
        tail = p.stdout.strip().splitlines()
        for line in tail[-14:]:
            print("   " + line)
        if p.returncode != 0:
            # An analysis with nothing to analyse yet is not a failure; the
            # scripts exit 0 and say so. A non-zero exit is a real problem.
            print(f"   FAILED (exit {p.returncode})")
            print("   " + (p.stderr.strip().splitlines() or ["<no stderr>"])[-1])
            failed.append(label)
        elif any("no runs" in l or "nothing to analyse" in l for l in tail):
            skipped.append(label)
        else:
            ok.append(label)

    manifest = ROOT / "results/analysis/FINAL_MANIFEST.md"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        f"# Evidence package — regenerated {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"Produced by `scripts/finalize.py`. Every manuscript number must come\n"
        f"from re-running this, not from memory.\n\n"
        f"## Coverage\n\n```\n{cov}\n```\n\n"
        f"## Steps\n\n"
        + "".join(f"- OK       {l}\n" for l in ok)
        + "".join(f"- no data  {l}\n" for l in skipped)
        + "".join(f"- FAILED   {l}\n" for l in failed))

    print("\n" + "=" * 78)
    print(f"{len(ok)} regenerated, {len(skipped)} awaiting data, "
          f"{len(failed)} failed  ({time.time()-t0:.0f}s)")
    print(f"manifest: {manifest}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
