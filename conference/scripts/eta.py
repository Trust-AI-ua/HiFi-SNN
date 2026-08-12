#!/usr/bin/env python3
"""Projected completion, and the C2 fallback rule evaluated mechanically.

The fallback exists so that a schedule decision cannot be contaminated by
results. It is written down and computable *before* C2 starts, so if the
schedule deteriorates the amendment is applied because the clock said so, not
because of whichever C2 cells happened to finish first.

    Rule (frozen 2026-08-12, before C2 began)
    -----------------------------------------
    IF projected completion is later than 2026-08-14 04:00,
    OR the program has accumulated more than FAILURE_BUDGET failed jobs,
    THEN reduce C2 to 3 seeds across BOTH CTU-13 folds — never 5 seeds on one.

Two folds beat five seeds because CTU-13's fold-to-fold difficulty dominates its
seed-to-seed variation: the scenario-disjoint folds span roughly 77 to 100
macro-F1 while seeds move results by a fraction of that. A 3-seed, 2-fold design
still asks whether the latency-family ranking survives two very different
scenario conditions; a 5-seed single-fold design would make the confirmation
look fold-specific, which is the criticism the CTU-13 work exists to answer.

    python scripts/eta.py            # project and evaluate the rule
    python scripts/eta.py --apply    # apply the amendment if the rule fires
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_program import PROGRAM, cost_h, jobs_for  # noqa: E402

DEADLINE = dt.datetime(2026, 8, 14, 4, 0)
FAILURE_BUDGET = 5
FALLBACK_SEEDS = [42, 43, 44]

LOG = ROOT / "results/runs/program.log"
STDOUT = ROOT / "results/runs/program_stdout.log"
AMENDMENT = ROOT / "results/runs/C2_AMENDMENT.md"


def measure() -> dict:
    starts = [l for l in LOG.read_text().splitlines() if "program start" in l]
    if not starts:
        raise SystemExit("no program start recorded yet")
    t0 = dt.datetime.strptime(
        re.search(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", starts[-1]).group(1),
        "%Y-%m-%d %H:%M:%S")
    now = dt.datetime.now()
    elapsed = (now - t0).total_seconds() / 3600

    lines = [l for l in STDOUT.read_text().splitlines() if re.match(r"^\[[BC]\d ", l)]
    retired, failures = 0.0, 0
    for l in lines:
        m = re.search(r"(ok  |FAIL) (\S+)/(\S+)/(\S+)/s(\d+)", l)
        if not m:
            continue
        retired += cost_h(m.group(2), m.group(3))
        failures += m.group(1) == "FAIL"

    total = {p: sum(cost_h(j[1], j[2]) for j in jobs_for(p)) for p in PROGRAM}
    rate = retired / elapsed if elapsed > 0 else 0.0
    remaining = sum(total.values()) - retired
    eta_h = remaining / rate if rate > 0 else float("inf")
    finish = now + dt.timedelta(hours=eta_h)
    c2_start = now + dt.timedelta(hours=(remaining - total["C2"]) / rate) if rate else None
    return {
        "t0": t0, "now": now, "elapsed_h": elapsed, "jobs_done": len(lines),
        "failures": failures, "retired_h": retired, "rate": rate,
        "remaining_h": remaining, "eta_h": eta_h, "finish": finish,
        "c2_start": c2_start, "c2_cost": total["C2"], "total": total,
    }


def apply_amendment(m: dict, reason: str) -> None:
    """Rewrite the C2 seed list in the program and record why, with a timestamp."""
    p = ROOT / "scripts/run_program.py"
    s = p.read_text()
    if "C2_SEEDS" not in s:
        s = s.replace(
            'SEEDS = [42, 43, 44, 45, 46]',
            'SEEDS = [42, 43, 44, 45, 46]\n'
            '#: C2 may be reduced by the frozen deadline rule in scripts/eta.py.\n'
            'C2_SEEDS = SEEDS')
        s = s.replace(
            '                for seed in SEEDS:',
            '                seeds = C2_SEEDS if phase == "C2" else SEEDS\n'
            '                for seed in seeds:')
    s = re.sub(r"^C2_SEEDS = .*$", f"C2_SEEDS = {FALLBACK_SEEDS}", s,
               flags=re.MULTILINE)
    p.write_text(s)
    AMENDMENT.write_text(
        f"""# C2 amendment — deadline-driven, {m['now']:%Y-%m-%d %H:%M}

Tier C2 reduced from 5 seeds to {len(FALLBACK_SEEDS)} ({FALLBACK_SEEDS}) across
**both** CTU-13 folds, per the rule frozen in `scripts/eta.py` before C2 began.

**Trigger:** {reason}

Measured at amendment time:
- elapsed {m['elapsed_h']:.2f} h, {m['jobs_done']} jobs done, {m['failures']} failed
- throughput {m['rate']:.2f} GPU-h per wall-hour
- projected finish {m['finish']:%Y-%m-%d %H:%M} against a {DEADLINE:%Y-%m-%d %H:%M} cutoff

This is a computational amendment, not a result-driven one. No C2 result
influenced it; the rule and its threshold were fixed in advance and evaluated
against the clock. Both folds are retained because CTU-13's fold-to-fold
difficulty dominates its seed-to-seed variation.
""")
    print(f"AMENDMENT APPLIED — C2 seeds now {FALLBACK_SEEDS}. Recorded in {AMENDMENT}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="apply the C2 amendment if the rule fires")
    args = ap.parse_args()
    m = measure()

    print(f"start            {m['t0']:%a %d %b %H:%M}")
    print(f"elapsed          {m['elapsed_h']:.2f} h   ({m['jobs_done']} jobs, "
          f"{m['failures']} failed)")
    print(f"throughput       {m['rate']:.2f} GPU-h per wall-hour")
    print(f"remaining        {m['remaining_h']:.1f} GPU-h")
    print(f"projected finish {m['finish']:%a %d %b %H:%M}")
    if m["c2_start"]:
        print(f"C2 starts        {m['c2_start']:%a %d %b %H:%M}  "
              f"({m['c2_cost']:.1f} GPU-h)")
    print(f"cutoff           {DEADLINE:%a %d %b %H:%M}")

    reasons = []
    if m["finish"] > DEADLINE:
        reasons.append(f"projected finish {m['finish']:%d %b %H:%M} is after the "
                       f"{DEADLINE:%d %b %H:%M} cutoff")
    if m["failures"] > FAILURE_BUDGET:
        reasons.append(f"{m['failures']} failed jobs exceeds the budget of "
                       f"{FAILURE_BUDGET}")

    if reasons:
        print(f"\nRULE FIRES: {'; '.join(reasons)}")
        if args.apply:
            apply_amendment(m, "; ".join(reasons))
        else:
            print("re-run with --apply to reduce C2 to 3 seeds across both folds")
        return 1
    margin = (DEADLINE - m["finish"]).total_seconds() / 3600
    print(f"\nrule does not fire — {margin:.1f} h of margin. C2 stays at 5 seeds "
          "x 2 folds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
