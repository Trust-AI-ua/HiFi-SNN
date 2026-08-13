#!/usr/bin/env python3
"""Enforce the manuscript's terminology rules on paper/*.tex.

The results are heterogeneous and the honest wording for them is narrow. Under
deadline pressure the natural drift is back toward the submitted paper's
vocabulary --- "consistently outperforms", "dominant", "energy savings" --- so
the banned phrases are checked rather than trusted to memory.

Each rule records why it exists, because a banned phrase with no reason attached
gets overridden by the next person in a hurry.

    python scripts/check_wording.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

#: (regex, why it is wrong given our actual results)
BANNED = [
    (r"statistically best",
     "14 of 27 configurations lie within one Nemenyi critical difference"),
    (r"universally superior|uniformly superior",
     "encoding preference inverts on CTU-13 fold 1"),
    (r"dominant (neuron|configuration|encoding)",
     "the neuron-axis lead is narrow and not per-block"),
    (r"consistently outperform",
     "Leaky wins more individual blocks than LeakyParallel"),
    (r"always faster|universally faster|generally enables earlier",
     "latency reaches T99 earlier on only 2 of 5 protocols"),
    (r"energy sav(ing|ings)(?!.{0,80}proxy)",
     "we measure synaptic operations, not energy; say 'operation-based proxy'"),
    (r"leakage[- ]free",
     "we removed specific leakage modes; absence of all leakage is not proven"),
    (r"generali[sz]es to unseen attacks",
     "unseen-type DR falls to 0.25-0.38 for every configuration"),
    (r"operationally low (FAR|false[- ]alarm)",
     "a validation-calibrated 1% FAR reaches 48% on CTU-13 fold 1"),
    (r"\bchampion\b",
     "informal; use 'top-ranked configuration' or 'aggregate leader'"),
    (r"fastest variant",
     "an inference-speed claim the strict timings do not support"),
]

#: Numbers that must never be typed literally — they have frozen macros.
HARDCODED = [
    (r"(?<![\d.])4\.03(?![\d])", r"\FNScrRank"),
    (r"(?<![\d.])1\.44(?![\d])", r"\FNEncRank"),
    (r"(?<![\d.])2\.70(?![\d])", r"\FNNeuRank"),
    (r"(?<![\d.])9\.28(?![\d])", r"\FNScrCD"),
    (r"92\.01", r"\FNCtuStrict"),
]

SKIP = {"sec_frozen_numbers.tex"}


def main() -> int:
    problems = []
    files = [p for p in sorted(PAPER.glob("*.tex")) if p.name not in SKIP]
    for path in files:
        text = path.read_text()
        # Strip comment lines: rationale in comments may quote a banned phrase.
        body = "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("%"))
        for pattern, why in BANNED:
            for m in re.finditer(pattern, body, flags=re.IGNORECASE):
                line = body[:m.start()].count("\n") + 1
                problems.append(f"{path.name}:~{line}  BANNED {m.group(0)!r}\n"
                                f"    why: {why}")
        for pattern, macro in HARDCODED:
            for m in re.finditer(pattern, body):
                line = body[:m.start()].count("\n") + 1
                problems.append(f"{path.name}:~{line}  HARD-CODED {m.group(0)!r}"
                                f" — use {macro}")

    if problems:
        print(f"{len(problems)} wording problem(s):\n")
        for p in problems:
            print("  " + p)
        return 1
    print(f"wording clean across {len(files)} tex files: no banned phrasing, "
          "no hard-coded frozen number")
    return 0


if __name__ == "__main__":
    sys.exit(main())
