#!/usr/bin/env python3
"""Check a manuscript against results/tables/numbers.tex.

Reports, per file:
  TYPED     a number typed in the text whose value equals a macro's value; it
            should be the macro, or a stale copy will survive the next regeneration
  UNDEFINED a \\JN... macro used in the text but not defined in numbers.tex
            (renamed or removed at the last regeneration)
  STALE?    a value on the scope-drift list (superseded by the leakage repair,
            the re-measurement or the operation-count corrections). Marked for
            review, because some coincide with a current value in another role

Comments are ignored. Exit status is non-zero if anything is found, so it can gate
a build.

    python scripts/check_numbers.py path/to/manuscript.tex [more.tex ...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NUM = REPO / "results/tables/numbers.tex"


def load():
    defs = {}
    for line in NUM.read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(JN[A-Za-z]+)\}\{(.*?)\}%", line)
        if m:
            v = re.sub(r"\\ensuremath\{([^}]*)\}", r"\1", m.group(2)).lstrip("+")
            defs[m.group(1)] = v
    return defs


STALE = {  # value: what it was
    "79.80": "pre-repair pooled accuracy", "79.86": "pre-repair pooled accuracy (also a current BACC)",
    "79.68": "pre-repair pooled accuracy", "88.78": "pre-repair Edge-IIoTset Tree-to-Spike",
    "89.87": "pre-repair Edge-IIoTset DNN-Tree", "75.94": "pre-repair CSE Tree-to-Spike",
    "65.93": "pre-repair CSE DNN-Tree", "93.18": "pre-repair accuracy",
    "3.32": "old IoT energy ratio (mixed conventions)", "4.98": "old energy figure",
    "53398": "fp32 artifact bytes", "53.4": "fp32 artifact KB", "40624": "old artifact bytes",
    "5.04": "native-JSON teacher MB", "3107.8": "K-per-leaf teacher KB", "2042.8": "K-per-leaf teacher KB",
    "1857.0": "K-per-leaf teacher KB", "82.13": "pre-repair teacher / Vanilla SNN",
    "81.32": "old teacher figure", "15.60": "old Friedman chi-square", "0.022": "asymptotic Friedman p",
    "2768": "old compact-ensemble pJ", "5654": "old student pJ", "76475": "old node count",
    "0.94": "old memory band (lower)", "2.06": "old memory band (upper)",
    "1.09": "pre-repair Edge-IIoTset non-inferiority value (ladder bootstrap); repaired: +3.34",
    "1.0855": "pre-repair Edge-IIoTset non-inferiority value (ladder bootstrap)",
}


def main(paths):
    defs = load()
    by_value = {}
    for k, v in defs.items():
        if re.fullmatch(r"-?[\d,]*\.\d+|-?\d{3,}(,\d{3})*", v):   # decimals, and integers >= 100
            by_value.setdefault(v.lstrip("-").replace(",", ""), []).append(k)
    found = 0
    for p in paths:
        for i, raw in enumerate(Path(p).read_text(errors="ignore").splitlines(), 1):
            line = re.sub(r"(?<!\\)%.*", "", raw)
            # LaTeX thousands separators: 2{,}768 and 2\,768 are the number 2,768
            line = re.sub(r"(?<=\d)(\{,\}|\\,)(?=\d{3})", ",", line)
            for m in re.finditer(r"\\(JN[A-Za-z]+)", line):
                if m.group(1) not in defs:
                    found += 1
                    print(f"{p}:{i}: UNDEFINED \\{m.group(1)}")
            for m in re.finditer(r"(?<![\w\\.])(\d[\d,]*\.\d+|\d{1,3}(?:,\d{3})+)(?![\w.])", line):
                key = m.group(1).replace(",", "")
                if key in STALE:
                    found += 1
                    print(f"{p}:{i}: STALE? {m.group(1)} ({STALE[key]})")
                if key in by_value:
                    found += 1
                    print(f"{p}:{i}: TYPED {m.group(1)} -> use \\{by_value[key][0]}"
                          + (f" (or {', '.join(by_value[key][1:3])})" if len(by_value[key]) > 1 else ""))
    print(f"{found} issue(s) in {len(paths)} file(s); {len(defs)} macros defined")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
