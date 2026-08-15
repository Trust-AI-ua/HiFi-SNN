#!/usr/bin/env python3
"""Verify every \\FN* macro the prose uses is defined, and archive provenance.

An undefined macro in LaTeX renders as nothing or as an error deep in a log
nobody reads at 2 a.m. Checking it here is cheaper. The reverse direction is
reported too: a defined-but-unused macro is not an error, but a long list of
them usually means the prose drifted away from the generated numbers.

Also writes `results/analysis/PROVENANCE.json`: a SHA-256 of every analysis
artifact the manuscript draws on, so the submitted numbers can be tied to exact
files after the fact.

    python scripts/check_macros.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ANALYSIS = ROOT / "results/analysis"
DEFS = PAPER / "sec_frozen_numbers.tex"



def check_regeneration_is_live() -> list[str]:
    """Refuse to pass if the frozen-numbers regeneration path is broken or stale.

    The manuscript's guarantee is that every number comes from a committed
    artifact. That guarantee is enforced by `make_frozen_numbers.py`, and on
    2026-08-14 it stopped working: a column was renamed, the script raised, and
    `finalize.py` faithfully recorded `FAILED` in a manifest nobody read. For a
    day the guarantee was not in force and nothing said so. Nothing had drifted,
    but only by luck.

    The failure class is *a guard reporting its own breakage into a channel with
    no reader*, so the fix is to make the breakage block rather than log. Two
    checks:

      1. The manifest's most recent entry for the step is not FAILED.
      2. Regenerating right now reproduces `sec_frozen_numbers.tex` byte for
         byte. This is direct verification rather than a staleness proxy: it
         catches a broken generator, a stale committed file, and a hand edit,
         and it cannot be fooled by an mtime that a checkout rewrote.
    """
    problems = []

    manifest = ANALYSIS / "FINAL_MANIFEST.md"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if "frozen manuscript numbers" in line and line.strip().startswith("- FAILED"):
                problems.append(
                    f"{manifest.name} records the frozen-numbers step as FAILED. "
                    "The manuscript's numbers are not currently guaranteed to "
                    "come from the artifacts. Run "
                    "`python scripts/make_frozen_numbers.py` and fix the error "
                    "it reports before building.")

    gen = ROOT / "scripts/make_frozen_numbers.py"
    target = PAPER / "sec_frozen_numbers.tex"
    if gen.exists():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "regenerated.tex"
            r = subprocess.run([sys.executable, str(gen), "--out", str(tmp),
                                "--quiet"], cwd=ROOT, capture_output=True,
                               text=True)
            if r.returncode != 0:
                last = (r.stderr.strip().splitlines() or ["<no stderr>"])[-1]
                problems.append(
                    f"make_frozen_numbers.py fails to run ({last}). Every macro "
                    "in the manuscript is currently unverifiable.")
            elif not target.exists():
                problems.append(f"{target.name} does not exist.")
            elif tmp.read_bytes() != target.read_bytes():
                problems.append(
                    f"{target.name} is not what make_frozen_numbers.py produces "
                    "right now --- it is stale, or it was edited by hand. "
                    "Re-run `python scripts/make_frozen_numbers.py`.")
    return problems


def main() -> int:
    defined = set(re.findall(r"\\newcommand\{\\(FN[A-Za-z]+)\}", DEFS.read_text()))
    used = {}
    for path in sorted(PAPER.glob("*.tex")):
        if path.name == DEFS.name:
            continue
        body = "\n".join(l for l in path.read_text().splitlines()
                         if not l.lstrip().startswith("%"))
        for m in re.finditer(r"\\(FN[A-Za-z]+)", body):
            used.setdefault(m.group(1), set()).add(path.name)

    undefined = {k: sorted(v) for k, v in used.items() if k not in defined}
    unused = sorted(defined - set(used))

    if undefined:
        print(f"{len(undefined)} UNDEFINED macro(s) used in prose:")
        for k, files in sorted(undefined.items()):
            print(f"  \\{k}  in {', '.join(files)}")
        return 1
    print(f"all {len(used)} macros used in prose are defined "
          f"({len(defined)} defined in total, {len(unused)} unused)")

    # ---- the regeneration path must be live, not merely reported ----
    stale = check_regeneration_is_live()
    if stale:
        print("FROZEN NUMBERS ARE NOT VERIFIED:")
        for v in stale:
            print("  -", v)
        return 1
    print("frozen numbers verified: regeneration reproduces "
          "sec_frozen_numbers.tex exactly")

    # ---- provenance ----
    entries = {}
    for p in sorted(ANALYSIS.rglob("*")):
        if p.is_file() and p.suffix in {".csv", ".json", ".md"}:
            entries[str(p.relative_to(ROOT))] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
    for p in sorted(PAPER.rglob("*.tex")):
        entries[str(p.relative_to(ROOT))] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
    out = ANALYSIS / "PROVENANCE.json"
    out.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("SHA-256 of every analysis artifact and manuscript source the "
                 "submitted numbers derive from. The experimental record was "
                 "closed on 2026-08-13 after 235 runs, 5 seeds per cell, "
                 "0 failures."),
        "n_files": len(entries),
        "files": entries,
    }, indent=2))
    print(f"provenance: {len(entries)} files hashed -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
