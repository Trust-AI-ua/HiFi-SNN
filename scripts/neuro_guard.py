#!/usr/bin/env python3
"""Run a command in the neuro environment only if it leaves the repository as it found it.

Added after the 2026-11-17 Lava incident (results/neuro/DEVIATIONS.md, D1). Run with the
system Python, isolated:

    python3 -I scripts/neuro_guard.py --allow results/neuro/a2_lava [--allow ...] -- <command ...>

Fails (exit 3) before the run if any tracked file is modified, staged or deleted
(`git status --porcelain --untracked-files=no` not empty). After the run, fails if
  * any tracked file is modified, staged or deleted;
  * `git status --porcelain --ignored` lists a path that is new since the start and lies
    outside the allowed prefixes;
  * any file under the repository (ignored files included) changed during the run outside the
    allowed prefixes (modification-time scan; catches an overwritten ignored file, which
    porcelain cannot see).
Paths written by jobs that run concurrently and by design (CONCURRENT) are exempt from the two
untracked checks, never from the tracked check.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONCURRENT = ("results/journal_r2/", "results/neuro/logs/", "results/neuro/b2/", "results/neuro/inputs/",
              "dataset/edge-iiotset/_stream/")
SKIP_DIRS = {".git", ".venv", ".venv-neuro", "__pycache__", "venv"}


def porcelain(*args):
    return subprocess.run(["git", "status", "--porcelain", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.splitlines()


def changed_since(t0, allowed):
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = Path(root) / f
            rel = p.relative_to(REPO).as_posix()
            if rel.startswith(allowed + CONCURRENT):
                continue
            try:
                if p.stat().st_mtime > t0:
                    hits.append(rel)
            except FileNotFoundError:
                pass
    return hits


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit(__doc__)
    i = argv.index("--")
    opts, cmd = argv[:i], argv[i + 1:]
    allowed = tuple(p.rstrip("/") + "/" for k, p in zip(opts[::2], opts[1::2]) if k == "--allow")
    tracked = porcelain("--untracked-files=no")
    if tracked:
        print("[neuro_guard] refusing to run: tracked files are modified:\n  " + "\n  ".join(tracked), file=sys.stderr)
        sys.exit(3)
    before = set(porcelain("--ignored"))
    t0 = time.time() - 1
    rc = subprocess.call(cmd, cwd=REPO)
    bad = []
    tracked = porcelain("--untracked-files=no")
    if tracked:
        bad.append("tracked files modified: " + ", ".join(tracked))
    new = [l for l in set(porcelain("--ignored")) - before
           if not l[3:].strip('"').startswith(allowed + CONCURRENT)]
    if new:
        bad.append("new paths outside the allowed outputs: " + ", ".join(sorted(new)))
    mt = changed_since(t0, allowed)
    if mt:
        bad.append("files changed outside the allowed outputs: " + ", ".join(sorted(mt)[:50]))
    if bad:
        print("[neuro_guard] FAILED after the run:\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(3)
    print(f"[neuro_guard] clean (command exit {rc})", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
