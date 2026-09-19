#!/usr/bin/env python3
"""Move a results path only if no running process could be using it.

Four path-level incidents in this project had the same shape: a directory was
moved, overwritten or repointed while a job still depended on it. A job can
depend on a path before it has any file open (it writes results.json only at the
end), so an open-file check alone is not enough. Refuses the move if any process
of this user

  * has an open file descriptor, or its working directory, under SRC, or
  * has a command line naming any guard pattern (default: the dataset names
    that appear as components of SRC, or its model names if it has none).

    python scripts/safe_mv.py SRC DST [--guard PATTERN ...] [--dry-run]
Exit status 0 on move, 2 when refused.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DATASETS = {"kddcup99", "nslkdd", "ctu13", "iscxids2012", "cicids2017", "cse-cic-ids2018",
            "unswnb15", "ciciot2023", "ciciomt2024", "edgeiiot", "toniot"}
MODELS = {"Tree-to-Spike", "DNN-Tree", "TASNN", "SNN_LeakyParallel"}


def holders(src: Path, guards):
    me, uid = os.getpid(), os.getuid()
    hits = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        p = Path("/proc") / pid
        try:
            if int(pid) in (me, os.getppid()) or p.stat().st_uid != uid:
                continue
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if "safe_mv.py" in cmd:
                continue
            why = []
            try:
                if Path(os.readlink(p / "cwd")).resolve().is_relative_to(src):
                    why.append("cwd")
            except OSError:
                pass
            try:
                for fd in os.listdir(p / "fd"):
                    t = os.readlink(p / "fd" / fd)
                    if t.startswith("/") and Path(t).resolve().is_relative_to(src):
                        why.append(f"fd {t}")
                        break
            except OSError:
                pass
            why += [f"cmdline ~ {g}" for g in guards if g in cmd]
            if why:
                hits.append((pid, cmd.strip()[:120], why))
        except (OSError, ValueError):
            continue
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--guard", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = Path(a.src).resolve()
    if not src.exists():
        sys.exit(f"[safe_mv] {src} does not exist")
    if a.guard is not None:
        guards = a.guard
    else:   # dataset names when the path has one (a model name alone matches every
            # job of that model on any dataset); model names otherwise
        guards = sorted({c for c in src.parts if c in DATASETS}) or \
                 sorted({c for c in src.parts if c in MODELS})
    hits = holders(src, guards)
    if hits:
        print(f"[safe_mv] REFUSED: {len(hits)} process(es) may depend on {src}")
        for pid, cmd, why in hits:
            print(f"  pid {pid}: {', '.join(why)} :: {cmd}")
        sys.exit(2)
    dst = Path(a.dst)
    if a.dry_run:
        print(f"[safe_mv] would move {src} -> {dst} (guards: {guards})")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"[safe_mv] moved {src} -> {dst} (guards checked: {guards})")


if __name__ == "__main__":
    main()
