"""Cross-process lock so two worker pools can share one job list.

A supplemental pool works the same cells as an already-running pool; the
"results.json exists" check alone cannot stop both from starting the same cell at
the same moment. ``claim`` creates a lock file atomically (O_EXCL) holding the
owner's pid, and treats a lock whose pid is gone as stale.

    with claim(path) as got:
        if not got:
            return           # another worker owns this cell
        ...
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def claim(lock_path):
    lock = Path(lock_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    got = False
    try:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = int(lock.read_text().strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and _alive(owner):
                yield False
                return
            lock.unlink(missing_ok=True)          # stale: owner died
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        got = True
        yield True
    finally:
        if got:
            lock.unlink(missing_ok=True)
