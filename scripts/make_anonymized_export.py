#!/usr/bin/env python3
"""Build an anonymized, history-free export of a tagged tree for double-blind review.

LOCAL ONLY: writes a directory and a .tar.gz outside the repository and publishes
nothing. Re-run after the tag moves.

  1. `git archive <tag>`: tracked files only, no git history, no author metadata.
  2. Exclude the author's notes and non-anonymous material: the handoff documents,
     TODO, the review rebuttals, the conference paper (prior, non-anonymous work),
     the archive, and results/superseded (stale by definition).
  3. Scrub, in text files: absolute paths and the account name; every commit hash of
     this repository (they are searchable on GitHub if the repository is public);
     the copyright holder in LICENSE; the README citation block and license line.
  4. Scan everything left, including binary files, for identifying strings and write
     REPORT.md listing each residual hit for a human decision. Nothing is guessed.

    python scripts/make_anonymized_export.py [--tag tnnls-submission] [--out ~/HiSNN_anon_export]
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDE = ["HANDOFF_JOURNAL.md", "HANDOFF_CTU13.md", "TODO.md", "rebuttal_response.tex",
           "rebuttal2_response.tex", "conference/", "archive/", "results/superseded/"]
IDENT = re.compile(r"(?i)\braj\b|patel|rpatel|ua\.edu|alabama|tuscaloosa|trust-ai|student3|/home/"
                   r"|akinrele|amebley|mitra|dibbo|rahimi")
TEXT_EXT = {".py", ".sh", ".md", ".txt", ".log", ".csv", ".json", ".yaml", ".yml", ".tex", ".ipynb", ".cfg",
            ".toml", ".ini", ".gitignore", ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tnnls-submission")
    ap.add_argument("--out", default=str(Path.home() / "HiSNN_anon_export"))
    a = ap.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", "--short", f"{a.tag}^{{commit}}"], cwd=REPO, text=True).strip()
    out = Path(a.out).expanduser() / f"export_{sha}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tar = subprocess.check_output(["git", "archive", a.tag], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        members = [m for m in tf.getmembers() if not any(m.name == e or m.name.startswith(e) for e in EXCLUDE)]
        tf.extractall(out, members=members)
    excluded = sorted({e for e in EXCLUDE})
    # every commit hash, full and short, longest first so a short one never splits a long one
    hashes = subprocess.check_output(["git", "log", "--all", "--format=%H"], cwd=REPO, text=True).split()
    hset = sorted({h for h in hashes} | {h[:7] for h in hashes} | {h[:8] for h in hashes}, key=len, reverse=True)
    hre = re.compile(r"(?<![0-9a-f])(" + "|".join(map(re.escape, hset)) + r")(?![0-9a-f])")
    repo_abs = str(REPO)
    counts = {"paths": 0, "hashes": 0, "files": 0}
    for p in out.rglob("*"):
        if not p.is_file() or (p.suffix not in TEXT_EXT and p.name not in {"LICENSE", ".gitignore"}):
            continue
        try:
            s = p.read_text()
        except UnicodeDecodeError:
            continue
        orig = s
        n = s.count(repo_abs) + s.count("/home/student3")
        s = s.replace(repo_abs, "<repo>").replace("/home/student3", "<home>").replace("student3", "<user>")
        s, nh = hre.subn("<commit>", s)
        if p.name == "LICENSE":
            s = re.sub(r"(?m)^Copyright \(c\) (\d{4}) .*$", r"Copyright (c) \1 Anonymous Authors", s)
        if p.relative_to(out).as_posix() == "README.md":
            s = re.sub(r"(?s)## Citation.*?(?=\n## )", "## Citation\n\nWithheld for double-blind review.\n\n", s)
            s = re.sub(r"(?m)^Released under the \*\*MIT License\*\* © (\d{4}) .*$",
                       r"Released under the **MIT License**, see [LICENSE](LICENSE).", s)
        if s != orig:
            p.write_text(s)
            counts["paths"] += n
            counts["hashes"] += nh
            counts["files"] += 1
    # residual scan, binary files included
    hits = []
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        data = p.read_bytes()
        for m in IDENT.finditer(data.decode("latin-1")):
            line = data[:m.start()].count(b"\n") + 1
            ctx = data.decode("latin-1")[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
            hits.append((p.relative_to(out).as_posix(), line, m.group(0), ctx))
    lines = [f"# Anonymized export of `{a.tag}` ({sha})", "",
             "Local only. Nothing has been published.", "",
             f"Excluded: {', '.join(excluded)}", "",
             f"Scrubbed {counts['files']} files: {counts['paths']} absolute paths, {counts['hashes']} commit hashes; "
             "LICENSE holder and README citation replaced.", "",
             f"## Residual identifying strings: {len(hits)}", "",
             "| File | Line | Match | Context |", "|---|---|---|---|"]
    lines += [f"| {f} | {ln} | `{mt}` | `{c.strip()[:90]}` |" for f, ln, mt, c in hits[:400]]
    (out.parent / f"REPORT_{sha}.md").write_text("\n".join(lines) + "\n")
    with tarfile.open(out.parent / f"export_{sha}.tar.gz", "w:gz") as t:
        t.add(out, arcname=f"export_{sha}")
    print(f"[export] {out}  ({sum(1 for _ in out.rglob('*') if _.is_file())} files)")
    print(f"[scrub] {counts}")
    print(f"[residual] {len(hits)} identifying strings -> {out.parent / f'REPORT_{sha}.md'}")


if __name__ == "__main__":
    main()
