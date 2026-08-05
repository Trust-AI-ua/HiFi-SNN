# STALE — phase-1 artifact, unrelated to the CTU-13 leakage experiment

These 5 runs (`latency/seed_42..46/results.json`, 2026-06-25) are leftovers from the
**phase-1 SNN architecture ablation**. They are *not* output of the CTU-13
three-condition leakage experiment or of the 300-run CTU-13 model matrix.

**Do not count these toward skip detection.** The layout here is
`<arm>/seed_<n>/results.json`, the same shape the ladder sweep uses, so a glob like
`results/ctu13/*/*/seed_*/results.json` will read them as completed cells and
silently skip real work. Skip detection must key on a `results.json` written by the
script that owns the run, never on directory existence under `results/ctu13/`.

See `STALE_DIRS` in `scripts/ctu13_stage1.py` and the state table in
`HANDOFF_CTU13.md` §1.

As of 2026-08-05 the CTU-13 model matrix has **never been launched** and has zero
result directories.
