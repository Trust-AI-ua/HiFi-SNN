#!/usr/bin/env python3
"""Gated Part B measurements: CICIoT2023 seed-42 deployment re-measurement (A1), B3 PLE-MLP
latency, B8 peak resident memory. Pre-registration 2026-09-28 (D9, A1).

Guards first (no gate needed; the script stops on any failure):
  Ga  scripts/deployment_measure.py as patched (model roots, artifact dumps) reproduces the
      committed version's measure() on TON-IoT seed 42 in every non-timing field.
  Gb  every CICIoT2023 block model (Tree-to-Spike, DNN-Tree, Vanilla SNN, TASNN, seed 42)
      rebuilt as the measurement rebuilds it reproduces its block record's test macro-F1
      (TASNN samples at inference: within 0.2 points with the seed the record used).
  Gc  each PLE-MLP seed-42 model rebuilt from its saved bins and checkpoint reproduces its
      record's fp32 test macro-F1.
Then the deployment gate (release file + one-minute load below 2.0 for 300 s), and in one
quiet window, single-threaded CPU, batch 1, the deployment workload (500 warm-up + 2,000
timed calls on the first 2,500 test rows):
  1. deployment_measure on CICIoT2023 seed 42 with the block's selected-checkpoint models,
     the block's prefix outputs and early-exit selection;
  2. deployment_measure on the other ten datasets, production models: the whole deployment
     table is re-measured in this session with the fixed procedure (teacher fitted before the
     thread cap and checked against the production teacher; one fresh process per dataset),
     on the user's decision of 2026-10-07; the published CSV is left as it was;
  3. B3: PLE-MLP latency per IoT dataset (selected configuration, seed 42, fp32);
  4. B8: peak resident memory per system and dataset, each in a fresh process
     (scripts/partB_b8_memchild.py).

    python scripts/partB_gated_measure.py
Output: results/journal_partB/B8/<run_id>/{guards.json, deployment_measurements_partB.csv,
        ple_latency.csv, peak_memory.csv, summary.json}
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

REPO = C.REPO
BLOCK = "results/journal_partB/B6/block"
FU = "results/journal_partB/B6/followup"
TIMING = {"p50_ms", "mean_ms", "p95_ms", "p99_ms", "n_timed"}


REF = "374e7d4a"   # last commit of scripts/deployment_measure.py before the Part B patch (9647f430)


def committed_measure_module():
    src = subprocess.check_output(["git", "show", f"{REF}:scripts/deployment_measure.py"], cwd=REPO, text=True)
    tmp = Path(tempfile.mkdtemp()) / "deployment_measure_committed.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("dm_committed", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPO, mod.OUT = REPO, REPO / "results" / "deployment"   # it would derive both from its temp path
    return mod


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def cell_rows(dataset, roots=None, artifact_dir=None, reference=False):
    """measure() for one dataset at seed 42 in a fresh process (patched script, or the
    pre-patch reference loaded from git); rows come back as JSON."""
    fd, out = tempfile.mkstemp(suffix=".json"); os.close(fd)
    env = {k: v for k, v in os.environ.items() if k not in C.THREAD_VARS}
    if reference:
        code = (f"import sys, json; sys.path.insert(0, {str(REPO)!r}); sys.path.insert(0, {str(REPO / 'scripts')!r}); "
                f"import partB_gated_measure as G; m = G.committed_measure_module(); "
                f"rows = m.measure({dataset!r}, 42); "
                f"open({out!r}, 'w').write(json.dumps(rows, default=lambda o: None if o is None else float(o)))")
        cmd = [sys.executable, "-c", code]
    else:
        cmd = [sys.executable, "-u", str(REPO / "scripts/deployment_measure.py"), "--datasets", dataset,
               "--seeds", "42", "--cell-out", out]
        for k, v in (roots or {}).items():
            cmd += ["--" + k.replace("_", "-") + ("-root" if k not in ("exit_csv", "out_csv") else ""), v]
        if artifact_dir:
            cmd += ["--artifact-dir", str(artifact_dir)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"measurement process failed for {dataset}: {r.stdout[-800:]} {r.stderr[-1500:]}")
    rows = json.loads(Path(out).read_text()); os.unlink(out)
    return rows


def guard_a():
    old, new = cell_rows("toniot", reference=True), cell_rows("toniot")
    if [r["system"] for r in old] != [r["system"] for r in new]:
        sys.exit("Ga: system lists differ")
    bad = []
    for o, n in zip(old, new):
        for k in set(o) | set(n):
            if k not in TIMING and not same(o.get(k), n.get(k)):
                bad.append((o["system"], k, o.get(k), n.get(k)))
    if bad:
        sys.exit(f"Ga failed: {bad[:5]}")
    return f"patched deployment_measure reproduces the pre-patch version (374e7d4a) on toniot/42 ({len(new)} systems, all non-timing fields; fresh processes)"


def f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", zero_division=0))


def guard_b():
    import src.training_treespike as tts
    import src.training_dnn as tdnn
    import src.training as tr
    import src.training_tasnn as tt
    from src.models import TreeGuidedSpikeEncoder, SNNStudent, DNNStudent, SNN, TASNN
    from src.encoding import get_encoder
    ds, seed = "ciciot2023", 42
    rec = lambda arm: json.loads((REPO / f"{BLOCK}/{arm}/{ds}/seed_{seed}/results.json").read_text())
    b, arrays = C.load_bundle(ds, seed)
    Xtr, Xte, yte = arrays[0], arrays[4], arrays[5]
    teacher, _ = C.fit_teacher_guarded(ds, seed, b, arrays)
    for arm in ("tree", "dnn"):
        if rec(arm)["test_plus"]["teacher_upper_bound"]["f1_macro"] != C.stored_record(ds, seed)["test_plus"]["teacher_upper_bound"]["f1_macro"]:
            sys.exit(f"Gb: block {arm} teacher differs from the stored teacher")
    cfg = C.t2s_config(ds, seed)
    enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"],
                                 jitter=cfg["jitter"]).fit(teacher, Xtr.shape[1], X=Xtr)
    out = {}
    dev = torch.device("cuda")
    st = SNNStudent(enc.out_dim, b.num_classes, hidden=tuple(cfg["hidden"]), beta=cfg["beta"], leaky=cfg["leaky"]).to(dev)
    st.load_state_dict(torch.load(REPO / f"{BLOCK}/tree/{ds}/seed_{seed}/checkpoint.pt", map_location=dev))
    p = tts._encode_forward_in_batches(st, enc, Xte, cfg["batch_size"], dev)[0]
    out["t2s"] = (f1(yte, p), rec("tree")["test_plus"]["metrics"]["f1_macro"])
    dcfg = yaml.safe_load(open(REPO / f"experiments/DNN-Tree/{ds}/config.yaml"))
    m = DNNStudent(enc.out_dim, b.num_classes, hidden=tuple(dcfg["hidden"]), clip_relu=dcfg.get("clip_relu"))
    m.load_state_dict(torch.load(REPO / f"{BLOCK}/dnn/{ds}/seed_{seed}/checkpoint.pt", map_location="cpu"))
    m.eval()
    want = rec("dnn")["test_plus"]["metrics"]["f1_macro"]
    got = [f1(yte, tdnn._forward_batches(m.to(d), enc, Xte, dcfg["batch_size"], torch.device(d))[0]) for d in ("cuda", "cpu")]
    out["dnn"] = (got[0] if got[0] == want else got[1], want)
    for arm, mod, path in (("vanilla", tr, f"configs/{ds}/SNN_LeakyParallel/latency/config.yaml"),
                           ("tasnn", tt, f"experiments/TASNN/{ds}/config.yaml")):
        r = rec(arm)
        c = yaml.safe_load(open(REPO / path)); c.update(r["config"]); c = mod._resolve(c)
        mod._set_seed(seed)
        from src.data import get_dataset
        bb = get_dataset(ds, **r["loader_kwargs"])
        if arm == "vanilla":
            mm = SNN(num_features=bb.num_features, hidden_size=c["hidden_size"], num_classes=bb.num_classes,
                     neuron_name=c["neuron"], num_steps=c["num_steps"]).to(dev)
            mm.load_state_dict(torch.load(REPO / f"{BLOCK}/{arm}/{ds}/seed_{seed}/checkpoint.pt", map_location=dev))
            o = tr._eval(mm, bb.test_loader, get_encoder(c["encoding"]), c["num_steps"], dev)
        else:
            mm = TASNN(num_features=bb.num_features, num_classes=bb.num_classes, num_steps=c["num_steps"],
                       d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"], ff=c["ff"],
                       dropout=c["dropout"], pop_M=c["pop_M"], stc_channels=c["stc_channels"],
                       stc_kernel=c["stc_kernel"], lif_hidden=tuple(c["lif_hidden"]), lif_beta=c["lif_beta"],
                       lif_v_th=c["lif_v_th"], lif_gamma=c["lif_gamma"], surrogate_beta=c["surrogate_beta"],
                       target_rate=c["target_rate"]).to(dev)
            mm.load_state_dict(torch.load(REPO / f"{BLOCK}/{arm}/{ds}/seed_{seed}/checkpoint.pt", map_location=dev))
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            o = tt._eval(mm, bb.test_loader, dev)
        out[arm] = (f1(o["y_true"], o["y_pred"]), r["test_plus"]["metrics"]["f1_macro"])
    bad = {k: v for k, v in out.items() if abs(v[0] - v[1]) > (0.002 if k == "tasnn" else 1e-12)}
    if bad:
        sys.exit(f"Gb failed (recomputed vs block record): {bad}")
    return {k: {"recomputed": v[0], "block_record": v[1]} for k, v in out.items()}


def ple_models():
    from partB_b3_ple import PLEMLP, predict
    sel = json.loads((REPO / "results/journal_partB/B3/selection.json").read_text())["selection"]
    out = {}
    for ds in C.IOT:
        d = REPO / f"results/journal_partB/B3/search/{ds}/cfg_{sel[ds]['config_id']:02d}"
        r = json.loads((d / "results.json").read_text())
        bz = torch.load(d / "bins.pt")
        b, arrays = C.load_bundle(ds, 42)
        m = PLEMLP(bz["bins"], bz["enc_cols"], bz["raw_cols"], r["config"]["width"], b.num_classes)
        m.load_state_dict(torch.load(d / "checkpoint_fp32.pt", map_location="cpu")); m.eval()
        p, _ = predict(m.to("cuda"), torch.from_numpy(arrays[4]).to("cuda"))
        got, want = f1(arrays[5], p), r["test"]["fp32"]["metrics"]["f1_macro"]
        if abs(got - want) > 1e-12:
            sys.exit(f"Gc failed {ds}: {got} vs {want}")
        out[ds] = dict(model=m.to("cpu"), bins=bz, width=r["config"]["width"], K=b.num_classes,
                       Xsub=arrays[4][:2500], f1=want, macs=r["model"]["macs_total"])
    return out


def main():
    C.no_thread_caps()
    st = C.stamp("B8")
    out = C.run_dir(st)
    art = out / "artifacts"
    import deployment_measure as DM
    guards = {}
    guards["Gb"] = guard_b()                      # first: nothing has capped threads in this process yet
    print("Gb:", guards["Gb"], flush=True)
    ple = ple_models()
    guards["Gc"] = {ds: v["f1"] for ds, v in ple.items()}
    print("Gc: PLE-MLP seed-42 models reproduce their records", flush=True)
    guards["Ga"] = guard_a()
    print("Ga:", guards["Ga"], flush=True)
    (out / "guards.json").write_text(json.dumps(guards, indent=1, default=float))

    gate = REPO / DM.GATE_FILE
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(f"released {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} for Part B gated measurements "
                    f"(run {st['run_id']}), on the user's instruction to stop other jobs\n")
    load0 = DM.wait_for_gate(REPO)

    # 1-2. the whole deployment table, one fresh process per dataset; CICIoT2023 with the block models
    block_roots = {"t2s": f"{BLOCK}/tree/{{d}}", "dnn": f"{BLOCK}/dnn/{{d}}", "vanilla": f"{BLOCK}/vanilla/{{d}}",
                   "tasnn": f"{BLOCK}/tasnn/{{d}}", "prefix": f"{FU}/prefix_outputs/{{d}}/treespike",
                   "exit_csv": f"{FU}/early_exit/selected_test.csv"}
    rows = []
    for ds in C.ALL:
        r = cell_rows(ds, roots=block_roots if ds == "ciciot2023" else None,
                      artifact_dir=art if ds in C.IOT else None)
        for x in r:
            x["models"] = ("CICIoT2023 block (A1: 20 epochs, validation-selected checkpoint)" if ds == "ciciot2023"
                           else "production")
        rows += r
        print(f"   measured {ds}: {len(r)} systems", flush=True)
    df = pd.DataFrame(rows).assign(run_id=st["run_id"], source_script="scripts/deployment_measure.py (Part B fixed procedure)")
    with open(out / "deployment_measurements_partB.csv", "w") as fh:
        fh.write(f"# run_id={st['run_id']} commit={st['commit']} load1_at_start={load0:.2f} teacher fitted before "
                 f"the thread cap and checked against the production teacher; one process per dataset\n")
        df.to_csv(fh, index=False)
    print("1-2. deployment table re-measured (11 datasets)", flush=True)

    # 3. B3 PLE-MLP latency (same protocol: one thread, batch 1, 500 + 2,000 calls)
    torch.set_num_threads(1)
    lat = []
    for ds, v in ple.items():
        m = v["model"]

        def fn(x, m=m):
            with torch.no_grad():
                return m(torch.from_numpy(x))
        lat.append(dict(dataset=ds, seed=42, system="ple_mlp", macs=v["macs"], **DM.time_fn(fn, v["Xsub"])))
        import pickle
        with open(art / f"{ds}_ple_mlp.pkl", "wb") as fh:
            pickle.dump(dict(system="ple_mlp", dataset=ds, seed=42, Xsub=v["Xsub"], kind="ple",
                             sd=m.state_dict(), bins=v["bins"]["bins"], enc_cols=v["bins"]["enc_cols"],
                             raw_cols=v["bins"]["raw_cols"], width=v["width"], K=v["K"]), fh, protocol=4)
    pd.DataFrame(lat).assign(run_id=st["run_id"]).to_csv(out / "ple_latency.csv", index=False)
    print("3. PLE-MLP latency measured", flush=True)

    # 4. B8 peak memory, one fresh process per artifact
    mem = []
    for pk in sorted(art.glob("*.pkl")):
        r = subprocess.run([sys.executable, "-u", str(REPO / "scripts/partB_b8_memchild.py"), str(pk)],
                           capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                                 if k not in C.THREAD_VARS})
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode or not line:
            sys.exit(f"memory child failed for {pk.name}: {r.stderr[-500:]}")
        mem.append(json.loads(line[-1]))
        print("   ", mem[-1], flush=True)
    pd.DataFrame(mem).assign(run_id=st["run_id"]).to_csv(out / "peak_memory.csv", index=False)
    load_end = os.getloadavg()[0]
    (out / "summary.json").write_text(json.dumps(dict(stamp=st, load1_at_start=load0, load1_at_end=load_end,
                                                      guards=guards), indent=1, default=float))
    gate.unlink()
    print(f"done; load at start {load0:.2f}, at end {load_end:.2f}; gate file removed", flush=True)


if __name__ == "__main__":
    main()
