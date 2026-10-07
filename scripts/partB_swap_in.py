#!/usr/bin/env python3
"""Put the Part B results in the reported places (amendments A1, A5; the 2026-10-07
deployment decision), moving what they replace aside. Nothing is deleted.

A1  CICIoT2023, 12 arms x seeds 42-46: the production run folders are moved to
    results/fixed_budget_ciciot2023/<same relative path> (the appendix's "original
    fixed-budget protocol") and the block runs (20 epochs, validation-selected checkpoint) are
    copied in their place. Also: the Tree-to-Spike prefix outputs, the M3 files of
    Tree-to-Spike and DNN-Tree (test_fp32_int8.json rebuilt from the block records'
    fp32 / per-tensor precisions; encoder_channels.json kept, after checking the channel
    count), and the M8 arm's test_fp32_int8.json.
A5  Edge-IIoTset seeds 45 and 46, Tree-to-Spike and DNN-Tree: the M3 run folders, curves and
    per-channel files are replaced by the default-environment reruns (which reproduce the
    production runs); the thread-capped originals go to results/superseded/m3_edge_threadcapped/.
D   results/deployment/deployment_measurements.csv is replaced by the 2026-10-07
    re-measurement (fixed teacher procedure, 11 datasets); the old file goes to
    results/superseded/deployment_2026-09-22_capped_teacher/.

Guards: every source exists; every block record has validation selection and three
precisions; split and teacher equal the replaced record's; the M3 channel count equals the
block record's. Writes results/journal_partB/swap_manifest.json. Refuses to run twice.

    python scripts/partB_swap_in.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

R = C.REPO
DS = "ciciot2023"
BLOCK = R / "results/journal_partB/B6/block"
FU = R / "results/journal_partB/B6/followup"
FIXED = R / "results/fixed_budget_ciciot2023"
SUP = R / "results/superseded"
ARMS = {"tree": "results/Tree-to-Spike/{d}", "dnn": "results/DNN-Tree/{d}",
        "vanilla": "results/{d}/SNN_LeakyParallel/latency", "tasnn": "results/TASNN/{d}",
        "dnn_tq": "results/DNN-TQ/{d}", "raw": "results/encoder_ablation/{d}/m8_raw",
        **{a: f"results/encoder_ablation/{{d}}/{a}" for a in ("quantile", "uniform", "all", "nokd", "entropy", "entropy_tree")}}
B8 = R / "results/journal_partB/B8/20261007T041558Z-66406a"
A5 = R / "results/journal_partB/A5_m3_edge"
MANIFEST = R / "results/journal_partB/swap_manifest.json"
moves = []


def mv(src: Path, dst: Path, dry):
    if not src.exists():
        sys.exit(f"missing {src}")
    if dst.exists():
        sys.exit(f"destination exists {dst}")
    moves.append(("move", str(src.relative_to(R)), str(dst.relative_to(R))))
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def cp(src: Path, dst: Path, dry):
    if not src.exists():
        sys.exit(f"missing {src}")
    moves.append(("copy", str(src.relative_to(R)), str(dst.relative_to(R))))
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(str(src), str(dst))


def m3_json(rec, model, seed):
    p = rec["test_plus"]["precisions"]
    g = lambda k, m: 100 * p[k]["metrics"][m]
    gb = lambda k, m: 100 * p[k]["binary_ids_metrics"][m]
    return dict(model=model, dataset=DS, seed=seed, fp_macro_f1=g("fp32", "f1_macro"),
                int8_macro_f1=g("per_tensor_int8", "f1_macro"), fp_mcc=g("fp32", "matthews_corrcoef"),
                int8_mcc=g("per_tensor_int8", "matthews_corrcoef"), fp_FAR=gb("fp32", "false_alarm_rate"),
                int8_FAR=gb("per_tensor_int8", "false_alarm_rate"), fp_DR=gb("fp32", "detection_rate"),
                int8_DR=gb("per_tensor_int8", "detection_rate"),
                stored_int8_macro_f1=100 * rec["test_plus"]["metrics"]["f1_macro"],
                reproduced=bool(p["per_tensor_int8"]["metrics"]["f1_macro"] == rec["test_plus"]["metrics"]["f1_macro"]),
                source="Part B A1 block record (20 epochs, validation-selected checkpoint)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dry = a.dry_run
    if MANIFEST.exists():
        sys.exit("swap already done (manifest exists)")
    sig = lambda r: (r["test_plus"]["n_samples"], json.dumps(r["data"]["class_counts_train"], sort_keys=True))
    # ---- A1 run folders
    for arm, patt in ARMS.items():
        for s in C.SEEDS:
            prod = R / patt.format(d=DS) / f"seed_{s}"
            blk = BLOCK / arm / DS / f"seed_{s}"
            rb, rp = json.loads((blk / "results.json").read_text()), json.loads((prod / "results.json").read_text())
            if not rb["training"].get("selection", {}).get("enabled") or "precisions" not in rb["test_plus"]:
                sys.exit(f"{arm}/{s}: block record lacks selection or precisions")
            if sig(rb) != sig(rp):
                sys.exit(f"{arm}/{s}: block split differs from the record it replaces")
            tb = rb["test_plus"].get("teacher_upper_bound", {}).get("f1_macro")
            tp = rp["test_plus"].get("teacher_upper_bound", {}).get("f1_macro")
            if tb != tp:
                if arm != "raw":
                    sys.exit(f"{arm}/{s}: teacher differs ({tb} vs {tp})")
                # The raw arm (raw_vanilla encoder, a_kd = a_rank = 0) fits the teacher but never
                # uses it, so its student does not depend on it. The production raw runs were
                # trained in the single-thread M8 pool and record the capped teacher; the block
                # runs record the production one. Recorded, not a refusal.
                moves.append(("note", f"raw/{s}: replaced record's teacher {tp} (single-thread pool), block {tb}; arm does not use the teacher", ""))
            mv(prod, FIXED / prod.relative_to(R / "results"), dry)
            for f in ("results.json", "checkpoint.pt", "checkpoint_fp32.pt"):
                if (blk / f).exists():
                    cp(blk / f, prod / f, dry)
            if arm == "raw":
                old = json.loads((FIXED / prod.relative_to(R / "results") / "test_fp32_int8.json").read_text()) if not dry \
                    else json.loads((prod / "test_fp32_int8.json").read_text())
                new = {k: old[k] for k in ("arm", "dataset", "seed", "encoder")}
                new.update({k: v for k, v in m3_json(rb, "m8_raw", s).items() if k.startswith(("fp_", "int8_"))})
                new.update(trainable_parameters=rb["model_size"]["trainable_parameters"],
                           encoder_channels=rb["data"]["encoder_channels"], input_features=old["input_features"],
                           source="Part B A1 block record (20 epochs, validation-selected checkpoint)")
                moves.append(("write", "", str((prod / "test_fp32_int8.json").relative_to(R))))
                if not dry:
                    (prod / "test_fp32_int8.json").write_text(json.dumps(new, indent=1))
    # ---- A1 prefix outputs (Tree-to-Spike)
    for s in C.SEEDS:
        prod = R / f"results/prefix_outputs/{DS}/treespike/seed_{s}"
        mv(prod, FIXED / "prefix_outputs" / DS / "treespike" / f"seed_{s}", dry)
        cp(FU / f"prefix_outputs/{DS}/treespike/seed_{s}", prod, dry)
    # ---- A1 M3 files (Tree-to-Spike, DNN-Tree)
    for model, arm in (("Tree-to-Spike", "tree"), ("DNN-Tree", "dnn")):
        vc = R / f"results/diagnostics/val_curves/{model}/{DS}"
        mv(vc, FIXED / "diagnostics/val_curves" / model / DS, dry)
        for s in C.SEEDS:
            src_old = (FIXED if not dry else R / "results") / ("diagnostics/val_curves" if not dry else "diagnostics/val_curves") / model / DS / f"runs/seed_{s}"
            rb = json.loads((BLOCK / arm / DS / f"seed_{s}/results.json").read_text())
            ch = json.loads((src_old / "encoder_channels.json").read_text())
            if len(ch["channels"]) != rb["data"]["encoder_channels"]:
                sys.exit(f"M3 {model}/{s}: channel count differs")
            run = vc / f"runs/seed_{s}"
            for f in ("results.json", "checkpoint.pt", "checkpoint_fp32.pt"):
                cp(BLOCK / arm / DS / f"seed_{s}" / f, run / f, dry)
            cp(src_old / "encoder_channels.json", run / "encoder_channels.json", dry)
            moves.append(("write", "", str((run / "test_fp32_int8.json").relative_to(R))))
            if not dry:
                (run / "test_fp32_int8.json").write_text(json.dumps(m3_json(rb, model, s), indent=2))
    # ---- A5 Edge-IIoTset M3 cells
    for model in ("Tree-to-Spike", "DNN-Tree"):
        for s in (45, 46):
            vc = R / f"results/diagnostics/val_curves/{model}/edgeiiot"
            mv(vc / f"runs/seed_{s}", SUP / "m3_edge_threadcapped/val_curves" / model / f"edgeiiot/runs/seed_{s}", dry)
            mv(vc / f"seed_{s}.csv", SUP / "m3_edge_threadcapped/val_curves" / model / f"edgeiiot/seed_{s}.csv", dry)
            cp(A5 / model / f"edgeiiot/runs/seed_{s}", vc / f"runs/seed_{s}", dry)
            cp(A5 / model / f"edgeiiot/seed_{s}.csv", vc / f"seed_{s}.csv", dry)
            pc = R / f"results/diagnostics/perchannel/{model}_edgeiiot_seed{s}.json"
            mv(pc, SUP / "m3_edge_threadcapped/perchannel" / pc.name, dry)
            cp(A5 / "perchannel" / pc.name, pc, dry)
    # ---- deployment table
    dep = R / "results/deployment/deployment_measurements.csv"
    mv(dep, SUP / "deployment_2026-09-22_capped_teacher/deployment_measurements.csv", dry)
    cp(B8 / "deployment_measurements_partB.csv", dep, dry)
    rec = dict(stamp=C.stamp("swap_in"), dry_run=dry, n=len(moves), moves=moves)
    if dry:
        print(json.dumps(dict(n=len(moves), first=moves[:6], last=moves[-4:]), indent=1))
    else:
        MANIFEST.write_text(json.dumps(rec, indent=1))
        print(f"swap done: {len(moves)} operations; manifest {MANIFEST.relative_to(R)}")


if __name__ == "__main__":
    main()
