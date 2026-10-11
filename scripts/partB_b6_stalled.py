#!/usr/bin/env python3
"""B6 diagnostic reruns of the three stalled entropy runs (pre-registered, Part B).

CICIoT2023 seeds 44 and 45 and CICIoMT2024 seed 46 (encoder_mode "entropy", E2 protocol)
are rerun unchanged through run_experiment, with an observer on the loop's own validation
pass (src.training_treespike._encode_forward_in_batches on Xval). At each epoch it records
the predicted-class distribution on validation and the per-layer firing rates (fraction of
hidden neurons spiking per timestep, layers 1 and 2) on the first 20,000 validation rows,
read through forward hooks. The observation runs under no_grad in eval mode and draws no
random numbers, so it cannot change the run.

Guard: the rerun must reproduce the stored run's training curve (train loss and validation
accuracy at every epoch) and its test macro-F1 exactly; otherwise the cell is reported as
not reproducible.

    python scripts/partB_b6_stalled.py <dataset> <seed>
Output: results/journal_partB/B6/stalled/<dataset>_seed<seed>/{observer.json,runs/...}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partB_common as C  # noqa: E402

CELLS = {("ciciot2023", 44), ("ciciot2023", 45), ("ciciomt2024", 46)}
N_OBS = 20_000


def main():
    import src.training_treespike as tts
    from encoder_ablation import ARM_CFG
    ds, seed = sys.argv[1], int(sys.argv[2])
    if (ds, seed) not in CELLS:
        sys.exit(f"{ds}/{seed} is not one of the pre-registered stalled cells")
    C.no_thread_caps()
    st = C.stamp("B6")
    out = C.REPO / f"results/journal_partB/B6/stalled/{ds}_seed{seed}"
    if (out / "observer.json").exists():
        print(f"[skip] {out}"); return
    out.mkdir(parents=True, exist_ok=True)
    cfg = C.t2s_config(ds, seed)
    cfg.update(ARM_CFG["entropy"])
    cfg.update(seed=seed, force=True, results_dir=str(out / "runs"), summary_csv=str(out / "runs" / "summary.csv"))

    captured, epochs = {}, []
    orig_arrays, orig_fwd = tts._arrays_from_bundle, tts._encode_forward_in_batches

    def arrays_hook(bundle):
        arrs = orig_arrays(bundle)
        captured["Xval"], captured["yval"], captured["K"] = arrs[2], arrs[3], bundle.num_classes
        return arrs

    def fwd_hook(student, encoder, X, batch_size, device):
        res = orig_fwd(student, encoder, X, batch_size, device)
        if X is captured.get("Xval") and len(epochs) < cfg["epochs"]:
            rates = {}
            hooks = [getattr(student, n).register_forward_hook(
                lambda m, i, o, n=n: rates.setdefault(n, []).append(float(o[0].float().mean())))
                for n in ("lif1", "lif2")]
            with torch.no_grad():
                for i in range(0, min(N_OBS, len(X)), batch_size):
                    student(encoder.encode(X[i:i + batch_size]).to(device))
            for h in hooks:
                h.remove()
            pred = res[0]
            epochs.append(dict(epoch=len(epochs) + 1,
                               pred_dist=np.bincount(pred, minlength=captured["K"]).tolist(),
                               true_dist=np.bincount(captured["yval"], minlength=captured["K"]).tolist(),
                               val_accuracy=float((pred == captured["yval"]).mean()),
                               firing_rate_l1=float(np.mean(rates["lif1"])), firing_rate_l2=float(np.mean(rates["lif2"]))))
            print(f"[observer] epoch {len(epochs)} val_acc {epochs[-1]['val_accuracy']:.4f} "
                  f"rates L1 {epochs[-1]['firing_rate_l1']:.4f} L2 {epochs[-1]['firing_rate_l2']:.4f}", flush=True)
            student.eval()
        return res

    tts._arrays_from_bundle, tts._encode_forward_in_batches = arrays_hook, fwd_hook
    rec = tts.run_experiment(cfg)
    stored = json.loads((C.REPO / f"results/encoder_ablation/{ds}/entropy/seed_{seed}/results.json").read_text())
    a, b = rec["training"]["training_curve"], stored["training"]["training_curve"]
    same_curve = len(a) == len(b) and all(x["train_loss"] == y["train_loss"] and x["val_accuracy"] == y["val_accuracy"]
                                          for x, y in zip(a, b))
    same_test = rec["test_plus"]["metrics"]["f1_macro"] == stored["test_plus"]["metrics"]["f1_macro"]
    result = dict(stamp=st, dataset=ds, seed=seed, reproduces_stored=bool(same_curve and same_test),
                  curve_identical=bool(same_curve), test_identical=bool(same_test), class_names=rec["data"]["class_names"],
                  epochs=epochs, loss_terms=[{k: v for k, v in e.items() if k.startswith("loss_") or k == "epoch"} for e in a])
    (out / "observer.json").write_text(json.dumps(result, indent=1))
    print(f"[done] {ds}/{seed}: reproduces stored run: {result['reproduces_stored']} "
          f"(curve {same_curve}, test {same_test})")


if __name__ == "__main__":
    main()
