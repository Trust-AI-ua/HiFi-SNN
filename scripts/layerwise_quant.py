#!/usr/bin/env python3
"""Quantize one DNN-Tree layer at a time (per-tensor and per-output-channel) and score on test.

Localizes a quantization loss to a layer. Used for TON-IoT seed 43, where per-channel
scaling does not repair the dense twin: the loss and the false-alarm spike are confined
to fc1. Scores on CUDA, where the M3 reruns were scored; the fp32 score must reproduce the
recorded one or the script stops.

    python scripts/layerwise_quant.py toniot 43
"""
import copy, json, sys, inspect, torch, yaml
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import src.training_dnn as tdnn
from src.models import TreeGuidedSpikeEncoder, DNNStudent
from src.data import get_dataset, DATASET_LOADERS
from src.metrics import compute_metrics, compute_binary_ids_metrics
ds, seed = sys.argv[1], int(sys.argv[2]); dev = torch.device("cuda")
run = str(REPO) + f"/results/diagnostics/val_curves/DNN-Tree/{ds}/runs/seed_{seed}"
rec = json.load(open(f"{run}/test_fp32_int8.json"))
cfg = yaml.safe_load(open(str(REPO) + f"/experiments/DNN-Tree/{ds}/config.yaml")); cfg.update(seed=seed, device="cuda"); cfg = tdnn._resolve(cfg)
lp = inspect.signature(DATASET_LOADERS[ds]).parameters
dk = dict(batch_size=cfg["batch_size"], val_fraction=cfg["val_fraction"], seed=seed)
for o in ("subsample_fraction", "test_fraction", "label_mode"):
    if o in cfg and o in lp: dk[o] = cfg[o]
b = get_dataset(ds, **dk); Xtr, ytr, _, _, Xte, yte = tdnn._arrays_from_bundle(b)
teacher, _, _ = tdnn._fit_teacher(Xtr, ytr, b.num_classes, cfg)
enc = TreeGuidedSpikeEncoder(T=cfg["T"], pop_max=cfg["pop_max"], prune_quantile=cfg["prune_quantile"], jitter=cfg["jitter"]).fit(teacher, Xtr.shape[1])
fp = DNNStudent(enc.out_dim, b.num_classes, hidden=tuple(cfg["hidden"]), clip_relu=cfg["clip_relu"])
fp.load_state_dict(torch.load(f"{run}/checkpoint_fp32.pt", map_location="cpu")); fp.eval()
def ev(m):
    m.eval()
    with torch.no_grad(): p, sc = tdnn._forward_batches(m.to(dev), enc, Xte, cfg["batch_size"], dev)[:2]
    mm = compute_metrics(yte, p, y_score=sc, class_names=b.class_names); bb = compute_binary_ids_metrics(yte, p, normal_class_index=0)
    return 100*mm["f1_macro"], 100*bb["false_alarm_rate"]
def q(m, names, per_channel):
    with torch.no_grad():
        for n, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and n in names:
                w = mod.weight.data
                s = (w.abs().amax(1, keepdim=True) if per_channel else w.abs().max()) / 127 + 1e-12
                mod.weight.data = torch.round(w / s).clamp(-127, 127) * s
    return m
lin = [n for n, mod in fp.named_modules() if isinstance(mod, torch.nn.Linear)]
f0 = ev(copy.deepcopy(fp)); assert abs(f0[0] - rec["fp_macro_f1"]) < 1e-6, "fp32 not reproduced"
print(f"{ds} seed {seed} fp32: F1 {f0[0]:.2f} FAR {f0[1]:.2f}   layers {lin}")
for n in lin:
    for pc in (False, True):
        r = ev(q(copy.deepcopy(fp), {n}, pc)); print(f"  only {n:6s} {'per-channel' if pc else 'per-tensor '}: F1 {r[0]:.2f} FAR {r[1]:.2f}")
r = ev(q(copy.deepcopy(fp), set(lin), False)); print(f"  all layers per-tensor : F1 {r[0]:.2f} FAR {r[1]:.2f} (recorded int8 {rec['int8_macro_f1']:.2f})")
