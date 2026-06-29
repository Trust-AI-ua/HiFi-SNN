#!/usr/bin/env python3
"""Fast synops_per_sample backfill for Vanilla SNN runs missing it: replays the
saved checkpoint on a capped test subset (synops is a per-sample mean, so a subset
is an accurate estimate) and writes it into results.json. Eval-only, no retraining.
"""
import glob, json, sys, inspect
from pathlib import Path
import numpy as np, torch, yaml
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from src.data import get_dataset, DATASET_LOADERS
from src.encoding import get_encoder
from src.models import SNN
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 10000
DSETS = ["cse-cic-ids2018", "unswnb15", "toniot", "ciciomt2024", "ciciot2023", "edgeiiot"]
for ds in DSETS:
    cfg = yaml.safe_load(open(REPO / f"configs/{ds}/SNN_LeakyParallel/latency/config.yaml"))
    lp = inspect.signature(DATASET_LOADERS[ds]).parameters
    bundle = None
    for seed in [42, 43, 44, 45, 46]:
        rjp = REPO / f"results/{ds}/SNN_LeakyParallel/latency/seed_{seed}/results.json"
        if not rjp.exists(): continue
        r = json.load(open(rjp))
        if r["test_plus"]["spike_stats"].get("synops_per_sample") is not None: continue
        dk = {"seed": seed}
        for o in ("subsample_fraction", "test_fraction", "label_mode"):
            if o in cfg and o in lp: dk[o] = cfg[o]
        b = get_dataset(ds, **dk)
        model = SNN(num_features=b.num_features, hidden_size=cfg["hidden_size"],
                    num_classes=b.num_classes, neuron_name="LeakyParallel",
                    num_steps=cfg["num_steps"]).to(DEV)
        model.load_state_dict(torch.load(rjp.with_name("checkpoint.pt"), map_location=DEV)); model.eval()
        Xte = b.test_loader.dataset.tensors[0][:N].to(DEV)
        enc = get_encoder("latency")
        with torch.no_grad():
            spikes = enc(Xte, cfg["num_steps"]).to(DEV)
            syn = float(model.predict_early_exit(spikes)[-1])
        r["test_plus"]["spike_stats"]["synops_per_sample"] = syn
        json.dump(r, open(rjp, "w"), indent=2)
        print(f"  {ds}/seed_{seed}: synops_per_sample={syn:.0f}  (n={Xte.shape[0]})", flush=True)
print("DONE")
