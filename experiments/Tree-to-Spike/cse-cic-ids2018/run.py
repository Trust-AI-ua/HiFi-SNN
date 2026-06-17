"""Tree-to-Spike (Design C) -- cse-cic-ids2018

Thin wrapper. Reads its sibling config.yaml and calls the shared
src.training_treespike.run_experiment. No experiment-specific logic lives here.
Multi-seed: pass HISNN_SEED=<n> (the orchestrator does this) or edit `seed`.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]          # experiments/Tree-to-Spike/<dataset>/run.py -> repo root
sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.training_treespike import run_experiment


def main():
    cfg = yaml.safe_load(open(HERE.with_name("config.yaml")))
    for key in ("results_dir", "summary_csv"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(REPO_ROOT / cfg[key])
    record = run_experiment(cfg)
    tp = record["test_plus"]
    m = tp["metrics"]; b = tp["binary_ids_metrics"]; sp = tp["spike_stats"]
    ee = tp["early_exit"]; tub = tp["teacher_upper_bound"]
    auc = m.get("roc_auc_weighted")
    auc_s = f"{auc:.4f}" if isinstance(auc, (int, float)) else "n/a"
    print(
        f"[done] Tree-to-Spike/{cfg['dataset']}"
        f" acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}"
        f" MCC={m['matthews_corrcoef']:.4f} auc={auc_s}"
        f" DR={b['detection_rate']:.4f} FAR={b['false_alarm_rate']:.4f}"
        f" sparsity={sp['spike_sparsity']:.3f} synops/sample={sp['synops_per_sample']:.0f}"
        f" exit={ee['mean_exit_step']:.1f}/{cfg['T']}"
        f" | teacher_f1m={tub['f1_macro']:.4f}"
        f" train_s={record['training']['train_time_s']:.1f}"
    )


if __name__ == "__main__":
    main()
