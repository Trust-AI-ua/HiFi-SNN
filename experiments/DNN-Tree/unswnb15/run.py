"""DNN-Tree (iso-architecture ANN twin of Tree-to-Spike) -- unswnb15

Thin wrapper. Reads its sibling config.yaml and calls the shared
src.training_dnn.run_experiment. Multi-seed: pass HISNN_SEED=<n>.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]          # experiments/DNN-Tree/<dataset>/run.py -> repo root
sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.training_dnn import run_experiment


def main():
    cfg = yaml.safe_load(open(HERE.with_name("config.yaml")))
    for key in ("results_dir", "summary_csv"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(REPO_ROOT / cfg[key])
    r = run_experiment(cfg)
    tp = r["test_plus"]; m = tp["metrics"]; b = tp["binary_ids_metrics"]
    c = tp["compute"]; tub = tp["teacher_upper_bound"]
    auc = m.get("roc_auc_weighted"); auc_s = f"{auc:.4f}" if isinstance(auc,(int,float)) else "n/a"
    print(
        f"[done] DNN-Tree/{cfg['dataset']}"
        f" acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f} MCC={m['matthews_corrcoef']:.4f} auc={auc_s}"
        f" DR={b['detection_rate']:.4f} FAR={b['false_alarm_rate']:.4f}"
        f" MACs={c['macs_per_sample']} E={c['energy_pj_horowitz']:.0f}pJ"
        f" lat_bs1={tp['inference_ms_per_sample_batch1']:.3f}ms"
        f" | teacher_f1m={tub['f1_macro']:.4f} train_s={r['training']['train_time_s']:.1f}"
    )


if __name__ == "__main__":
    main()
