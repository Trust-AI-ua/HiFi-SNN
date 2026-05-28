"""CTU-13 · SNN_SConv2dLSTM · latency

Thin wrapper. Reads its matching config from configs/ and runs the shared
src.training.run_experiment. No experiment-specific logic lives here.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.training import run_experiment


def _config_path() -> Path:
    rel = HERE.relative_to(REPO_ROOT)
    parts = list(rel.parts)
    parts[0] = "configs"
    parts[-1] = "config.yaml"
    return REPO_ROOT.joinpath(*parts)


def main():
    cfg = yaml.safe_load(open(_config_path()))
    for key in ("results_dir", "summary_csv"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(REPO_ROOT / cfg[key])
    record = run_experiment(cfg)
    tp = record["test_plus"]
    m = tp["metrics"]
    b = tp["binary_ids_metrics"]
    sp = tp["spike_stats"]
    auc = m.get("roc_auc_weighted")
    auc_s = f"{auc:.4f}" if isinstance(auc, (int, float)) else "n/a"
    print(
        f"[done] {cfg['dataset']}/SNN_{cfg['neuron']}/{cfg['encoding']}"
        f" acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}"
        f" MCC={m['matthews_corrcoef']:.4f} auc={auc_s}"
        f" DR={b['detection_rate']:.4f} FAR={b['false_alarm_rate']:.4f}"
        f" sparsity={sp['spike_sparsity']:.3f}"
        f" train_s={record['training']['train_time_s']:.1f}"
    )


if __name__ == "__main__":
    main()
