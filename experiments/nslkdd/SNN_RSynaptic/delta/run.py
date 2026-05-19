"""NSL-KDD · SNN_RSynaptic · delta

Thin wrapper. Reads the matching config from configs/ and runs the shared
src.training.run_experiment. No experiment-specific logic lives here.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]                       # …/HiSNN
sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.training import run_experiment


def _config_path() -> Path:
    # experiments/{dataset}/SNN_{neuron}/{encoding}/run.py
    #     ↓ swap "experiments" → "configs", "run.py" → "config.yaml"
    rel = HERE.relative_to(REPO_ROOT)
    parts = list(rel.parts)
    parts[0] = "configs"
    parts[-1] = "config.yaml"
    return REPO_ROOT.joinpath(*parts)


def main():
    cfg = yaml.safe_load(open(_config_path()))
    # Re-anchor relative paths to repo root so the script works from any CWD.
    for key in ("results_dir", "summary_csv"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(REPO_ROOT / cfg[key])
    record = run_experiment(cfg)
    m = record["metrics"]
    print(
        f"[done] {cfg['dataset']}/SNN_{cfg['neuron']}/{cfg['encoding']}"
        f" acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}"
        f" auc={m['roc_auc_weighted'] if m['roc_auc_weighted'] is not None else float('nan'):.4f}"
        f" energy={record['energy_spikes_per_sample']:.1f}/sample"
        f" train_s={record['train_time_s']:.1f}"
    )


if __name__ == "__main__":
    main()
