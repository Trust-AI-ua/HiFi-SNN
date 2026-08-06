from src.data.nslkdd import load_nslkdd
from src.data.kddcup99 import load_kddcup99
from src.data.cicids2017 import load_cicids2017
from src.data.cseids2018 import load_cseids2018
from src.data.iscxids2012 import load_iscxids2012
from src.data.ctu13 import load_ctu13
from src.data.edgeiiot import load_edgeiiot
from src.data.unswnb15 import load_unswnb15
from src.data.ciciomt2024 import load_ciciomt2024
from src.data.ciciot2023 import load_ciciot2023
from src.data.toniot import load_toniot
from functools import partial as _partial
from src.data.leakage import load_leakage_bundle

DATASET_LOADERS = {
    "nslkdd": load_nslkdd,
    "kddcup99": load_kddcup99,
    "cicids2017": load_cicids2017,
    "cse-cic-ids2018": load_cseids2018,
    "iscxids2012": load_iscxids2012,
    "ctu13": load_ctu13,
    "edgeiiot": load_edgeiiot,
    "unswnb15": load_unswnb15,
    "ciciomt2024": load_ciciomt2024,
    "ciciot2023": load_ciciot2023,
    "toniot": load_toniot,
}

# Leakage-resistant protocols (round-2 reviewer item 1). Registered as first-class
# dataset names so every trainer can consume them unchanged:
#   <ds>_hostdisjoint_binary : constrained source-IP-disjoint 4-fold, binary task
#   <ds>_flow_binary         : binary flow-level CONTROL (same rows/dedup/encoding,
#                              random split) -- isolates the partition effect from
#                              the change in task granularity
for _ds in ("toniot", "edgeiiot"):
    DATASET_LOADERS[f"{_ds}_hostdisjoint_binary"] = _partial(
        load_leakage_bundle, dataset=_ds, split_mode="host", label_mode="binary")
    DATASET_LOADERS[f"{_ds}_flow_binary"] = _partial(
        load_leakage_bundle, dataset=_ds, split_mode="flow", label_mode="binary")
    # Fold-pinned variants: the dataset name fixes the outer partition, freeing the
    # trainer's `seed` to act purely as an optimisation seed. This is what makes the
    # 4 folds x 5 seeds design expressible without modifying any trainer.
    for _f in range(4):
        DATASET_LOADERS[f"{_ds}_hostdisjoint_binary_f{_f}"] = _partial(
            load_leakage_bundle, dataset=_ds, split_mode="host",
            label_mode="binary", fold_override=_f)
        DATASET_LOADERS[f"{_ds}_flow_binary_f{_f}"] = _partial(
            load_leakage_bundle, dataset=_ds, split_mode="flow",
            label_mode="binary", fold_override=_f)


# CTU-13 three-condition protocols (round-2 follow-up). The dataset name pins the
# condition, the outer fold AND the feature block, so the trainer's `seed` stays a
# pure optimisation seed and no trainer reconstructs a partition of its own.
#   ctu13_<condition>_f<fold>_<base|combined>
# The base/combined pair is what makes the host-context interaction measurable:
#   d_host = (combined_scenario - base_scenario) - (combined_random - base_random)
from src.data.ctu13_conditions import (load_ctu13_condition as _load_ctu13_cond,
                                       CONDITIONS as _CTU13_CONDITIONS,
                                       FEATURE_SETS as _CTU13_FEATURE_SETS)

for _c in _CTU13_CONDITIONS:
    for _f in range(4):
        for _fs in _CTU13_FEATURE_SETS:
            DATASET_LOADERS[f"ctu13_{_c}_f{_f}_{_fs}"] = _partial(
                _load_ctu13_cond, condition=_c, fold=_f, feature_set=_fs)


def get_dataset(name: str, **kwargs):
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name](**kwargs)
