from src.data.nslkdd import load_nslkdd
from src.data.kddcup99 import load_kddcup99
from src.data.cicids2017 import load_cicids2017
from src.data.cseids2018 import load_cseids2018
from src.data.iscxids2012 import load_iscxids2012
from src.data.ctu13 import load_ctu13
from src.data.edgeiiot import load_edgeiiot

DATASET_LOADERS = {
    "nslkdd": load_nslkdd,
    "kddcup99": load_kddcup99,
    "cicids2017": load_cicids2017,
    "cse-cic-ids2018": load_cseids2018,
    "iscxids2012": load_iscxids2012,
    "ctu13": load_ctu13,
    "edgeiiot": load_edgeiiot,
}


def get_dataset(name: str, **kwargs):
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name](**kwargs)
