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


def get_dataset(name: str, **kwargs):
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name](**kwargs)
