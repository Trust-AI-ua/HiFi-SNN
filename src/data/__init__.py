from src.data.nslkdd import load_nslkdd

DATASET_LOADERS = {
    "nslkdd": load_nslkdd,
}


def get_dataset(name: str, **kwargs):
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name](**kwargs)
