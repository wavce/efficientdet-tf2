from .dataset import Dataset

from .objects365_dataset import Objects365Dataset


DATASET = {
    "objects365": Objects365Dataset 
}


def build_dataset(name, **kwargs):
    return DATASET[name](**kwargs).dataset()


__all__ = [
    "Dataset",
    "build_dataset"
]
