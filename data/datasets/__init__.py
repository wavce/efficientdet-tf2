from .dataset import Dataset

from .objects365_dataset import Objects365Dataset


DATASET = {
    "objects365": Objects365Dataset 
}


def build_dataset(dataset, **kwargs):
    return DATASET[dataset](**kwargs).dataset()


__all__ = [
    "Dataset",
    "build_dataset"
]
