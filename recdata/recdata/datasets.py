from .base import SequentialDataset
from .yandex import YambdaDataset
from .amazon import AmazonBeautyDataset

DATASETS = {
    "YambdaDataset": YambdaDataset,
    "AmazonBeautyDataset": AmazonBeautyDataset,
}


def build_dataset(class_name: str, **json_args) -> SequentialDataset:
    if class_name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{class_name}'. Available: {sorted(DATASETS)}"
        )
    return DATASETS[class_name](**json_args)
