import polars as pl

from .loader import get_amazon_data


class AmazonBeautyDataset:
    """Amazon Beauty interactions.

    Returns ``(train, test, test_targets)`` in the same canonical shape as
    :class:`recdata.yandex.dataset.YambdaDataset`. The actual loading/splitting
    lives in :func:`recdata.amazon.loader.get_amazon_data` (not implemented yet).
    """

    def __init__(
        self,
        path_interactions: str,
        core_min_interaction_per_item: int,
        test_interval_seconds: int,
    ):
        self.path_interactions = path_interactions
        self.core_min_interaction_per_item = core_min_interaction_per_item
        self.test_interval_seconds = test_interval_seconds

    def load(self) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
        return get_amazon_data(
            path_interactions=self.path_interactions,
            core_min_interaction_per_item=self.core_min_interaction_per_item,
            test_interval_seconds=self.test_interval_seconds,
        )
