import polars as pl

from .loader import get_amazon_data


class AmazonBeautyDataset:

    def __init__(
        self,
        path_interactions: str,
        core_min_interaction_per_item: int,
        test_interval_seconds: int,
        window_start: int,
        window_end: int,
    ):
        self.path_interactions = path_interactions
        self.core_min_interaction_per_item = core_min_interaction_per_item
        self.test_interval_seconds = test_interval_seconds
        self.window_start = window_start
        self.window_end = window_end

    def load(self) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
        return get_amazon_data(
            path_interactions=self.path_interactions,
            core_min_interaction_per_item=self.core_min_interaction_per_item,
            test_interval_seconds=self.test_interval_seconds,
            window_start=self.window_start,
            window_end=self.window_end,
        )
