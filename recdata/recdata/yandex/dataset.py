import polars as pl

from ..base import CANONICAL_COLUMNS
from .loader import get_general_data


class YambdaDataset:

    def __init__(
        self,
        path_interactions: str,
        path_embeddings: str,
        path_artists: str,
        core_min_interaction_per_item: int,
        test_interval_seconds: int,
    ):
        self.path_interactions = path_interactions
        self.path_embeddings = path_embeddings
        self.path_artists = path_artists
        self.core_min_interaction_per_item = core_min_interaction_per_item
        self.test_interval_seconds = test_interval_seconds

    def load(self) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
        train, test, _embeddings, _artists, test_targets = get_general_data(
            path_interactions=self.path_interactions,
            path_embeddings=self.path_embeddings,
            path_artists=self.path_artists,
            core_min_interaction_per_item=self.core_min_interaction_per_item,
            test_interval_seconds=self.test_interval_seconds,
        )
        return (
            train.select(CANONICAL_COLUMNS),
            test.select(CANONICAL_COLUMNS),
            test_targets,
        )
