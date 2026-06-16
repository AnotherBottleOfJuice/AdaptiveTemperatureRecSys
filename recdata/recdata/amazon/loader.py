import polars as pl
from datasets import load_dataset


def get_amazon_data(
    path_interactions: str,
    core_min_interaction_per_item: int,
    test_interval_seconds: int,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Load and split the Amazon Beauty interactions.

    TODO: implement. Must return ``(train, test, test_targets)`` with train/test
    carrying canonical columns ``uid``, ``item_id``, ``timestamp`` and
    ``test_targets`` mapping ``uid`` -> numpy array of held-out ``item_id`` values
    (see :data:`recdata.base.CANONICAL_COLUMNS`).
    """
    
    
    
