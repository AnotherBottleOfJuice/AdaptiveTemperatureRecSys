from typing import Protocol, runtime_checkable

import polars as pl

CANONICAL_COLUMNS = ["uid", "item_id", "timestamp"]


@runtime_checkable
class SequentialDataset(Protocol):

    def load(self) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
        ...
