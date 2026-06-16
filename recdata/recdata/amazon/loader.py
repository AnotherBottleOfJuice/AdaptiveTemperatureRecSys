import numpy as np
import polars as pl

_MS_PER_SECOND = 1000


def get_amazon_data(
    path_interactions: str,
    core_min_interaction_per_item: int,
    test_interval_seconds: int,
    window_start: int,
    window_end: int,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    interactions = (
        pl.scan_ndjson(path_interactions)
        .select(["rating", "asin", "user_id", "timestamp"])
        .filter(pl.col("rating") >= 4)
        .filter(
            (pl.col("timestamp") >= window_start)
            & (pl.col("timestamp") < window_end)
        )
        .select(
            pl.col("user_id").alias("uid"),
            pl.col("asin").alias("item_id"),
            pl.col("timestamp"),
        )
        .collect()
    )

    count = interactions.group_by("item_id").len()

    interactions = interactions.join(count, on="item_id", how="left").filter(
        pl.col("len") > core_min_interaction_per_item
    ).drop("len")

    cutoff = window_end - test_interval_seconds * _MS_PER_SECOND

    train = interactions.filter(pl.col("timestamp") <= cutoff)
    test = interactions.filter(pl.col("timestamp") > cutoff)
    test = test.filter(pl.col("uid").is_in(train["uid"].implode()))

    test_targets = {
        i[0]: np.array(i[1]) for i in test.group_by("uid").agg("item_id").rows()
    }

    return train, test, test_targets
