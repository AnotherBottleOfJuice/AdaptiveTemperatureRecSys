import numpy as np
import polars as pl


def get_general_data(
    path_interactions: str,
    path_embeddings: str,
    path_artists: str,
    core_min_interaction_per_user: int,
    test_interval_seconds: int,
):
    interactions = pl.read_parquet(path_interactions)
    embeddings = pl.read_parquet(path_embeddings)
    artists = pl.read_parquet(path_artists)

    interactions = interactions.join(embeddings, on="item_id", how="semi")

    count = interactions.group_by("item_id").len()

    interactions = interactions.join(count, on="item_id", how="left").filter(
        pl.col("len") >= core_min_interaction_per_user
    )

    interactions = interactions.join(artists, on="item_id", how="semi")

    end = interactions["timestamp"].max()

    train = interactions.filter(pl.col("timestamp") < end - test_interval_seconds)
    test = interactions.filter(pl.col("timestamp") >= end - test_interval_seconds)
    test = test.filter(pl.col("uid").is_in(train["uid"].implode()))

    test_targets = {
        i[0]: np.array(i[1]) for i in test.group_by("uid").agg("item_id").rows()
    }

    embeddings = embeddings.filter(
        pl.col("item_id").is_in(interactions["item_id"].implode())
    )

    return train, test, embeddings, artists, test_targets


def get_item_to_freq(train: pl.DataFrame):
    return train.select("item_id").to_series().value_counts(normalize=True, name="freq")


def get_item_to_token(train: pl.DataFrame, vocab_size: int | None):
    return (
        train.select("item_id")
        .group_by("item_id")
        .len()
        .rename({"len": "count"})
        .sort("item_id")
        .reverse()
        .sort("count")
        .reverse()
        .slice(0, vocab_size if vocab_size else None)
        .with_row_index(name="token_id")
        .with_columns(pl.col("token_id") + 1)
        .select(("item_id", "token_id"))
    )
