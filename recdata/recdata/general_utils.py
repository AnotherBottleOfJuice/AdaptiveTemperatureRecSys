import polars as pl


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
