import polars as pl


def get_test_users(test: pl.DataFrame) -> pl.DataFrame:
    return test.select("uid").unique()


def get_test_users_events(
    test: pl.DataFrame, train_events: pl.DataFrame, max_len: int
) -> pl.DataFrame:
    return (
        train_events.join(get_test_users(test), on="uid", how="semi")
        .sort("timestamp")
        .group_by("uid", maintain_order=True)
        .tail(n=max_len - 2)
        .select(["uid", "token_id"])
    )


def get_test_histories(
    test: pl.DataFrame, train_events: pl.DataFrame, bos: int, max_len: int
) -> pl.DataFrame:
    test_histories = (
        pl.concat(
            [
                get_test_users(test).with_columns(
                    pl.lit(bos, dtype=pl.UInt32).alias("token_id")
                ),
                get_test_users_events(test, train_events, max_len),
            ]
        )
        .group_by("uid", maintain_order=True)
        .agg("token_id")
    )

    return test_histories.with_columns(length=pl.col("token_id").list.len()).sort(
        "length", descending=True
    )
