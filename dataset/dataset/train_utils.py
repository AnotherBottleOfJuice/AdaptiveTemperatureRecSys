import polars as pl

from config import MAX_TRAIN_EVENTS_PER_USER, BOS


def get_train_events(train: pl.DataFrame, item_to_token_id: pl.DataFrame,
                     max_events_per_user: int = MAX_TRAIN_EVENTS_PER_USER) -> pl.DataFrame:
    train_events = (
        train
        .join(item_to_token_id, on='item_id', how='inner')
        .sort('timestamp')
        .drop(['is_organic', 'len', 'item_id'])
    )

    return (
        train_events
        .group_by('uid', maintain_order=True)
        .tail(max_events_per_user - 1)
    )


def get_train_histories(train_events: pl.DataFrame,
                        bos: int = BOS) -> pl.DataFrame:
    return (
        pl.concat([
            (
                train_events.select('uid').unique()
                .with_columns(pl.lit(bos, dtype=pl.UInt32).alias('token_id'))
            ),
            train_events.drop('timestamp')
        ])
        .group_by('uid', maintain_order=True)
        .agg('token_id')
    )
