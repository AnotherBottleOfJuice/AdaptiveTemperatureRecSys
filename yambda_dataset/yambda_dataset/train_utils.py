import polars as pl

from yambda_dataset.config import MAX_TRAIN_EVENTS_PER_USER, BOS


def get_train_events(train: pl.DataFrame, item_to_token_id: pl.DataFrame,
                     item_to_freq: pl.DataFrame | None = None,
                     max_events_per_user: int = MAX_TRAIN_EVENTS_PER_USER,) -> pl.DataFrame:
    train_events = (
        train
        .join(item_to_token_id, on='item_id', how='inner')
        .sort('timestamp')
        .drop(['is_organic', 'len', 'item_id'])
    )

    if item_to_freq is not None:
        train_events = train_events.join(
            item_to_token_id.join(item_to_freq, on='item_id', how='left').drop('item_id'),
            on='token_id', how='left')

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
                .with_columns(pl.lit(1, dtype=pl.Float64).alias('freq'))
            ),
            train_events.drop('timestamp')
        ])
        .group_by('uid', maintain_order=True)
        .agg('token_id', ('freq' if 'freq' in train_events.columns else None))
    )
