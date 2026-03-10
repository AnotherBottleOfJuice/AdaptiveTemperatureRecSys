import polars as pl

from config import *


def get_general_data():
    interactions = pl.read_parquet(PATH_INTERACTIONS)
    embeddings = pl.read_parquet(PATH_EMBEDDINGS)
    artists = pl.read_parquet(PATH_ARTISTS)

    interactions = interactions.join(embeddings, on="item_id", how="semi")

    count = interactions.group_by('item_id').len()

    interactions = (interactions.join(count, on='item_id', how='left')
                    .filter(pl.col('len') >= CORE_MIN_INTERACTIONS_PER_ITEM))

    interactions = interactions.join(artists, on='item_id', how='semi')

    end = interactions['timestamp'].max()

    train = interactions.filter(pl.col('timestamp') < end - TEST_INTERVAL_SECONDS)
    test = interactions.filter(pl.col('timestamp') >= end - TEST_INTERVAL_SECONDS)
    test = test.filter(pl.col('uid').is_in(train['uid'].implode()))

    test_targets = {
        i[0]: i[1]
        for i in test.group_by('uid').agg('item_id').rows()
    }

    embeddings = embeddings.filter(pl.col('item_id').is_in(interactions['item_id'].implode()))

    return train, test, embeddings, artists, test_targets


def get_item_to_token(train):
    return (
        train
        .select("item_id")  # берём только item_id (проще и быстрее)
        .group_by('item_id').len().rename({'len': 'count'})
        .sort('item_id')
        .reverse()
        .sort('count')
        .reverse()
        .head(VOCAB_ITEMS)
        .with_row_index(name='token_id')
        .with_columns(pl.col('token_id') + 1)
        .select(('item_id', 'token_id'))
    )
