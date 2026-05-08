import numpy as np
import polars as pl

from ..config import (PATH_EMBEDDINGS, PATH_ARTISTS, PATH_INTERACTIONS,
                      CORE_MIN_INTERACTIONS_PER_ITEM, TEST_INTERVAL_SECONDS)


def get_general_data(path_interactions: str = PATH_INTERACTIONS,
                     path_embeddings: str = PATH_EMBEDDINGS,
                     path_artists: str = PATH_ARTISTS,
                     core_min_interaction_per_user: int = CORE_MIN_INTERACTIONS_PER_ITEM,
                     test_interval_seconds: int = TEST_INTERVAL_SECONDS, ):
    interactions = pl.read_parquet(path_interactions)
    embeddings = pl.read_parquet(path_embeddings)
    artists = pl.read_parquet(path_artists)

    interactions = interactions.join(embeddings, on="item_id", how="semi")

    count = interactions.group_by('item_id').len()

    interactions = (interactions.join(count, on='item_id', how='left')
                    .filter(pl.col('len') >= core_min_interaction_per_user))

    interactions = interactions.join(artists, on='item_id', how='semi')

    end = interactions['timestamp'].max()

    train = interactions.filter(pl.col('timestamp') < end - test_interval_seconds)
    test = interactions.filter(pl.col('timestamp') >= end - test_interval_seconds)
    test = test.filter(pl.col('uid').is_in(train['uid'].implode()))

    test_targets = {
            i[0]: np.array(i[1])
            for i in test.group_by('uid').agg('item_id').rows()
    }
    
    embeddings = embeddings.filter(pl.col('item_id').is_in(interactions['item_id'].implode()))

    return train, test, embeddings, artists, test_targets

def get_item_to_freq(train : pl.DataFrame):
    return (
        train
        .select("item_id")
        .to_series()
        .value_counts(normalize=True, name='freq')
    )


def get_item_to_token(train : pl.DataFrame, vocab_size: int = None):
    return (
        train
        .select("item_id")
        .group_by('item_id').len().rename({'len': 'count'})
        .sort('item_id')
        .reverse()
        .sort('count')
        .reverse()
        .slice(0, vocab_size if vocab_size else None)
        .with_row_index(name='token_id')
        .with_columns(pl.col('token_id') + 1)
        .select(('item_id', 'token_id'))
    )
