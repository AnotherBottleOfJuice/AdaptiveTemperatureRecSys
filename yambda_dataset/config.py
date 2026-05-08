import os
import numpy as np

# Data paths (relative to root project directory)
DATA_DIR = ".."
PATH_INTERACTIONS = os.path.join(DATA_DIR, "interactions.parquet")
PATH_EMBEDDINGS = os.path.join(DATA_DIR, "embeddings.parquet")
PATH_ARTISTS = os.path.join(DATA_DIR, "artists.parquet")

# Dataset parameters
VOCAB_SIZE = 157_157
CORE_MIN_INTERACTIONS_PER_ITEM = 5

# Data split parameters
TEST_INTERVAL_SECONDS = 7 * 24 * 60 * 60

# Sequence parameters (used by dataset modules)
BOS = 0
MAX_LEN = 100  # including BOS
MAX_TRAIN_EVENTS_PER_USER = 100  # including BOS

# Negative sampling parameters
UNIFORM_NEGATIVES_NUM = 25_000
IN_BATCH_NEGATIVES_NUM = 5_000

# Reproducibility
np.random.seed(42)

__all__ = [
    "DATA_DIR",
    "PATH_INTERACTIONS",
    "PATH_EMBEDDINGS",
    "PATH_ARTISTS",
    "VOCAB_SIZE",
    "CORE_MIN_INTERACTIONS_PER_ITEM",
    "TEST_INTERVAL_SECONDS",
    "BOS",
    "MAX_LEN",
    "MAX_TRAIN_EVENTS_PER_USER",
    "UNIFORM_NEGATIVES_NUM",
    "IN_BATCH_NEGATIVES_NUM",
]
