import os
import numpy as np

# Пути к данным (ожидается, что они лежат рядом с ноутбуком)
DATA_DIR = "."
PATH_INTERACTIONS = os.path.join(DATA_DIR, "interactions.parquet")
PATH_EMBEDDINGS = os.path.join(DATA_DIR, "embeddings.parquet")
PATH_ARTISTS = os.path.join(DATA_DIR, "artists.parquet")

# Глобальные параметры
TOPK = 100
CORE_MIN_INTERACTIONS_PER_ITEM = 5
TEST_INTERVAL_SECONDS = 7 * 24 * 60 * 60
VOCAB_ITEMS = 30_000
MAX_EVENTS_PER_USER = 100  # включая BOS
MAX_LEN_PER_USER = 100  # итоговая длина истории, включая BOS
BOS = 0

# Для воспроизводимости
np.random.seed(42)

__all__ = [
    "DATA_DIR",
    "PATH_INTERACTIONS",
    "PATH_EMBEDDINGS",
    "PATH_ARTISTS",
    "TOPK",
    "CORE_MIN_INTERACTIONS_PER_ITEM",
    "TEST_INTERVAL_SECONDS",
    "VOCAB_ITEMS",
    "MAX_EVENTS_PER_USER",
    "MAX_LEN_PER_USER",
    "BOS",
]
