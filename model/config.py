import os
from dotenv import load_dotenv

load_dotenv()

# Model architecture parameters
LOG_Q_CORRECTION = 1.0

# Training parameters
TOPK = 100

# API
COMET_API_KEY = os.getenv("COMET_API_KEY", "")

__all__ = [
    "LOG_Q_CORRECTION",
    "TOPK",
    "COMET_API_KEY",
]
