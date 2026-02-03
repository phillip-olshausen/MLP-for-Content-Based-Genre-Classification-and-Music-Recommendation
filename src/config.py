# src/config.py
from dataclasses import dataclass
import torch
import numpy as np
import random
import os

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_COL = "track_genre"

NUM_COLS = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

CAT_COLS = ["explicit", "key", "mode", "time_signature"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    hidden_dims: list
    optimizer_name: str
    lr: float
    batch_size: int
    epochs: int
    patience: int
