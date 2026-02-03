# src/data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset

from .config import NUM_COLS, CAT_COLS, TARGET_COL, SEED


def load_spotify_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # if you do any initial filtering in the notebook, do it here
    return df


def preprocess_and_split(df: pd.DataFrame):
    """
    Returns:
        df2: cleaned dataframe
        X_train_num, X_val_num, X_test_num
        X_train_cat, X_val_cat, X_test_cat
        y_train, y_val, y_test
        scaler: fitted StandardScaler
        le: fitted LabelEncoder for track_genre
    """
    df2 = df.copy()

    # label encode target
    le = LabelEncoder()
    df2["genre_idx"] = le.fit_transform(df2[TARGET_COL])

    # numeric / categorical matrices
    X_num = df2[NUM_COLS].values.astype(np.float32)
    X_cat = df2[CAT_COLS].values

    y = df2["genre_idx"].values

    X_train_num, X_temp_num, X_train_cat, X_temp_cat, y_train, y_temp = \
        train_test_split(X_num, X_cat, y, test_size=0.30,
                         stratify=y, random_state=SEED)

    X_val_num, X_test_num, X_val_cat, X_test_cat, y_val, y_test = \
        train_test_split(X_temp_num, X_temp_cat, y_temp, test_size=0.50,
                         stratify=y_temp, random_state=SEED)

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_val_num = scaler.transform(X_val_num)
    X_test_num = scaler.transform(X_test_num)

    # map categories to integer IDs (same mapping as in your notebook)
    cat_maps = {}
    X_train_cat_int = np.zeros_like(X_train_cat, dtype=np.int64)
    X_val_cat_int = np.zeros_like(X_val_cat, dtype=np.int64)
    X_test_cat_int = np.zeros_like(X_test_cat, dtype=np.int64)

    for j, col in enumerate(CAT_COLS):
        vals = df2[col].astype(str).unique()
        mapping = {v: i + 1 for i, v in enumerate(vals)}  # 0 = "unknown"
        cat_maps[col] = mapping

        def map_array(arr):
            return np.array([mapping.get(str(v), 0) for v in arr], dtype=np.int64)

        X_train_cat_int[:, j] = map_array(X_train_cat[:, j])
        X_val_cat_int[:, j] = map_array(X_val_cat[:, j])
        X_test_cat_int[:, j] = map_array(X_test_cat[:, j])

    return (
        df2,
        X_train_num, X_val_num, X_test_num,
        X_train_cat_int, X_val_cat_int, X_test_cat_int,
        y_train, y_val, y_test,
        scaler, le, cat_maps,
    )


class SpotifyTabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_num[idx],
            self.X_cat[idx],
            self.y[idx],
        )
