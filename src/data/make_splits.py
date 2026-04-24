make_splits_text = """
import pandas as pd
from src.config import TRAIN_SPLIT_CSV, VAL_SPLIT_CSV, TEST_SPLIT_CSV


def load_saved_splits():
    train_df = pd.read_csv(TRAIN_SPLIT_CSV)
    val_df = pd.read_csv(VAL_SPLIT_CSV)
    test_df = pd.read_csv(TEST_SPLIT_CSV)

    return train_df, val_df, test_df
"""

with open("/content/movie_genre_classification/src/data/make_splits.py", "w") as f:
    f.write(make_splits_text)

print("make_splits.py updated")
