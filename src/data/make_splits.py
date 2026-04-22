import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import SPLITS_DIR, ID_COLUMN, LABEL_COLUMN


def create_splits(
    df,
    test_size=0.2,
    val_size=0.2,
    random_state=42
):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[LABEL_COLUMN]
    )

    adjusted_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_df[LABEL_COLUMN]
    )

    return train_df, val_df, test_df


def save_split_ids(train_df, val_df, test_df):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df[[ID_COLUMN]].to_csv(SPLITS_DIR / "train_ids.csv", index=False)
    val_df[[ID_COLUMN]].to_csv(SPLITS_DIR / "val_ids.csv", index=False)
    test_df[[ID_COLUMN]].to_csv(SPLITS_DIR / "test_ids.csv", index=False)

    train_df.to_csv(SPLITS_DIR / "train_split.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val_split.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test_split.csv", index=False)


def load_saved_splits():
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    return train_df, val_df, test_df
