import pandas as pd
from src.config import DATA_CSV, TEXT_COLUMN, ID_COLUMN, LABEL_COLUMN, GENRE_COLUMN


def load_dataset(csv_path=None):
    csv_path = csv_path or DATA_CSV
    df = pd.read_csv(csv_path)

    required_columns = [ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN, GENRE_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def basic_dataset_check(df):
    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing descriptions:")
    print(df[TEXT_COLUMN].isna().sum())

    print("\nGenre distribution:")
    print(df[GENRE_COLUMN].value_counts())

    print("\nLabel distribution:")
    print(df[LABEL_COLUMN].value_counts().sort_index())
