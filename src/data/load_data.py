from pathlib import Path
import pandas as pd
from src.config import DATASET_ROOT


CSV_NAME = "IMDB_four_genre_larger_plot_description.csv"
POSTERS_DIR_NAME = "IMDB four_genre_posters"
VALID_GENRES = {"action", "comedy", "horror", "romance"}


def get_csv_path() -> Path:
    csv_path = DATASET_ROOT / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return csv_path


def get_posters_root() -> Path:
    posters_root = DATASET_ROOT / POSTERS_DIR_NAME
    if not posters_root.exists():
        raise FileNotFoundError(f"Posters folder not found: {posters_root}")
    return posters_root


def load_metadata() -> pd.DataFrame:
    csv_path = get_csv_path()
    df = pd.read_csv(csv_path)

    # standardize columns
    df.columns = [col.strip().lower() for col in df.columns]
    df["genre"] = df["genre"].str.strip().str.lower()
    df["movie_id"] = df["movie_id"].astype(str).str.strip()

    return df


def build_image_path(row, posters_root: Path) -> Path:
    genre = row["genre"]
    movie_id = row["movie_id"]
    return posters_root / genre.capitalize() / f"{movie_id}.jpg"


def attach_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    posters_root = get_posters_root()

    df = df.copy()
    df["image_path"] = df.apply(lambda row: build_image_path(row, posters_root), axis=1)
    df["image_exists"] = df["image_path"].apply(lambda p: p.exists())

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    print("Dataset shape:", df.shape)
    print("\nGenres:")
    print(df["genre"].value_counts())

    print("\nMissing descriptions:", df["description"].isnull().sum())
    print("Missing genres:", df["genre"].isnull().sum())
    print("Missing movie_ids:", df["movie_id"].isnull().sum())

    missing_images = (~df["image_exists"]).sum()
    print("\nRows with missing images:", missing_images)

    if missing_images > 0:
        print("\nSample missing image rows:")
        print(df.loc[~df["image_exists"], ["movie_id", "genre", "image_path"]].head())


def load_full_dataset() -> pd.DataFrame:
    df = load_metadata()
    df = attach_image_paths(df)
    return df


if __name__ == "__main__":
    df = load_full_dataset()
    validate_dataset(df)

    print("\nFirst 5 rows:")
    print(df.head())