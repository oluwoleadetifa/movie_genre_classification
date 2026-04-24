import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.config import PROCESSED_DATA_DIR
from src.data.load_data import load_full_dataset


def preprocess_dataset() -> pd.DataFrame:
    df = load_full_dataset()

    # keep only valid rows
    df = df[df["image_exists"]].copy()

    # remove duplicate movie ids if any
    df = df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)

    # simple text cleanup
    df["description"] = df["description"].astype(str).str.strip()

    # encode labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["genre"])

    print("Label mapping:")
    for genre, label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f"{genre}: {label}")

    return df


def save_processed_dataset(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "movies_with_posters.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed dataset to: {output_path}")


if __name__ == "__main__":
    df = preprocess_dataset()
    print("\nProcessed shape:", df.shape)
    print(df.head())

    save_processed_dataset(df)