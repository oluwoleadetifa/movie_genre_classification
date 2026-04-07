import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_DIR, SPLITS_DIR


def make_splits():
    input_path = PROCESSED_DATA_DIR / "movies_with_posters.csv"
    df = pd.read_csv(input_path)

    # train 70, val 15, test 15
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["genre"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["genre"]
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print("Saved splits:")
    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)

    print("\nTrain genre distribution:")
    print(train_df["genre"].value_counts())

    print("\nVal genre distribution:")
    print(val_df["genre"].value_counts())

    print("\nTest genre distribution:")
    print(test_df["genre"].value_counts())


if __name__ == "__main__":
    make_splits()