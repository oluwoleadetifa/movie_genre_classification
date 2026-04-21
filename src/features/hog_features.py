from pathlib import Path
import numpy as np
import pandas as pd

from skimage.feature import hog

from src.data.preprocess_image import preprocess_image, to_grayscale
from src.config import PROCESSED_DATA_DIR


def extract_hog_features(image_array: np.ndarray) -> np.ndarray:
    """
    Extract HOG features from a grayscale image.
    """
    features = hog(
        image_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return features


def build_hog_dataset(df: pd.DataFrame):
    """
    Convert dataset into HOG feature matrix X and labels y.
    """
    X = []
    y = []

    for idx, row in df.iterrows():
        try:
            image = preprocess_image(row["image_path"])
            gray = to_grayscale(image)

            features = extract_hog_features(gray)

            X.append(features)
            y.append(row["label"])

        except Exception as e:
          print(f"\nError at index {idx}")
          print("movie_id:", row.get("movie_id"))
          print("image_path:", row.get("image_path"))
          print("error:", repr(e))
          raise

    X = np.array(X)
    y = np.array(y)

    return X, y


if __name__ == "__main__":
    dataset_path = PROCESSED_DATA_DIR / "movies_with_posters.csv"
    df = pd.read_csv(dataset_path)

    print("Building HOG dataset...")

    X, y = build_hog_dataset(df)

    print("\nHOG Feature Matrix Shape:", X.shape)
    print("Labels Shape:", y.shape)

    print("\nSample feature vector length:", X.shape[1])

    np.save(PROCESSED_DATA_DIR / "hog_features.npy", X)
    np.save(PROCESSED_DATA_DIR / "hog_labels.npy", y)

    print("\nSaved HOG features and labels.")