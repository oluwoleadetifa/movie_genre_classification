from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def show_image(img_array: np.ndarray, title: str = "Processed Image") -> None:
    """
    Display a processed image array.
    """
    plt.imshow(img_array)
    plt.title(title)
    plt.axis("off")
    plt.show()

def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from disk and convert it to RGB.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    return img

def resize_image(img: Image.Image, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Resize image to a fixed size.
    """
    return img.resize(size)

def normalize_image(img: Image.Image) -> np.ndarray:
    """
    Convert PIL image to float32 NumPy array in range [0, 1].
    """
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array

def preprocess_image(image_path: Union[str, Path], size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Load image
    2. Convert to RGB
    3. Resize
    4. Normalize to [0, 1]

    Returns:
        np.ndarray of shape (H, W, 3)
    """
    img = load_image(image_path)
    img = resize_image(img, size=size)
    img_array = normalize_image(img)
    return img_array

def to_grayscale(img_array: np.ndarray) -> np.ndarray:
    """
    Convert RGB image array to grayscale using luminance weighting.
    Output shape: (H, W)
    """
    gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    return gray.astype(np.float32)

# /Users/oluwoleadetifa/Library/CloudStorage/GoogleDrive-adetifaoluwole@gmail.com/My Drive/Movie_Genre_Project/dataset_raw/IMDB four_genre_posters/Action/tt15734582.jpg
if __name__ == "__main__":
    from src.config import PROCESSED_DATA_DIR
    import pandas as pd
    dataset_path = PROCESSED_DATA_DIR / "movies_with_posters.csv"
    df = pd.read_csv(dataset_path)

    sample_row = df.sample(1, random_state=42).iloc[0]
    image_path = sample_row["image_path"]

    print("Movie ID:", sample_row["movie_id"])
    print("Genre:", sample_row["genre"])
    print("Image path:", image_path)

    image = preprocess_image(image_path)
    gray_image = to_grayscale(image)

    print("\nRGB image shape:", image.shape)
    print("RGB dtype:", image.dtype)
    print("RGB min:", image.min())
    print("RGB max:", image.max())

    print("\nGrayscale image shape:", gray_image.shape)
    print("Grayscale dtype:", gray_image.dtype)
    print("Grayscale min:", gray_image.min())
    print("Grayscale max:", gray_image.max())

    show_image(image, title=f"RGB: {sample_row['genre']} - {sample_row['movie_id']}")
    show_image(gray_image, title=f"Grayscale: {sample_row['genre']} - {sample_row['movie_id']}")