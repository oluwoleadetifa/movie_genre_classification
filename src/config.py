config_text = """
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

for folder in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

DATA_CSV = PROCESSED_DATA_DIR / "movies_with_posters.csv"

TRAIN_SPLIT_CSV = SPLITS_DIR / "train.csv"
VAL_SPLIT_CSV = SPLITS_DIR / "val.csv"
TEST_SPLIT_CSV = SPLITS_DIR / "test.csv"

TEXT_COLUMN = "description"
ID_COLUMN = "movie_id"
LABEL_COLUMN = "label"
GENRE_COLUMN = "genre"
IMAGE_PATH_COLUMN = "image_path"

CLASS_NAMES = ["action", "comedy", "horror", "romance"]
"""

with open("/content/movie_genre_classification/src/config.py", "w") as f:
    f.write(config_text)

print("config.py updated")
