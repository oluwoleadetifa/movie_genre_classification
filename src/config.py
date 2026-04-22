from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
<<<<<<< HEAD

=======
load_dotenv(PROJECT_ROOT / ".env")

# Core directories
>>>>>>> 7c73183 (HOG validation)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
<<<<<<< HEAD
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

DATA_CSV = RAW_DATA_DIR / "IMDB_four_genre_larger_plot_description.csv"

TEXT_COLUMN = "description"
ID_COLUMN = "movie_id"
LABEL_COLUMN = "genre"

CLASS_NAMES = ["action", "comedy", "horror", "romance"]
=======
SPLITS_DIR = DATA_DIR / "splits"   # 🔥 THIS was missing
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# External dataset path (Google Drive or local)
drive_data_dir = os.getenv("DRIVE_DATA_DIR")

if not drive_data_dir:
    raise ValueError("DRIVE_DATA_DIR is not set. Please define it in your .env file.")

DATASET_ROOT = Path(drive_data_dir).expanduser().resolve()
>>>>>>> 7c73183 (HOG validation)
