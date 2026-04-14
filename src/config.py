from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", DATA_DIR / "raw"))
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

for path in [PROCESSED_DATA_DIR, SPLITS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DATA_CSV = RAW_DATA_DIR / "IMDB_four_genre_larger_plot_description.csv"

TEXT_COLUMN = "description"
ID_COLUMN = "movie_id"
LABEL_COLUMN = "genre"

CLASS_NAMES = ["action", "comedy", "horror", "romance"]
