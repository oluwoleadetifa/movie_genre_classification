from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Optional external shared drive path
DRIVE_DATA_DIR = Path(os.getenv("DRIVE_DATA_DIR", RAW_DATA_DIR))