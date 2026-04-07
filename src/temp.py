from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
root = Path(os.getenv("DATASET_ROOT"))

print("Dataset root:", root)
print("\nTop-level contents:")
for item in root.iterdir():
    print("-", item.name)