from models import get_loaders
import shutil
from pathlib import Path

if __name__ == "__main__":
    if Path("models/data").exists():
        print("Data already loaded.")
        exit(0)
    print("Downloading Fashion MNIST dataset...")
    get_loaders()

    # move data to models for usage
    shutil.move("data", "models/data")

    print("Download finished. Dataset can be found in models/data")