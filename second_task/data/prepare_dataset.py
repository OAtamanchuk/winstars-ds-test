# This script renames the folders in the dataset from Italian to English. 
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "animals10", "raw-img")

TRANSLATE = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider"   
}

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory not found: {DATA_DIR}")
        print("Make sure you unzipped Animals-10 to second_task/data/animals10/")
        sys.exit(1)

    print(f"Processing folder: {DATA_DIR}\n")

    renamed = 0
    for old, new in TRANSLATE.items():
        old_path = os.path.join(DATA_DIR, old)
        new_path = os.path.join(DATA_DIR, new)

        if os.path.exists(old_path):
            if os.path.exists(new_path):
                print(f"Folder '{new}' already exists, skipping '{old}'")
            else:
                os.rename(old_path, new_path)
                print(f"Renamed: {old} to {new}")
                renamed += 1
        else:
            print(f"Folder '{old}' not found, skipping")

    print(f"\nDone! Renamed {renamed} folders.")
    print("You can now run eda.ipynb or start training.")

if __name__ == "__main__":
    main()