import os
import numpy as np
import warnings
from PIL import Image
from config import DATA_FOLDERS, VALID_EXTENSIONS, SYSTEM_FILES, BLANK_THRESHOLD

def clean_corrupt_images(folders):
    total_removed = 0
    print("Starting data cleaning for corrupt and non-image files:")
    
    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        
        if not os.path.isdir(folder):
            print(f"WARNING: Folder not found: {folder}. Skipping")
            continue
        
        files_to_check = os.listdir(folder)

        for filename in files_to_check:
            file_path = os.path.join(folder, filename)

            if filename in SYSTEM_FILES:
                try:
                    os.remove(file_path)
                    total_removed += 1
                    print(f"Removed system file: {filename}")
                except OSError as e:
                    print(f"Error removing file {filename}: {e}")
                continue

            if os.path.splitext(filename)[1].lower() not in VALID_EXTENSIONS:
                try:
                    os.remove(file_path)
                    total_removed += 1
                    print(f"Removed non-image file: {filename}")
                except OSError as e:
                    print(f"Error removing file {filename}: {e}")
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=UserWarning)
                try:
                    img = Image.open(file_path).convert('RGB')
                    img.load()
                    img_array = np.array(img)
                    pixel_std = np.std(img_array)
                    if pixel_std < BLANK_THRESHOLD:
                        os.remove(file_path)
                        total_removed += 1
                        print(f"Removed BLANK/UNIFORM file: {filename}")
                        continue
                except Exception as e:
                    try:
                        os.remove(file_path)
                        total_removed += 1
                        print(f"Removed CORRUPT file: {file_path} (Error: {e})")
                    except OSError as err:
                        print(f"Error removing corrupt file {filename}: {err}")

    print(f"\nCleaning Complete")
    print(f"Total files removed across all folders: {total_removed}")

if __name__ == '__main__':
    clean_corrupt_images(DATA_FOLDERS)