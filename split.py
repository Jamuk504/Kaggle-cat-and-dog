import os
import shutil
import random

def split_files(source_dir, train_target_dir, valid_target_dir):
    files = os.listdir(source_dir)
    random.shuffle(files)
    split_point = int(len(files) * (0.8))
    train_files = files[:split_point]
    valid_files = files[split_point:]

    print(f"Moving {len(train_files)} training files from {source_dir}:")
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_target_dir, file))

    print(f"Moving {len(valid_files)} validation files from {source_dir}:")
    for file in valid_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(valid_target_dir, file))

os.makedirs(os.path.join('data/train', 'cat'), exist_ok=True)
os.makedirs(os.path.join('data/train', 'dog'), exist_ok=True)
os.makedirs(os.path.join('data/valid', 'cat'), exist_ok=True)
os.makedirs(os.path.join('data/valid', 'dog'), exist_ok=True)

print("Splitting Cat images:")
split_files('data/petImages/Cat', os.path.join('data/train', 'cat'), os.path.join('data/valid', 'cat'))

print("Splitting Dog images:")
split_files('data/petImages/Dog', os.path.join('data/train', 'dog'), os.path.join('data/valid', 'dog'))

print("Data split complete.")
