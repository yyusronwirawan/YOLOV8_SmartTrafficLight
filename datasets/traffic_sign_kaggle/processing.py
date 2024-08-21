import os
import shutil
from random import shuffle
from shutil import copyfile


def split_data(train_ratio, test_ratio, directory, root_directory):
    """
    directory : your desired dataset folder
    root_directory : your current data folder
    """
    # Create subdirectories
    os.makedirs(os.path.join(directory, "datasets/images", "train"), exist_ok=True)
    os.makedirs(os.path.join(directory, "datasets/images", "valid"), exist_ok=True)
    os.makedirs(os.path.join(directory, "datasets/images", "test"), exist_ok=True)

    os.makedirs(os.path.join(directory, "datasets/labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(directory, "datasets/labels", "valid"), exist_ok=True)
    os.makedirs(os.path.join(directory, "datasets/labels", "test"), exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(root_directory) if f.endswith('.jpg')]

    # Shuffle the files
    shuffle(image_files)

    # Calculate split indices
    total_samples = len(image_files)
    train_split = int(train_ratio * total_samples)
    valid_split = int((1 - (train_ratio + test_ratio)) * total_samples)
    print(valid_split)
    # Split the files
    train_files = image_files[:train_split]
    valid_files = image_files[train_split:train_split + valid_split]
    test_files = image_files[train_split + valid_split:]

    # Move image files to respective directories
    for file in train_files:
        copyfile(os.path.join(root_directory, file), os.path.join(directory, "datasets/images", "train", file))
        copyfile(os.path.join(root_directory, file.replace('.jpg', '.txt')),
                 os.path.join(directory, "datasets/labels", "train", file.replace('.jpg', '.txt')))

    for file in valid_files:
        copyfile(os.path.join(root_directory, file), os.path.join(directory, "datasets/images", "valid", file))
        copyfile(os.path.join(root_directory, file.replace('.jpg', '.txt')),
                 os.path.join(directory, "datasets/labels", "valid", file.replace('.jpg', '.txt')))

    for file in test_files:
        copyfile(os.path.join(root_directory, file), os.path.join(directory, "datasets/images", "test", file))
        copyfile(os.path.join(root_directory, file.replace('.jpg', '.txt')),
                 os.path.join(directory, "datasets/labels", "test", file.replace('.jpg', '.txt')))



split_data(
    train_ratio=0.6, test_ratio=0.2,
    directory='/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign_kaggle/dataset',
    root_directory="/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign_kaggle/raw_dataset/ts/ts"
)