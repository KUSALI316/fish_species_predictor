import os
import shutil
import random

# Paths
raw_data_dir = "Balanced_Dataset"  # Original dataset location
dataset_dir = "Dataset_for_train"  # New structured dataset location

train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

# Create train, val, test directories
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# Get fish type folders from the raw dataset
fish_types = [f for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))]

# Split Ratios
train_ratio = 0.7  # 70% training
val_ratio = 0.15   # 15% validation
test_ratio = 0.15  # 15% testing

# Process each fish type
for fish_type in fish_types:
    fish_folder = os.path.join(raw_data_dir, fish_type)
    images = [f for f in os.listdir(fish_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not images:
        print(f"⚠ No images found in {fish_folder}. Skipping...")
        continue

    random.shuffle(images)

    total_images = len(images)
    train_idx = int(total_images * train_ratio)
    val_idx = int(total_images * (train_ratio + val_ratio))

    train_images = images[:train_idx]
    val_images = images[train_idx:val_idx]
    test_images = images[val_idx:]

    # Create class folders in train, val, test directories
    for folder, img_list in zip([train_dir, val_dir, test_dir], [train_images, val_images, test_images]):
        class_folder = os.path.join(folder, fish_type)
        os.makedirs(class_folder, exist_ok=True)

        for img in img_list:
            src_path = os.path.join(fish_folder, img)
            dest_path = os.path.join(class_folder, img)
            shutil.copy(src_path, dest_path)  # Use copy for testing

print("Dataset successfully organized into train, validation, and test sets!")
