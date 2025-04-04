import os
import shutil
import random
from PIL import Image

# Paths
raw_data_dir = "fish_dataset"  # Original dataset location
balanced_data_dir = "Balanced_Dataset"  # Directory for balanced dataset

# Create the balanced dataset directory
os.makedirs(balanced_data_dir, exist_ok=True)

# Get fish type folders from the raw dataset
fish_types = [f for f in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, f))]

# Define the target number of images per category
TARGET_IMAGES = 186

# Process each fish type
for fish_type in fish_types:
    fish_folder = os.path.join(raw_data_dir, fish_type)
    balanced_folder = os.path.join(balanced_data_dir, fish_type)
    os.makedirs(balanced_folder, exist_ok=True)

    images = [f for f in os.listdir(fish_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    num_images = len(images)

    if num_images == 0:
        print(f"⚠ No images found in {fish_folder}. Skipping...")
        continue

    # Copy existing images first
    for img in images:
        src_path = os.path.join(fish_folder, img)
        dest_path = os.path.join(balanced_folder, img)
        shutil.copy(src_path, dest_path)
    
    # If images are less than TARGET_IMAGES, oversample
    if num_images < TARGET_IMAGES:
        print(f"Balancing {fish_type}: {num_images} -> {TARGET_IMAGES}")
        while len(os.listdir(balanced_folder)) < TARGET_IMAGES:
            img_to_duplicate = random.choice(images)
            src_path = os.path.join(fish_folder, img_to_duplicate)
            new_img_name = f"copy_{random.randint(1000, 9999)}_{img_to_duplicate}"
            dest_path = os.path.join(balanced_folder, new_img_name)
            shutil.copy(src_path, dest_path)

print("Dataset successfully balanced! Each category now has 186 images.")
