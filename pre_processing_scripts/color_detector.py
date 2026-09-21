import os
from pathlib import Path
import re
import cv2
import numpy as np

CROP_LEFT = 320
CROP_UPPER = 28

SOURCE_ROOT = Path("../Dataset/instrument_2017_test/instrument_2017_test")

def natural_sort_key(path):
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 0

def detect_colors_in_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    h, w, _ = img.shape
    right = w - 320
    lower = h - 28
    cropped = img[CROP_UPPER:lower, CROP_LEFT:right]

    # BGR color masks with relaxed thresholding
    color_ranges = {
        "Cyan": (np.array([150, 150, 0]), np.array([255, 255, 50])),
        "Yellow": (np.array([0, 150, 150]), np.array([50, 255, 255])),
        "Dark Blue": (np.array([50, 0, 0]), np.array([255, 50, 50])),
        "Red": (np.array([0, 0, 150]), np.array([50, 50, 255])),
        "Green": (np.array([0, 150, 0]), np.array([50, 255, 50])),
    }

    found_colors = []
    for color_name, (lower_bound, upper_bound) in color_ranges.items():
        mask = cv2.inRange(cropped, lower_bound, upper_bound)
        if cv2.countNonZero(mask) > 10:
            found_colors.append(color_name)

    return found_colors

dataset_folders = sorted(
    [f for f in SOURCE_ROOT.glob("instrument_dataset_*") if f.is_dir()],
    key=natural_sort_key
)

print("Scanning TypeSegmentationRescaled folders for colors...\n")

for ds_folder in dataset_folders:
    type_seg_path = ds_folder / "TypeSegmentationRescaled"
    if not type_seg_path.exists():
        continue

    image_paths = sorted(list(type_seg_path.glob("*.png")), key=natural_sort_key)
    if not image_paths:
        continue

    print(f"--- {ds_folder.name} ---")
    
    # Track unique colors found across this specific dataset folder
    dataset_colors = set()
    for img_path in image_paths:
        colors = detect_colors_in_image(img_path)
        dataset_colors.update(colors)

    if dataset_colors:
        print(f"  Found colors: {list(dataset_colors)}")
    else:
        print("  No target colors detected.")

print("\nScan complete!")