import os
from pathlib import Path
import re
import cv2
import numpy as np

# --- CONFIGURATION ---
TOOL_BASE_IDS = {
    "Bipolar_Forceps": 0,
    "Prograsp_Forceps": 3,
    "Large_Needle_Driver": 6,
    "Vessel_Sealer": 9,
    "Grasping_Retractor": 12,
    "Monopolar_Curved_Scissors": 15,
}

PART_OFFSETS = {
    30: 0,   # Shaft
    100: 1,  # Wrist
    255: 2,  # Claspers
}

COLOR_TO_TOOL = {
    "Cyan": "Prograsp_Forceps",
    "Yellow": "Bipolar_Forceps",
    "Dark Blue": "Large_Needle_Driver",
    "Red": "Vessel_Sealer",
    "Green": "Monopolar_Curved_Scissors",
}

CROP_LEFT = 320
CROP_UPPER = 28

SOURCE_ROOT = Path("./Dataset/instrument_2017_test/instrument_2017_test")
OUTPUT_DIR = Path("cropped_data/test_data/labels")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

global_frame_count = 0

print("Starting label conversion with fixed flat-file processing...")

def natural_sort_key(path):
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else 0

def identify_tools_from_mask(type_mask_crop):
    matched_tools = []
    cyan_mask = cv2.inRange(type_mask_crop, np.array([240, 240, 0]), np.array([255, 255, 20]))
    yellow_mask = cv2.inRange(type_mask_crop, np.array([0, 240, 240]), np.array([20, 255, 255]))
    dark_blue_mask = cv2.inRange(type_mask_crop, np.array([100, 0, 0]), np.array([255, 30, 30]))
    red_mask = cv2.inRange(type_mask_crop, np.array([0, 0, 240]), np.array([20, 20, 255]))
    green_mask = cv2.inRange(type_mask_crop, np.array([0, 240, 0]), np.array([20, 255, 20]))
    
    if cv2.countNonZero(cyan_mask) > 10: matched_tools.append(COLOR_TO_TOOL["Cyan"])
    if cv2.countNonZero(yellow_mask) > 10: matched_tools.append(COLOR_TO_TOOL["Yellow"])
    if cv2.countNonZero(dark_blue_mask) > 10: matched_tools.append(COLOR_TO_TOOL["Dark Blue"])
    if cv2.countNonZero(red_mask) > 10: matched_tools.append(COLOR_TO_TOOL["Red"])
    if cv2.countNonZero(green_mask) > 10: matched_tools.append(COLOR_TO_TOOL["Green"])
    return matched_tools

dataset_folders = sorted([f for f in SOURCE_ROOT.glob("instrument_dataset_*") if f.is_dir()], key=natural_sort_key)

for ds_folder in dataset_folders:
    parts_seg_path = ds_folder / "PartsSegmentation"
    type_seg_path = ds_folder / "TypeSegmentationRescaled"
    
    if not parts_seg_path.exists() or not type_seg_path.exists():
        continue

    print(f"Processing Dataset: {ds_folder.name}")

    image_paths = sorted(list(parts_seg_path.glob("*.png")), key=natural_sort_key)
    if not image_paths:
        continue

    for img_path in image_paths:
        frame_name = img_path.name
        yolo_lines = []

        # 1. Read TypeSegmentationRescaled frame to see which tool colors are active
        type_mask_path = type_seg_path / frame_name
        active_tools = []
        if type_mask_path.exists():
            type_img = cv2.imread(str(type_mask_path))
            if type_img is not None:
                h, w, _ = type_img.shape
                cropped_type_img = type_img[CROP_UPPER:h-28, CROP_LEFT:w-320]
                active_tools = identify_tools_from_mask(cropped_type_img)

        # 2. Read PartsSegmentation image
        mask = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape
        cropped_mask = mask[CROP_UPPER:h-28, CROP_LEFT:w-320]
        ch, cw = cropped_mask.shape

        # 3. For every active tool found in the type mask, extract component parts
        for tool_name in active_tools:
            base_id = TOOL_BASE_IDS.get(tool_name)
            if base_id is None:
                continue

            for gray_val, offset in PART_OFFSETS.items():
                final_class_id = base_id + offset
                part_mask = cv2.inRange(cropped_mask, gray_val, gray_val)
                contours, _ = cv2.findContours(part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    if cv2.contourArea(c) < 20:
                        continue
                    
                    polygon = [f"{p[0][0]/cw:.6f} {p[0][1]/ch:.6f}" for p in c]
                    if polygon:
                        yolo_lines.append(f"{final_class_id} {' '.join(polygon)}\n")

        new_filename = f"frame{global_frame_count:04d}.txt"
        with open(OUTPUT_DIR / new_filename, "w") as f:
            f.writelines(yolo_lines)

        global_frame_count += 1

print(f"\nSuccess! Created {global_frame_count} populated segmentation label files in {OUTPUT_DIR}")