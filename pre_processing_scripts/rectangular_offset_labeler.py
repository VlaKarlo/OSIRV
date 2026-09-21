import os
from pathlib import Path
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
    10: 0,  # Shaft
    20: 1,  # Wrist
    30: 2,  # Claspers
}

# Crop boundaries matching your working image cropper
CROP_LEFT = 320
CROP_UPPER = 28

SOURCE_ROOT = Path("./Dataset/trainingData")
OUTPUT_DIR = Path("cropped_data/training_data/labels")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

global_frame_count = 0

print("Starting robust label generation (ignoring Probe and Other)...")

dataset_folders = sorted(list(SOURCE_ROOT.rglob("instrument_dataset_*")))

for ds_folder in dataset_folders:
  if not ds_folder.is_dir():
    continue

  gt_path = ds_folder / "ground_truth"
  if not gt_path.exists():
    continue

  print(f"Processing Dataset: {ds_folder.name}")

  sample_tool_folder = next(gt_path.iterdir())
  frame_names = sorted([f.name for f in sample_tool_folder.glob("*.png")])

  for frame_name in frame_names:
    yolo_lines = []

    for tool_folder in gt_path.iterdir():
      if not tool_folder.is_dir():
        continue

      # --- IGNORE PROBE AND OTHER FOLDERS ---
      if "Probe" in tool_folder.name or "Other" in tool_folder.name:
        continue

      base_id = None
      for tool_key, b_id in TOOL_BASE_IDS.items():
        if tool_key in tool_folder.name:
          base_id = b_id
          break

      if base_id is None:
        continue

      mask_path = tool_folder / frame_name
      if not mask_path.exists():
        continue

      # Read original 1280x1024 mask
      mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
      if mask is None:
        continue

      # 1. APPLY EXACT SAME CROP AS YOUR IMAGES
      h, w = mask.shape
      right = w - 320
      lower = h - 28
      cropped_mask = mask[CROP_UPPER:lower, CROP_LEFT:right]

      ch, cw = cropped_mask.shape  # Should be exactly 968 x 640

      for gray_val, offset in PART_OFFSETS.items():
        final_class_id = base_id + offset
        part_mask = cv2.inRange(cropped_mask, gray_val, gray_val)
        contours, _ = cv2.findContours(
            part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
          if cv2.contourArea(c) < 20:
            continue

          x, y, bw, bh = cv2.boundingRect(c)

          x_center = (x + bw / 2) / cw
          y_center = (y + bh / 2) / ch
          norm_w = bw / cw
          norm_h = bh / ch

          yolo_lines.append(
              f"{final_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f}"
              f" {norm_h:.6f}\n"
          )

    new_filename = f"frame{global_frame_count:04d}.txt"
    with open(OUTPUT_DIR / new_filename, "w") as f:
      f.writelines(yolo_lines)

    global_frame_count += 1

print(
    f"\nSuccess! Created {global_frame_count} accurately cropped label files in"
    f" {OUTPUT_DIR}"
)