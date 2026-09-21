from pathlib import Path

labels_dir = Path("../cropped_data/training_data/labels")
flagged_count = 0

print("Scanning for segmentation polygons with fewer than 3 points...")

for label_path in labels_dir.glob("*.txt"):
    if label_path.stat().st_size == 0:
        continue
        
    with open(label_path, "r") as f:
        lines = f.readlines()
        
    for line_idx, line in enumerate(lines):
        parts = line.strip().split()
        if parts:
            # YOLO segmentation requires class_id + at least 3 coordinate pairs (1 + 6 = 7 values minimum)
            if len(parts) < 7:
                print(f"Invalid segmentation polygon (too few points) in: {label_path.name} at line {line_idx + 1}")
                flagged_count += 1

print(f"\nScan complete. Total invalid polygon files found: {flagged_count}")