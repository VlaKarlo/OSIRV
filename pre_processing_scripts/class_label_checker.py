from pathlib import Path

# Path to your training labels directory (updated to match pre_processing_scripts location context)
labels_dir = Path("../cropped_data/training_data/labels")

max_allowed_class = 17
flagged_count = 0

print(f"Scanning for label files containing negative class IDs or class IDs greater than {max_allowed_class}...")

for label_path in labels_dir.glob("*.txt"):
    if label_path.stat().st_size == 0:
        continue  # Skip empty files

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                if class_id < 0 or class_id > max_allowed_class:
                    print(f"Invalid class ID {class_id} found in: {label_path.name}")
                    flagged_count += 1
                    break  # Stop checking this file once an invalid class is found

print(f"\nScan complete. Total files with out-of-bounds classes: {flagged_count}")