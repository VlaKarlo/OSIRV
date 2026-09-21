from pathlib import Path

labels_dir = Path("../cropped_data/training_data/labels")
images_dir = Path("../cropped_data/training_data/images")

deleted_count = 0

for label_path in labels_dir.glob("*.txt"):
    # Check if file is empty
    if label_path.stat().st_size == 0:
        print(f"Found empty label: {label_path.name}")
        
        # Delete empty label file
        label_path.unlink()
        
        # Optionally find and remove the matching image to prevent dataset mismatch
        img_path = images_dir / f"{label_path.stem}.png"
        if img_path.exists():
            img_path.unlink()
            print(f"Removed corresponding image: {img_path.name}")
            
        deleted_count += 1

print(f"\nCleanup complete. Removed {deleted_count} empty label files and their images.")