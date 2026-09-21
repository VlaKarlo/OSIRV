from pathlib import Path

# --- CONFIGURATION ---
IMAGES_DIR = Path("cropped_data/training_data/images")
MODEL_DATA_DIR = Path("cropped_folds")
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_SIZE = 225  # Each video contains 225 frames
NUM_FOLDS = 4

print("Generating 4-fold cross-validation files (2 validation videos per fold)...")

# 1. Gather all existing frame images sorted by name
image_files = sorted(list(IMAGES_DIR.glob("frame*.png")) + list(IMAGES_DIR.glob("frame*.jpg")))

if not image_files:
    print(f"Error: No images found in {IMAGES_DIR}")
    exit(1)

total_images = len(image_files)
print(f"Total images found: {total_images}")

# Convert all image paths to resolved absolute string paths with newline formatting
all_image_paths = [f"{img.resolve()}\n" for img in image_files]

# 2. Create 4 folds where Fold i uses Video (i+1) and Video (i+5) for validation
for fold in range(NUM_FOLDS):
    # Determine the two video indices (0-indexed) for validation
    # Fold 0 -> Videos 0 and 4 (which are the 1st and 5th videos)
    # Fold 1 -> Videos 1 and 5 (2nd and 6th videos), etc.
    val_video_indices = [fold, fold + 4]
    
    val_files = []
    val_ranges_desc = []
    
    for v_idx in val_video_indices:
        start = v_idx * VIDEO_SIZE
        end = min((v_idx + 1) * VIDEO_SIZE, total_images)
        if start < total_images:
            val_files.extend(all_image_paths[start:end])
            val_ranges_desc.append(f"Video {v_idx + 1} [{start}:{end}]")
            
    # Training files are everything NOT in the validation set for this fold
    val_set_set = set(val_files)
    train_files = [img for img in all_image_paths if img not in val_set_set]
    
    # Define file names for each fold (0 to 3)
    train_txt_path = MODEL_DATA_DIR / f"fold{fold}_train.txt"
    val_txt_path = MODEL_DATA_DIR / f"fold{fold}_val.txt"
    
    # Write paths to text files
    with open(train_txt_path, "w") as f:
        f.writelines(train_files)
        
    with open(val_txt_path, "w") as f:
        f.writelines(val_files)
        
    print(f"Fold {fold}: Val -> {', '.join(val_ranges_desc)} ({len(val_files)} samples) | Train samples: {len(train_files)}")

print(f"\nSuccess! Created 4 fold text files inside {MODEL_DATA_DIR}/")