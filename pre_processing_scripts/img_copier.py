import shutil
from pathlib import Path

# --- CONFIGURATION ---
# Target directories based on your previous structure
source_root = Path("../Dataset/testingData")
destination_dir = Path("../modelData/test_data/images")

# Create the destination directory if it doesn't exist
destination_dir.mkdir(parents=True, exist_ok=True)

print("Starting renamed copy process...")

# Initialize a global counter for renaming
frame_count = 0

# Find all 'left_frames' directories within the trainingData folder
dataset_folders = sorted(list(source_root.rglob("instrument_dataset_*")))

for ds_folder in dataset_folders:
    left_frames_folder = ds_folder / "left_frames"
    
    if left_frames_folder.exists():
        print(f"Processing in order: {ds_folder.name}")
        
        # 2. Sort the individual frames within that folder
        for image_file in sorted(left_frames_folder.iterdir()):
            if image_file.is_file() and image_file.suffix.lower() in ['.png', '.jpg']:
                
                new_filename = f"frame{frame_count:04d}{image_file.suffix}"
                dest_path = destination_dir / new_filename
                
                shutil.copy2(image_file, dest_path)
                frame_count += 1

print(f"Success! {frame_count} images copied and renamed to {destination_dir}")