import os

# --- CONFIGURATION ---
BASE_PATH = r'Dataset/instrument_2017_test/instrument_2017_test/instrument_dataset_1/BinarySegmentation'

TARGET_FOLDERS = [
    'left_frames',
    'ground_truth/Bipolar_Forceps_labels',
    'ground_truth/Left_Grasping_Retractor_labels',
    'ground_truth/Right_Grasping_Retractor_labels',
    'ground_truth/Monopolar_Curved_Scissors_labels',
]

OFFSET = -225

def rename_files_with_offset(base_path, target_subfolders, offset):
    for subfolder in target_subfolders:
        folder_path = os.path.join(base_path, subfolder)
        
        if not os.path.exists(folder_path):
            print(f"Skipping: {subfolder} (Path not found)")
            continue

        print(f"Processing: {subfolder}...")
        
        # Sort files to maintain order
        files = sorted(os.listdir(folder_path))
        
        for filename in files:
            name, ext = os.path.splitext(filename)
            
            # Remove "frame" prefix to get the number
            # This handles "frame000" -> "000"
            clean_name = name.replace('frame', '')
            
            try:
                current_num = int(clean_name)
                new_num = current_num + offset
                
                # Format with leading zeros to keep things neat (frame000 -> frame225)
                # :03d ensures it stays at least 3 digits long
                new_filename = f"frame{new_num:03d}{ext}"
                
                old_file = os.path.join(folder_path, filename)
                new_file = os.path.join(folder_path, new_filename)
                
                os.rename(old_file, new_file)
            except ValueError:
                print(f"  Still skipping {filename}: Could not find a number in the name.")

    print("\nRenaming complete!")

if __name__ == "__main__":
    rename_files_with_offset(BASE_PATH, TARGET_FOLDERS, OFFSET)