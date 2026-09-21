import os
import re

# Use your actual folder structure
image_dir = os.path.abspath('modelData/training_data/images')
output_dir = os.path.abspath('modelData') # Where the .txt files will go

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], key=natural_sort_key)

frames_per_video = 225
num_videos = 8
videos = [all_images[i*frames_per_video:(i+1)*frames_per_video] for i in range(num_videos)]

fold_val_indices = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 4)]

for i, val_idx in enumerate(fold_val_indices):
    train_frames, val_frames = [], []
    for vid_id in range(num_videos):
        if vid_id in val_idx:
            val_frames.extend(videos[vid_id])
        else:
            train_frames.extend(videos[vid_id])
            
    # Save inside modelData/
    for name, frames in [('train', train_frames), ('val', val_frames)]:
        file_path = os.path.join(output_dir, f'fold_{i}_{name}.txt')
        with open(file_path, 'w') as f:
            for img in frames:
                f.write(os.path.join(image_dir, img) + '\n')

print(f"Txt files generated in {output_dir}")