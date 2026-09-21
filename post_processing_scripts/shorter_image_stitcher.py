import cv2
import os

# 1. Configuration
image_folder = 'runs/segment/predict3/' 
video_name = 'l50_fold_1_segmentation_cleaner_slow.mp4'
fps = 2           # Your current speed (slow slideshow)
max_seconds = 60  # Your time limit

# 2. Get and Sort Images
images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))]
images.sort() 

# 3. Calculate Limit
max_frames = fps * max_seconds
# Slice the list to only include up to the limit
images_to_process = images[:max_frames]

# 4. Initialize Video Writer
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'avc1') 
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# 5. Write Frames
print(f"Creating video with {len(images_to_process)} frames ({max_seconds} seconds at {fps} FPS)...")

for image in images_to_process:
    img_path = os.path.join(image_folder, image)
    video.write(cv2.imread(img_path))

video.release()
cv2.destroyAllWindows()

print(f"Video saved as {video_name}")