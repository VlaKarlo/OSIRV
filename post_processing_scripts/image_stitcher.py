import cv2
import os

# 1. Path to the folder where YOLO saved your PREDICTED images
image_folder = 'runs/segment/my_predictions/surgical_test_results' 
video_name = 'test_video.mp4'

# 2. Get a sorted list of images
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
images.sort() # Crucial to keep the surgery steps in order!

# 3. Read the first image to get the size
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 4. Define the codec and create VideoWriter object

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' works well for .mp4 files
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # avc1 best quality size ratio
video = cv2.VideoWriter(video_name, fourcc, 2, (width, height)) # 30 is the FPS

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

print(f"Video saved as {video_name}")