import os
from pathlib import Path
import cv2
import numpy as np

# --- CONFIGURATION ---
TARGET_FRAME = "frame0224"  # Change this to any frame name you want to view

CROP_LEFT = 320
CROP_UPPER = 28

IMAGE_DIR = Path("../cropped_data/training_data/images")
LABEL_DIR = Path("../cropped_data/training_data/labels")

print(f"Visualizing specific frame: {TARGET_FRAME}...")

# 1. Define paths for the specific image and its matching label file
img_path = IMAGE_DIR / f"{TARGET_FRAME}.png"
if not img_path.exists():
    img_path = IMAGE_DIR / f"{TARGET_FRAME}.jpg"

label_path = LABEL_DIR / f"{TARGET_FRAME}.txt"

if not img_path.exists():
    print(f"Error: Image for {TARGET_FRAME} not found in {IMAGE_DIR}")
    exit()

if not label_path.exists():
    print(f"Error: Label file for {TARGET_FRAME} not found in {LABEL_DIR}")
    exit()

# 2. Load the image
img = cv2.imread(str(img_path))
if img is None:
    print(f"Error: Could not read image {img_path}")
    exit()

h, w, _ = img.shape

# Generate random distinct colors for classes
np.random.seed(40)
COLORS = np.random.randint(0, 255, size=(30, 3), dtype=int).tolist()

# 3. Read polygon lines from the specific YOLO label file
with open(label_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    if not parts:
        continue
    
    class_id = int(parts.pop(0))
    coords = [float(p) for p in parts]
    
    # Reshape into an array of points (N, 1, 2)
    pts = []
    for i in range(0, len(coords), 2):
        px = int(coords[i] * w)
        py = int(coords[i+1] * h)
        pts.append([px, py])
        
    pts = np.array(pts, dtype=np.int32)
    if len(pts) > 0:
        pts = pts.reshape((-1, 1, 2))
        color = COLORS[class_id % len(COLORS)]
        
        # Draw filled polygon with transparency, and outline
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        
        # Put class ID text near the first point
        cv2.putText(img, f"ID: {class_id}", (pts[0][0][0], pts[0][0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# 4. Display the single target image until a key is pressed
cv2.imshow(f"Target Frame: {TARGET_FRAME} (Press any key to exit)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Visualization finished.")