import cv2
import numpy as np


def verify_labels(image_path, label_path):
  img = cv2.imread(image_path)
  if img is None:
    print(f"Error loading image: {image_path}")
    return
  h, w, _ = img.shape

  with open(label_path, "r") as f:
    lines = f.readlines()

  for line in lines:
    parts = list(map(float, line.strip().split()))
    if len(parts) < 5:
      continue

    class_id = int(parts[0])
    x_center, y_center, norm_w, norm_h = parts[1:5]

    # Denormalize back to pixel values relative to the cropped image size
    bw = norm_w * w
    bh = norm_h * h
    cx = x_center * w
    cy = y_center * h

    # Calculate top-left and bottom-right corners for OpenCV rectangle
    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    # Draw YOLO bounding box rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"ID: {class_id}",
        (x1, max(y1 - 10, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

  cv2.imshow("YOLO Verification", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


# Test it on frame 0
verify_labels(
    "croppedData/training_data/images/frame0224.png",
    "croppedData/training_data/labels/frame0224.txt",
)