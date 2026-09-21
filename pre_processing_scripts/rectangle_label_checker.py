import cv2
import numpy as np
import os


def verify_labels(image_path, label_path):
  img = cv2.imread(image_path)
  if img is None:
    print(f"Error: Could not load image at {image_path}")
    return

  h, w, _ = img.shape

  if not os.path.exists(label_path):
    print(f"Error: Label file not found at {label_path}")
    return

  with open(label_path, "r") as f:
    lines = f.readlines()

  for line in lines:
    parts = list(map(float, line.strip().split()))
    if len(parts) < 5:
      continue

    class_id = int(parts[0])
    x_center, y_center, box_w, box_h = parts[1:5]

    # Denormalize and convert from YOLO center format to corner coordinates
    bw = box_w * w
    bh = box_h * h
    xc = x_center * w
    yc = y_center * h

    x_min = int(xc - bw / 2)
    y_min = int(yc - bh / 2)
    x_max = int(xc + bw / 2)
    y_max = int(yc + bh / 2)

    # Draw the bounding box
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Put class ID text near the top-left of the box
    cv2.putText(
        img,
        f"ID: {class_id}",
        (x_min, max(y_min - 5, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

  cv2.imshow("Verification", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


# Test it on frame 1100 (or whichever frame you want to inspect)
verify_labels(
    "cropped_data/training_data/images/frame0224.png",
    "cropped_rectangular_bounding_boxes/frame0224.txt",
)