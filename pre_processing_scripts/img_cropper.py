import os
from PIL import Image

# Define source and destination root directories
source_root = "modelData"
dest_root = "croppedData"

print("Starting image cropping...")

for root, dirs, files in os.walk(source_root):
  for file in files:
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
      src_file_path = os.path.join(root, file)

      # Replicate the nested directory structure inside croppedData
      rel_path = os.path.relpath(root, source_root)
      dest_dir = os.path.join(dest_root, rel_path)
      os.makedirs(dest_dir, exist_ok=True)

      dest_file_path = os.path.join(dest_dir, file)

      try:
        with Image.open(src_file_path) as img:
          img.load()
          width, height = img.size

          # Correct crop box based on 1280x1024 resolution with 320 side margins and 28 top/bottom margins
          # Format: (left, upper, right, lower)
          left = 320
          upper = 28
          right = width - 320
          lower = height - 28

          cropped_img = img.crop((left, upper, right, lower))
          cropped_img.save(dest_file_path)
          print(f"Cropped & Saved: {dest_file_path}")
      except Exception as e:
        print(f"Error processing {src_file_path}: {e}")

print("\nAll images have been successfully cropped and saved to 'croppedData'!")