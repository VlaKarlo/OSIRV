import cv2
import numpy as np

# Load the image in grayscale mode
image_path = "Dataset/instrument_2017_test/instrument_2017_test/instrument_dataset_1/PartsSegmentation/frame225.png"  # Replace with your actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
  print("Error: Could not load image. Check the file path.")
else:
  # Find unique intensity values and their counts
  unique_values, counts = np.unique(image, return_counts=True)

  print("Unique Grayscale Intensities found in the image:")
  print("-" * 40)
  print(f"{'Intensity Value':<20} | {'Pixel Count':<20}")
  print("-" * 40)

  for val, count in zip(unique_values, counts):
    print(f"{val:<20} | {count:<20}")