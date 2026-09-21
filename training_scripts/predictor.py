from ultralytics import YOLO
import torch
import os

model = YOLO("runs/segment/Surgical_Instrument_Segmentation/l50_folded_1/weights/best.pt")

results = model.predict(
    source='cropped_data/test_data/images',
    save=True,
    project="my_predictions",
    name="surgical_test_results",
    show_boxes=True,       # Must be True for labels to show
    show_labels=True,      
    show_conf=True,
    stream=True,       
    line_width=1           # Set to 1 to make the boxes nearly invisible
)