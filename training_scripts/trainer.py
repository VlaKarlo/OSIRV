from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_model():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    yaml_file = 'final.yaml'
    print(f"\n--- STARTING TRAINING WITH {yaml_file} ---")

    cache_path = 'cropped_data/training_data/labels.cache' 
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Removed old cache: {cache_path}")

    model = YOLO('yolov8l-seg.pt')

    # Dynamically find the absolute path of the main project root 
    # (assuming model_scripts is one level deep from the root)
    project_root = Path(__file__).resolve().parent.parent
    main_runs_dir = project_root / 'runs' / 'segment'

    results = model.train(
        data=yaml_file,
        epochs=50,
        imgsz=[640, 968],
        batch=2,
        device=device,
        amp=False,
        
        # --- ABSOLUTE PATH FIX ---
        project=str(main_runs_dir),
        name='final_model_50',
        
        # --- ANTI-OVERFITTING SUITE ---
        dropout=0.15,
        weight_decay=0.001,
        label_smoothing=0.1,
        
        # --- AGGRESSIVE AUGMENTATION ---
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        degrees=20.0,
        fliplr=0.5,
        flipud=0.2,
        hsv_s=0.7,
        hsv_v=0.4,
        
        workers=8 
    )

if __name__ == '__main__':
    train_model()