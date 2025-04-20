#!/usr/bin/env python3

from ultralytics import YOLO

def main():
    # Load the official YOLOv10‑n model from Hugging Face
    model = YOLO("jameslahm/yolov10n")
    
    # Train on your custom data with proper augmentation parameters
    model.train(
        data="config.yaml",
        epochs=200,
        batch=32,
        project="yolo_train_run",
        name="augmented",
        exist_ok=True,
        augment=True,       # enables mosaic, mixup, HSV-jitter, perspective
        fliplr=0.5,         # 50% chance horizontal flip (replaces flip=True)
        flipud=0.0,         # 0% chance vertical flip
        degrees=10.0,       # ±10° random rotations (replaces rotate=10)
        translate=0.1,      # ±10% shifts
        scale=0.5           # random zoom in/out"""  """
    )

if __name__ == "__main__":
    main()
