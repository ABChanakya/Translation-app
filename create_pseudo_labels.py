#!/usr/bin/env python3
"""
create_pseudo_labels.py

Generate pseudo-labels (YOLO-format bounding boxes) and visualizations for unlabeled images using a pretrained Ultralytics YOLO model.

Features:
1. Configurable via CLI arguments
2. Uses Ultralytics YOLOv8 API for inference
3. Saves annotations in YOLO txt format: <class> <x_center> <y_center> <width> <height>
4. Optionally draws bounding boxes with class labels/confidences on images
5. Supports recursive image discovery and custom class names
6. Organizes label files into a "labels" subfolder and visualizations into output directory
7. Robust error handling and logging

Usage:
    python create_pseudo_labels.py \
        --input data/needs \
        --output pseudo_labels/run2 \
        --model yolo_train_run/augmented/weights/best.pt \
        --conf 0.25 \
        --batch-size 16 \
        --names classes.txt \
        --vis-thickness 2

"""
import os
import sys
import argparse
import logging
from pathlib import Path
import glob
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# --------------------------- Logging ---------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ------------------------- Argument Parsing -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate YOLO-format pseudo-labels and optional visualizations"
    )
    parser.add_argument('--input', '-i', required=True,
                        help="Directory with unlabeled images")
    parser.add_argument('--output', '-o', required=True,
                        help="Directory to save pseudo-label text files and visualizations")
    parser.add_argument('--model', '-m', required=True,
                        help="Path to YOLO .pt model file")
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help="Batch size for YOLO inference")
    parser.add_argument('--names', '-n', default=None,
                        help="Path to class names text file (one per line)")
    parser.add_argument('--vis-thickness', type=int, default=2,
                        help="Bounding box line thickness for visualizations")
    return parser.parse_args()

# --------------------- Utility Functions ---------------------
def find_images(input_dir):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    return sorted(paths)


def load_names(names_path):
    if not names_path or not os.path.isfile(names_path):
        return None
    with open(names_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def save_annotations(txt_path, boxes):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w') as f:
        for cls, x_center, y_center, width, height in boxes:
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def draw_boxes(image_path, boxes, vis_path, names=None, thickness=2):
    """Draw bounding boxes and labels on image and save to vis_path."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for cls, x1, y1, x2, y2, conf in boxes:
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=thickness)
        label = f"{names[cls] if names else cls} {conf:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x1
        text_y = y1 - text_h if y1 - text_h > 0 else y1
        draw.rectangle(
            [(text_x, text_y), (text_x + text_w, text_y + text_h)],
            fill="red"
        )
        draw.text((text_x, text_y), label, fill="white", font=font)

    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    img.save(vis_path)

# ----------------------------- Main -----------------------------
def main():
    setup_logging()
    args = parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create labels subfolder
    label_dir = out_dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.is_file():
        logging.error(f"Model file not found: {model_path}")
        sys.exit(1)

    names = load_names(args.names)
    if args.names and not names:
        logging.warning(f"Names file not found or empty: {args.names}")

    try:
        model = YOLO(str(model_path))
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    image_paths = find_images(str(input_dir))
    if not image_paths:
        logging.error(f"No images found in {input_dir}")
        sys.exit(1)
    logging.info(f"Found {len(image_paths)} images for inference")

    results = model.predict(
        source=image_paths,
        conf=args.conf,
        batch=args.batch_size,
        device=''
    )

    for res in results:
        img_path = Path(res.path if hasattr(res, 'path') else res.orig_img_path)
        h_img, w_img = res.orig_shape
        boxes_norm, boxes_vis = [], []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
            conf = float(box.conf.cpu().item())
            cls = int(box.cls.cpu().item())
            if conf < args.conf:
                continue
            x_center = ((x1 + x2) / 2) / w_img
            y_center = ((y1 + y2) / 2) / h_img
            width = (x2 - x1) / w_img
            height = (y2 - y1) / h_img
            boxes_norm.append((cls, x_center, y_center, width, height))
            boxes_vis.append((cls, x1, y1, x2, y2, conf))

        # Save annotation txt in labels subfolder
        txt_path = label_dir / f"{img_path.stem}.txt"
        save_annotations(str(txt_path), boxes_norm)

        # Save visualization image
        vis_path = out_dir / f"{img_path.stem}.jpg"
        if boxes_vis:
            draw_boxes(str(img_path), boxes_vis, str(vis_path), names, args.vis_thickness)

    logging.info("Pseudo-label generation complete.")

if __name__ == '__main__':
    main()
