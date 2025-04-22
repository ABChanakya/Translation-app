#!/usr/bin/env python3
"""
convert_yolo_to_pixels_single_output.py

Converts YOLO format (normalized) bounding boxes into pixel coordinates
and stores all results in a single output file.

Usage:
    python3 convert_yolo_to_pixels.py \
        --images path/to/images \
        --labels path/to/yolo_labels \
        --outfile path/to/output.txt
"""

import os
import cv2
import glob
import argparse

def convert_yolo_to_pixels(img_path, txt_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h, w = img.shape[:2]
    boxes = []
    with open(txt_path) as f:
        for line in f:
            cls, xc, yc, bw, bh = line.strip().split()
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
            cx = xc * w
            cy = yc * h
            box_w = bw * w
            box_h = bh * h
            x1 = int(cx - box_w / 2)
            y1 = int(cy - box_h / 2)
            x2 = int(cx + box_w / 2)
            y2 = int(cy + box_h / 2)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='Directory with .jpg images')
    parser.add_argument('--labels', required=True, help='Directory with YOLO .txt files')
    parser.add_argument('--outfile', required=True, help='Single output file to write all pixel boxes')
    args = parser.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.images, '*.jpg')))
    print(f"[INFO] Found {len(img_paths)} image(s)")

    with open(args.outfile, 'w') as out_f:
        for img_path in img_paths:
            base = os.path.basename(img_path)
            txt_path = os.path.join(args.labels, base.replace('.jpg', '.txt'))
            if not os.path.isfile(txt_path):
                print(f"[WARN] Skipping {base}: no label file found")
                continue

            try:
                pixel_boxes = convert_yolo_to_pixels(img_path, txt_path)
                for cls, x1, y1, x2, y2 in pixel_boxes:
                    out_f.write(f"{base} {cls} {x1} {y1} {x2} {y2}\n")
                print(f"[OK] Processed {base}")
            except Exception as e:
                print(f"[ERROR] Skipping {base}: {e}")
