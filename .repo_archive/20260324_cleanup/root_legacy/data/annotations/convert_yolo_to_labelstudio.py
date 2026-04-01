# convert_yolo_to_labelstudio.py
"""
Convert a directory of images and YOLO-format .txt labels into a Label Studio import JSON.

Usage:
    python convert_yolo_to_labelstudio.py \
      --image-dir data/needs \
      --label-dir pseudo_labels/run2/labels \
      --classes classes.txt \
      --output import_ls.json

Requirements:
    pip install pillow pyyaml

This script outputs a JSON array of tasks:
[
  {
    "id": 1,
    "data": {"image": "file:///absolute/path/to/image.jpg"},
    "annotations": [
      {"result": [ {"value": {"x":10, "y":20, "width":30, "height":40, "rotation":0}, "id": "...", "from_name": "label", "to_name": "image", "type": "rectanglelabels", "origin": "manual", "category": "Dialogue" } ] }
    ]
  },
  ...
]

You can then import this JSON into Label Studio, correct labels, and export back to YOLO.
"""
import os
import json
import argparse
from pathlib import Path
from PIL import Image


def load_classes(classes_path):
    cls = []
    with open(classes_path, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                cls.append(name)
    return cls


def load_yolo_labels(txt_path):
    labels = []
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                c, xc, yc, w, h = parts
                labels.append((int(c), float(xc), float(yc), float(w), float(h)))
    return labels


def yolo_to_labelstudio(image_path, label_path, classes):
    img = Image.open(image_path)
    w, h = img.size
    tasks = []
    # build a single task dict
    data = {"image": f"file://{os.path.abspath(image_path)}"}
    results = []
    for (c, xc, yc, bw, bh) in load_yolo_labels(label_path):
        x = (xc - bw/2) * 100
        y = (yc - bh/2) * 100
        width = bw * 100
        height = bh * 100
        # Label Studio uses percentages
        result = {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {"x": x, "y": y, "width": width, "height": height, "rotation": 0},
            "origin": "prediction",
            "id": f"box_{c}_{xc:.2f}_{yc:.2f}",
            "category": classes[c]
        }
        results.append(result)
    annotation = {"result": results}
    return {"id": None, "data": data, "annotations": [annotation]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--label-dir', required=True)
    parser.add_argument('--classes', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    classes = load_classes(args.classes)
    tasks = []
    img_files = sorted(Path(args.image_dir).glob('*'))
    task_id = 1
    for img_path in img_files:
        if not img_path.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.tif','.tiff']:
            continue
        txt_path = Path(args.label_dir) / f"{img_path.stem}.txt"
        task = yolo_to_labelstudio(str(img_path), str(txt_path), classes)
        task['id'] = task_id
        tasks.append(task)
        task_id += 1

    with open(args.output, 'w') as f:
        json.dump(tasks, f, indent=2)
    print(f"Exported {len(tasks)} tasks to {args.output}")

if __name__ == '__main__':
    main()
