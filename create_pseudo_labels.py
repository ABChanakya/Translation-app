import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def detect_bubbles(img: np.ndarray, model: YOLO, conf: float = 0.25) -> list[tuple[int,int,int,int]]:
    """
    Run YOLOv8 model to detect speech bubbles and return bounding boxes.
    """
    results = model.predict(img, conf=conf, verbose=False, save=False, save_txt=False, show=False)
    if not results or not results[0].boxes:
        return []
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    h, w = img.shape[:2]
    clamped = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            clamped.append((x1, y1, x2, y2))
    return clamped


def write_yolo_labels(boxes: list[tuple[int,int,int,int]], img_shape: tuple[int,int], label_path: Path, class_id: int = 0):
    """
    Write YOLO-format labels: class_id x_center y_center width height (normalized)
    """
    h, w = img_shape
    lines = []
    for x1, y1, x2, y2 in boxes:
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines))


def gather_images(input_dir: Path) -> list[Path]:
    """Return sorted list of .jpg, .jpeg, .png in input_dir."""
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels and prepare YOLOv8 dataset with custom config.yaml"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Folder with original images (JPG/PNG)"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output folder root for images, labels, and config.yaml"
    )
    parser.add_argument(
        "-m", "--model",
        default="yolo_train_run/augmented/weights/best.pt",
        help="Path to YOLOv8 checkpoint (.pt)"
    )
    parser.add_argument(
        "-c", "--conf", type=float, default=0.25,
        help="Detection confidence threshold"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    for d in (output_dir, images_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{args.model}'...")
    model = YOLO(args.model)

    image_paths = gather_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for img_path in tqdm(image_paths, desc="Processing images"):
        # Copy image
        dst_img = images_dir / img_path.name
        shutil.copy(img_path, dst_img)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}")
            continue

        # Detect bubbles and write labels
        boxes = detect_bubbles(img, model, conf=args.conf)
        label_file = labels_dir / f"{img_path.stem}.txt"
        write_yolo_labels(boxes, img.shape[:2], label_file)

    # Write custom config.yaml as specified
    config_path = output_dir / 'config.yaml'
    config_content = '''nc: 3

train: "/home/chanakya/chanakya/UNI/translation tool/data/images/train"
val:   "/home/chanakya/chanakya/UNI/translation tool/data/images/val"

names:
  0: Dialogue
  1: Sound Effects
  2: Signs/labels
'''
    with open(config_path, 'w') as f:
        f.write(config_content)

    print("Done. Dataset prepared:")
    print(f"  Images -> {images_dir}")
    print(f"  Labels -> {labels_dir}")
    print(f"  Config -> {config_path}")

if __name__ == "__main__":
    main()
