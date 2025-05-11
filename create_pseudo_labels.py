#!/usr/bin/env python3
"""
create_pseudo_labels.py

Generate YOLO-format pseudo-labels and optional visualizations for unlabeled images
using a pretrained Ultralytics YOLO model, with memory-aware chunking and inference options.

Features:
  • Chunked inference to prevent OOM (--chunk-size)
  • Configurable batch size (--batch-size)
  • NMS IoU (--iou), confidence threshold (--conf)
  • Inference image size (--imgsz) and max detections (--max-det)
  • FP16 half-precision (--half)
  • Test-time augmentation (--augment)
  • Device selection (--device)
  • Handles truncated/corrupt images
  • Streams inference results (--stream)
  • Saves YOLO-format .txt labels and optional visualization .jpg files

Example:
    python create_pseudo_labels.py \
      --input data/needs \
      --output pseudo_labels/run10/images \
      --model yolo_train_run/full_finetune_phase21/weights/best.pt \
      --conf 0.25 --iou 0.3 \
      --imgsz 640 --max-det 300 \
      --batch-size 4 --chunk-size 64 \
      --device 0 --names classes.txt \
      --vis-thickness 5 
"""
import os
import sys
import argparse
import logging
import glob
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont, ImageFile

# Optional: torch for device management and cache clearing
torch = None
try:
    import torch
except ImportError:
    pass

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate YOLO pseudo-labels (memory-aware)")
    parser.add_argument('-i', '--input',       required=True,  help='Input directory with images')
    parser.add_argument('-o', '--output',      required=True,  help='Output directory for labels & visuals')
    parser.add_argument('-m', '--model',       required=True,  help='Path to YOLO .pt model')
    parser.add_argument('-c', '--conf',        type=float, default=0.25, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou',                type=float, default=0.5,  help='NMS IoU threshold (0.0-1.0)')
    parser.add_argument('--imgsz',              type=int,   default=640,  help='Inference image size (square, px)')
    parser.add_argument('--max-det',            type=int,   default=1000, help='Max detections per image')
    parser.add_argument('-b', '--batch-size',   type=int,   default=4,    help='Batch size for YOLO (GPU micro-batch)')
    parser.add_argument('--chunk-size',         type=int,   default=64,   help='Images per inference chunk (0=all)')
    parser.add_argument('--device',             default='',       help="Device: ''=auto, 'cpu', '0', 'cuda:0', etc.")
    parser.add_argument('--half',               action='store_true', help='Use FP16 precision on CUDA')
    parser.add_argument('--augment',            action='store_true', help='Enable test-time augmentation')
    parser.add_argument('-n','--names',         default=None,     help='Path to class names .txt (one per line)')
    parser.add_argument('--vis-thickness',      type=int,   default=2,    help='Visualization box line thickness')
    parser.add_argument('--vis-scale',          type=float, default=1.0,  help='Scale factor for output visualizations (e.g. 0.5 for half size)')
    return parser.parse_args()


def find_images(root: Path):
    patterns = ['*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff']
    files = []
    for pat in patterns:
        files.extend(glob.glob(str(root / '**' / pat), recursive=True))
    return sorted(files)


def load_names(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path,'r') as f:
        return [line.strip() for line in f]


def save_yolo(txt_path, boxes):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w') as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")



def draw_boxes(img_path, boxes, save_path, names, thickness, scale=1.0):
    try:
        img = Image.open(img_path).convert('RGB')
    except OSError as e:
        logging.warning(f"Skipping visualization {img_path}: {e}")
        return
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', size=14)
    except IOError:
        font = ImageFont.load_default()
    for cls, x1, y1, x2, y2, conf in boxes:
        draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=thickness)
        label = f"{names[cls] if names else cls} {conf:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        w_box = bbox[2] - bbox[0]
        h_box = bbox[3] - bbox[1]
        y_text = y1 - h_box if y1 >= h_box else y1
        draw.rectangle([(x1, y_text), (x1 + w_box, y_text + h_box)], fill='red')
        draw.text((x1, y_text), label, fill='white', font=font)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img.save(save_path)


def normalize_device(dev: str) -> str:
    if dev == '':
        return ''
    return f"cuda:{dev}" if dev.isdigit() else dev




def main():
    setup_logging()
    args = parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_dir = output_dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)

    names = load_names(args.names)

    model = YOLO(args.model)
    device_str = normalize_device(args.device)
    if device_str:
        model.to(device_str)
    if args.half and torch and torch.cuda.is_available() and 'cuda' in str(model.device):
        model.half()
        logging.info('Using FP16 precision (FP16)')

    images = find_images(input_dir)
    if not images:
        logging.error(f"No images found in {input_dir}")
        sys.exit(1)
    logging.info(f"Total images: {len(images)}")

    chunk = args.chunk_size if args.chunk_size > 0 else len(images)
    for start in range(0, len(images), chunk):
        slice_imgs = images[start:start+chunk]
        logging.info(f"Processing images {start+1}-{start+len(slice_imgs)} (batch={args.batch_size}, chunk={chunk})")
        try:
            results = model.predict(
                source=slice_imgs,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                max_det=args.max_det,
                batch=args.batch_size,
                device=device_str,
                augment=args.augment,
                stream=True
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logging.error("CUDA OOM: try lowering --batch-size or --chunk-size or use --device cpu")
                sys.exit(1)
            raise

        for res in results:
            img_path = Path(res.path if hasattr(res, 'path') else res.orig_img_path)
            h, w = res.orig_shape
            norm_boxes, vis_boxes = [], []
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy.cpu().numpy().flatten()
                conf = float(b.conf.cpu().item())
                cls = int(b.cls.cpu().item())
                if conf < args.conf:
                    continue
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                norm_boxes.append((cls, xc, yc, bw, bh))
                vis_boxes.append((cls, x1, y1, x2, y2, conf))

            label_file = label_dir / f"{img_path.stem}.txt"
            save_yolo(str(label_file), norm_boxes)

            if vis_boxes:
                vis_file = output_dir / f"{img_path.stem}.jpg"
                draw_boxes(str(img_path), vis_boxes, str(vis_file), names, args.vis_thickness, scale=args.vis_scale)

        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info("Pseudo-label generation complete.")

if __name__ == '__main__':
    main()
