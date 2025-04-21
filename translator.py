#!/usr/bin/env python3
"""
translator.py – detect speech bubbles, OCR them, translate, and overlay the
translation back onto the page.

Dependencies::

    pip install ultralytics manga-ocr googletrans==4.0.0-rc1 pillow opencv-python tqdm

Usage examples
--------------
Translate every image under ``data/needs`` and write the results next to them::

    python translator.py \
        --input data/needs \
        --model yolo_train_run/full_finetune/weights/best.pt\
        --lang en \
        --output translated_pages

Translate a *single* file::

    python translator.py -i page01.png -o out.png

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
Crop = Tuple[np.ndarray, BBox]

################################################################################
# 1. Bubble detection                                                          #
################################################################################

def detect_bubbles(img: np.ndarray, model: YOLO, conf: float = 0.25) -> List[BBox]:
    results = model.predict(img, conf=conf, verbose=False)
    if not results:
        return []
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
    h, w = img.shape[:2]
    return [[max(0, x1), max(0, y1), min(w, x2), min(h, y2)] for x1, y1, x2, y2 in boxes]

################################################################################
# 2. Crop extraction                                                           #
################################################################################

def extract_crops(img: np.ndarray, boxes: List[BBox]) -> List[Crop]:
    return [(img[y1:y2, x1:x2], (x1, y1, x2, y2)) for x1, y1, x2, y2 in boxes]

################################################################################
# 3. OCR                                                                       #
################################################################################

def ocr_bubbles(crops: List[Crop], ocr_engine: MangaOcr) -> List[str]:
    texts: List[str] = []
    for crop_arr, _ in crops:
        crop_pil = Image.fromarray(cv2.cvtColor(crop_arr, cv2.COLOR_BGR2RGB))
        ann = ocr_engine(crop_pil)
        if isinstance(ann, str):
            text = ann
        else:
            try:
                if all(isinstance(el, str) for el in ann):
                    text = "".join(ann)
                elif all(isinstance(el, (tuple, list)) and len(el) >= 1 for el in ann):
                    text = "".join(el[0] for el in ann)
                else:
                    text = " ".join(str(el) for el in ann)
            except Exception:
                text = str(ann)
        texts.append(text)
    return texts

################################################################################
# 4. Translation                                                               #
################################################################################
def translate_texts(texts: List[str], target_lang: str = "en") -> List[str]:
    """
    Translate a list of strings into `target_lang` using deep-translator.
    Falls back to the original text if an exception occurs.
    """
    translations: List[str] = []
    for t in texts:
        try:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(t)
            translations.append(translated)
        except Exception as e:
            logging.warning(f"Translation failed for: '{t}' – {e}")
            translations.append(t)
    return translations

################################################################################
# 5. Overlay                                                                   #
################################################################################

def overlay_translations(page_img: np.ndarray, crops: List[Crop], translations: List[str]) -> np.ndarray:
    # ——— sanitize translations so we never have None ———
    translations = [t if isinstance(t, str) else "" for t in translations]

    pil_base = Image.fromarray(cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    txt_layer = Image.new("RGBA", pil_base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    font = ImageFont.load_default()

    def get_text_size(text: str, multiline: bool = False, spacing: int = 0) -> tuple[int, int]:
        try:
            if multiline:
                bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
            else:
                bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # fallback for older Pillow
            if multiline:
                return draw.multiline_textsize(text, font=font, spacing=spacing)
            else:
                return draw.textsize(text, font=font)

    for (_, (x1, y1, x2, y2)), text in zip(crops, translations):
        # guard against None or blank strings
        t = (text or "").strip()
        if not t:
            continue

        # word‑wrap logic unchanged
        max_w = max(10, x2 - x1 - 10)
        words = t.split()
        lines: List[str] = []
        curr = ""
        for w in words:
            test = (curr + " " + w).strip()
            if get_text_size(test)[0] <= max_w:
                curr = test
            else:
                if curr:
                    lines.append(curr)
                curr = w
        lines.append(curr)
        final_text = "\n".join(lines)

        # center the box
        w, h = get_text_size(final_text, multiline=True, spacing=2)
        tx = x1 + max(0, ((x2 - x1) - w) // 2)
        ty = y1 + max(0, ((y2 - y1) - h) // 2)

        draw.rectangle([tx - 2, ty - 2, tx + w + 2, ty + h + 2], fill=(255, 255, 255, 200))
        draw.multiline_text((tx, ty), final_text, fill=(0, 0, 255, 255),
                            font=font, spacing=2, align="center")

    combined = Image.alpha_composite(pil_base, txt_layer).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


################################################################################
# Utility                                                                      #
################################################################################

def gather_images(path: Path) -> List[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg")
    files: List[Path] = []
    for ext in exts:
        files.extend(path.glob(ext))
    return sorted(files)

################################################################################
# Main                                                                         #
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Detect, OCR & translate speech bubbles on manga/comic pages")
    parser.add_argument("-i", "--input", required=True, help="Image file or directory with images")
    parser.add_argument("-m", "--model", default="yolov8n.pt", help="YOLOv8 checkpoint (.pt)")
    parser.add_argument("-l", "--lang", default="en", help="Target language (ISO‑639‑1 code)")
    parser.add_argument("-c", "--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("-o", "--output", required=True, help="Output file or directory")
    args = parser.parse_args()

    inp = Path(args.input).expanduser()
    out = Path(args.output).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if inp.is_dir():
        files = gather_images(inp)
        if not files:
            logging.error(f"No images found in {inp}")
            sys.exit(1)
    else:
        if not inp.exists():
            logging.error(f"Input file not found: {inp}")
            sys.exit(1)
        files = [inp]

    logging.info("Loading models … (this can take a few seconds)")
    model_path = "yolo_train_run/augmented/weights/best.pt"
    logging.info(f"Using YOLO model: {model_path}")
    detector = YOLO(model_path)
    ocr_engine = MangaOcr()

    for img_path in tqdm(files, desc="Pages"):
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning(f"Skipping unreadable image: {img_path}")
            continue
        boxes = detect_bubbles(img, detector, conf=args.conf)
        if not boxes:
            logging.warning(f"No bubbles detected on {img_path.name}")
            continue
        crops = extract_crops(img, boxes)
        texts = ocr_bubbles(crops, ocr_engine)
        translations = translate_texts(texts, target_lang=args.lang)
        result = overlay_translations(img, crops, translations)
        out_path = out / img_path.name
        cv2.imwrite(str(out_path), result)
        logging.info(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
