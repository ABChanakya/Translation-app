#!/usr/bin/env python3
"""
translator.py – detect speech bubbles, OCR them, translate, and overlay the
translation back onto the page.

Dependencies::

    pip install ultralytics manga-ocr deep-translator pillow opencv-python tqdm

Usage examples
--------------
Translate every image under ``data/needs`` and write the results next to them::

    python translator.py \
        --input data/needs \
        --model yolo_train_run/full_finetune_phase2/weights/best.pt \
        --lang en \
        --output translated_pages

Translate a *single* file::

    python translator.py -i page01.png -o out.png
"""

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

def detect_bubbles(
    img: np.ndarray,
    model: YOLO,
    conf: float = 0.25,
    iou: float  = 0.3,   # <-- use a sensible NMS IoU threshold
    max_det: int = 50
) -> List[BBox]:
    """
    Run YOLO inference on `img`, keeping all boxes >= conf
    and applying NMS at `iou` to suppress only very close duplicates.
    """
    results = model.predict(
        source=img,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False
    )
    if not results:
        return []
    # extract and clip boxes
    h, w = img.shape[:2]
    raw = results[0].boxes.xyxy.cpu().numpy().astype(int)
    return [
        [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
        for x1, y1, x2, y2 in raw
    ]

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
            texts.append(ann)
        else:
            # flatten whatever form MangaOcr returns
            try:
                if all(isinstance(el, str) for el in ann):
                    texts.append("".join(ann))
                elif all(isinstance(el, (tuple, list)) for el in ann):
                    texts.append("".join(el[0] for el in ann))
                else:
                    texts.append(" ".join(str(el) for el in ann))
            except Exception:
                texts.append(str(ann))
    return texts

################################################################################
# 4. Translation                                                               #
################################################################################

def translate_texts(texts: List[str], target_lang: str = "en") -> List[str]:
    translations: List[str] = []
    for t in texts:
        try:
            translations.append(
                GoogleTranslator(source='auto', target=target_lang).translate(t)
            )
        except Exception as e:
            logging.warning(f"Translation failed for '{t}': {e}")
            translations.append(t)
    return translations

################################################################################
# 5. Overlay                                                                   #
################################################################################

def overlay_translations(
    page_img: np.ndarray,
    crops: List[Crop],
    translations: List[str]
) -> np.ndarray:
    pil_base = Image.fromarray(
        cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)
    ).convert("RGBA")
    txt_layer = Image.new("RGBA", pil_base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    font = ImageFont.load_default()

    def text_size(text: str, multiline=False, spacing=0):
        if multiline:
            return draw.multiline_textbbox((0,0), text, font=font, spacing=spacing)[2:]
        else:
            return draw.textbbox((0,0), text, font=font)[2:]

    for (_, (x1, y1, x2, y2)), text in zip(crops, translations):
        t = (text or "").strip()
        if not t:
            continue
        # word-wrap
        max_w = max(10, x2 - x1 - 10)
        words, lines, curr = t.split(), [], ""
        for w in words:
            test = (curr + " " + w).strip()
            if text_size(test)[0] <= max_w:
                curr = test
            else:
                lines.append(curr); curr = w
        lines.append(curr)
        final = "\n".join(lines)
        w, h = text_size(final, multiline=True, spacing=2)
        tx = x1 + max(0, ((x2 - x1) - w)//2)
        ty = y1 + max(0, ((y2 - y1) - h)//2)
        # draw box + text
        draw.rectangle([tx-2, ty-2, tx+w+2, ty+h+2], fill=(255,255,255,200))
        draw.multiline_text((tx, ty), final, fill=(0,0,255,255),
                            font=font, spacing=2, align="center")

    combined = Image.alpha_composite(pil_base, txt_layer).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

################################################################################
# Utility                                                                      #
################################################################################

def gather_images(path: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.png","*.jpg","*.jpeg"):
        files.extend(path.glob(ext))
    return sorted(files)

################################################################################
# Main                                                                         #
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Detect, OCR & translate speech bubbles on manga pages"
    )
    parser.add_argument("-i","--input", required=True,
                        help="Image file or directory")
    parser.add_argument("-m","--model", default="yolov8n.pt",
                        help="YOLO .pt checkpoint")
    parser.add_argument("-l","--lang", default="en",
                        help="Target ISO‑639‑1 language code")
    parser.add_argument("-c","--conf", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("-o","--output", required=True,
                        help="Output file or directory")
    args = parser.parse_args()

    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if inp.is_dir():
        files = gather_images(inp)
        if not files:
            logging.error(f"No images in {inp}"); sys.exit(1)
    else:
        if not inp.exists():
            logging.error(f"File not found: {inp}"); sys.exit(1)
        files = [inp]

    logging.info("Loading YOLO model…")
    detector = YOLO(args.model)
    ocr = MangaOcr()

    for img_path in tqdm(files, desc="Pages"):
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning(f"Cannot read {img_path}"); continue
        boxes = detect_bubbles(img, detector, conf=args.conf)
        if not boxes:
            logging.warning(f"No bubbles on {img_path.name}"); continue
        crops      = extract_crops(img, boxes)
        texts      = ocr_bubbles(crops, ocr)
        translations = translate_texts(texts, target_lang=args.lang)
        result     = overlay_translations(img, crops, translations)
        out_file   = out / img_path.name
        cv2.imwrite(str(out_file), result)
        logging.info(f"Saved ↴ {out_file}")

if __name__ == "__main__":
    main()
