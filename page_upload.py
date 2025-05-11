#!/usr/bin/env python3
"""
website_multi.py – Streamlit demo for batch-detecting speech bubbles, OCRing,
translating, and overlaying the translation on multiple pages.

Dependencies::

    pip install streamlit ultralytics manga-ocr deep-translator pillow opencv-python tqdm

Run with:

    streamlit run website_multi.py
"""
import os
import asyncio
import glob
from typing import List, Tuple

# -----------------------
# Disable inotify watchers (avoid ENOSPC errors)
# -----------------------
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# -----------------------
# Async Fix
# -----------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -----------------------
# Imports
# -----------------------
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from ultralytics import YOLO
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator

# -----------------------
# Utility Functions
# -----------------------
from typing import List, Tuple

def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    """Convert a hex color string (#RRGGBB) to an RGB tuple."""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# -----------------------
# Model Loading & OCR
# -----------------------
@st.cache_resource(show_spinner=False)
def load_models(path: str):
    return YOLO(path), MangaOcr()


def detect_and_ocr(
    pil_img: Image.Image,
    yolo_model,
    ocr_engine,
    conf: float,
    iou: float
) -> Tuple[List[Tuple[int,int,int,int]], List[str]]:
    np_img = np.array(pil_img)[:, :, ::-1]
    preds = yolo_model.predict(source=np_img, conf=conf, iou=iou, max_det=100, verbose=False)
    boxes, texts = [], []
    if preds:
        raw = preds[0].boxes.xyxy.cpu().numpy().astype(int)
        for x1, y1, x2, y2 in raw:
            crop = pil_img.crop((x1, y1, x2, y2))
            gray = ImageOps.grayscale(crop)
            enh = ImageEnhance.Contrast(gray).enhance(1.5)
            ann = ocr_engine(enh)
            txt = ann if isinstance(ann, str) else ("".join(ann) if all(isinstance(el, str) for el in ann) else " ".join(map(str, ann)))
            boxes.append((x1, y1, x2, y2))
            texts.append(txt or "")
    return boxes, texts

# -----------------------
# Overlay Translation
# -----------------------

def overlay(
    pil_img: Image.Image,
    boxes: List[Tuple[int,int,int,int]],
    texts: List[str],
    sizes: List[int],
    colors: List[Tuple[int,int,int]],
    lang: str
) -> Image.Image:
    # Ensure texts are strings
    texts = [t or "" for t in texts]
    base = pil_img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (255,255,255,0))
    draw = ImageDraw.Draw(layer)

    # Font paths
    script_fonts = {
        "ja": "/usr/share/fonts/opentype/noto/NotoSansJP-Regular.otf",
        "ko": "/usr/share/fonts/opentype/noto/NotoSansKR-Regular.otf",
        "hi": "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.otf",
    }
    default_font = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"

    for idx, ((x1,y1,x2,y2), txt) in enumerate(zip(boxes, texts)):
        if not txt.strip():
            continue
        # Choose font
        font_path = script_fonts.get(lang, default_font)
        try:
            font = ImageFont.truetype(font_path, sizes[idx])
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Wrap text
        max_w = (x2 - x1) - 10
        words = txt.split()
        lines, curr = [], ""
        for w in words:
            candidate = (curr + " " + w).strip()
            w_px = draw.textbbox((0,0), candidate, font=font)[2] if hasattr(draw, 'textbbox') else draw.textsize(candidate, font=font)[0]
            if w_px <= max_w:
                curr = candidate
            else:
                lines.append(curr)
                curr = w
        lines.append(curr)
        final = "\n".join(lines)

        # Measure block
        if hasattr(draw, 'multiline_textbbox'):
            wt, ht = draw.multiline_textbbox((0,0), final, font=font, spacing=2)[2:]
        else:
            wt, ht = draw.multiline_textsize(final, font=font, spacing=2)

        # Position
        tx = x1 + max(0, ((x2-x1) - wt)//2)
        ty = y1 + max(0, ((y2-y1) - ht)//2)

        # Draw background & text
        draw.rectangle([tx-2, ty-2, tx+wt+2, ty+ht+2], fill=(255,255,255,200))
        draw.multiline_text((tx, ty), final, font=font, fill=colors[idx], spacing=2, align="center")

    return Image.alpha_composite(base, layer).convert("RGB")

# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(page_title="Manga Batch Translation Demo", layout="wide")
st.title("🧰 Manga/Manhwa Batch Translation Demo")

# Sidebar
st.sidebar.header("⚙️ Settings")
lang = st.sidebar.selectbox("Target Language", ["en","es","ar","pt","id","fr","ja","ru","de","ko","hi"])  
conf = st.sidebar.slider("Detection Confidence", 0.05, 1.0, 0.25, 0.05)
iou  = st.sidebar.slider("NMS IoU Threshold",     0.0, 0.9, 0.3, 0.05)
font_size   = st.sidebar.slider("Overlay Font Size", 12, 48, 20)
col         = st.sidebar.color_picker("Overlay Font Color", '#0000FF')
font_color  = hex_to_rgb(col)

# File uploader
uploads = st.file_uploader("Upload pages (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)
if not uploads:
    st.info("Please upload one or more images above.")
    st.stop()

# Load models
tf_path = "yolo_train_run/full_finetune_phase2/weights/best.pt"
if not os.path.exists(tf_path):
    st.error(f"YOLO model not found at `{tf_path}`")
    st.stop()
with st.spinner("Loading models…"):
    yolo_model, ocr_engine = load_models(tf_path)

# Process pages
for idx, file in enumerate(uploads, start=1):
    try:
        img = Image.open(file).convert("RGB")
    except:
        st.error(f"Page {idx}: could not open image.")
        continue

    st.subheader(f"Page {idx} - Original")
    st.image(img, use_container_width=True)

    boxes, texts = detect_and_ocr(img, yolo_model, ocr_engine, conf, iou)
    if not boxes:
        st.warning(f"No bubbles detected on page {idx}.")
        continue

    translations = [
        GoogleTranslator(source='auto', target=lang).translate(txt) if txt.strip() else ""
        for txt in texts
    ]

    result = overlay(img, boxes, translations, [font_size]*len(boxes), [font_color]*len(boxes), lang)

    st.subheader(f"Page {idx} - Translated")
    st.image(result, use_container_width=True)

st.success("🎉 Batch translation complete!")
