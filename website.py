#!/usr/bin/env python3
"""
website.py – Streamlit demo for detecting speech bubbles, OCRing,
translating, and overlaying the translation back onto the page.

Dependencies::

    pip install streamlit ultralytics manga-ocr deep-translator pillow opencv-python tqdm

Run with:

    streamlit run website.py
"""

import os
import asyncio

# -----------------------
# Environment & Async Fix
# -----------------------
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -----------------------
# Imports
# -----------------------
import streamlit as st
import requests
import io
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from ultralytics import YOLO
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator

# -----------------------
# Utility Functions
# -----------------------
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert a hex color string (#RRGGBB) to an RGB tuple."""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(
    page_title="Manga Translation Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧰 Manga/Manhwa Live Translation Demo")

# Sidebar for global settings
st.sidebar.header("⚙️ Settings")
lang = st.sidebar.selectbox(
    "Target Language",
    [
        "en",  # English
        "es",  # Spanish
        "ar",  # Arabic
        "pt",  # Portuguese
        "id",  # Indonesian
        "fr",  # French
        "ja",  # Japanese
        "ru",  # Russian
        "de",  # German
        "ko",  # Korean
        "hi",  # Hindi
        "te",  # Telugu
        "ta",  # Tamil
        "bn",  # Bengali
        "mr",  # Marathi
        "gu",  # Gujarati
        "ml",  # Malayalam
        "kn",  # Kannada
        "pa",  # Punjabi
        "ur",  # Urdu
        "th",  # Thai
        "vi",  # Vietnamese
    ],
    help="Select the language code to translate the detected text into."
)
conf_threshold = st.sidebar.slider(
    "Detection Confidence", 0.05, 1.0, 0.25, 0.05,
    help="Minimum confidence score for YOLO detections. Lower shows more boxes, higher filters weaker ones."
)
iou_threshold  = st.sidebar.slider(
    "NMS IoU Threshold", 0.0, 0.9, 0.3, 0.05,
    help="Non-Maximum Suppression IoU threshold: boxes with IoU > this will be suppressed to reduce duplicates."
)

st.markdown("""
Upload a manga/manhwa page (JPEG/PNG) or paste an image URL.  
This demo will:
1. Detect speech bubbles
2. OCR their contents
3. Translate into your language
4. Overlay the translation back onto the page
""" )

# -----------------------
# 1. Input
# -----------------------
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload page", type=["jpg","jpeg","png"])
with col2:
    url = st.text_input("—or paste image URL")

if not (uploaded or url):
    st.info("Please upload an image or enter a URL above.")
    st.stop()

try:
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
    else:
        resp = requests.get(url)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
except Exception as e:
    st.error(f"Could not load image: {e}")
    st.stop()

st.image(img, caption="Original Page", use_column_width=True)

# -----------------------
# 2. Load Models
# -----------------------
@st.cache_resource(show_spinner=False)
def load_models(path: str):
    yolo = YOLO(path)
    ocr  = MangaOcr()
    return yolo, ocr

model_path = "yolo_train_run/full_finetune_phase21/weights/best.pt"
if not os.path.exists(model_path):
    st.error(f"YOLO model not found at `{model_path}`")
    st.stop()

with st.spinner("Loading models…"):
    yolo_model, ocr_engine = load_models(model_path)

# -----------------------
# 3. Detect & OCR
# -----------------------
st.header("Step 1: Detect & OCR")
def detect_and_ocr(pil_img: Image.Image):
    np_img = np.array(pil_img)[:, :, ::-1]
    preds = yolo_model.predict(
        source=np_img,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=100,
        verbose=False
    )
    boxes, texts = [], []
    if preds:
        raw = preds[0].boxes.xyxy.cpu().numpy().astype(int)
        for (x1,y1,x2,y2) in raw:
            crop = pil_img.crop((x1,y1,x2,y2))
            gray = ImageOps.grayscale(crop)
            crop = ImageEnhance.Contrast(gray).enhance(1.5)
            ann = ocr_engine(crop)
            if isinstance(ann, str):
                txt = ann
            else:
                try:
                    if all(isinstance(el,str) for el in ann): txt = "".join(ann)
                    else: txt = " ".join(str(el) for el in ann)
                except:
                    txt = str(ann)
            boxes.append((x1,y1,x2,y2))
            texts.append(txt)
    return boxes, texts

with st.spinner("Running detection & OCR…"):
    boxes, ocr_texts = detect_and_ocr(img)

if not boxes:
    st.warning("No speech bubbles detected.")
    st.stop()

# Per-bubble style controls
bubble_font_sizes = []
bubble_font_colors = []
st.header("Step 1b: Style Each Bubble")
for i, ((x1,y1,x2,y2), txt) in enumerate(zip(boxes,ocr_texts), 1):
    st.subheader(f"Bubble {i}")
    st.image(img.crop((x1,y1,x2,y2)), width=200)
    st.text_area("OCR Text", value=txt, key=f"ocr_{i}", height=80)
    size = st.slider(
        f"Font Size for Bubble {i}", 10, 72, 20, 1,
        key=f"size_{i}",
        help="Adjust font size for this bubble's translation."
    )
    color = st.color_picker(
        f"Font Color for Bubble {i}", '#0000FF', key=f"color_{i}",
        help="Choose a color for the translated text."
    )
    bubble_font_sizes.append(size)
    bubble_font_colors.append(hex_to_rgb(color))

# -----------------------
# 4. Translate
# -----------------------
st.header("Step 2: Translation")
translations = []
for txt in ocr_texts:
    try:
        if txt:  # Ensure txt is not None or empty
            tr = GoogleTranslator(source='auto', target=lang).translate(txt)
        else:
            tr = ""
    except:
        tr = txt or ""
    translations.append(tr)
for i, (orig, tr) in enumerate(zip(ocr_texts, translations), 1):
    st.subheader(f"Bubble {i} Translations")
    st.markdown(f"- **Original:** {orig}")
    st.markdown(f"- **Translated:** {tr}")

# -----------------------
# 5. Overlay Translation
# -----------------------
st.header("Step 3: Overlay Translated Text")
def overlay(
    pil_img: Image.Image,
    boxes, texts, sizes, colors
) -> Image.Image:
    base = pil_img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (255,255,255,0))
    draw = ImageDraw.Draw(layer)

    # Font lookup
    script_fonts = {
        "ja": "/usr/share/fonts/opentype/noto/NotoSansJP-Regular.otf",
        "ko": "/usr/share/fonts/opentype/noto/NotoSansKR-Regular.otf",
        "hi": "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.otf",
        # …etc…
    }
    default_font = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
    serif_font  = "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf"


    for idx, ((x1,y1,x2,y2), txt) in enumerate(zip(boxes, texts)):
        # 1) Guard against None
        txt = txt or ""
        if not txt.strip():
            # no text → nothing to draw
            continue

        font_size  = sizes[idx]
        font_color = colors[idx]
        font_path  = script_fonts.get(lang, default_font)

        # 2) Load font, fallback to default PIL font
        try:
            font = ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # 3) Wrap text to fit bubble width
        max_width = (x2 - x1) - 10
        words = txt.split()
        lines = []
        curr = ""
        for w in words:
            candidate = (curr + " " + w).strip()
            try:
                w_px = draw.textbbox((0,0), candidate, font=font)[2]
            except AttributeError:
                w_px = draw.textsize(candidate, font=font)[0]

            if w_px <= max_width:
                curr = candidate
            else:
                lines.append(curr)
                curr = w
        lines.append(curr)
        final = "\n".join(lines)

        # 4) Measure block size
        try:
            wt, ht = draw.multiline_textbbox((0,0), final, font=font, spacing=2)[2:]
        except AttributeError:
            wt, ht = draw.multiline_textsize(final, font=font, spacing=2)

        # 5) Center inside the bubble
        tx = x1 + max(0, ((x2 - x1) - wt) // 2)
        ty = y1 + max(0, ((y2 - y1) - ht) // 2)

        # 6) Draw background & text
        draw.rectangle([tx-2, ty-2, tx+wt+2, ty+ht+2], fill=(255,255,255,200))
        draw.multiline_text((tx, ty), final, font=font, fill=font_color, spacing=2, align="center")

    return Image.alpha_composite(base, layer).convert("RGB")

with st.spinner("Compositing translated page…"):
    result = overlay(img, boxes, translations, bubble_font_sizes, bubble_font_colors)

st.image(result, caption="Translated Page", use_column_width=True)
st.success("🎉 Translation complete!")