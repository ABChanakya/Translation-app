import os
import asyncio

# =========================
# Environment & Async Fixes
# =========================

# Force Streamlit to use polling instead of its default file watcher
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Ensure an asyncio event loop exists (fixes "no running event loop" errors)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# =========================
# Import Libraries
# =========================

import streamlit as st
import requests
import io
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont

# PyTorch & YOLO imports for detection
import cv2
from ultralytics import YOLO
from manga_ocr import MangaOcr  # Ensure manga_ocr is installed and set up

# =========================
# Utility & Stub Functions
# =========================

def preprocess_image(img):
    """
    Pre-process the image to enhance OCR performance:
      - Convert to grayscale.
      - Increase contrast.
    """
    img_gray = ImageOps.grayscale(img)
    enhancer = ImageEnhance.Contrast(img_gray)
    return enhancer.enhance(1.5)

def detect_bubbles(img_pil, yolo_model):
    """
    Detect speech bubbles using YOLOv8. Converts the PIL image to OpenCV BGR format,
    runs detection, and then crops each detected bubble as a separate PIL image.
    """
    img_cv2 = np.array(img_pil.convert("RGB"))[:, :, ::-1]  # Convert PIL -> OpenCV BGR
    results = yolo_model(img_cv2)
    bubbles = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        bubble_crop = img_pil.crop((x1, y1, x2, y2))
        bubbles.append(bubble_crop)
    return bubbles

def extract_text_from_bubbles(bubbles, ocr):
    """
    Process each detected bubble:
      - Pre-process the bubble image.
      - Extract text using manga_ocr.
    Returns a list of tuples (bubble_image, extracted_text).
    """
    bubble_texts = []
    for bubble in bubbles:
        enhanced = preprocess_image(bubble)
        text = ocr(enhanced)
        bubble_texts.append((bubble, text))
    return bubble_texts

def classify_text(text):
    """
    Dummy text classification that distinguishes between dialogue, sound effects, etc.
    """
    return "Sound Effect" if len(text.split()) <= 3 else "Dialogue"

def translate_text(text, target_language="en"):
    """
    Dummy translation stub.
    """
    return f"[Translated to {target_language}]: {text}"

def detect_characters(img):
    """
    Placeholder for character detection/recognition.
    """
    return []

def replace_text_on_image(img, texts, positions):
    """
    Overlay translated text onto an image at given positions.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    for text, pos in zip(texts, positions):
        draw.text(pos, text, fill="red", font=font)
    return img_copy

# =========================
# Main Pipeline Implementation
# =========================

st.title("🧰 Manga/Manhwa Translation Pipeline")

st.markdown("""
This pipeline processes uploaded manga/manhwa pages:
1. **Input & Pre-processing:** Upload a manga/manhwa page.
2. **Text Detection & Categorization:** Detect speech bubbles, run OCR, and classify text.
3. **Context & Character Recognition:** (Stub) Detect characters and scene context.
4. **Translation & Style Customization:** (Stub) Translate extracted text.
5. **Text Replacement & Image Reconstruction:** (Stub) Overlay translation on image.
6. **Output:** Display results.
""")

# 1. Input & Pre-processing
uploaded_file = st.file_uploader(
    "Upload your manga/manhwa page (JPEG, PNG)", type=["jpeg", "jpg", "png"]
)
url = st.text_input("Or paste an image URL")

if uploaded_file:
    try:
        original_img = Image.open(uploaded_file)
        st.success("File uploaded successfully!")
        st.image(original_img, caption="Original Image")
    except Exception as e:
        st.error(f"Failed to open uploaded image: {e}")
        st.stop()
elif url:
    try:
        response = requests.get(url)
        response.raise_for_status()
        original_img = Image.open(io.BytesIO(response.content))
        st.success("Image fetched successfully from URL!")
        st.image(original_img, caption="Original Image from URL")
    except Exception as e:
        st.error(f"Failed to fetch image from URL: {e}")
        st.stop()
else:
    st.info("Please upload an image or provide a URL.")
    st.stop()

# 2. Model Loading
yolo_model_path = "/home/chanakya/chanakya/UNI/translation tool/yolo_train_run/train/weights/best.pt"

if not os.path.exists(yolo_model_path):
    st.error("YOLOv8 model file not found.")
    st.stop()

try:
    bubble_model = YOLO(yolo_model_path)
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

try:
    ocr = MangaOcr()
except Exception as e:
    st.error(f"Failed to initialize Manga OCR: {e}")
    st.stop()

# 3. Text Detection & Categorization
st.header("Step 2: Text Detection & Categorization")

with st.spinner("Detecting speech bubbles..."):
    bubbles = detect_bubbles(original_img, bubble_model)

if not bubbles:
    st.warning("No speech bubbles detected.")
    st.stop()

bubble_texts = extract_text_from_bubbles(bubbles, ocr)
classified_results = [(b, t, classify_text(t)) for b, t in bubble_texts]

st.subheader("Detected Speech Bubbles & OCR Text")
for i, (bubble, text, category) in enumerate(classified_results):
    st.image(bubble, caption=f"Bubble {i+1}")
    st.text_area(
        f"Extracted Text (Category: {category})",  # visible label
        text,
        height=100,
        key=f"extracted_{category}_{i}"              # unique key!
    )

# 4. Context & Character Recognition (Stub)
st.header("Step 3: Context & Character Recognition (Stub)")
st.write("Detected Characters:", detect_characters(original_img))

# 5. Translation & Style Customization (Stub)
st.header("Step 4: Translation & Style Customization (Stub)")
translated_texts = [translate_text(t) for _, t, _ in classified_results]
for idx, ((_, orig_text, _), tr_text) in enumerate(zip(classified_results, translated_texts)):
    st.write(f"Original [{idx+1}]: {orig_text}")
    st.write(f"Translated [{idx+1}]: {tr_text}")

# 6. Text Replacement & Image Reconstruction (Stub)
st.header("Step 5: Text Replacement & Image Reconstruction (Stub)")
positions = [(50, 50 + j * 30) for j in range(len(translated_texts))]
reconstructed_img = replace_text_on_image(original_img, translated_texts, positions)
st.image(reconstructed_img, caption="Reconstructed Image with Translated Text")

# 7. Final Output & Next Steps
st.header("Step 6: Output & Next Steps")
st.markdown("""
- **Quality Assurance:** Add a human‑review step.
- **Performance Optimization:** Batch processing, GPU acceleration.
- **Data Privacy:** Define clear handling policies.
""")

