import os
import asyncio

# =========================
# Environment & Async Fixes
# =========================
# Force Streamlit watcher to polling
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"
# Ensure an asyncio loop exists
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# =========================
# Imports
# =========================
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

# =========================
# Helper: Draw detections on image
# =========================
def draw_detections(img: Image.Image, model: YOLO, font_size: int) -> Image.Image:
    # Run detection
    cv_img = np.array(img.convert("RGB"))
    results = model.predict(source=cv_img, conf=0.25, iou=0.45, verbose=False)[0]
    # Map each class to its color and name
    class_info = {
        0: ("Dialogue", "#00d8ff"),
        1: ("Sound Effects", "#ff0000"),
        2: ("Signs", "#cbf8003d"),
        3: ("Text", "#3df53d"),
        4: ("Removal", "#f786d4"),
    }
    draw = ImageDraw.Draw(img)
    # Prepare font for labels
    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)  # increased font size for labels
    except Exception:
        label_font = ImageFont.load_default()
    # Draw detection boxes with labels
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        name, color = class_info.get(cls, ("Unknown", "red"))
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # Draw label background
        text_size = draw.textbbox((0,0), name, font=label_font)
        text_w = text_size[2] - text_size[0]
        text_h = text_size[3] - text_size[1]
        # Make sure label background is visible
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color)
        # Draw label text
        draw.text((x1 + 2, y1 - text_h - 2), name, fill="white", font=label_font)
    return img

# =========================
# Streamlit App
# =========================
st.title("🧰 Manga Bubble Detection")

# Control for label font size
label_size = st.sidebar.slider(
    "Label Font Size", min_value=10, max_value=40, value=24, step=2
)

# Image upload
uploaded = st.file_uploader("Upload a page", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Please upload an image.")
    st.stop()
try:
    img = Image.open(uploaded).convert("RGB")
except Exception as e:
    st.error(f"Could not open image: {e}")
    st.stop()

# Show original
st.image(img, caption="Original Image", use_container_width=True)

# Load model (cached)
@st.cache_resource
def load_model(path: str) -> YOLO:
    return YOLO(path)

yolo_path = "yolo_train_run/full_finetune_phase20/weights/best.pt"
if not os.path.exists(yolo_path):
    st.error(f"Model not found at {yolo_path}")
    st.stop()
model = load_model(yolo_path)

# Run detection and display
with st.spinner("Detecting bubbles..."):
    det_img = draw_detections(img.copy(), model, label_size)
st.image(det_img, caption="Detected Bubbles", use_container_width=True)

# Display legend below the image
# st.markdown(
#     "**Legend:**<br>"
#     "<span style='color:#00d8ff;font-size:1.2em;'>&#9632;</span> Dialogue&nbsp;&nbsp;"
#     "<span style='color:#ff0000;font-size:1.2em;'>&#9632;</span> Sound Effects&nbsp;&nbsp;"
#     "<span style='color:#f8f800;font-size:1.2em;'>&#9632;</span> Signs&nbsp;&nbsp;"
#     "<span style='color:#3df53d;font-size:1.2em;'>&#9632;</span> Text&nbsp;&nbsp;"
#     "<span style='color:#f786d4;font-size:1.2em;'>&#9632;</span> Removal",
#     unsafe_allow_html=True
#)
