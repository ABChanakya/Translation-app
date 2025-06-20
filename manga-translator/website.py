"""
website.py – Self‑contained dual‑front‑end (Streamlit / Gradio) manga‑page translator.

Installation (once):
    pip install streamlit gradio ultralytics manga-ocr pillow numpy opencv-python \
        argostranslate==1.8.0 transformers==4.* sentencepiece torch

Run Streamlit (default):
    streamlit run website.py

Run Gradio:
    WEB_UI=gradio python website.py
"""



from __future__ import annotations

import os
import io
import math
import json
import time
import shutil
import tempfile
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any
import textwrap

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageColor


import torch
from ultralytics import YOLO
import torchvision.ops as ops
from manga_ocr import MangaOcr

# UI libs (always import; both are installed via requirements)
import streamlit as st


from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianTokenizer,
    MarianMTModel,
)
from pathlib import Path
from enum import Enum
import ollama
import base64

import argostranslate.package
import argostranslate.translate
# Add these imports at the top of your file:
try:
    from googletrans import Translator as GoogleTranslator
except (ImportError, AttributeError):
    GoogleTranslator = None
    # and later in your translate() function:
    # if engine=="Google" and GoogleTranslator is None: fall back to another engine
# ─── googletrans helper ────────────────────────────────────────────────────────
if GoogleTranslator is not None:
    _GT = GoogleTranslator()          # one reusable client
else:
    _GT = None                        # will trigger fallback later


import deepl
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
import requests





# ---------------------------------- CONFIG ---------------------------------- #
YOLO_MODEL_PATH = "yolo_train_run/full_finetune_phase20/weights/best.pt"        # <––– PUT your YOLOv8/9 weights here
# Example file structure:
# yolo_train_run/
# ├── full_finetune_phase20/
# │   ├── weights/
# │   │   ├── best.pt
# For more details, refer to the YOLOv8/YOLOv9 documentation: https://docs.ultralytics.com/
CACHE_DIR = os.path.join(tempfile.gettempdir(), "manga_translator_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SIDE = 1600          # shrink page to <=1600 px on longest side

# ---------------------------------------------------------------------------- #

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "P6vGDWzSCEIJ02uKfWRE")   # <-- set in your shell
PROJECT_SLUG     = "manga-panels-jxuz5-lijsg-1gosm-ccatd"  # <-- your model ID
VERSION          = "1"         # Roboflow model version
RF_CONFIDENCE    = 0.4         # 0-1 (probability threshold)
RF_OVERLAP       = 30          # NMS overlap in pixels


# Translation model IDs
MARIAN_PREFIX = "Helsinki-NLP/opus-mt"
NLLB_ID = "facebook/nllb-200-distilled-600M"
DEEPL_API_KEY        = os.getenv("DEEPL_API_KEY", "")
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY", "")
AZURE_ENDPOINT       = os.getenv("AZURE_ENDPOINT", "")
# Define default font path for text rendering.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Define detection classes
# SFX = 0
# SIGN = 1
# TEXT = 2
# REMOVAL = 3
# DIALOGUE = 4
DIALOGUE = 0
SOUND_EFFECTS = 1
SIGNS = 2
TEXT = 3
REMOVAL = 4
SFX = SOUND_EFFECTS
SIGN = SIGNS

# Class IDs
@st.cache_resource(ttl=86400)
def get_yolo_model(conf: float = .25, iou: float = .45):
    model = YOLO(YOLO_MODEL_PATH)   # 1. load
    model.fuse()                    # 2. fuse ‑ in‑place, ignore return
    if DEVICE == "cuda":
        model.to("cuda").half()     # 3. move / cast
    model.predict(conf=conf, iou=iou)   # 4. warm‑up
    return model

# ---------------------------------------------------------------------------- #
# ---------- 1. keep ONE get_yolo_model ------------------------------------
@st.cache_resource(ttl=86400, show_spinner=False)
def get_yolo_model(conf: float = .25, iou: float = .45):
    model = YOLO(YOLO_MODEL_PATH)   # 1️⃣ create model
    model.fuse()                    # 2️⃣ optimise layers in‑place
    if DEVICE == "cuda":
        model.to("cuda").half()     # 3️⃣ move & cast
    model.predict(conf=conf, iou=iou)   # 4️⃣ warm‑up
    return model                    # 5️⃣ return the real object


@st.cache_resource(show_spinner=False, ttl=None)
def get_ocr():
    ocr = MangaOcr()
    if DEVICE == "cuda":
        ocr.model.to("cuda", dtype=torch.float16)
    return ocr



@lru_cache(maxsize=4)
def _load_marian(src: str, tgt: str):
    model_name = f"{MARIAN_PREFIX}-{src}-{tgt}"
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(model_name)
    return tok, mdl


@lru_cache(maxsize=2)
def _load_nllb():
    tok = AutoTokenizer.from_pretrained(NLLB_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID)
    return tok, mdl


def _ensure_argos_pkg(src: str, tgt: str):
    """Download Argos model pair lazily into cache dir."""
    installed = {(p.from_code, p.to_code) for p in argostranslate.package.get_installed_packages()}
    if (src, tgt) in installed:
        return
    # tiny network call; if offline will except and fallback
    pkg_url = f"https://huggingface.co/argosopentech/argos-translate-{src}_{tgt}/resolve/main/{src}_{tgt}.argos"
    try:
        fn = os.path.join(CACHE_DIR, f"{src}_{tgt}.argos")
        if not os.path.exists(fn):
            r = requests.get(pkg_url, timeout=10)
            r.raise_for_status()
            with open(fn, "wb") as f:
                f.write(r.content)
        argostranslate.package.install_from_path(fn)
    except Exception:
        pass  # fallback later


# ---------------------------------------------------------------------------- #
#                              UTILITY HELPERS                                 #
# ---------------------------------------------------------------------------- #
def whitest_pixel(arr: np.ndarray) -> Tuple[int, int, int]:
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Input array must have three color channels (H x W x 3).")
    idx = np.argmax(arr.sum(axis=-1))
    h, w, _ = arr.shape
    y, x = divmod(idx, w)
    return tuple(map(int, arr[y, x]))


def median_color(arr: np.ndarray) -> Tuple[int, int, int]:
    med = np.median(arr.reshape(-1, 3), axis=0)
    return tuple(int(x) for x in med)


def overlay(
    pil_img: Image.Image,
    boxes,            # Iterable[(x1,y1,x2,y2), ...]
    texts,            # Iterable[str, …]  – already wrapped
    sizes,            # Iterable[int, …]  – font sizes per text
    colors            # Iterable[Tuple[int,int,int,int], …]  RGBA
) -> Image.Image:
    """
    Draws *texts* on a transparent layer and composites it on top of *pil_img*.

    Parameters
    ----------
    pil_img : PIL.Image
        Original manga page (RGB or RGBA).
    boxes : list[tuple[int,int,int,int]]
        Bounding boxes where each string should be rendered.
    texts : list[str]
        Text strings (one per box).
    sizes : list[int]
        Font size per string.
    colors : list[tuple[int,int,int,int]]
        RGBA text colours, e.g. (0,0,255,255).

    Returns
    -------
    PIL.Image
        The original page with the overlay composited (mode “RGB”).
    """
    base   = pil_img.convert("RGBA")
    layer  = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw   = ImageDraw.Draw(layer)

    for (x1, y1, x2, y2), txt, sz, col in zip(boxes, texts, sizes, colors):
        font = load_font(sz)
        wbox = draw.textbbox((0, 0), txt, font=font)
        tw, th = wbox[2] - wbox[0], wbox[3] - wbox[1]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.multiline_text(
            (cx, cy),
            txt,
            font=font,
            fill=col,
            anchor="mm",
            align="center",
        )

    # composite and drop alpha
    return Image.alpha_composite(base, layer).convert("RGB")


def load_font(size:int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()
# ---------------------------------------------------------------------------- #
#                    FITTING TEXT INTO BOUNDING BOXES                          #
# ---------------------------------------------------------------------------- #


def fit_and_wrap_text(
    draw: ImageDraw.Draw,
    text: str,
    box: Tuple[int, int, int, int],
    font_path: str = FONT_PATH,
    max_size: int = 60,
    min_size: int = 12,
    step: int = 2
) -> Tuple[str, ImageFont.FreeTypeFont]:
    """
    Returns a (wrapped_text, font) pair such that:
      • wrapped_text is broken into lines that never exceed box width
      • font is the largest size between max_size and min_size that
        makes the rendered block fit inside box (w × h)
    """
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1

    # Estimate avg char width by measuring “M” once per size
    for size in range(max_size, min_size - 1, -step):
        try:
            font = ImageFont.truetype(font_path, size=size)
        except OSError:
            continue  # skip sizes if ttf missing
        
        # Compute bounding box for a reference character "M"
        m_bbox = draw.textbbox((0, 0), "M", font=font)
        avg_char_w = max(5, m_bbox[2] - m_bbox[0])  # Ensure a minimum threshold for avg_char_w
        max_chars = max(1, box_w // avg_char_w)     # Avoid unexpected behavior
        # Optionally recompute avg_char_w and max_chars (if needed)
        avg_char_w = max(1, m_bbox[2] - m_bbox[0])
        max_chars = box_w // avg_char_w or 1
        
        # Wrap by words using textwrap
        wrapped = "\n".join(textwrap.wrap(text, width=max_chars,break_long_words=False, break_on_hyphens=False))
        
        # Measure the whole block
        tb = draw.multiline_textbbox((0, 0), wrapped, font=font)
        text_w, text_h = tb[2] - tb[0], tb[3] - tb[1]
        
        if text_w <= box_w and text_h <= box_h:
            return wrapped, font

    # Fallback: use the smallest size
    font = ImageFont.truetype(font_path, size=min_size)
    m_bbox = draw.textbbox((0, 0), "M", font=font)
    avg_char_w = max(1, m_bbox[2] - m_bbox[0])
    wrapped = "\n".join(textwrap.wrap(text, width=box_w // avg_char_w or 1))
    return wrapped, font



import requests, io

def detect_panels_via_roboflow(pil_image: Image.Image):
    """
    Returns a list of panel bounding-boxes (x1,y1,x2,y2) using Roboflow’s
    hosted model. Falls back to a single full-page box if the request fails.
    """
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY.startswith("YOUR"):
        return [(0, 0, pil_image.width, pil_image.height)]

    url = f"https://detect.roboflow.com/{PROJECT_SLUG}/{VERSION}"
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    try:
        r = requests.post(
            url,
            params=dict(
                api_key=ROBOFLOW_API_KEY,
                confidence=int(RF_CONFIDENCE * 100),
                overlap=RF_OVERLAP,
            ),
            files={"file": img_bytes},
            timeout=30,
        )
        r.raise_for_status()
        preds = r.json().get("predictions", [])
        boxes = [
            (int(p["x"] - p["width"]  / 2),
             int(p["y"] - p["height"] / 2),
             int(p["x"] + p["width"]  / 2),
             int(p["y"] + p["height"] / 2))
            for p in preds
        ]
        return boxes or [(0, 0, pil_image.width, pil_image.height)]
    except Exception as e:
        st.warning(f"Roboflow panel detection failed: {e}")
        return [(0, 0, pil_image.width, pil_image.height)]


# ---------------------------------------------------------------------------- #
#                    GEMMA3 TRANSLATION INTEGRATION                            #
# ---------------------------------------------------------------------------- #
class ResponseFormat(Enum): JSON = 'json_object'; TEXT = 'text'

def call_model(
    prompt:str='', image_path:Path|None=None,
    response_format:ResponseFormat=ResponseFormat.TEXT,
    system_prompt:str=''
) -> str:
    msgs=[]
    if system_prompt: msgs.append({'role':'system','content':system_prompt})
    if image_path:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        uri = f"data:image/png;base64,{b64}"
        msgs.append({'role':'user','content':f'![Image]({uri})\n\n{prompt}'})
    else:
        msgs.append({'role':'user','content':prompt})
    resp = ollama.chat(
        model='gemma3:4b', messages=msgs, keep_alive='1h',
        format='' if response_format==ResponseFormat.TEXT else 'json',
        options={'temperature':1.0,'min_p':0.01,'repeat_penalty':1.0,'top_k':64,'top_p':0.95}
    )
    reply = resp.message.content
    return reply


# ---------------------------------------------------------------------------- #
#                           TRANSLATION ABSTRACTION                            #
# ---------------------------------------------------------------------------- ##
def translate(text: str, src: str, tgt: str, engine: str) -> str:
    """
    Translate *text* from *src* → *tgt* using the chosen *engine*.
    If that engine fails, fall back to Argos → Google in that order.
    If Google also fails, return the original text unchanged.
    """
    text = text.strip()
    if not text:
        return ""

    try:
        if engine == "Gemma3":
            # 1) System prompt now mentions both src and tgt
            system_prompt = (
                f"You are a world-class translator with deep expertise in {src} and {tgt}. "
                f"Translate the following {src} text into fluent, idiomatic {tgt}, preserving nuance and tone. "
                "Output ONLY the translated text, with no commentary or formatting."
            )

            # 2) Wrap the text and label it clearly
            user_prompt = (
                f"=== Begin {src} text ===\n"
                f"{text}\n"
                f"=== End {src} text ==="
            )

            # 3) Call the model
            reply = call_model(
                prompt=user_prompt,
                response_format=ResponseFormat.TEXT,
                system_prompt=system_prompt
            )
            # 4) Debug log of what we got back
            print("<<< GEMMA3 OUTPUT:\n", reply)
            # 5) Return the actual reply
            return reply
        # ── Google Translate (unofficial) ────────────────────────────────
        if engine == "Google":
            #if _GT is None:                           # library missing
                # Check if Argos is installed before falling back
                #if not argostranslate.package.get_installed_packages():
                    #raise RuntimeError("Argos Translate is not installed. Please install it to use as a fallback.")
                #return translate(text, src, tgt, "Argos")

            
            return _GT.translate(text, src=src or "auto", dest=tgt).text
            
                #return translate(text, src, tgt, "Argos")

        # ── DeepL ────────────────────────────────────────────────────────
        elif engine == "DeepL":
            translator = deepl.Translator(DEEPL_API_KEY)
            resp = translator.translate_text(
                text, source_lang=src.upper(), target_lang=tgt.upper()
            )
            return resp.text

        # ── Azure Translator ─────────────────────────────────────────────
        elif engine == "Azure":
            cred   = AzureKeyCredential(AZURE_TRANSLATOR_KEY)
            client = TextTranslationClient(endpoint=AZURE_ENDPOINT, credential=cred)
            result = client.translate(content=[text], from_parameter=src, to=[tgt])
            return result[0].translations[0].text

        # ── Argos (offline) ──────────────────────────────────────────────
        elif engine == "Argos":
            _ensure_argos_pkg(src, tgt)
            return argostranslate.translate.translate(text, src, tgt)

        # ── MarianMT (Helsinki) ──────────────────────────────────────────
        elif engine == "MarianMT":
            tok, mdl = _load_marian(src, tgt)
            out = mdl.generate(**tok(text, return_tensors="pt"), max_length=256)
            return tok.decode(out[0], skip_special_tokens=True)

        # ── NLLB (Meta) ──────────────────────────────────────────────────
        elif engine == "NLLB":
            tok, mdl = _load_nllb()
            if src not in tok.lang_code_to_id:
                raise ValueError(f"Unsupported src lang {src!r} for NLLB")
            inp = tok(text, return_tensors="pt")
            inp["forced_bos_token_id"] = tok.lang_code_to_id.get(tgt, 0)
            out = mdl.generate(**inp, max_length=256)
            return tok.decode(out[0], skip_special_tokens=True)

        # ── Unknown engine name ──────────────────────────────────────────
        else:
            raise ValueError(f"Unknown translation engine: {engine!r}")

    # ── If the chosen engine blew up, fall back hierarchically ──────────
    except Exception:
        if engine not in ("Google", "Argos"):
            return translate(text, src, tgt, "Argos")
        if engine != "Google":
            return translate(text, src, tgt, "Google")
        return text  # All fallbacks failed


# ---------------------------------------------------------------------------- #
#                          PROCESSING                                          #
# ---------------------------------------------------------------------------- #
#def group_boxes_by_class(result, conf_thresh=.25, iou_thresh=.45):
    groups = {int(k): [] for k in range(len(result.names))}

    # collect boxes
    for box, cls, conf in zip(result.boxes.xyxy.cpu(),
                              result.boxes.cls.int().cpu(),
                              result.boxes.conf.cpu()):
        if conf < conf_thresh:
            continue
        groups[int(cls)].append((tuple(map(float, box)), float(conf)))

    if iou_thresh:
        for cls, items in groups.items():
            if not items:
                continue
            # → float32 on the SAME device as torchvision
            device = "cuda" if DEVICE == "cuda" else "cpu"
            b = torch.tensor([bb for bb, _ in items],
                             dtype=torch.float32, device=device)
            s = torch.tensor([ss for _, ss in items],
                             dtype=torch.float32, device=device)

            keep = ops.nms(b, s, iou_thresh).tolist()
            groups[cls] = [items[i] for i in keep]

    return groups

def group_boxes_by_class(result,
                          conf_thresh: float = 0.25,
                          iou_thresh: float = 0.45) -> Dict[int, List[Tuple]]:
    """Return {class_id: [(x1,y1,x2,y2), score]} with **native ints**."""
    groups = {int(i): [] for i in range(len(result.names))}

    # ➊ filter boxes
    for box, cls, conf in zip(result.boxes.xyxy.cpu(),
                              result.boxes.cls.cpu(),
                              result.boxes.conf.cpu()):
        if conf < conf_thresh:
            continue
        # -> turn each coordinate into a python int via .item()
        bbox = tuple(int(v.item()) for v in box)
        groups[int(cls.item())].append((bbox, float(conf.item())))

    # ➋ optional per‑class NMS
    if iou_thresh:
        for cid, items in groups.items():
            if not items:
                continue
            b = torch.tensor([bb for bb, _ in items], dtype=torch.float32)
            s = torch.tensor([sc for _, sc in items])
            keep = ops.nms(b, s, iou_thresh)
            groups[cid] = [items[i] for i in keep]

    return groups
# ---------------------------------------------------------------------------- #



def find_containing_dialogue(
    rem_box: Tuple[int, int, int, int],
    dialogues: List[Tuple[Tuple[int, int, int, int], float]]
) -> Optional[Tuple[int, int, int, int]]:
    """
    Given a removal‐region box and a list of dialogue bubbles (each as (box, confidence)),
    return the first dialogue‐box that fully contains rem_box, or None if there is none.
    """
    rx1, ry1, rx2, ry2 = rem_box
    for (dx1, dy1, dx2, dy2), _ in dialogues:
        if rx1 >= dx1 and ry1 >= dy1 and rx2 <= dx2 and ry2 <= dy2:
            return (dx1, dy1, dx2, dy2)
    return None
from simple_lama_inpainting import SimpleLama

@st.cache_resource(ttl=86400, show_spinner=False)
def get_lama_inpainter():
    # This downloads/caches the LaMa weights on first use
    return SimpleLama()

# ---------------------------------------------------------------------------- #
#                          PROCESSING                                          #
# ---------------------------------------------------------------------------- #
def _process_single_panel(
    img: Image.Image,
    src: str,
    tgt: str,
    engine: str,
    gem_tag: str,
    conf: float,
    iou: float,
    color: str
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Exactly the old body of process_page, but working on *one* panel 'img'.
    Returns (translated_panel_image, logs_for_that_panel).
    """
    # ── overlay buffers (collect first, render once at the end) ──────────
    ov_boxes:  list[tuple[int, int, int, int]] = []
    ov_texts:  list[str]                       = []
    ov_sizes:  list[int]                       = []
    ov_colors: list[tuple[int, int, int, int]] = []

    # ── prep  ────────────────────────────────────────────────────────────
    model   = get_yolo_model(conf, iou)
    ocr     = get_ocr()
    np_img  = np.array(img.convert("RGB"))
    out_img = img.copy()
    draw    = ImageDraw.Draw(out_img)
    logs: list[Dict[str, Any]] = []
    text_rgb = ImageColor.getrgb(color)

    # ── detect & group  ──────────────────────────────────────────────────
    det    = model.predict(source=np_img, conf=conf, iou=iou, verbose=False)[0]
    groups = group_boxes_by_class(det, conf_thresh=conf, iou_thresh=iou)

    # ── per-region processing  ───────────────────────────────────────────
    for cls in (SFX, SIGN, TEXT, REMOVAL):
        name = {SFX: "SFX", SIGN: "SIGN", TEXT: "TEXT", REMOVAL: "REMOVAL"}[cls]

        for (x1, y1, x2, y2), _ in groups[cls]:
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

              # ── If this is SFX or SIGN, inpaint the background first ───────
            if cls in (SFX, SIGN,TEXT):
                lama      = get_lama_inpainter()
                panel_np  = np.array(img)                   # full panel as np
                mask      = np.zeros(panel_np.shape[:2], np.uint8)
                mask[y1:y2, x1:x2] = 255
                # call the model via __call__; it may return PIL.Image or ndarray
                inpainted = lama(panel_np, mask)

                # handle both return types:
                if isinstance(inpainted, Image.Image):
                    img = inpainted
                    np_img = np.array(inpainted)
                else:
                    img = Image.fromarray(inpainted)
                    np_img = inpainted

                # reset your drawing canvas on the cleaned panel
                out_img = img.copy()
                draw    = ImageDraw.Draw(out_img)

            # ① crop for OCR (optionally use parent dialogue bubble)
            if cls == REMOVAL and DIALOGUE in groups:
                parent = find_containing_dialogue((x1, y1, x2, y2), groups[DIALOGUE])
                crop_box = parent if parent else (x1, y1, x2, y2)
            else:
                crop_box = (x1, y1, x2, y2)
            sub_img = img.crop(crop_box)

            # ② OCR
            try:
                src_text = ocr(sub_img) or ""
            except Exception:
                src_text = ""

            # ③ translate
            tgt_text = translate(src_text, src, tgt, engine) if src_text else ""

            # ④ log
            logs.append({
                "class":    name,
                "src_lang": src,
                "src_text": src_text,
                "tgt_lang": tgt,
                "tgt_text": tgt_text,
            })

            # ⑤ blank background (white for REMOVAL, median colour otherwise)
            patch = np_img[y1:y2, x1:x2]
            fill  = whitest_pixel(patch) if cls == REMOVAL else None
            draw.rectangle([x1, y1, x2, y2], fill=fill)

            # ⑥ collect overlay info (render later)
            if tgt_text:
                wrapped, font = fit_and_wrap_text(draw, tgt_text, (x1, y1, x2, y2))
                ov_boxes.append((x1, y1, x2, y2))
                ov_texts.append(wrapped)
                ov_sizes.append(font.size)
                ov_colors.append((*text_rgb, 255))

    # ── single overlay render ────────────────────────────────────────────
    if ov_boxes:
        out_img = overlay(out_img, ov_boxes, ov_texts, ov_sizes, ov_colors)

    return out_img, logs


def process_page(
    pil_img: Image.Image,
    src: str,
    tgt: str,
    engine: str,
    gem_tag: str,
    conf: float,
    iou: float,
    color: str = "#0000FF",
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Top-level page processor: detect panels, run per-panel logic, then reassemble.
    """
    # 1) Detect panels first
    panel_boxes = detect_panels_via_roboflow(pil_img)

    # 2) Prepare final image & logs
    final_logs: List[Dict[str, Any]] = []
    final_image = pil_img.copy()

    # 3) Process each panel independently
    for px1, py1, px2, py2 in panel_boxes:
        panel_crop = pil_img.crop((px1, py1, px2, py2))

        translated_panel, panel_logs = _process_single_panel(
            panel_crop, src, tgt, engine, gem_tag,
            conf, iou, color
        )

        # paste translated panel back into page
        final_image.paste(translated_panel, (px1, py1))
        final_logs.extend(panel_logs)

    return final_image, final_logs



# ---------------------------------------------------------------------------- #
#                             STREAMLIT FRONT‑END                              #
# ---------------------------------------------------------------------------- #


def build_streamlit() -> None:
    st.set_page_config("Manga Translator")
    st.title("📖 Manga Translator (Offline · Open Source)")

    # ── sidebar inputs ───────────────────────────────────────────────────────
    with st.sidebar:
        src_lang   = st.text_input("Source language code",  value="ja")
        tgt_lang   = st.text_input("Target language code",  value="en")
        engine     = st.selectbox(
            "Translation engine",
            ["Gemma3","MarianMT", "Google", "DeepL", "Azure", "Argos", "NLLB"],
        )
        conf       = st.slider("YOLO confidence", 0.1, 1.0, 0.25, 0.05)
        iou_thr    = st.slider("NMS IoU threshold", 0.1, 1.0, 0.45, 0.05)
        text_color = st.color_picker("Overlay text color", "#0000FF")

    # ── file input ───────────────────────────────────────────────────────────
    img_file = st.file_uploader("Upload manga page",
                                type=["png", "jpg", "jpeg"])

    # ‼️ everything that touches the file stays **inside** this guard
    if img_file:
        # 1️⃣ preview original
        st.image(img_file, caption="Original Page",
                 use_container_width=True)

        # 2️⃣ run the translation pipeline
        pil_img = Image.open(img_file).convert("RGB")
        with st.spinner("Detecting & Translating …"):
            out_img, logs = process_page(
            pil_img,
            src=src_lang,
            tgt=tgt_lang,
            engine=engine,
            gem_tag="gemma3:4b",
            conf=conf,
            iou=iou_thr,
            color=text_color,
            )

        # 3️⃣ show translated page
        st.image(out_img, caption="Translated",
                 use_container_width=True)

        # 4️⃣ show per‑bubble logs
        for log in logs:
            with st.expander(f"{log['class']}  "
                             f"({log['src_lang']} → {log['tgt_lang']})"):
                st.write("Src:", log["src_text"] or "—")
                st.write("Tgt:", log["tgt_text"] or "—")

    st.caption("All processing is local · models are cached for speed.")


# ---------------------------------------------------------------------------- #
#                               GRADIO FRONT‑END                               #
# ---------------------------------------------------------------------------- #
def build_gradio() -> None:
    import gradio as gr
    def gr_process(img, src_lang, tgt_lang, engine, conf, iou, text_color):
        if img is None:
            return None, "No image"
        # ⬇ REPLACE the old call (src_lang= …) with this:
        pil_img = Image.fromarray(img)
        out_img, logs = process_page(
        pil_img,
        src=src_lang,
        tgt=tgt_lang,
        engine=engine,
        gem_tag="gemma3:4b",
        conf=conf,
        iou=iou,
        color=text_color,
    )

        return np.array(out_img), json.dumps(logs, ensure_ascii=False, indent=2)

    with gr.Blocks(title="Manga Translator") as demo:
        gr.Markdown("# 📖 Manga Translator (Offline · Open Source)")
        with gr.Row():
            img_in  = gr.Image(type="numpy", label="Input page")
            img_out = gr.Image(type="numpy", label="Translated")
        with gr.Row():
            src_lang   = gr.Textbox(value="ja", label="Source lang code")
            tgt_lang   = gr.Textbox(value="en", label="Target lang code")
            engine     = gr.Dropdown(
                ["Gemma3","Google", "DeepL", "Azure", "Argos", "MarianMT", "NLLB"],
                value="Gemma3",
                label="Engine"
            )
            conf       = gr.Slider(0.1, 1.0, 0.25, label="YOLO confidence")
            iou        = gr.Slider(0.1, 1.0, 0.45, label="NMS IoU threshold")
            text_color = gr.ColorPicker(value="#0000FF", label="Overlay text color")
        logs_box = gr.Textbox(label="Logs (JSON)")

        gr.Button("Translate").click(
            gr_process,
            inputs=[img_in, src_lang, tgt_lang, engine, conf, iou, text_color],
            outputs=[img_out, logs_box],
        )

    demo.launch(share=False)


# ---------------------------------------------------------------------------- #
#                                      MAIN                                    #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    ui = os.getenv("WEB_UI", "streamlit")
    ui = ui.lower() if isinstance(ui, str) else "streamlit"
    if ui == "gradio":
        build_gradio()
    else:
        build_streamlit()