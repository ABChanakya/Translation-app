#!/usr/bin/env python3
"""
website.py – Self‑contained manga-page translator **with optional colourization**.

This version now **auto‑installs or discovers** the Manga‑Colorization‑v2 repo so
"ModuleNotFoundError: model" will not appear again.

Run:
    streamlit run website.py         # default Streamlit UI
or:
    WEB_UI=gradio python website.py  # optional Gradio UI
"""

from __future__ import annotations

# ── stdlib / typing ─────────────────────────────────────────────────────────
import os, sys, json, textwrap, base64, tempfile, subprocess, requests
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

# ── third‑party ─────────────────────────────────────────────────────────────
import numpy as np
import torch, torchvision.transforms as T, torchvision.ops as ops
from PIL import Image, ImageDraw, ImageFont, ImageColor
from ultralytics import YOLO
from manga_ocr import MangaOcr
import streamlit as st

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    MarianTokenizer, MarianMTModel,
)
import deepl
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential

try:                      # optional googletrans
    from googletrans import Translator as _GoogleTranslator
except (ImportError, AttributeError):
    _GoogleTranslator = None

import argostranslate.package as _argos_pkg
import argostranslate.translate as _argos_tx

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
YOLO_MODEL    = ROOT/"yolo_train_run/full_finetune_phase20/weights/best.pt"
GEN_WEIGHTS   = ROOT/"colorizer_v2/generator.pth"
EXT_WEIGHTS   = ROOT/"colorizer_v2/extractor.pth"
COLOR_REPO    = ROOT/"manga-colorization-v2"  # will be git‑cloned if missing

CACHE_DIR     = Path(tempfile.gettempdir())/"manga_translator_cache"
CACHE_DIR.mkdir(exist_ok=True)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
FONT_PATH     = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Translation back‑ends
MARIAN_PREFIX = "Helsinki-NLP/opus-mt"
NLLB_ID       = "facebook/nllb-200-distilled-600M"
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY", "")
AZURE_ENDPOINT       = os.getenv("AZURE_ENDPOINT", "")

# YOLO class ids (adapt to your training)
DIALOGUE, SOUND_EFFECTS, SIGNS, TEXT, REMOVAL = range(5)

# ────────────────────────────────────────────────────────────────────────────
# UTILS – Repo auto‑ensure
# ────────────────────────────────────────────────────────────────────────────

def _ensure_color_repo():
    """Clone manga‑colorization‑v2 if not present & add to sys.path."""
    if COLOR_REPO.exists():
        sys.path.append(str(COLOR_REPO))
        return
    # attempt shallow clone (no history) – works offline if repo pre‑downloaded
    try:
        print("[setup] Cloning manga-colorization-v2 …")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/qweasdd/manga-colorization-v2.git",
            str(COLOR_REPO)
        ], check=True)
        sys.path.append(str(COLOR_REPO))
    except Exception as e:
        raise ModuleNotFoundError(
            "Could not find or clone manga-colorization-v2 repo – "
            "please install it manually.\nOrig error: " + str(e)
        )

# ────────────────────────────────────────────────────────────────────────────
# CACHED MODELS
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=86400, show_spinner=False)
def get_yolo(conf=.25, iou=.45):
    m = YOLO(str(YOLO_MODEL))
    m.fuse();
    if DEVICE == "cuda":
        m.to("cuda").half()
    m.predict(conf=conf, iou=iou)
    return m

@st.cache_resource(show_spinner=False)
def get_ocr():
    ocr = MangaOcr();
    if DEVICE == "cuda":
        ocr.model.to("cuda", dtype=torch.float16)
    return ocr

@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def get_colorizer():
    """Return a generator-only model that maps [B,1,H,W]→[B,3,H,W]."""
    _ensure_color_repo()

    model_path = COLOR_REPO / "model"
    if str(COLOR_REPO) not in sys.path:
        sys.path.append(str(COLOR_REPO))
    if str(model_path) not in sys.path:
        sys.path.append(str(model_path))

    from models import Generator, get_seresnext_extractor

    gen = Generator()
    gen.load_state_dict(torch.load(GEN_WEIGHTS, map_location="cpu"))
    ext = get_seresnext_extractor()
    ext.load_state_dict(torch.load(EXT_WEIGHTS, map_location="cpu"))

    class Colorizer(torch.nn.Module):
        def __init__(self, g, e):
            super().__init__()
            self.g, self.e = g.eval(), e.eval()

        @torch.no_grad()
        def forward(self, x):
            return self.g(x, self.e(x))

    col = Colorizer(gen, ext)
    if DEVICE == "cuda":
        col.to("cuda").half()
    return col


# ────────────────────────────────────────────────────────────────────────────
# COLOUR HELPER
# ────────────────────────────────────────────────────────────────────────────

def colorize_page(pil: Image.Image, side=256):
    g = pil.convert("L"); w,h = g.size
    t = T.ToTensor()(g.resize((side, side))).unsqueeze(0)*2-1
    rgb = get_colorizer()(t.to(DEVICE))[0].clamp(-1,1)
    rgb = (rgb+1)*0.5
    out = T.ToPILImage()(rgb.cpu()).resize((w,h), Image.BICUBIC)
    return out

# ────────────────────────────────────────────────────────────────────────────
# … REST OF YOUR ORIGINAL FUNCTIONS (translate, group_boxes_by_class, overlay,
#     process_page, Streamlit & Gradio UIs) – UNCHANGED – KEEP THEM HERE …
# ────────────────────────────────────────────────────────────────────────────

# For brevity they are omitted in this snippet – copy them from previous
# canvas version.

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ui = os.getenv("WEB_UI", "streamlit").lower()
    from website import build_streamlit, build_gradio  # if placed in website.py
    (build_gradio if ui=="gradio" else build_streamlit)()
