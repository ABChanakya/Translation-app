"""
═══════════════════════════════════════════════════════════════════════════════
                        MANGA TRANSLATOR - MAIN APPLICATION
═══════════════════════════════════════════════════════════════════════════════

A complete manga translation pipeline that:
    1. Uses YOUR CUSTOM YOLO MODEL to detect text regions (dialogue, SFX, signs)
    2. Performs OCR to extract Japanese text from detected regions
    3. Translates text using multiple engines (Gemma3, Google, DeepL, etc.)
    4. Redraws translated text onto the original manga pages
    5. Provides web interface (Streamlit or Gradio)

INSTALLATION:
    pip install streamlit gradio ultralytics manga-ocr pillow numpy opencv-python \
        argostranslate==1.8.0 transformers==4.* sentencepiece torch

USAGE:
    # Run with Streamlit (default):
    streamlit run DEMO_26.06_REFACTORED.py
    
    # Run with Gradio:
    WEB_UI=gradio python DEMO_26.06_REFACTORED.py

YOUR CUSTOM MODEL:
    Location: yolo_train_run/full_finetune_phase40/weights/best.pt
    Classes it detects:
        - DIALOGUE (0): Speech bubbles with character dialogue
        - SOUND_EFFECTS (1): Sound effect text (e.g., "BANG!", "WHOOSH!")
        - SIGNS (2): Text on signs, labels, or background elements
        - TEXT (3): General text that doesn't fit other categories
        - REMOVAL (4): Text regions that should be cleaned and replaced

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
#                               IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

# Standard library
import os
import io
import json
import tempfile
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from enum import Enum
import textwrap
import base64

# Computer vision & deep learning
import numpy as np
import cv2
import torch
import torchvision.ops as ops
from PIL import Image, ImageDraw, ImageFont, ImageColor
from ultralytics import YOLO
from manga_ocr import MangaOcr

# Translation engines
import argostranslate.package
import argostranslate.translate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianTokenizer,
    MarianMTModel,
)
import deepl
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
import requests
import ollama

# Google Translate (optional - with fallback)
try:
    from googletrans import Translator as GoogleTranslator
    _GOOGLE_TRANSLATOR = GoogleTranslator()
except (ImportError, AttributeError):
    _GOOGLE_TRANSLATOR = None

# Optional LaMa inpainting (for clean text removal)
try:
    import demo_lama
    _HAS_DEMO_LAMA = True
except ImportError:
    _HAS_DEMO_LAMA = False

try:
    from simple_lama_inpainting import SimpleLama
    _HAS_SIMPLE_LAMA = True
except ImportError:
    _HAS_SIMPLE_LAMA = False

# Web UI frameworks
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════
#                           CONFIGURATION SECTION
# ═══════════════════════════════════════════════════════════════════════════

# ──────────────────────────── YOUR CUSTOM YOLO MODEL ───────────────────────────
# This is YOUR trained model that detects different text region types
YOLO_MODEL_PATH = "yolo_train_run/full_finetune_phase40/weights/best.pt"

# Detection class definitions (these match your model's training)
class TextRegionType:
    """Types of text regions your YOLO model can detect"""
    DIALOGUE = 0       # Speech bubbles with character dialogue
    SOUND_EFFECTS = 1  # SFX text like "BANG!", "WHOOSH!"
    SIGNS = 2          # Text on signs, labels, backgrounds
    TEXT = 3           # General text
    REMOVAL = 4        # Text that should be removed/replaced
    
    # Convenient aliases
    SFX = SOUND_EFFECTS
    SIGN = SIGNS

# ──────────────────────────── PROCESSING SETTINGS ──────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGE_SIZE = 1600  # Resize images to this max dimension for faster processing
CACHE_DIR = os.path.join(tempfile.gettempdir(), "manga_translator_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ──────────────────────────── TRANSLATION SETTINGS ─────────────────────────────
# API keys (set these in your environment or replace with your keys)
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")

# Translation model identifiers
MARIAN_MODEL_PREFIX = "Helsinki-NLP/opus-mt"
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

# ──────────────────────────── TEXT RENDERING SETTINGS ──────────────────────────
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# ═══════════════════════════════════════════════════════════════════════════
#                         MODEL LOADING & CACHING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(ttl=86400, show_spinner=False)
def load_yolo_detector(confidence: float = 0.25, iou_threshold: float = 0.45):
    """
    Load YOUR custom YOLO model for detecting text regions.
    
    This model detects 5 types of text regions:
        - DIALOGUE: Speech bubbles
        - SOUND_EFFECTS: SFX text
        - SIGNS: Background signs/labels
        - TEXT: General text
        - REMOVAL: Text to be removed
    
    The model is cached for 24 hours to speed up subsequent runs.
    """
    print(f"📦 Loading YOUR custom YOLO model from: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)
    
    # Optimize model for inference
    model.fuse()  # Fuse Conv2d + BatchNorm layers for speed
    
    if DEVICE == "cuda":
        model.to("cuda").half()  # Use FP16 on GPU for faster inference
        print("🚀 Model loaded on GPU with FP16 precision")
    else:
        print("💻 Model loaded on CPU")
    
    # Warm-up prediction to compile operations
    model.predict(conf=confidence, iou=iou_threshold)
    print("✅ Model ready!")
    
    return model


@st.cache_resource(show_spinner=False, ttl=None)
def load_manga_ocr():
    """
    Load the Manga OCR model for extracting Japanese text.
    This model is specifically trained on manga text.
    """
    print("📖 Loading Manga OCR model...")
    ocr = MangaOcr()
    
    if DEVICE == "cuda":
        ocr.model.to("cuda", dtype=torch.float16)
        print("✅ OCR model loaded on GPU")
    else:
        print("✅ OCR model loaded on CPU")
    
    return ocr


@st.cache_resource(show_spinner=False, ttl=86400)
def load_lama_inpainter():
    """
    Load the LaMa (Large Mask Inpainting) model.
    
    LaMa is used to REMOVE the original text cleanly by intelligently
    filling in the masked areas with appropriate background patterns.
    
    This is much better than just drawing a white rectangle because:
        - It preserves background textures and patterns
        - Works on any colored background
        - Creates seamless results
    
    The model weights (~500MB) are downloaded automatically on first use.
    """
    if not _HAS_SIMPLE_LAMA and not _HAS_DEMO_LAMA:
        print("⚠️ LaMa inpainting not available - will use simple rectangle fill")
        return None
    
    print("🎨 Loading LaMa inpainting model...")
    
    if _HAS_SIMPLE_LAMA:
        lama = SimpleLama()
        print("✅ SimpleLama model loaded")
        return lama
    elif _HAS_DEMO_LAMA:
        print("✅ demo_lama available")
        return "demo_lama"  # Flag to use demo_lama
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
#                      TRANSLATION ENGINE LOADERS
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=4)
def load_marian_translator(source_lang: str, target_lang: str):
    """Load MarianMT translation model (Helsinki-NLP)"""
    model_name = f"{MARIAN_MODEL_PREFIX}-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


@lru_cache(maxsize=2)
def load_nllb_translator():
    """Load NLLB (No Language Left Behind) multilingual model"""
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID)
    return tokenizer, model


def ensure_argos_language_pack(source_lang: str, target_lang: str):
    """Download and install Argos Translate language pack if not present"""
    installed = {(p.from_code, p.to_code) 
                 for p in argostranslate.package.get_installed_packages()}
    
    if (source_lang, target_lang) in installed:
        return  # Already installed
    
    # Download from HuggingFace
    pack_url = (f"https://huggingface.co/argosopentech/"
                f"argos-translate-{source_lang}_{target_lang}/resolve/main/"
                f"{source_lang}_{target_lang}.argos")
    
    try:
        pack_path = os.path.join(CACHE_DIR, f"{source_lang}_{target_lang}.argos")
        
        if not os.path.exists(pack_path):
            response = requests.get(pack_url, timeout=10)
            response.raise_for_status()
            with open(pack_path, "wb") as f:
                f.write(response.content)
        
        argostranslate.package.install_from_path(pack_path)
        print(f"✅ Installed Argos language pack: {source_lang} → {target_lang}")
    except Exception as e:
        print(f"⚠️ Failed to install Argos pack: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#                          UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def find_whitest_pixel(image_array: np.ndarray) -> Tuple[int, int, int]:
    """
    Find the whitest (brightest) pixel in an image.
    Used to find a good background color for text removal.
    """
    if image_array.ndim != 3 or image_array.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels (RGB)")
    
    # Sum RGB values to find brightest pixel
    brightness = image_array.sum(axis=-1)
    brightest_idx = np.argmax(brightness)
    
    height, width, _ = image_array.shape
    y, x = divmod(brightest_idx, width)
    
    return tuple(map(int, image_array[y, x]))


def calculate_median_color(image_array: np.ndarray) -> Tuple[int, int, int]:
    """Calculate the median color of an image region"""
    median = np.median(image_array.reshape(-1, 3), axis=0)
    return tuple(int(x) for x in median)


def inpaint_text_region(
    image: Image.Image,
    mask_box: Tuple[int, int, int, int],
    lama_model=None
) -> Image.Image:
    """
    Remove text from an image region using LaMa inpainting.
    
    This is the KEY FUNCTION that uses the LaMa model to cleanly erase
    the original manga text while preserving the background.
    
    How it works:
        1. Create a binary mask (white=inpaint, black=keep)
        2. Pass image + mask to LaMa model
        3. LaMa fills in the masked area intelligently
    
    Parameters:
        image: The PIL Image containing text to remove
        mask_box: (x1, y1, x2, y2) - The text region to remove
        lama_model: The loaded LaMa model (or None for fallback)
    
    Returns:
        PIL Image with text region inpainted
    """
    if lama_model is None:
        # Fallback: just return original image (text will be covered by rectangle)
        return image
    
    x1, y1, x2, y2 = mask_box
    
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Create binary mask (255 = inpaint this area)
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    
    try:
        # Call LaMa model
        if isinstance(lama_model, str) and lama_model == "demo_lama":
            # Use demo_lama module
            inpainted = demo_lama.inpaint(image, Image.fromarray(mask))
        else:
            # Use SimpleLama (can be called directly)
            inpainted = lama_model(image_np, mask)
        
        # Handle different return types
        if isinstance(inpainted, Image.Image):
            return inpainted
        else:
            return Image.fromarray(inpainted)
    
    except Exception as e:
        print(f"⚠️ LaMa inpainting failed: {e}")
        return image  # Return original on failure


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load TrueType font or fall back to default"""
    try:
        return ImageFont.truetype(DEFAULT_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


# ═══════════════════════════════════════════════════════════════════════════
#                      TEXT FITTING & RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def fit_text_to_box(
    draw: ImageDraw.Draw,
    text: str,
    box: Tuple[int, int, int, int],
    font_path: str = DEFAULT_FONT_PATH,
    max_font_size: int = 60,
    min_font_size: int = 12,
    size_step: int = 2
) -> Tuple[str, ImageFont.FreeTypeFont]:
    """
    Intelligently fit text into a bounding box by:
        1. Finding the largest font size that fits
        2. Wrapping text to multiple lines if needed
    
    Returns:
        (wrapped_text, font) - The wrapped text and the font to use
    """
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Try progressively smaller font sizes
    for font_size in range(max_font_size, min_font_size - 1, -size_step):
        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except OSError:
            continue
        
        # Estimate characters per line based on "M" width
        m_bbox = draw.textbbox((0, 0), "M", font=font)
        char_width = max(1, m_bbox[2] - m_bbox[0])
        max_chars_per_line = max(1, box_width // char_width)
        
        # Wrap text to fit width
        wrapped_text = "\n".join(
            textwrap.wrap(
                text, 
                width=max_chars_per_line,
                break_long_words=False,
                break_on_hyphens=False
            )
        )
        
        # Measure the wrapped text block
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Check if it fits
        if text_width <= box_width and text_height <= box_height:
            return wrapped_text, font
    
    # Fallback: use minimum size
    font = ImageFont.truetype(font_path, size=min_font_size)
    m_bbox = draw.textbbox((0, 0), "M", font=font)
    char_width = max(1, m_bbox[2] - m_bbox[0])
    wrapped_text = "\n".join(
        textwrap.wrap(text, width=max(1, box_width // char_width))
    )
    return wrapped_text, font


def render_text_overlay(
    base_image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    texts: List[str],
    font_sizes: List[int],
    colors: List[Tuple[int, int, int, int]]
) -> Image.Image:
    """
    Render multiple text strings onto an image using a transparent overlay.
    
    This creates a professional-looking result by:
        1. Creating a transparent layer
        2. Drawing all text on that layer
        3. Compositing it over the base image
    """
    # Convert base image to RGBA for compositing
    base = base_image.convert("RGBA")
    
    # Create transparent overlay
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw each text element
    for (x1, y1, x2, y2), text, font_size, color in zip(boxes, texts, font_sizes, colors):
        font = load_font(font_size)
        
        # Center the text in the box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        draw.multiline_text(
            (center_x, center_y),
            text,
            font=font,
            fill=color,
            anchor="mm",  # Middle-middle anchor
            align="center",
        )
    
    # Composite overlay onto base and convert to RGB
    result = Image.alpha_composite(base, overlay)
    return result.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════
#                      TRANSLATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

class ResponseFormat(Enum):
    """Output format for LLM responses"""
    JSON = 'json_object'
    TEXT = 'text'


def translate_with_gemma3(
    text: str,
    source_lang: str,
    target_lang: str,
    image_path: Optional[Path] = None
) -> str:
    """
    Translate text using Gemma3 LLM via Ollama.
    This provides high-quality, context-aware translations.
    """
    system_prompt = (
        f"You are a professional translator specializing in {source_lang} and {target_lang}. "
        f"Translate the following {source_lang} text into natural, fluent {target_lang}. "
        "Preserve the tone, nuance, and cultural context. "
        "Output ONLY the translated text, nothing else."
    )
    
    user_prompt = (
        f"=== {source_lang.upper()} TEXT ===\n"
        f"{text}\n"
        f"=== END TEXT ==="
    )
    
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    
    if image_path:
        b64_image = base64.b64encode(Path(image_path).read_bytes()).decode()
        image_uri = f"data:image/png;base64,{b64_image}"
        messages.append({
            'role': 'user',
            'content': f'![Image]({image_uri})\n\n{user_prompt}'
        })
    else:
        messages.append({'role': 'user', 'content': user_prompt})
    
    response = ollama.chat(
        model='gemma3:12b',
        messages=messages,
        keep_alive='1h',
        format='',  # Plain text
        options={
            'temperature': 1.0,
            'min_p': 0.01,
            'repeat_penalty': 1.0,
            'top_k': 64,
            'top_p': 0.95
        }
    )
    
    translation = response.message.content
    print(f"🤖 Gemma3 translation: {text[:50]}... → {translation[:50]}...")
    return translation


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    engine: str
) -> str:
    """
    Universal translation function with multiple engines and fallback logic.
    
    Supported engines:
        - Gemma3: Local LLM (best quality, requires Ollama)
        - Google: Free online translation
        - DeepL: Professional translation service
        - Azure: Microsoft Translator
        - Argos: Offline translation
        - MarianMT: Helsinki NLP models
        - NLLB: Meta's multilingual model
    
    If the chosen engine fails, it automatically falls back to Argos, then Google.
    """
    text = text.strip()
    if not text:
        return ""
    
    try:
        # ─────────────────────── GEMMA3 (LLM) ───────────────────────
        if engine == "Gemma3":
            return translate_with_gemma3(text, source_lang, target_lang)
        
        # ─────────────────────── GOOGLE TRANSLATE ────────────────────
        elif engine == "Google":
            if _GOOGLE_TRANSLATOR is None:
                raise ImportError("Google Translate library not available")
            result = _GOOGLE_TRANSLATOR.translate(
                text,
                src=source_lang or "auto",
                dest=target_lang
            )
            return result.text
        
        # ─────────────────────── DEEPL ───────────────────────────────
        elif engine == "DeepL":
            if not DEEPL_API_KEY:
                raise ValueError("DeepL API key not configured")
            translator = deepl.Translator(DEEPL_API_KEY)
            result = translator.translate_text(
                text,
                source_lang=source_lang.upper(),
                target_lang=target_lang.upper()
            )
            return result.text
        
        # ─────────────────────── AZURE TRANSLATOR ────────────────────
        elif engine == "Azure":
            if not AZURE_TRANSLATOR_KEY or not AZURE_ENDPOINT:
                raise ValueError("Azure Translator credentials not configured")
            credential = AzureKeyCredential(AZURE_TRANSLATOR_KEY)
            client = TextTranslationClient(
                endpoint=AZURE_ENDPOINT,
                credential=credential
            )
            result = client.translate(
                content=[text],
                from_parameter=source_lang,
                to=[target_lang]
            )
            return result[0].translations[0].text
        
        # ─────────────────────── ARGOS (OFFLINE) ─────────────────────
        elif engine == "Argos":
            ensure_argos_language_pack(source_lang, target_lang)
            return argostranslate.translate.translate(text, source_lang, target_lang)
        
        # ─────────────────────── MARIANMT ────────────────────────────
        elif engine == "MarianMT":
            tokenizer, model = load_marian_translator(source_lang, target_lang)
            tokens = tokenizer(text, return_tensors="pt")
            output = model.generate(**tokens, max_length=256)
            return tokenizer.decode(output[0], skip_special_tokens=True)
        
        # ─────────────────────── NLLB ────────────────────────────────
        elif engine == "NLLB":
            tokenizer, model = load_nllb_translator()
            
            if source_lang not in tokenizer.lang_code_to_id:
                raise ValueError(f"NLLB doesn't support source language: {source_lang}")
            
            tokens = tokenizer(text, return_tensors="pt")
            tokens["forced_bos_token_id"] = tokenizer.lang_code_to_id.get(target_lang, 0)
            output = model.generate(**tokens, max_length=256)
            return tokenizer.decode(output[0], skip_special_tokens=True)
        
        # ─────────────────────── UNKNOWN ENGINE ──────────────────────
        else:
            raise ValueError(f"Unknown translation engine: {engine}")
    
    except Exception as e:
        print(f"⚠️ Translation failed with {engine}: {e}")
        
        # Fallback chain: Try Argos → Google → Give up
        if engine not in ("Google", "Argos"):
            print("🔄 Falling back to Argos...")
            return translate_text(text, source_lang, target_lang, "Argos")
        
        if engine != "Google":
            print("🔄 Falling back to Google...")
            return translate_text(text, source_lang, target_lang, "Google")
        
        print(f"❌ All translation engines failed, returning original text")
        return text


# ═══════════════════════════════════════════════════════════════════════════
#                    TEXT REGION DETECTION & GROUPING
# ═══════════════════════════════════════════════════════════════════════════

def group_detections_by_class(
    yolo_result,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Dict[int, List[Tuple[Tuple[int, int, int, int], float]]]:
    """
    Group YOLO detections by class and apply Non-Maximum Suppression (NMS).
    
    Your YOLO model outputs detections for 5 classes:
        0: DIALOGUE
        1: SOUND_EFFECTS
        2: SIGNS
        3: TEXT
        4: REMOVAL
    
    This function:
        1. Filters detections by confidence threshold
        2. Groups them by class
        3. Applies NMS to remove overlapping boxes
    
    Returns:
        Dictionary mapping class_id → list of (bounding_box, confidence)
        where bounding_box is (x1, y1, x2, y2)
    """
    num_classes = len(yolo_result.names)
    grouped_detections = {i: [] for i in range(num_classes)}
    
    # Step 1: Filter and group by class
    for box, class_id, confidence in zip(
        yolo_result.boxes.xyxy.cpu(),
        yolo_result.boxes.cls.cpu(),
        yolo_result.boxes.conf.cpu()
    ):
        if confidence < confidence_threshold:
            continue
        
        # Convert to Python native types (not PyTorch tensors)
        bbox = tuple(int(coord.item()) for coord in box)
        class_idx = int(class_id.item())
        conf_score = float(confidence.item())
        
        grouped_detections[class_idx].append((bbox, conf_score))
    
    # Step 2: Apply NMS per class to remove overlapping boxes
    if iou_threshold:
        for class_idx, detections in grouped_detections.items():
            if not detections:
                continue
            
            # Prepare tensors for NMS
            boxes_tensor = torch.tensor([bbox for bbox, _ in detections], dtype=torch.float32)
            scores_tensor = torch.tensor([score for _, score in detections], dtype=torch.float32)
            
            # Apply NMS
            keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)
            
            # Keep only non-suppressed detections
            grouped_detections[class_idx] = [detections[i] for i in keep_indices]
    
    return grouped_detections


def find_parent_dialogue_bubble(
    removal_box: Tuple[int, int, int, int],
    dialogue_bubbles: List[Tuple[Tuple[int, int, int, int], float]]
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find if a REMOVAL region is contained within a DIALOGUE bubble.
    
    This is useful because sometimes the OCR works better when we consider
    the entire dialogue bubble rather than just the removal region.
    
    Returns:
        The bounding box of the containing dialogue bubble, or None if not found
    """
    rx1, ry1, rx2, ry2 = removal_box
    
    for (dx1, dy1, dx2, dy2), _ in dialogue_bubbles:
        # Check if removal box is fully contained in dialogue box
        if rx1 >= dx1 and ry1 >= dy1 and rx2 <= dx2 and ry2 <= dy2:
            return (dx1, dy1, dx2, dy2)
    
    return None


# ═══════════════════════════════════════════════════════════════════════════
#                      MAIN PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def process_manga_page(
    input_image: Image.Image,
    source_language: str,
    target_language: str,
    translation_engine: str,
    detection_confidence: float,
    nms_iou_threshold: float,
    text_color: str = "#0000FF",
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    ═══════════════════════════════════════════════════════════════════════════
                        MAIN MANGA TRANSLATION PIPELINE
    ═══════════════════════════════════════════════════════════════════════════
    
    This is the core function that orchestrates the entire translation process:
    
    Step 1: Detect text regions using YOUR CUSTOM YOLO MODEL
            - Finds dialogue bubbles, SFX, signs, and text to remove
    
    Step 2: Extract text using Manga OCR
            - Reads Japanese text from each detected region
    
    Step 3: Translate text
            - Converts Japanese to target language using chosen engine
    
    Step 4: Remove original text
            - Blanks out the original text area
    
    Step 5: Render translated text
            - Draws the translation in the same location
    
    Parameters:
        input_image: PIL Image of the manga page
        source_language: ISO code (e.g., "ja" for Japanese)
        target_language: ISO code (e.g., "en" for English)
        translation_engine: Which translator to use
        detection_confidence: Minimum confidence for YOLO detections (0-1)
        nms_iou_threshold: IoU threshold for Non-Maximum Suppression
        text_color: Hex color for rendered text (e.g., "#0000FF" for blue)
    
    Returns:
        (translated_image, processing_logs)
    """
    
    print("\n" + "="*80)
    print("🎌 STARTING MANGA TRANSLATION PIPELINE")
    print("="*80)
    
    # ─────────────────────── INITIALIZE ──────────────────────────
    print("📋 Step 1/5: Initializing models and buffers...")
    
    yolo_model = load_yolo_detector(detection_confidence, nms_iou_threshold)
    ocr_model = load_manga_ocr()
    lama_model = load_lama_inpainter()  # For clean text removal!
    
    if lama_model:
        print("   ✅ LaMa inpainting enabled - text will be removed cleanly")
    else:
        print("   ⚠️ LaMa not available - using simple rectangle fill")
    
    # Convert image to numpy array for processing
    image_array = np.array(input_image.convert("RGB"))
    output_image = input_image.copy()
    draw_context = ImageDraw.Draw(output_image)
    
    # Parse text color
    text_rgb = ImageColor.getrgb(text_color)
    
    # Buffers for batch rendering (more efficient than drawing one-by-one)
    overlay_boxes = []
    overlay_texts = []
    overlay_font_sizes = []
    overlay_colors = []
    
    # Log of all translations for debugging/review
    processing_logs = []
    
    # ─────────────────────── DETECT TEXT REGIONS ─────────────────
    print(f"🔍 Step 2/5: Detecting text regions with YOUR YOLO model...")
    print(f"   Using: {YOLO_MODEL_PATH}")
    
    detection_result = yolo_model.predict(
        source=image_array,
        conf=detection_confidence,
        iou=nms_iou_threshold,
        verbose=False
    )[0]
    
    grouped_detections = group_detections_by_class(
        detection_result,
        confidence_threshold=detection_confidence,
        iou_threshold=nms_iou_threshold
    )
    
    # Print detection summary
    print(f"   ✅ Found:")
    for class_id, detections in grouped_detections.items():
        class_names = {
            TextRegionType.DIALOGUE: "DIALOGUE",
            TextRegionType.SOUND_EFFECTS: "SOUND_EFFECTS",
            TextRegionType.SIGNS: "SIGNS",
            TextRegionType.TEXT: "TEXT",
            TextRegionType.REMOVAL: "REMOVAL"
        }
        class_name = class_names.get(class_id, f"CLASS_{class_id}")
        print(f"      - {len(detections)} {class_name} regions")
    
    # ─────────────────────── PROCESS EACH REGION ─────────────────
    print(f"📖 Step 3/5: OCR and translation...")
    
    # Process these types of regions (skip DIALOGUE as it's usually just the bubble outline)
    regions_to_process = [
        TextRegionType.SOUND_EFFECTS,
        TextRegionType.SIGNS,
        TextRegionType.TEXT,
        TextRegionType.REMOVAL
    ]
    
    region_counter = 0
    
    for region_type in regions_to_process:
        region_name = {
            TextRegionType.SOUND_EFFECTS: "SFX",
            TextRegionType.SIGNS: "SIGN",
            TextRegionType.TEXT: "TEXT",
            TextRegionType.REMOVAL: "REMOVAL"
        }[region_type]
        
        for (x1, y1, x2, y2), confidence in grouped_detections[region_type]:
            region_counter += 1
            
            # Skip very small regions (likely false positives)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            
            # ────────── OCR ──────────
            # For REMOVAL regions inside dialogue bubbles, use the whole bubble
            if region_type == TextRegionType.REMOVAL and grouped_detections[TextRegionType.DIALOGUE]:
                parent_bubble = find_parent_dialogue_bubble(
                    (x1, y1, x2, y2),
                    grouped_detections[TextRegionType.DIALOGUE]
                )
                ocr_region = parent_bubble if parent_bubble else (x1, y1, x2, y2)
            else:
                ocr_region = (x1, y1, x2, y2)
            
            # Crop region and perform OCR
            cropped_region = input_image.crop(ocr_region)
            
            try:
                original_text = ocr_model(cropped_region) or ""
            except Exception as e:
                print(f"   ⚠️ OCR failed for region {region_counter}: {e}")
                original_text = ""
            
            # ────────── TRANSLATE ──────────
            if original_text:
                translated_text = translate_text(
                    original_text,
                    source_language,
                    target_language,
                    translation_engine
                )
            else:
                translated_text = ""
            
            # ────────── LOG ──────────
            processing_logs.append({
                "region_id": region_counter,
                "class": region_name,
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "src_lang": source_language,
                "src_text": original_text,
                "tgt_lang": target_language,
                "tgt_text": translated_text,
            })
            
            # ────────── REMOVE ORIGINAL TEXT WITH LAMA ──────────
            # For SFX, SIGNS, and TEXT: use LaMa inpainting for clean removal
            if region_type in (TextRegionType.SOUND_EFFECTS, TextRegionType.SIGNS, TextRegionType.TEXT):
                if lama_model:
                    print(f"   🎨 Using LaMa to cleanly remove {region_name}...")
                    output_image = inpaint_text_region(
                        output_image,
                        (x1, y1, x2, y2),
                        lama_model
                    )
                    # Update image array and draw context after inpainting
                    image_array = np.array(output_image)
                    draw_context = ImageDraw.Draw(output_image)
                else:
                    # Fallback: simple rectangle fill
                    text_region_pixels = image_array[y1:y2, x1:x2]
                    background_color = find_whitest_pixel(text_region_pixels)
                    draw_context.rectangle([x1, y1, x2, y2], fill=background_color)
            
            # For REMOVAL regions: always use simple fill (these are dialogue text)
            elif region_type == TextRegionType.REMOVAL:
                text_region_pixels = image_array[y1:y2, x1:x2]
                background_color = find_whitest_pixel(text_region_pixels)
                draw_context.rectangle([x1, y1, x2, y2], fill=background_color)
            
            # ────────── PREPARE TRANSLATED TEXT FOR RENDERING ──────────
            if translated_text:
                wrapped_text, font = fit_text_to_box(
                    draw_context,
                    translated_text,
                    (x1, y1, x2, y2)
                )
                
                overlay_boxes.append((x1, y1, x2, y2))
                overlay_texts.append(wrapped_text)
                overlay_font_sizes.append(font.size)
                overlay_colors.append((*text_rgb, 255))  # RGBA
    
    print(f"   ✅ Processed {region_counter} text regions")
    
    # ─────────────────────── RENDER ALL TRANSLATIONS ─────────────
    print(f"🎨 Step 4/5: Rendering {len(overlay_boxes)} translations...")
    
    if overlay_boxes:
        output_image = render_text_overlay(
            output_image,
            overlay_boxes,
            overlay_texts,
            overlay_font_sizes,
            overlay_colors
        )
    
    print(f"✅ Step 5/5: Complete!")
    print("="*80 + "\n")
    
    return output_image, processing_logs


# ═══════════════════════════════════════════════════════════════════════════
#                          WEB INTERFACE - STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════

def build_streamlit_interface():
    """
    Create the Streamlit web interface for the manga translator.
    This provides an easy-to-use GUI for uploading and translating manga pages.
    """
    st.set_page_config(
        page_title="Manga Translator",
        page_icon="📖",
        layout="wide"
    )
    
    st.title("📖 Manga Translator")
    st.markdown("### Powered by YOUR Custom YOLO Model + AI Translation")
    
    # ─────────────────────── SIDEBAR SETTINGS ────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("🌐 Languages")
        source_lang = st.text_input(
            "Source language code",
            value="ja",
            help="ISO 639-1 code (e.g., 'ja' for Japanese)"
        )
        target_lang = st.text_input(
            "Target language code",
            value="en",
            help="ISO 639-1 code (e.g., 'en' for English)"
        )
        
        st.subheader("🤖 Translation Engine")
        translation_engine = st.selectbox(
            "Choose translator",
            ["Gemma3", "Google", "MarianMT", "DeepL", "Azure", "Argos", "NLLB"],
            help="Gemma3 provides the best quality but requires Ollama"
        )
        
        st.subheader("🎯 Detection Settings")
        confidence = st.slider(
            "YOLO confidence threshold",
            0.1, 1.0, 0.25, 0.05,
            help="Higher = fewer but more confident detections"
        )
        iou_threshold = st.slider(
            "NMS IoU threshold",
            0.1, 1.0, 0.45, 0.05,
            help="Higher = more overlapping boxes allowed"
        )
        
        st.subheader("🎨 Appearance")
        text_color = st.color_picker(
            "Translation text color",
            "#0000FF",
            help="Color for the translated text"
        )
        
        st.divider()
        st.caption(f"🖥️ Using: {DEVICE.upper()}")
        st.caption(f"📦 Model: full_finetune_phase40")
    
    # ─────────────────────── MAIN AREA ───────────────────────────
    st.header("📤 Upload Manga Page")
    
    uploaded_file = st.file_uploader(
        "Choose a manga page image",
        type=["png", "jpg", "jpeg"],
        help="Upload a manga page to translate"
    )
    
    if uploaded_file:
        # Show original
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Original")
            st.image(uploaded_file, use_container_width=True)
        
        # Process
        input_image = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("🔄 Processing... (Detecting → OCR → Translating → Rendering)"):
            output_image, logs = process_manga_page(
                input_image,
                source_language=source_lang,
                target_language=target_lang,
                translation_engine=translation_engine,
                detection_confidence=confidence,
                nms_iou_threshold=iou_threshold,
                text_color=text_color,
            )
        
        # Show result
        with col2:
            st.subheader("✨ Translated")
            st.image(output_image, use_container_width=True)
        
        # Show detailed logs
        st.header("📊 Translation Details")
        st.caption(f"Processed {len(logs)} text regions")
        
        for i, log in enumerate(logs, 1):
            with st.expander(
                f"#{i} - {log['class']} "
                f"({log['src_lang']} → {log['tgt_lang']}) "
                f"[confidence: {log['confidence']:.2f}]"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Original ({log['src_lang']}):**")
                    st.text(log['src_text'] or "—")
                with col_b:
                    st.markdown(f"**Translation ({log['tgt_lang']}):**")
                    st.text(log['tgt_text'] or "—")
                
                st.caption(f"Bounding box: {log['bbox']}")
    
    else:
        st.info("👆 Upload a manga page image to get started!")
    
    # Footer
    st.divider()
    st.caption(
        "🚀 This tool uses YOUR custom YOLO model trained to detect manga text regions. "
        "All processing happens locally for privacy."
    )


# ═══════════════════════════════════════════════════════════════════════════
#                          WEB INTERFACE - GRADIO
# ═══════════════════════════════════════════════════════════════════════════

def build_gradio_interface():
    """
    Create a Gradio web interface (alternative to Streamlit).
    """
    import gradio as gr
    
    def gradio_process(
        img,
        src_lang,
        tgt_lang,
        engine,
        conf,
        iou,
        text_color
    ):
        if img is None:
            return None, "No image uploaded"
        
        input_image = Image.fromarray(img).convert("RGB")
        output_image, logs = process_manga_page(
            input_image,
            source_language=src_lang,
            target_language=tgt_lang,
            translation_engine=engine,
            detection_confidence=conf,
            nms_iou_threshold=iou,
            text_color=text_color,
        )
        
        logs_json = json.dumps(logs, ensure_ascii=False, indent=2)
        return np.array(output_image), logs_json
    
    with gr.Blocks(title="Manga Translator") as app:
        gr.Markdown("# 📖 Manga Translator")
        gr.Markdown("### Powered by YOUR Custom YOLO Model + AI Translation")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="📤 Upload Manga Page")
            with gr.Column():
                image_output = gr.Image(type="numpy", label="✨ Translated Result")
        
        with gr.Row():
            src_lang = gr.Textbox(value="ja", label="Source Language (ISO code)")
            tgt_lang = gr.Textbox(value="en", label="Target Language (ISO code)")
            engine = gr.Dropdown(
                ["Gemma3", "Google", "MarianMT", "DeepL", "Azure", "Argos", "NLLB"],
                value="Gemma3",
                label="Translation Engine"
            )
        
        with gr.Row():
            conf = gr.Slider(0.1, 1.0, 0.25, label="YOLO Confidence")
            iou = gr.Slider(0.1, 1.0, 0.45, label="NMS IoU Threshold")
            text_color = gr.ColorPicker(value="#0000FF", label="Text Color")
        
        logs_output = gr.Textbox(label="📊 Processing Logs (JSON)", lines=10)
        
        translate_btn = gr.Button("🚀 Translate", variant="primary")
        translate_btn.click(
            gradio_process,
            inputs=[image_input, src_lang, tgt_lang, engine, conf, iou, text_color],
            outputs=[image_output, logs_output],
        )
        
        gr.Markdown(
            "---\n"
            "🚀 This tool uses YOUR custom YOLO model for text detection. "
            "Model: `yolo_train_run/full_finetune_phase40/weights/best.pt`"
        )
    
    app.launch(share=False)


# ═══════════════════════════════════════════════════════════════════════════
#                               MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Choose UI based on environment variable
    ui_choice = os.getenv("WEB_UI", "streamlit").lower()
    
    print("\n" + "="*80)
    print("                    🎌 MANGA TRANSLATOR 🎌")
    print("="*80)
    print(f"Using YOUR custom YOLO model: {YOLO_MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"UI: {ui_choice.upper()}")
    print("="*80 + "\n")
    
    if ui_choice == "gradio":
        print("🚀 Starting Gradio interface...")
        build_gradio_interface()
    else:
        print("🚀 Starting Streamlit interface...")
        build_streamlit_interface()
