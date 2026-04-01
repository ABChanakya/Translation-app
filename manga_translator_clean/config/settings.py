"""Configuration settings for the Manga Translator application."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

# ═══════════════════════════════════════════════════════════════════════════
#                           YOLO MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CANDIDATES = [
    PROJECT_ROOT / "yolo_train_run" / "full_finetune_phase40" / "weights" / "best.pt",
    PROJECT_ROOT / "yolo_train_run" / "full_finetune_60_20" / "weights" / "best.pt",
    PROJECT_ROOT / "models" / "checkpoints" / "custom_yolo_best.pt",
]


def _resolve_yolo_model_path() -> str:
    env_override = os.getenv("YOLO_MODEL_PATH")
    if env_override:
        candidate = Path(env_override)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return str(candidate)

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    return str(DEFAULT_MODEL_CANDIDATES[0])


# Prefer the 5-class checkpoint when available so inference matches the dataset config.
YOLO_MODEL_PATH = _resolve_yolo_model_path()

# Text region types detected by your model
class TextRegionType:
    """Text region classes detected by the YOLO model"""
    DIALOGUE = 0
    SOUND_EFFECTS = 1
    SIGNS = 2
    TEXT = 3
    REMOVAL = 4


# ═══════════════════════════════════════════════════════════════════════════
#                           PROCESSING SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image processing
MAX_IMAGE_SIZE = 1600  # Max dimension for processing (larger = slower but more accurate)

# Detection thresholds
DEFAULT_CONFIDENCE = 0.10  # Lower default to catch more bubbles
DEFAULT_IOU_THRESHOLD = 0.55  # Stricter NMS to limit duplicates

# Cache directory
CACHE_DIR = os.path.join(tempfile.gettempdir(), "manga_translator_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Default training/evaluation dataset
DEFAULT_DATASET_YAML = str((PROJECT_ROOT / "training" / "datasets" / "custom_manga.yaml").resolve())


# ═══════════════════════════════════════════════════════════════════════════
#                           TRANSLATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

# API Keys (read from environment variables)
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")

# Translation model identifiers
MARIAN_MODEL_PREFIX = "Helsinki-NLP/opus-mt"
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

# Gemma3 settings (via Ollama)
GEMMA_MODEL = "gemma3:12b"  # Using the 12B model for better quality
GEMMA_KEEP_ALIVE = "1h"


# ═══════════════════════════════════════════════════════════════════════════
#                           TEXT RENDERING SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

# Font configuration
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# 60 was far too large — a 60px font overflows small speech bubbles and
# bleeds into artwork. 28 is the practical upper bound for manga bubble text.
DEFAULT_FONT_SIZE_MAX = 28
DEFAULT_FONT_SIZE_MIN = 8
FONT_SIZE_STEP = 2

# Default text color — black reads cleanly on the white background of
# cleared speech bubbles. Blue was jarring over manga art.
DEFAULT_TEXT_COLOR = "#000000"


# ═══════════════════════════════════════════════════════════════════════════
#                           UI SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

# Streamlit configuration
STREAMLIT_PAGE_TITLE = "Auto Manga Translation"
STREAMLIT_PAGE_ICON = "📖"
STREAMLIT_LAYOUT = "wide"
PRODUCT_NAME = "Auto Manga Translation"

# Available translation engines
AVAILABLE_ENGINES = [
    "Gemma3",
    "Google",
    "DeepL",
    "Argos",
    "MarianMT",
    "NLLB",
    "Azure",
]

# Default engine
DEFAULT_ENGINE = "Gemma3"


# ═══════════════════════════════════════════════════════════════════════════
#                           LAMA INPAINTING SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

# Which region types should use LaMa inpainting
USE_LAMA_FOR_REGIONS = [
    TextRegionType.DIALOGUE,       # Enable for dialogue bubbles!
    TextRegionType.SOUND_EFFECTS,
    TextRegionType.SIGNS,
    TextRegionType.TEXT
]

# External LaMa microservice endpoint
LAMA_SERVICE_URL = os.getenv("LAMA_SERVICE_URL", "http://127.0.0.1:5001")

# Model cache TTL (seconds)
MODEL_CACHE_TTL = 86400  # 24 hours


# ═══════════════════════════════════════════════════════════════════════════
#                           SETTINGS CLASS
# ═══════════════════════════════════════════════════════════════════════════

class Settings:
    """Wrapper class for all configuration settings"""
    
    # YOLO Model Configuration
    YOLO_MODEL_PATH = YOLO_MODEL_PATH
    TextRegionType = TextRegionType
    
    # Processing Settings
    DEVICE = DEVICE
    MAX_IMAGE_SIZE = MAX_IMAGE_SIZE
    DEFAULT_CONFIDENCE = DEFAULT_CONFIDENCE
    DEFAULT_IOU_THRESHOLD = DEFAULT_IOU_THRESHOLD
    CACHE_DIR = CACHE_DIR
    PROJECT_ROOT = str(PROJECT_ROOT)
    DEFAULT_DATASET_YAML = DEFAULT_DATASET_YAML
    
    # Translation Settings
    DEEPL_API_KEY = DEEPL_API_KEY
    AZURE_TRANSLATOR_KEY = AZURE_TRANSLATOR_KEY
    AZURE_ENDPOINT = AZURE_ENDPOINT
    MARIAN_MODEL_PREFIX = MARIAN_MODEL_PREFIX
    NLLB_MODEL_ID = NLLB_MODEL_ID
    GEMMA_MODEL = GEMMA_MODEL
    GEMMA_KEEP_ALIVE = GEMMA_KEEP_ALIVE
    
    # Text Rendering Settings
    FONT_PATH = FONT_PATH
    DEFAULT_FONT_SIZE_MAX = DEFAULT_FONT_SIZE_MAX
    DEFAULT_FONT_SIZE_MIN = DEFAULT_FONT_SIZE_MIN
    FONT_SIZE_STEP = FONT_SIZE_STEP
    DEFAULT_TEXT_COLOR = DEFAULT_TEXT_COLOR
    
    # UI Settings
    STREAMLIT_PAGE_TITLE = STREAMLIT_PAGE_TITLE
    STREAMLIT_PAGE_ICON = STREAMLIT_PAGE_ICON
    STREAMLIT_LAYOUT = STREAMLIT_LAYOUT
    PRODUCT_NAME = PRODUCT_NAME
    AVAILABLE_ENGINES = AVAILABLE_ENGINES
    DEFAULT_ENGINE = DEFAULT_ENGINE
    
    # LAMA Inpainting Settings
    USE_LAMA_FOR_REGIONS = USE_LAMA_FOR_REGIONS
    LAMA_SERVICE_URL = LAMA_SERVICE_URL
    MODEL_CACHE_TTL = MODEL_CACHE_TTL
