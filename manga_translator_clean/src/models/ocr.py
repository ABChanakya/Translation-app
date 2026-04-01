"""
Manga OCR model for Japanese text extraction.
"""

import os

# Apply PyTorch patch FIRST before any torch imports
try:
    from . import torch_patch
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import torch_patch

# Allow torch.load with weights_only=True (workaround for PyTorch 2.5.x security requirement)
# This is safe because we're loading trusted models from HuggingFace
# Must be set BEFORE importing torch
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

import torch
from manga_ocr import MangaOcr
from functools import lru_cache

from config.settings import DEVICE, MODEL_CACHE_TTL


@lru_cache(maxsize=1)
def load_manga_ocr():
    """
    Load and cache the Manga OCR model.
    
    This model is specifically trained on manga text and handles:
    - Vertical text
    - Stylized fonts
    - Small text
    - Text with effects
    
    Returns:
        Loaded MangaOcr model
    """
    print("📖 Loading Manga OCR model...")
    
    # Set HuggingFace to use local files only (avoid auth issues)
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    try:
        ocr = MangaOcr(force_cpu=False)
        
        if DEVICE == "cuda":
            ocr.model.to("cuda", dtype=torch.float16)
            print("✅ OCR model loaded on GPU")
        else:
            print("✅ OCR model loaded on CPU")
        
        return ocr
        
    except Exception as e:
        print(f"⚠️ Error loading manga-ocr in offline mode: {e}")
        print("💡 Trying online mode (may require HuggingFace login)...")
        
        # Try online mode
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        os.environ.pop('HF_DATASETS_OFFLINE', None)
        
        try:
            ocr = MangaOcr(force_cpu=(DEVICE != "cuda"))
            if DEVICE == "cuda":
                ocr.model.to("cuda", dtype=torch.float16)
                print("✅ OCR model loaded from HuggingFace")
            else:
                print("✅ OCR model loaded on CPU from HuggingFace")
            return ocr
        except Exception as e2:
            raise RuntimeError(
                f"❌ Could not load manga-ocr model.\n"
                f"Error: {e2}\n\n"
                f"Solutions:\n"
                f"1. Run: huggingface-cli login\n"
                f"2. See: /home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/QUICKSTART.md (OCR Model Access)"
            )


class OCRExtractor:
    """Wrapper class for Manga OCR"""
    
    def __init__(self):
        self.model = load_manga_ocr()
    
    def extract_text(self, image):
        """
        Extract Japanese text from an image.
        
        Args:
            image: PIL Image
        
        Returns:
            Extracted text string (or empty string if no text found)
        """
        try:
            text = self.model(image)
            return text if text else ""
        except Exception as e:
            print(f"⚠️ OCR extraction failed: {e}")
            return ""
