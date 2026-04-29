"""
Smart OCR preprocessing and confidence checking.
Improves manga-ocr accuracy with better crops and retry logic.
"""

import cv2
import json
import unicodedata
import numpy as np
from pathlib import Path
from typing import Tuple
from PIL import Image


def preprocess_for_ocr(
    page: np.ndarray,
    text_bbox: Tuple[int, int, int, int]
) -> Image.Image:
    """
    Preprocesses a region for manga-ocr.
    Handles upscaling, adaptive thresholding, and contrast enhancement.

    Args:
        page: BGR image array
        text_bbox: (x1, y1, x2, y2) bounding box

    Returns:
        PIL Image (grayscale binary) ready for OCR
    """
    x1, y1, x2, y2 = text_bbox

    # Add padding around text region
    padding = 8
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(page.shape[1], x2 + padding)
    y2 = min(page.shape[0], y2 + padding)

    crop = page[y1:y2, x1:x2].copy()

    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale if crop is too small
    h, w = gray.shape
    min_dim = 64
    if h < min_dim or w < min_dim:
        scale = max(min_dim / h, min_dim / w)
        new_h, new_w = int(h * scale), int(w * scale)
        gray = cv2.resize(
            gray,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

    # Adaptive thresholding for low-contrast text
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Check polarity: text should be black on white for OCR
    if binary.mean() < 128:
        binary = cv2.bitwise_not(binary)

    return Image.fromarray(binary)


def remove_furigana(crop_image: Image.Image) -> Image.Image:
    """
    Detects and removes furigana (small ruby text) before OCR.
    Furigana columns are narrower than main text columns.

    Args:
        crop_image: PIL Image of crop region

    Returns:
        PIL Image with furigana blanked out
    """
    gray = np.array(crop_image.convert('L'))
    h, w = gray.shape

    if h < 20 or w < 20:
        # Too small to reliably detect furigana
        return crop_image

    # Vertical projection: count dark pixels per column
    col_sums = (gray < 128).sum(axis=0)

    if col_sums.max() == 0:
        return crop_image

    # Find columns with significant text
    threshold = col_sums.max() * 0.1
    text_cols = col_sums > threshold

    # Find connected groups of columns
    groups = []
    in_group = False
    start = 0

    for i, is_text in enumerate(text_cols):
        if is_text and not in_group:
            start = i
            in_group = True
        elif not is_text and in_group:
            if i - start > 2:  # Only groups wider than 2px
                groups.append((start, i, i - start))
            in_group = False

    if len(groups) < 2:
        # Not enough groups to identify furigana
        return crop_image

    # Furigana columns are narrower than main text
    widths = [g[2] for g in groups]
    median_width = sorted(widths)[len(widths) // 2]

    # Blank out columns narrower than 40% of median
    result = gray.copy()
    for start, end, width in groups:
        if width < median_width * 0.4:
            result[:, start:end] = 255  # White out furigana

    return Image.fromarray(result)


def _is_japanese_text(text: str) -> bool:
    """Check if text contains valid Japanese characters."""
    for char in text:
        # CJK Unified Ideographs and Japanese ranges
        code = ord(char)
        # Hiragana, Katakana, CJK Unified Ideographs
        if (0x3040 <= code <= 0x309F or  # Hiragana
            0x30A0 <= code <= 0x30FF or  # Katakana
            0x4E00 <= code <= 0x9FFF or  # CJK Unified
            0x3400 <= code <= 0x4DBF):   # CJK Extension A
            return True
    return False


def ocr_with_confidence(
    manga_ocr_model,
    crop: Image.Image,
    min_confidence: float = 0.3
) -> Tuple[str, float]:
    """
    Runs manga-ocr and estimates confidence.
    Retries with different preprocessing if confidence is low.

    Args:
        manga_ocr_model: manga-ocr model with __call__ method
        crop: PIL Image of text region
        min_confidence: Minimum confidence to accept result

    Returns:
        (ocr_text, confidence_score)
    """
    # Primary attempt
    result = manga_ocr_model(crop)
    result = result.strip()

    # Estimate confidence by checking for Japanese characters
    japanese_count = sum(1 for c in result if _is_japanese_text(c))
    total_count = len(result)

    if total_count == 0:
        confidence = 0.0
    else:
        confidence = japanese_count / total_count

    # If confidence is low and we have text, retry with inverted
    if confidence < min_confidence and total_count > 0:
        # Invert the image
        inverted_arr = cv2.bitwise_not(np.array(crop))
        inverted = Image.fromarray(inverted_arr)

        result_retry = manga_ocr_model(inverted)
        result_retry = result_retry.strip()

        japanese_retry = sum(1 for c in result_retry if _is_japanese_text(c))
        confidence_retry = (japanese_retry / len(result_retry)) if len(result_retry) > 0 else 0.0

        if confidence_retry > confidence:
            result = result_retry
            confidence = confidence_retry

    return result, confidence


def log_ocr_result(
    bubble_idx: int,
    bbox: Tuple[int, int, int, int],
    ocr_text: str,
    confidence: float,
    log_dir: str = "."
) -> None:
    """
    Logs OCR results for monitoring and debugging.

    Args:
        bubble_idx: Region index
        bbox: Bounding box
        ocr_text: Extracted text
        confidence: Confidence score
        log_dir: Directory to write ocr_log.jsonl
    """
    log_path = Path(log_dir) / "ocr_log.jsonl"

    entry = {
        "bubble_idx": bubble_idx,
        "bbox": list(bbox),
        "ocr_text": ocr_text,
        "confidence": float(confidence),
        "flagged": confidence < 0.5
    }

    # Append to log file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if confidence < 0.3:
        print(f"⚠️  Low OCR confidence ({confidence:.2f}) on bubble {bubble_idx}: '{ocr_text}'")


def ocr_region_with_preprocessing(
    manga_ocr_model,
    page: np.ndarray,
    text_bbox: Tuple[int, int, int, int],
    bubble_idx: int,
    log_dir: str = "."
) -> Tuple[str, float]:
    """
    Complete OCR pipeline: preprocess → remove furigana → OCR → log.

    Args:
        manga_ocr_model: manga-ocr model
        page: BGR image
        text_bbox: Bounding box
        bubble_idx: Region index for logging
        log_dir: Directory for ocr_log.jsonl

    Returns:
        (ocr_text, confidence)
    """
    # Preprocess
    crop = preprocess_for_ocr(page, text_bbox)

    # Remove furigana
    crop = remove_furigana(crop)

    # OCR with confidence checking
    text, confidence = ocr_with_confidence(manga_ocr_model, crop)

    # Log
    log_ocr_result(bubble_idx, text_bbox, text, confidence, log_dir)

    return text, confidence
