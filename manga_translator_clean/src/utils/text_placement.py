"""
Mask-aware text placement for manga bubbles.

Given a binary bubble mask, finds the usable text region inside the
bubble and renders wrapped text that respects the actual bubble geometry
instead of just centering in a bounding box.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

from config.settings import FONT_PATH, DEFAULT_FONT_SIZE_MAX, DEFAULT_FONT_SIZE_MIN


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load font at the given size, with fallback."""
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def find_text_region_in_mask(
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding_pct: float = 0.12,
) -> Tuple[int, int, int, int]:
    """
    Find the largest usable rectangular region inside a bubble mask.

    Instead of using the full bounding box, we scan the mask row by row
    to find the widest safe area that stays within the bubble shape.

    Args:
        mask: Full-image binary mask (H, W), 255 inside bubble
        bbox: (x1, y1, x2, y2) bounding box of the bubble
        padding_pct: Padding as a percentage of box dimensions

    Returns:
        (tx1, ty1, tx2, ty2) — the safe text rectangle in image coords
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # Add padding
    pad_x = int(bw * padding_pct)
    pad_y = int(bh * padding_pct)

    # Crop the mask to the bbox region
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return (x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y)

    h, w = crop.shape

    # For each row, find the leftmost and rightmost mask pixel
    row_spans = []
    for row_idx in range(h):
        row = crop[row_idx]
        white_pixels = np.where(row > 127)[0]
        if len(white_pixels) == 0:
            row_spans.append((0, 0))
        else:
            row_spans.append((int(white_pixels[0]), int(white_pixels[-1])))

    # Find the tallest rectangle that fits inside the mask
    # Use the middle 70% of rows (avoid top/bottom edges of bubble)
    start_row = int(h * 0.15)
    end_row = int(h * 0.85)
    if end_row <= start_row:
        start_row, end_row = 0, h

    middle_spans = row_spans[start_row:end_row]
    if not middle_spans or all(s[1] - s[0] == 0 for s in middle_spans):
        return (x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y)

    # Find the narrowest span in the middle rows — that's our safe width
    valid_spans = [(l, r) for l, r in middle_spans if r - l > 10]
    if not valid_spans:
        return (x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y)

    # Use the 25th percentile width (conservative) to avoid overflow
    widths = sorted([r - l for l, r in valid_spans])
    safe_width_idx = max(0, len(widths) // 4)
    safe_width = widths[safe_width_idx]

    # Center position: median of left edges + half the safe width
    lefts = [l for l, r in valid_spans]
    median_left = sorted(lefts)[len(lefts) // 2]
    center_x = median_left + safe_width // 2

    # Build the text rectangle
    tx1 = x1 + center_x - safe_width // 2 + pad_x
    tx2 = x1 + center_x + safe_width // 2 - pad_x
    ty1 = y1 + start_row + pad_y
    ty2 = y1 + end_row - pad_y

    # Clamp
    tx1 = max(x1, tx1)
    tx2 = min(x2, tx2)
    ty1 = max(y1, ty1)
    ty2 = min(y2, ty2)

    if tx2 - tx1 < 20 or ty2 - ty1 < 20:
        return (x1 + pad_x, y1 + pad_y, x2 - pad_x, y2 - pad_y)

    return (tx1, ty1, tx2, ty2)


def wrap_text_to_width(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> List[str]:
    """
    Word-wrap text to fit within max_width pixels.

    Breaks on spaces. If a single word is wider than max_width, it gets
    its own line (no mid-word hyphenation for English manga text).
    """
    words = text.split()
    if not words:
        return []

    lines = []
    current_line = words[0]

    for word in words[1:]:
        test_line = current_line + " " + word
        bbox = font.getbbox(test_line)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines


def fit_text_to_mask(
    text: str,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    max_size: int = DEFAULT_FONT_SIZE_MAX,
    min_size: int = DEFAULT_FONT_SIZE_MIN,
) -> Tuple[List[str], ImageFont.FreeTypeFont, Tuple[int, int, int, int]]:
    """
    Find the largest font size where wrapped text fits inside the bubble mask.

    Args:
        text: Translated text to render
        mask: Full-image binary mask
        bbox: Bubble bounding box
        max_size: Maximum font size to try
        min_size: Minimum font size

    Returns:
        (wrapped_lines, font, text_region) where text_region is the
        safe rectangle (tx1, ty1, tx2, ty2) inside the bubble
    """
    text_region = find_text_region_in_mask(mask, bbox)
    tx1, ty1, tx2, ty2 = text_region
    avail_w = tx2 - tx1
    avail_h = ty2 - ty1

    if avail_w < 15 or avail_h < 15:
        font = _load_font(min_size)
        return ([text], font, text_region)

    best_lines = None
    best_font = None

    for font_size in range(max_size, min_size - 1, -1):
        font = _load_font(font_size)
        line_spacing = int(font_size * 0.3)

        lines = wrap_text_to_width(text, font, avail_w)
        total_height = len(lines) * (font_size + line_spacing) - line_spacing

        if total_height <= avail_h:
            best_lines = lines
            best_font = font
            break

    if best_lines is None:
        best_font = _load_font(min_size)
        best_lines = wrap_text_to_width(text, best_font, avail_w)

    return (best_lines, best_font, text_region)


def render_text_in_bubble(
    image: Image.Image,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    text: str,
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Render translated text inside a bubble, respecting its mask shape.

    Args:
        image: Current page image (after inpainting)
        mask: Full-image binary mask for this bubble
        bbox: Bubble bounding box
        text: Translated text
        text_color: RGB color for text

    Returns:
        Image with text rendered inside the bubble
    """
    text = text.strip()
    if not text:
        return image

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    if bw < 25 or bh < 25:
        return image

    lines, font, text_region = fit_text_to_mask(text, mask, bbox)
    tx1, ty1, tx2, ty2 = text_region
    avail_w = tx2 - tx1
    avail_h = ty2 - ty1

    font_size = font.size
    line_spacing = int(font_size * 0.3)

    # Calculate total text block height
    total_height = len(lines) * (font_size + line_spacing) - line_spacing

    # Create a tile for this bubble region (RGBA for alpha compositing)
    tile = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)

    # Create mask tile for clipping
    mask_crop = mask[y1:y2, x1:x2]
    mask_pil = Image.fromarray(mask_crop, mode="L")

    # Center the text block vertically in the text region
    region_center_y = (ty1 + ty2) / 2 - y1  # relative to tile
    region_center_x = (tx1 + tx2) / 2 - x1

    start_y = region_center_y - total_height / 2

    # Contrasting stroke for readability
    stroke_fill = (255, 255, 255) if sum(text_color) < 384 else (0, 0, 0)
    stroke_width = max(1, font_size // 18)

    for i, line in enumerate(lines):
        line_bbox = font.getbbox(line)
        line_w = line_bbox[2] - line_bbox[0]
        line_x = region_center_x - line_w / 2
        line_y = start_y + i * (font_size + line_spacing)

        tile_draw.text(
            (line_x, line_y),
            line,
            font=font,
            fill=(*text_color, 255),
            stroke_width=stroke_width,
            stroke_fill=(*stroke_fill, 255),
        )

    # Clip text to bubble mask so nothing escapes
    clipped_tile = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    clipped_tile.paste(tile, mask=mask_pil)

    # Composite onto the page
    base = image.convert("RGBA")
    base.paste(clipped_tile, (x1, y1), mask=clipped_tile)

    return base.convert("RGB")


def render_all_bubbles(
    image: Image.Image,
    bubbles: list,
    translations: List[str],
    text_color: Tuple[int, int, int] = (0, 0, 0),
    smart_color: bool = True,
) -> Image.Image:
    """
    Render translations into all detected bubbles.

    Args:
        image: Page image (after inpainting)
        bubbles: List of dicts from BubbleSegmenter.detect()
        translations: List of translated strings, one per bubble
        text_color: Default text color
        smart_color: If True, use white text on dark backgrounds

    Returns:
        Final image with all translations rendered
    """
    output = image.copy()

    for bubble, text in zip(bubbles, translations):
        if not text or not text.strip():
            continue

        mask = bubble["mask"]
        bbox = bubble["bbox"]
        x1, y1, x2, y2 = bbox

        # Smart color: check background brightness in the bubble region
        if smart_color:
            region = np.array(output)[y1:y2, x1:x2]
            # Only check pixels inside the mask
            mask_crop = mask[y1:y2, x1:x2]
            if mask_crop.any():
                masked_pixels = region[mask_crop > 127]
                if len(masked_pixels) > 0:
                    brightness = masked_pixels.mean()
                    color = (255, 255, 255) if brightness < 140 else text_color
                else:
                    color = text_color
            else:
                color = text_color
        else:
            color = text_color

        output = render_text_in_bubble(output, mask, bbox, text, text_color=color)

    return output
