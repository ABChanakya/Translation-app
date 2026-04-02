"""
Text fitting and rendering utilities.
"""

import textwrap
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont

from config.settings import (
    FONT_PATH,
    DEFAULT_FONT_SIZE_MAX,
    DEFAULT_FONT_SIZE_MIN,
    FONT_SIZE_STEP
)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Load TrueType font or fall back to default.
    
    Args:
        size: Font size in points
    
    Returns:
        ImageFont object
    """
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def fit_text_to_box(
    draw: ImageDraw.Draw,
    text: str,
    box: Tuple[int, int, int, int],
    max_size: int = DEFAULT_FONT_SIZE_MAX,
    min_size: int = DEFAULT_FONT_SIZE_MIN,
    step: int = FONT_SIZE_STEP
) -> Tuple[str, ImageFont.FreeTypeFont]:
    """
    Fit text into a bounding box by finding the largest font size
    and wrapping text across multiple lines if needed.
    
    Args:
        draw: ImageDraw context
        text: Text to fit
        box: (x1, y1, x2, y2) bounding box
        max_size: Maximum font size to try
        min_size: Minimum font size
        step: Font size decrement step
    
    Returns:
        Tuple of (wrapped_text, font)
    """
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Try progressively smaller fonts
    for font_size in range(max_size, min_size - 1, -step):
        font = load_font(font_size)
        
        # Estimate characters per line
        m_bbox = draw.textbbox((0, 0), "M", font=font)
        char_width = max(1, m_bbox[2] - m_bbox[0])
        max_chars_per_line = max(1, box_width // char_width)
        
        # Wrap text
        wrapped_text = "\n".join(
            textwrap.wrap(
                text,
                width=max_chars_per_line,
                break_long_words=False,
                break_on_hyphens=False
            )
        )
        
        # Measure wrapped text
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Check if it fits
        if text_width <= box_width and text_height <= box_height:
            return wrapped_text, font
    
    # Fallback: use minimum size
    font = load_font(min_size)
    m_bbox = draw.textbbox((0, 0), "M", font=font)
    char_width = max(1, m_bbox[2] - m_bbox[0])
    wrapped_text = "\n".join(
        textwrap.wrap(text, width=max(1, box_width // char_width))
    )
    return wrapped_text, font


def render_text_overlay(
    base_image: Image.Image,
    boxes: list,
    texts: list,
    font_sizes: list,
    colors: list
) -> Image.Image:
    """
    Render multiple text strings onto an image, each clipped to its bounding box.

    Each translation is drawn into a small tile the exact size of the detection
    box, then composited back at the correct position. This prevents any text
    from bleeding into adjacent panels or artwork outside the bubble.

    Args:
        base_image: PIL Image
        boxes: List of (x1, y1, x2, y2) tuples
        texts: List of text strings
        font_sizes: List of font sizes
        colors: List of RGBA color tuples

    Returns:
        PIL Image with text overlay composited
    """
    base = base_image.convert("RGBA")

    for (x1, y1, x2, y2), text, font_size, color in zip(
        boxes, texts, font_sizes, colors
    ):
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        # Tile sized exactly to the detection box — text cannot escape it
        tile = Image.new("RGBA", (box_w, box_h), (255, 255, 255, 0))
        tile_draw = ImageDraw.Draw(tile)

        font = load_font(font_size)
        cx = box_w // 2
        cy = box_h // 2

        # Contrasting stroke improves readability on screentones and dark backgrounds
        stroke_fill = (255, 255, 255) if color[:3] == (0, 0, 0) else (0, 0, 0)
        stroke_width = max(1, font_size // 20)
        tile_draw.multiline_text(
            (cx, cy),
            text,
            font=font,
            fill=color,
            anchor="mm",
            align="center",
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

        # Paste the tile back at the correct position on the base image
        base.paste(tile, (x1, y1), mask=tile)

    return base.convert("RGB")
