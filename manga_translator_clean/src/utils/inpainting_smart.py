"""
Smart inpainting with background detection and artifact reduction.
Handles screentone, white, and artwork backgrounds differently.
"""

import cv2
import numpy as np
from typing import Literal, Tuple
from PIL import Image


BackgroundType = Literal['white', 'light', 'screentone', 'artwork']


def create_inpainting_mask(
    page: np.ndarray,
    text_bbox: Tuple[int, int, int, int],
    bubble_mask: np.ndarray
) -> np.ndarray:
    """
    Creates a tight, clean mask for inpainting.
    Only masks the actual text pixels, not the whole bubble.

    Args:
        page: BGR image array
        text_bbox: (x1, y1, x2, y2) bounding box
        bubble_mask: Full-image binary mask of bubble shape

    Returns:
        Binary mask (H, W) of text strokes to inpaint
    """
    x1, y1, x2, y2 = text_bbox

    # Extract region
    region = page[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Threshold to find dark pixels (Japanese text is black)
    _, text_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Morphological closing to connect broken strokes
    kernel = np.ones((3, 3), np.uint8)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dilate slightly to ensure full text coverage
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)

    # Intersect with bubble mask to avoid touching border
    # Erode bubble mask by 3px before intersecting
    bubble_crop = bubble_mask[y1:y2, x1:x2]
    bubble_eroded = cv2.erode(bubble_crop, kernel, iterations=3)
    text_mask = cv2.bitwise_and(text_mask, bubble_eroded)

    # Place back into full-page mask
    full_mask = np.zeros(page.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = text_mask

    return full_mask


def detect_bubble_background(
    page: np.ndarray,
    bubble_mask: np.ndarray
) -> BackgroundType:
    """
    Detects the type of background inside a bubble.

    Returns:
        'white', 'light', 'screentone', or 'artwork'
    """
    # Extract bubble interior pixels
    region = cv2.bitwise_and(page, page, mask=bubble_mask)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Get non-zero pixels only
    pixels = gray[bubble_mask > 0]

    if len(pixels) == 0:
        return 'white'

    mean_brightness = float(pixels.mean())
    std_brightness = float(pixels.std())

    # Clean white bubble
    if mean_brightness > 230 and std_brightness < 20:
        return 'white'

    # Light background
    if mean_brightness > 180 and std_brightness < 50:
        return 'light'

    # Check for screentone (periodic pattern)
    # Use FFT to detect peaks away from center
    try:
        f = np.fft.fft2(gray.astype(float))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = magnitude.shape
        # Center region (DC component + low frequencies)
        center_region = magnitude[max(0, h//2-20):min(h, h//2+20),
                                  max(0, w//2-20):min(w, w//2+20)]
        center_max = float(center_region.max()) if center_region.size > 0 else 0
        outer_max = float(magnitude.max())

        # Strong periodic pattern: outer peaks >> center
        if outer_max > center_max * 2.5:
            return 'screentone'
    except Exception:
        pass

    return 'artwork'


def inpaint_white_bubble(
    page: np.ndarray,
    text_mask: np.ndarray
) -> np.ndarray:
    """Simple white fill for clean white bubbles."""
    result = page.copy()
    result[text_mask > 0] = 255
    return result


def inpaint_screentone(
    page: np.ndarray,
    bubble_mask: np.ndarray,
    text_mask: np.ndarray
) -> np.ndarray:
    """
    For screentone bubbles, sample the pattern from clean regions
    and tile it over text areas.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find clean region of bubble (not text)
    clean_region_mask = cv2.bitwise_and(
        bubble_mask,
        cv2.bitwise_not(text_mask)
    )

    # Sample a patch from clean region
    moments = cv2.moments(clean_region_mask)
    if moments['m00'] == 0:
        # No clean region, fall back to simple fill
        return inpaint_white_bubble(page, text_mask)

    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    patch_size = 32

    # Extract patch, with bounds checking
    py1 = max(0, cy - patch_size // 2)
    py2 = min(h, cy + patch_size // 2)
    px1 = max(0, cx - patch_size // 2)
    px2 = min(w, cx + patch_size // 2)
    patch = gray[py1:py2, px1:px2]

    if patch.size == 0:
        return inpaint_white_bubble(page, text_mask)

    # Tile the patch across the page
    tiled = np.tile(patch, (h // patch_size + 3, w // patch_size + 3))
    tiled = tiled[:h, :w]
    tiled_bgr = cv2.cvtColor(tiled, cv2.COLOR_GRAY2BGR)

    # Apply tiled pattern where text_mask is active
    result = page.copy()
    result[text_mask > 0] = tiled_bgr[text_mask > 0]

    # Feather edges for smooth blending
    feather_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    feather_mask = cv2.GaussianBlur(text_mask.astype(float), (7, 7), 2) / 255.0

    result = (
        result.astype(float) * feather_mask[:, :, np.newaxis] +
        page.astype(float) * (1 - feather_mask[:, :, np.newaxis])
    ).astype(np.uint8)

    return result


def post_inpainting_cleanup(
    page: np.ndarray,
    bubble_mask: np.ndarray,
    background_type: BackgroundType
) -> np.ndarray:
    """
    Removes small artifacts after inpainting.
    """
    if background_type not in ('white', 'light'):
        return page

    result = page.copy()
    region = cv2.bitwise_and(result, result, mask=bubble_mask)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Find dark pixels that shouldn't be there
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, bubble_mask)

    # Remove small isolated dark blobs (artifacts)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 50:  # Small artifact
            component_mask = (labels == i).astype(np.uint8) * 255
            result[component_mask > 0] = 255

    return result


def smart_inpaint_bubble(
    page: np.ndarray,
    bubble_mask: np.ndarray,
    text_bbox: Tuple[int, int, int, int],
    lama_inpainter=None
) -> Tuple[np.ndarray, BackgroundType]:
    """
    Smart inpainting strategy based on background type.

    Args:
        page: BGR image
        bubble_mask: Full-image binary mask
        text_bbox: (x1, y1, x2, y2)
        lama_inpainter: Optional LaMa inpainter object with inpaint_region() method

    Returns:
        (inpainted_page, background_type)
    """
    # Create tight inpainting mask
    text_mask = create_inpainting_mask(page, text_bbox, bubble_mask)

    # Detect background type
    background = detect_bubble_background(page, bubble_mask)

    # Inpaint based on background
    if background == 'white':
        result = inpaint_white_bubble(page, text_mask)
    elif background == 'light':
        # Try LaMa if available, else white fill
        if lama_inpainter and lama_inpainter.available:
            try:
                result = lama_inpainter.inpaint_region(page, text_bbox)
            except Exception:
                result = inpaint_white_bubble(page, text_mask)
        else:
            result = inpaint_white_bubble(page, text_mask)
    elif background == 'screentone':
        result = inpaint_screentone(page, bubble_mask, text_mask)
    else:  # artwork
        if lama_inpainter and lama_inpainter.available:
            try:
                result = lama_inpainter.inpaint_region(page, text_bbox)
            except Exception:
                result = inpaint_white_bubble(page, text_mask)
        else:
            result = inpaint_white_bubble(page, text_mask)

    # Post-processing cleanup
    result = post_inpainting_cleanup(result, bubble_mask, background)

    return result, background
