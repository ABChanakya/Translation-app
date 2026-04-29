"""
Bubble segmentation using YOLOv8-seg.

Returns pixel-level masks for each speech bubble, not just bounding boxes.
This allows text to be placed inside the actual bubble shape instead of
forcing it into a rectangle.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
from functools import lru_cache

from config.settings import DEVICE


# Default path — can be overridden
_DEFAULT_SEG_MODEL = str(
    Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "yolov8m_seg_bubble.pt"
)


@lru_cache(maxsize=1)
def _load_seg_model(model_path: str):
    """Load and cache the YOLOv8-seg model."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    if DEVICE == "cuda":
        model.to("cuda")
    return model


class BubbleSegmenter:
    """Detect speech bubbles and return pixel-level masks."""

    def __init__(
        self,
        model_path: str = _DEFAULT_SEG_MODEL,
        confidence: float = 0.30,
        iou_threshold: float = 0.50,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.available = Path(model_path).exists()

        if self.available:
            try:
                self.model = _load_seg_model(model_path)
                print(f"✅ Bubble segmenter loaded: {Path(model_path).name}")
            except Exception as e:
                print(f"⚠️  Bubble segmenter failed to load: {e}")
                self.available = False

    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Detect bubbles and return masks.

        Args:
            image: PIL Image (RGB)

        Returns:
            List of dicts, each with:
                bbox: (x1, y1, x2, y2) in original image coords
                mask: np.ndarray (H, W) binary uint8, same size as input image
                confidence: float
                contour: np.ndarray from cv2.findContours (largest contour)
        """
        if not self.available:
            return []

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        results = self.model.predict(
            source=img_array,
            conf=self.confidence,
            iou=self.iou_threshold,
            max_det=50,
            verbose=False,
        )
        result = results[0]

        if result.masks is None or len(result.boxes) == 0:
            return []

        bubbles = []
        for i in range(len(result.boxes)):
            # Bounding box in original image coords
            box = result.boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            conf = float(result.boxes.conf[i])

            # Mask: model returns at reduced resolution, resize to original
            raw_mask = result.masks.data[i].cpu().numpy()  # (mask_h, mask_w)
            mask_resized = cv2.resize(
                raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

            # Clean up the mask
            binary_mask = self._clean_mask(binary_mask)

            # Find the largest contour
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area < 400:  # Skip tiny artifacts
                continue

            bubbles.append({
                "bbox": (x1, y1, x2, y2),
                "mask": binary_mask,
                "confidence": conf,
                "contour": largest_contour,
                "area": area,
            })

        # Sort by reading order: top-to-bottom, right-to-left (manga order)
        bubbles.sort(key=lambda b: (b["bbox"][1] // 100, -b["bbox"][0]))

        return bubbles

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """Morphological cleanup: fill holes, smooth edges."""
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)

        return mask
