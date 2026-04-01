"""Client wrapper around the external LaMa inpainting service."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import requests
from PIL import Image

from config.settings import LAMA_SERVICE_URL

# How many pixels to expand the mask beyond the detected bounding box.
# Manga text often bleeds 2-4 px outside the YOLO box; expanding the mask
# ensures clean removal without leaving a fringe of original characters.
_MASK_EXPAND_PX = 8


def _clamp_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    """Ensure the mask box stays inside image bounds."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x1 >= x2 or y1 >= y2:
        return 0, 0, 0, 0
    return x1, y1, x2, y2


@dataclass
class TextInpainter:
    """Proxy that forwards inpainting requests to a dedicated LaMa service."""

    service_url: str = LAMA_SERVICE_URL.rstrip("/")
    # Number of consecutive failures before giving up for the rest of this run
    _fail_count: int = field(default=0, init=False, repr=False)
    _max_failures: int = field(default=3, init=False, repr=False)

    def __post_init__(self) -> None:
        self.available = self._check_health()
        if self.available:
            print(f"✅ LaMa service detected at {self.service_url}")
        else:
            print("⚠️ LaMa service not reachable - falling back to flat fills")

    def _check_health(self) -> bool:
        try:
            response = requests.get(f"{self.service_url}/health", timeout=3)
            if response.status_code == 200 and response.json().get("model_loaded"):
                return True
        except Exception as exc:  # noqa: broad-except - connectivity issues expected
            print(f"⚠️ Unable to contact LaMa service: {exc}")
        return False

    def try_reconnect(self) -> bool:
        """Re-check the LaMa service health. Call this if LaMa was started after the pipeline."""
        self.available = self._check_health()
        self._fail_count = 0
        if self.available:
            print(f"✅ LaMa service reconnected at {self.service_url}")
        return self.available

    def inpaint_region(self, image: Image.Image, mask_box: Tuple[int, int, int, int]) -> Image.Image:
        """Request background inpainting for the provided region.

        The mask is expanded by _MASK_EXPAND_PX pixels on each side so that
        character strokes that slightly overflow the YOLO bounding box are
        also removed cleanly.
        """
        if not self.available:
            return image

        width, height = image.size

        # Expand the mask beyond the bounding box for cleaner removal
        x1, y1, x2, y2 = mask_box
        expanded = _clamp_box(
            (x1 - _MASK_EXPAND_PX, y1 - _MASK_EXPAND_PX,
             x2 + _MASK_EXPAND_PX, y2 + _MASK_EXPAND_PX),
            width, height,
        )
        if expanded == (0, 0, 0, 0):
            return image
        ex1, ey1, ex2, ey2 = expanded

        mask_array = np.zeros((height, width), dtype=np.uint8)
        mask_array[ey1:ey2, ex1:ex2] = 255

        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        mask_image = Image.fromarray(mask_array)
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_buffer.seek(0)

        files = {
            "image": ("image.png", image_buffer, "image/png"),
            "mask": ("mask.png", mask_buffer, "image/png"),
        }

        try:
            response = requests.post(f"{self.service_url}/inpaint", files=files, timeout=60)
            if response.status_code != 200:
                print(f"⚠️ LaMa service response {response.status_code}: {response.text}")
                self._fail_count += 1
                if self._fail_count >= self._max_failures:
                    print("⚠️ LaMa service disabled after repeated failures")
                    self.available = False
                return image

            self._fail_count = 0  # reset on success
            result_image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return result_image
        except Exception as exc:  # noqa: broad-except - network failures possible
            print(f"⚠️ LaMa inpainting failed: {exc}")
            self._fail_count += 1
            if self._fail_count >= self._max_failures:
                print("⚠️ LaMa service disabled after repeated failures")
                self.available = False
            return image
