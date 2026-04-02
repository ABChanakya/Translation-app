"""
Tier 2 VLM Translation Pipeline.

Simpler alternative to the main pipeline (Tier 1). Uses YOLO for bounding-box
detection, then sends each cropped region to Gemma 3 vision for combined
OCR + translation in a single call. No manga-ocr dependency needed.

Tradeoff vs Tier 1:
  - ~20-30% lower OCR accuracy (VLM vs manga-ocr)
  - Simpler code path, no separate OCR model
  - Useful for handwritten/unusual fonts where manga-ocr struggles

Usage:
    from src.vlm_pipeline import VLMTranslationPipeline
    pipeline = VLMTranslationPipeline(model="gemma3:12b")
    output_image, logs = pipeline.process(page_image)
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from config.settings import GEMMA_MODEL, USE_LAMA_FOR_REGIONS
from src.models.detector import TextDetector
from src.models.inpainter import TextInpainter
from src.pipeline import group_detections_by_class, resolve_detection_class_names
from src.utils.image import find_whitest_pixel
from src.utils.text import fit_text_to_box, render_text_overlay


class VLMTranslationPipeline:
    """
    Tier 2: YOLO bounding boxes + Gemma 3 vision per region + LaMa inpainting.

    Each detected text region is cropped, padded, and sent to Gemma 3 vision
    with a "read and translate" prompt. The response is parsed for a
    「Japanese」→"English" pair. Inpainting and text rendering then follow
    the same three-pass approach as the Tier 1 pipeline.
    """

    # Concise prompt — long prompts trigger Gemma3's infinite-repetition bug
    _REGION_PROMPT = (
        "Read the Japanese text in this image. Ignore furigana (small phonetic characters). "
        "Translate it to English. "
        'Reply only with: 「Japanese text」→"English translation"'
    )

    def __init__(
        self,
        model: str = GEMMA_MODEL,
        target_lang: str = "en",
        detection_confidence: float = 0.25,
        nms_iou_threshold: float = 0.45,
        lama_service_url: str = "http://127.0.0.1:5001",
    ):
        self.model = model
        self.target_lang = target_lang
        self.detector = TextDetector(detection_confidence, nms_iou_threshold)
        self.inpainter = TextInpainter(service_url=lama_service_url)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _add_margin(img: Image.Image, pct: float = 0.15) -> Image.Image:
        """Add white border to prevent Pan & Scan from cropping edge text."""
        w, h = img.size
        pw, ph = int(w * pct), int(h * pct)
        result = Image.new("RGB", (w + 2 * pw, h + 2 * ph), (255, 255, 255))
        result.paste(img, (pw, ph))
        return result

    @staticmethod
    def _parse_pair(raw: str) -> Tuple[str, str]:
        """Parse 「Japanese」→"English" — falls back to (empty, raw) on mismatch."""
        m = re.search(r'[「\[](.+?)[」\]]\s*[→\->=]+\s*["\u201c\u201d](.+?)["\u201c\u201d]', raw, re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # Looser fallback: anything after → or ->
        m2 = re.search(r'[→\->=]+\s*(.+)', raw)
        if m2:
            return "", m2.group(1).strip().strip('"\'「」')
        return "", raw.strip()

    def _vlm_ocr_translate(self, region_img: Image.Image) -> Tuple[str, str]:
        """
        Send one cropped region to Gemma 3 vision.
        Returns (japanese_text, english_translation). Both may be empty on failure.
        """
        try:
            import ollama  # type: ignore
        except ImportError:
            return "", ""

        padded = self._add_margin(region_img)
        buf = io.BytesIO()
        padded.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": self._REGION_PROMPT,
                    "images": [b64],
                }],
                options={"temperature": 0.3, "num_ctx": 4096},
            )
            return self._parse_pair(resp.message.content)
        except Exception as e:
            print(f"   ⚠️  VLM OCR+translate failed: {e}")
            return "", ""

    # ── main pipeline ─────────────────────────────────────────────────────────

    def process(
        self,
        image: Image.Image,
        story_context: str = "",
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Translate a manga page.

        Returns:
            (output_image, logs) where logs is a list of dicts with
            bbox, src_text, tgt_text per region.
        """
        print("\n" + "=" * 70)
        print("🎌 VLM PIPELINE (Tier 2): YOLO + Gemma 3 Vision + LaMa")
        print("=" * 70)

        image_array = np.array(image.convert("RGB"))
        output = image.copy()
        logs: List[Dict[str, Any]] = []

        # Pass 1 — detect + VLM OCR+translate each region
        print("🔍 Detecting text regions...")
        detection_result = self.detector.detect(image_array)
        grouped = group_detections_by_class(detection_result)
        class_names = resolve_detection_class_names(detection_result)

        ready: List[Tuple[Tuple[int, int, int, int], int, str]] = []

        for region_type in sorted(grouped.keys()):
            for (x1, y1, x2, y2), confidence in grouped[region_type]:
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                crop = image.crop((x1, y1, x2, y2))
                print(f"   📍 [{class_names.get(region_type, str(region_type))}] "
                      f"({x1},{y1})→({x2},{y2}) conf={confidence:.0%}")
                ja, en = self._vlm_ocr_translate(crop)
                print(f"      「{ja}」 → \"{en}\"" if ja or en else "      (no text)")
                logs.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": class_names.get(region_type, str(region_type)),
                    "class_id": region_type,
                    "confidence": confidence,
                    "src_text": ja,
                    "tgt_text": en,
                })
                if en:
                    ready.append(((x1, y1, x2, y2), region_type, en))

        # Pass 2 — inpainting (brightness check → LaMa or flat-fill)
        print(f"\n🎨 Inpainting {len(ready)} regions...")
        for (x1, y1, x2, y2), region_type, _ in ready:
            region_pixels = np.array(output)[y1:y2, x1:x2]
            mean_brightness = region_pixels.mean()
            if mean_brightness >= 240:
                print(f"   ⚡ Brightness skip ({mean_brightness:.0f}) — white fill")
                ImageDraw.Draw(output).rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
            elif region_type in USE_LAMA_FOR_REGIONS and self.inpainter.available:
                print(f"   🎨 LaMa inpainting...")
                output = self.inpainter.inpaint_region(output, (x1, y1, x2, y2))
            else:
                bg = find_whitest_pixel(region_pixels)
                ImageDraw.Draw(output).rectangle([x1, y1, x2, y2], fill=bg)

        # Pass 3 — render translated text
        print(f"\n✍️  Rendering {len(ready)} translations...")
        draw = ImageDraw.Draw(output)
        boxes, texts, sizes, colors = [], [], [], []
        for (x1, y1, x2, y2), _, en in ready:
            wrapped, font = fit_text_to_box(draw, en, (x1, y1, x2, y2))
            bg = np.array(output)[y1:y2, x1:x2].mean()
            color = (255, 255, 255, 255) if bg < 160 else (0, 0, 0, 255)
            boxes.append((x1, y1, x2, y2))
            texts.append(wrapped)
            sizes.append(font.size)
            colors.append(color)

        if boxes:
            output = render_text_overlay(output, boxes, texts, sizes, colors)

        print("✅ VLM pipeline complete!\n")
        return output, logs
