"""Main manga translation pipeline."""

import re
import unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from typing import List, Dict, Any, Tuple, Optional

from config.settings import TextRegionType, USE_LAMA_FOR_REGIONS
from src.models.detector import TextDetector
from src.models.ocr import OCRExtractor
from src.models.inpainter import TextInpainter
from src.translators.base import TranslatorFactory
from src.utils.image import find_whitest_pixel
from src.utils.text import fit_text_to_box, render_text_overlay


# Patterns that indicate Gemma3 returned a meta-response rather than a
# real translation. These happen when OCR reads garbage or near-empty text.
_META_RESPONSE_FRAGMENTS = (
    "please provide",
    "i'm sorry",
    "i need the text",
    "ja text",
    "end text",
    "japanese text you want",
    "cannot translate",
    "no text provided",
)


def _is_garbage_ocr(text: str) -> bool:
    """Return True if the OCR output is too short or contains only punctuation/symbols."""
    if not text or len(text.strip()) < 2:
        return True
    # Count actual CJK / Latin letters (not punctuation or whitespace)
    letters = sum(
        1 for ch in text
        if unicodedata.category(ch).startswith(("L", "N"))
    )
    return letters < 2


def _is_meta_response(text: str) -> bool:
    """Return True if the translation engine returned a refusal/meta-response."""
    lower = text.lower()
    return any(frag in lower for frag in _META_RESPONSE_FRAGMENTS)


def group_detections_by_class(yolo_result) -> Dict[int, List[Tuple]]:
    """
    Simply group YOLO detections by class - YOLO already did NMS.
    Returns {class_id: [(bbox, confidence), ...]} with native Python ints.
    """
    num_classes = len(yolo_result.names)
    grouped = {i: [] for i in range(num_classes)}
    
    # Just group by class - YOLO already filtered by confidence and applied NMS
    for box, class_id, confidence in zip(
        yolo_result.boxes.xyxy.cpu(),
        yolo_result.boxes.cls.cpu(),
        yolo_result.boxes.conf.cpu()
    ):
        bbox = tuple(int(coord.item()) for coord in box)
        class_idx = int(class_id.item())
        conf_score = float(confidence.item())
        grouped[class_idx].append((bbox, conf_score))
    
    return grouped


def resolve_detection_class_names(yolo_result) -> Dict[int, str]:
    """Return display-friendly class names from the active YOLO model."""
    raw_names = getattr(yolo_result, "names", {}) or {}
    if isinstance(raw_names, list):
        raw_names = {index: name for index, name in enumerate(raw_names)}

    fallback_names = {
        TextRegionType.DIALOGUE: "Dialogue",
        TextRegionType.SOUND_EFFECTS: "Sound Effects",
        TextRegionType.SIGNS: "Signs",
        TextRegionType.TEXT: "Text",
        TextRegionType.REMOVAL: "Removal",
    }

    class_names: Dict[int, str] = {}
    for class_id in range(max(len(raw_names), len(fallback_names))):
        raw_name = raw_names.get(class_id, fallback_names.get(class_id, f"Class {class_id}"))
        class_names[class_id] = str(raw_name).replace("_", " ").strip()

    return class_names


class MangaTranslationPipeline:
    """Complete manga translation pipeline"""

    # Hard cap on regions processed per page. At very low confidence the
    # detector produces many false positives and each one runs OCR + a
    # network translation call. Without a cap a single page can take
    # minutes and the batch request times out in the browser.
    MAX_REGIONS_PER_PAGE = 40

    def __init__(
        self,
        source_lang: str = "ja",
        target_lang: str = "en",
        translation_engine: str = "Gemma3",
        detection_confidence: float = 0.25,
        nms_iou_threshold: float = 0.45,
        text_color: str = "#0000FF",
        story_context: Optional[str] = None,
        vlm_context_enabled: bool = False,
    ):
        """
        Initialize the translation pipeline.
        
        Args:
            source_lang: Source language code (e.g., "ja")
            target_lang: Target language code (e.g., "en")
            translation_engine: Translation engine to use
            detection_confidence: YOLO confidence threshold
            nms_iou_threshold: NMS IoU threshold
            text_color: Hex color for translated text
            story_context: Optional global story context (characters, plot, glossary, etc.)
                          This is included in the system prompt for all translations
        """
        print("\n" + "="*80)
        print("🎌 INITIALIZING MANGA TRANSLATION PIPELINE")
        print("="*80)
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.text_color = text_color
        self.story_context = story_context
        self.vlm_context_enabled = vlm_context_enabled
        
        # Initialize models
        self.detector = TextDetector(detection_confidence, nms_iou_threshold)
        self.ocr = OCRExtractor()
        self.inpainter = TextInpainter()
        self.translator = TranslatorFactory.create(
            translation_engine,
            source_lang,
            target_lang
        )
        
        print(f"✅ Pipeline ready!")
        print(f"   Translator: {self.translator.name}")
        print(f"   LaMa Inpainting: {'✅ Enabled' if self.inpainter.available else '⚠️ Disabled'}")
        print("="*80 + "\n")
    
    def process(
        self,
        image: Image.Image,
        previous_page_context: Optional[List[str]] = None
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Process a manga page and translate all text.
        
        Args:
            image: PIL Image of manga page
            previous_page_context: Optional list of translations from 1-2 previous pages
                                   to provide narrative continuity
        
        Returns:
            Tuple of (translated_image, processing_logs)
        """
        print("\n" + "="*80)
        print("🎌 PROCESSING MANGA PAGE")
        print("="*80)
        
        # Prepare image
        image_array = np.array(image.convert("RGB"))
        output_image = image.copy()
        draw_context = ImageDraw.Draw(output_image)
        text_rgb = ImageColor.getrgb(self.text_color)
        
        # Buffers for text overlay
        overlay_boxes = []
        overlay_texts = []
        overlay_font_sizes = []
        overlay_colors = []
        
        processing_logs = []
        
        # Step 1: Detect text regions
        print("🔍 Step 1/5: Detecting text regions...")
        detection_result = self.detector.detect(image_array)
        grouped_detections = group_detections_by_class(detection_result)
        
        class_names = resolve_detection_class_names(detection_result)
        
        total_detections = sum(len(v) for v in grouped_detections.values())
        for class_id, detections in grouped_detections.items():
            if detections:
                print(f"   - Found {len(detections)} {class_names.get(class_id, f'Class {class_id}')} regions")

        # Guard against false-positive explosion at low confidence thresholds.
        # Sort each class bucket by descending confidence and keep only the
        # top MAX_REGIONS_PER_PAGE detections total across all classes.
        if total_detections > self.MAX_REGIONS_PER_PAGE:
            print(f"   ⚠️  {total_detections} detections exceed cap of "
                  f"{self.MAX_REGIONS_PER_PAGE}. Keeping highest-confidence regions.")
            # Flatten → sort → take top N → rebuild grouped dict
            all_dets = [
                (class_id, bbox, conf)
                for class_id, dets in grouped_detections.items()
                for bbox, conf in dets
            ]
            all_dets.sort(key=lambda x: x[2], reverse=True)
            all_dets = all_dets[:self.MAX_REGIONS_PER_PAGE]
            num_classes = len(grouped_detections)
            grouped_detections = {i: [] for i in range(num_classes)}
            for class_id, bbox, conf in all_dets:
                grouped_detections[class_id].append((bbox, conf))

        # ── Optional: VLM visual context (1 Ollama vision call per page) ──────────
        vlm_context = ""
        if self.vlm_context_enabled:
            from src.models.vlm_ocr import PageContextExtractor
            print("🖼️  Extracting visual context from page (VLM)...")
            vlm_context = PageContextExtractor(
                model=getattr(self.translator, "model", "gemma3:12b")
            ).extract_context(output_image)
            if vlm_context:
                print(f"   📝 Context: {vlm_context[:120]}{'…' if len(vlm_context) > 120 else ''}")

        # ── PASS 1: OCR (all regions) then one batch Translation call ────────────
        print("📖 Step 2/4: OCR (all regions)...")

        regions_to_process = sorted(grouped_detections.keys())
        region_counter = 0

        # Build narrative context once for the whole page
        context_parts = []
        if previous_page_context:
            context_parts.append(
                f"[Previous page context:\n{chr(10).join(previous_page_context[-20:])}]"
            )
        if vlm_context:
            context_parts.append(f"[Visual context: {vlm_context}]")
        context_prompt = "\n".join(context_parts)

        # Collect OCR results before calling the translator
        ocr_pending = []   # (region_type, region_name, bbox, confidence, original_text)
        sfx_logs    = []   # sound effects logged but not rendered

        for region_type in regions_to_process:
            region_name = class_names.get(region_type, f"Class {region_type}")
            if not grouped_detections.get(region_type):
                continue
            for (x1, y1, x2, y2), confidence in grouped_detections[region_type]:
                region_counter += 1
                print(f"\n📍 Region #{region_counter} ({region_name}) conf={confidence:.0%}")
                if x2 - x1 < 20 or y2 - y1 < 20:
                    print(f"   ⏭️  Too small, skipped")
                    continue
                print(f"   👁️  OCR...")
                original_text = self.ocr.extract_text(image.crop((x1, y1, x2, y2)))
                print(f"   📖 '{original_text}'")
                if _is_garbage_ocr(original_text):
                    print(f"   ⏭️  Garbage OCR, skipped")
                    continue
                ocr_pending.append((region_type, region_name, (x1, y1, x2, y2), confidence, original_text))

        # ── One batch translation call for the entire page ────────────────────
        print(f"\n🌐 Step 2b/4: Translating {len(ocr_pending)} texts in one batch call...")
        texts_to_translate = [r[4] for r in ocr_pending]
        try:
            translations = self.translator.translate_batch(
                texts_to_translate,
                context_prompt=context_prompt,
                story_context=self.story_context,
            )
        except Exception as e:
            import traceback
            print(f"⚠️ Batch translation failed ({type(e).__name__}): {e}")
            print(traceback.format_exc())
            translations = texts_to_translate  # fall back to originals

        # ── Merge OCR + translations ──────────────────────────────────────────
        ready_regions = []
        for (region_type, region_name, (x1, y1, x2, y2), confidence, original_text), translated_text in zip(ocr_pending, translations):
            print(f"   ✅ [{region_name}] '{original_text}' → '{translated_text}'")

            if _is_meta_response(translated_text):
                print(f"   ⏭️  Meta-response, skipped")
                continue

            if region_type == TextRegionType.SOUND_EFFECTS:
                processing_logs.append({
                    "region_id": region_counter, "class": region_name,
                    "class_id": region_type, "bbox": (x1, y1, x2, y2),
                    "confidence": confidence, "src_text": original_text,
                    "tgt_text": translated_text,
                })
                continue

            processing_logs.append({
                "region_id": region_counter, "class": region_name,
                "class_id": region_type, "bbox": (x1, y1, x2, y2),
                "confidence": confidence, "src_text": original_text,
                "tgt_text": translated_text,
            })
            ready_regions.append((region_type, (x1, y1, x2, y2), translated_text))

        print(f"\n   ✅ OCR+Translation done — {len(ready_regions)} regions to render")

        # ── Unload LLM from GPU so LaMa can use the VRAM for inpainting ──────────
        print("🧹 Unloading translation model from GPU...")
        self.translator.unload()
        # Re-check LaMa availability now that VRAM is free (lazy-loads on first call)
        if not self.inpainter.available:
            self.inpainter.try_reconnect()

        # ── PASS 2: Inpainting (brightness check → LaMa or fast flat-fill) ─────────
        print(f"🎨 Step 3/4: Inpainting {len(ready_regions)} regions...")

        for region_type, (x1, y1, x2, y2), _ in ready_regions:
            region_pixels = np.array(output_image)[y1:y2, x1:x2]
            mean_brightness = region_pixels.mean()
            if mean_brightness >= 240:
                # Plain white bubble — instant flat-fill, no LaMa needed (~60-70% of regions)
                print(f"   ⚡ Brightness skip ({mean_brightness:.0f}) — flat white fill")
                ImageDraw.Draw(output_image).rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
            elif region_type in USE_LAMA_FOR_REGIONS and self.inpainter.available:
                print(f"   🎨 LaMa inpainting ({x1},{y1})→({x2},{y2})...")
                output_image = self.inpainter.inpaint_region(output_image, (x1, y1, x2, y2))
            else:
                background_color = find_whitest_pixel(region_pixels)
                ImageDraw.Draw(output_image).rectangle([x1, y1, x2, y2], fill=background_color)

        # Rebuild draw context once after all inpainting is done
        image_array = np.array(output_image)
        draw_context = ImageDraw.Draw(output_image)

        # ── PASS 3: Render translated text ───────────────────────────────────────
        print(f"✍️  Step 4/4: Rendering {len(ready_regions)} translations...")

        for _, (x1, y1, x2, y2), translated_text in ready_regions:
            if translated_text:
                wrapped_text, font = fit_text_to_box(draw_context, translated_text, (x1, y1, x2, y2))
                overlay_boxes.append((x1, y1, x2, y2))
                overlay_texts.append(wrapped_text)
                overlay_font_sizes.append(font.size)
                # Smart text color: dark background → white text, light → user's chosen color
                region_arr = np.array(output_image)[y1:y2, x1:x2]
                bg_brightness = region_arr.mean()
                region_color = (255, 255, 255, 255) if bg_brightness < 160 else (*text_rgb, 255)
                overlay_colors.append(region_color)

        if overlay_boxes:
            output_image = render_text_overlay(
                output_image, overlay_boxes, overlay_texts,
                overlay_font_sizes, overlay_colors,
            )

        print("✅ Complete!")
        print("="*80 + "\n")

        return output_image, processing_logs

    def process_image(
        self,
        input_path: str,
        output_path: str,
        previous_page_context: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a manga image from file and save the result.
        
        Args:
            input_path: Path to input manga image
            output_path: Path to save translated image
            previous_page_context: Optional list of translations from previous pages
            **kwargs: Additional arguments (for batch processor compatibility)
        
        Returns:
            Dictionary with processing statistics
        """
        # Load image
        image = Image.open(input_path)
        
        # Process with narrative context
        output_image, processing_logs = self.process(image, previous_page_context=previous_page_context)
        
        # Save result
        output_image.save(output_path)
        
        # Generate statistics
        stats = {
            "bubbles_detected": len(processing_logs),  # For frontend display
            "regions_detected": len(processing_logs),
            "regions_by_type": {},
            # Keep translation records structured so future accessibility work
            # can add readout/TTS layers without rewriting the pipeline output.
            "translations": [],
            "processing_time": "N/A"  # Could add timing later
        }
        
        for log in processing_logs:
            region_class = log["class"]
            if region_class not in stats["regions_by_type"]:
                stats["regions_by_type"][region_class] = 0
            stats["regions_by_type"][region_class] += 1
            
            stats["translations"].append({
                "id": log["region_id"],
                "type": region_class,
                "class_id": log["class_id"],
                "bbox": list(log["bbox"]),
                "original": log["src_text"],
                "translated": log["tgt_text"],
                "confidence": log["confidence"]
            })
        
        return stats
