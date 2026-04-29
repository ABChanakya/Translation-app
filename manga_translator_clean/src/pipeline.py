"""Main manga translation pipeline."""

import re
import unicodedata
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageColor
from typing import List, Dict, Any, Tuple, Optional, Callable

from config.settings import TextRegionType, USE_LAMA_FOR_REGIONS
from src.models.detector import TextDetector
from src.models.ocr import OCRExtractor
from src.models.inpainter import TextInpainter
from src.models.bubble_segmenter import BubbleSegmenter
from src.translators.base import TranslatorFactory
from src.utils.image import find_whitest_pixel
from src.utils.text import fit_text_to_box, render_text_overlay
from src.utils.text_placement import render_all_bubbles
from src.utils.inpainting_smart import smart_inpaint_bubble
from src.utils.ocr_smart import ocr_region_with_preprocessing
from src.vertical_text import VerticalTextDetector, VerticalTextRotator


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


def _bbox_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Compute IoU between two (x1,y1,x2,y2) bboxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _find_matching_bubble(bbox: Tuple[int,int,int,int], bubble_masks: dict, threshold: float = 0.3) -> Optional[dict]:
    """Find the bubble mask that best overlaps the given YOLO bbox."""
    best_iou, best_bubble = 0.0, None
    for bkey, bubble in bubble_masks.items():
        iou = _bbox_iou(bbox, bkey)
        if iou > best_iou:
            best_iou, best_bubble = iou, bubble
    return best_bubble if best_iou >= threshold else None


def _compute_quality_score(
    ocr_confidence: float,
    japanese_text: str,
    translated_text: str,
) -> float:
    """
    Composite 0.0–1.0 quality score:
      40% — OCR confidence
      30% — translation length ratio sanity (JP→EN ~1–2.5×)
      30% — meta-response penalty
    """
    ocr_score = max(0.0, min(1.0, ocr_confidence))

    jp_len = len(japanese_text.strip())
    en_len = len(translated_text.strip())
    if jp_len == 0:
        length_score = 0.0
    else:
        ratio = en_len / jp_len
        if 0.5 <= ratio <= 3.0:
            length_score = 1.0
        elif ratio < 0.5:
            length_score = ratio / 0.5
        else:
            length_score = max(0.0, 1.0 - (ratio - 3.0) / 3.0)

    meta_score = 0.0 if _is_meta_response(translated_text) else 1.0

    return round(0.4 * ocr_score + 0.3 * length_score + 0.3 * meta_score, 3)


def _sort_manga_reading_order(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort pipeline logs in manga RTL reading order:
    right-to-left columns (rightmost x-centre first), then top-to-bottom.
    Column buckets snap every 80 px so nearby bubbles share a column.
    """
    if not logs:
        return logs

    def _key(log: Dict[str, Any]) -> Tuple[float, float]:
        x1, y1, x2, y2 = log.get("bbox", (0, 0, 0, 0))
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        col_bucket = -(cx // 80)   # negative → rightmost first
        return (col_bucket, cy)

    return sorted(logs, key=_key)


def _find_narration_columns(
    image_array: np.ndarray,
    existing_detections: Dict[int, List[Tuple]],
    min_height: int = 150,
    min_aspect_ratio: float = 4.0,
    max_width: int = 90,
) -> List[Tuple]:
    """
    Scan for vertical narration text columns that YOLO missed.
    These are tall, narrow strips of stacked kanji on panel margins.
    Returns list of (bbox, confidence) tuples with synthetic confidence 0.5.
    """
    import cv2
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Dilate vertically to merge stacked characters into one blob per column
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    existing_bboxes = [
        bbox for dets in existing_detections.values() for bbox, _ in dets
    ]

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < min_height or w > max_width:
            continue
        if h / max(w, 1) < min_aspect_ratio:
            continue
        region = binary[y:y + h, x:x + w]
        if region.size == 0:
            continue
        # Require meaningful ink density — rules out panel borders / thin lines
        density = float(region.sum()) / (255.0 * region.size)
        if density < 0.04:
            continue
        bbox = (x, y, x + w, y + h)
        # Skip if well-covered by an existing YOLO detection
        if any(_bbox_iou(bbox, eb) > 0.3 for eb in existing_bboxes):
            continue
        candidates.append((bbox, 0.50))

    return candidates


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
        text_color: str = "#000000",
        story_context: Optional[str] = None,
        vlm_context_enabled: bool = False,
        manga_profile: Optional["MangaProfile"] = None,
        chapter_num: int = 1,
        page_num: int = 1,
        progress_callback: Optional[Callable[[str, str], None]] = None,
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
            manga_profile: Optional MangaProfile for glossary, characters, and rolling memory
            chapter_num: Chapter number (used with manga_profile)
            page_num: Starting page number (incremented per page in batch)
        """
        print("\n" + "="*80)
        print("🎌 INITIALIZING MANGA TRANSLATION PIPELINE")
        print("="*80)

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.text_color = text_color
        self.story_context = story_context
        self.vlm_context_enabled = vlm_context_enabled
        self.manga_profile = manga_profile
        self.chapter_num = chapter_num
        self.page_num = page_num
        self.progress_callback = progress_callback

        # Initialize models
        self.detector = TextDetector(detection_confidence, nms_iou_threshold)
        self.ocr = OCRExtractor()
        self.inpainter = TextInpainter()
        self.segmenter = BubbleSegmenter()
        self.vertical_detector = VerticalTextDetector()
        self.translator = TranslatorFactory.create(
            translation_engine,
            source_lang,
            target_lang
        )

        print(f"✅ Pipeline ready!")
        print(f"   Translator: {self.translator.name}")
        print(f"   Bubble Segmenter: {'✅ Enabled' if self.segmenter.available else '⚠️ Disabled (bbox fallback)'}")
        print(f"   LaMa Inpainting: {'✅ Enabled' if self.inpainter.available else '⚠️ Disabled'}")
        print("="*80 + "\n")

    def _log(self, stage: str, message: str) -> None:
        """Emit a pipeline progress event to stdout and to any attached callback."""
        print(message)
        cb = getattr(self, "progress_callback", None)
        if cb is not None:
            try:
                cb(stage, message)
            except Exception:
                pass
    
    def process(
        self,
        image: Image.Image,
        previous_page_context: Optional[List[str]] = None,
        clean_save_path: Optional[str] = None,
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """
        Process a manga page and translate all text.

        Args:
            image: PIL Image of manga page
            previous_page_context: Optional list of translations from 1-2 previous pages
                                   to provide narrative continuity
            clean_save_path: If provided, save a copy of the page *after* inpainting
                             but *before* any translated text is rendered. This gives
                             the review UI a pristine base to re-composite from when
                             a user applies a revised translation, so the old rendered
                             text is cleanly wiped before the new text is drawn.

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
        
        # Step 1: Detect text regions + bubble masks
        self._log("detecting", "🔍 Step 1/5: Detecting text regions...")
        detection_result = self.detector.detect(image_array)
        grouped_detections = group_detections_by_class(detection_result)

        class_names = resolve_detection_class_names(detection_result)

        total_detections = sum(len(v) for v in grouped_detections.values())
        for class_id, detections in grouped_detections.items():
            if detections:
                self._log(
                    "detecting",
                    f"   - Found {len(detections)} {class_names.get(class_id, f'Class {class_id}')} regions",
                )

        # Run bubble segmentation for mask-aware text placement
        bubble_masks = {}  # bbox_key → bubble dict with mask
        if self.segmenter.available:
            self._log("segmenting", "🫧 Running bubble segmentation...")
            bubbles = self.segmenter.detect(image)
            self._log("segmenting", f"   Found {len(bubbles)} bubble masks")
            # Index bubbles for fast lookup by YOLO bbox matching
            for bubble in bubbles:
                bubble_masks[bubble["bbox"]] = bubble

        # Supplement YOLO with narration columns it may have missed
        narration_candidates = _find_narration_columns(image_array, grouped_detections)
        if narration_candidates:
            self._log(
                "detecting",
                f"   📜 {len(narration_candidates)} narration column(s) found (not detected by YOLO)",
            )
            grouped_detections.setdefault(TextRegionType.TEXT, []).extend(narration_candidates)
            total_detections += len(narration_candidates)

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
                model=getattr(self.translator, "model", "gemma4:latest")
            ).extract_context(output_image)
            if vlm_context:
                print(f"   📝 Context: {vlm_context[:120]}{'…' if len(vlm_context) > 120 else ''}")

        # ── PASS 1: OCR (all regions) then one batch Translation call ────────────
        self._log("ocr", "📖 Step 2/4: OCR (all regions)...")

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

        # If a MangaProfile is active, build enriched story context
        # from glossary, characters, chapter summaries, and recent translations
        effective_story_context = self.story_context or ""
        if self.manga_profile:
            profile_parts = []
            glossary = self.manga_profile.get_glossary_as_prompt_block()
            if glossary:
                profile_parts.append(glossary)
            chars = self.manga_profile.get_characters_as_prompt_block()
            if chars:
                profile_parts.append(chars)
            summaries = self.manga_profile.get_recent_chapter_summaries(n=2)
            if summaries:
                profile_parts.append(summaries)
            recent = self.manga_profile.get_recent_translations_as_prompt_block(n=10)
            if recent:
                profile_parts.append(recent)
            if profile_parts:
                profile_block = "\n\n".join(profile_parts)
                if effective_story_context:
                    effective_story_context += "\n\n" + profile_block
                else:
                    effective_story_context = profile_block
                print(f"   📚 Profile context injected ({len(profile_parts)} blocks)")

        # Collect OCR results before calling the translator
        ocr_pending = []   # (region_type, region_name, bbox, detection_conf, original_text, ocr_conf)
        sfx_logs    = []   # sound effects logged but not rendered

        for region_type in regions_to_process:
            region_name = class_names.get(region_type, f"Class {region_type}")
            if not grouped_detections.get(region_type):
                continue
            for (x1, y1, x2, y2), confidence in grouped_detections[region_type]:
                region_counter += 1
                self._log(
                    "ocr_region",
                    f"📍 Region #{region_counter} ({region_name}) conf={confidence:.0%}",
                )
                if x2 - x1 < 20 or y2 - y1 < 20:
                    self._log("ocr_region", "   ⏭️  Too small, skipped")
                    continue
                # Detect orientation — vertical narration text needs rotation before OCR
                orientation = self.vertical_detector.detect_orientation((x1, y1, x2, y2))
                if orientation.is_vertical:
                    print(f"   ↕️  Vertical text (aspect {orientation.bbox_aspect_ratio:.1f}x) — rotating for OCR")
                    rotated_crop = VerticalTextRotator.rotate_for_ocr(
                        image_array, (x1, y1, x2, y2), orientation.rotation_angle
                    )
                    original_text = self.ocr.model(Image.fromarray(rotated_crop))
                    ocr_confidence = 0.85
                else:
                    # Smart OCR with preprocessing, furigana removal, and confidence checking
                    original_text, ocr_confidence = ocr_region_with_preprocessing(
                        self.ocr.model,
                        image_array,
                        (x1, y1, x2, y2),
                        region_counter,
                        log_dir=str(Path.cwd())
                    )
                self._log(
                    "ocr_region",
                    f"   📖 '{original_text}' (confidence: {ocr_confidence:.2f})",
                )
                if _is_garbage_ocr(original_text):
                    self._log("ocr_region", "   ⏭️  Garbage OCR, skipped")
                    continue
                ocr_pending.append((region_type, region_name, (x1, y1, x2, y2), confidence, original_text, ocr_confidence))

        # ── One batch translation call for the entire page ────────────────────
        self._log(
            "translating",
            f"🌐 Step 2b/4: Translating {len(ocr_pending)} texts in one batch call...",
        )
        texts_to_translate = [r[4] for r in ocr_pending]
        try:
            translations = self.translator.translate_batch(
                texts_to_translate,
                context_prompt=context_prompt,
                story_context=effective_story_context,
            )
        except Exception as e:
            import traceback
            print(f"⚠️ Batch translation failed ({type(e).__name__}): {e}")
            print(traceback.format_exc())
            translations = texts_to_translate  # fall back to originals

        # ── Glossary compliance validation + retry ───────────────────────────
        if self.manga_profile and texts_to_translate:
            from src.translation.validator import (
                validate_and_retry_translations,
                log_violations,
            )
            # Pass ollama_client=None so we don't retry via LLM here
            # (the model has already been called once; retrying doubles latency).
            # Instead we do a fast check and force-inject any missed terms.
            translations, violations = validate_and_retry_translations(
                japanese_texts=texts_to_translate,
                english_texts=translations,
                profile=self.manga_profile,
                ollama_client=None,   # no LLM retry, just detect + force-inject
                max_retries=0,
            )
            if violations:
                # Force-inject missing terms so they appear in the output
                from src.translation.validator import force_inject_terms
                for v in violations:
                    idx = v["index"]
                    translations[idx] = force_inject_terms(
                        translations[idx], v["violations"]
                    )
                    print(f"   ⚠️  Glossary violation in bubble {idx + 1}: "
                          f"force-injected "
                          + ", ".join(vv["expected_english"] for vv in v["violations"]))
                log_violations(violations, self.page_num, self.chapter_num)

        # ── Merge OCR + translations ──────────────────────────────────────────
        ready_regions = []
        for (region_type, region_name, (x1, y1, x2, y2), confidence, original_text, ocr_conf), translated_text in zip(ocr_pending, translations):
            self._log(
                "translated",
                f"   ✅ [{region_name}] '{original_text}' → '{translated_text}'",
            )

            if _is_meta_response(translated_text):
                self._log("translated", "   ⏭️  Meta-response, skipped")
                continue

            quality_score = _compute_quality_score(ocr_conf, original_text, translated_text)

            if region_type == TextRegionType.SOUND_EFFECTS:
                processing_logs.append({
                    "region_id": region_counter, "class": region_name,
                    "class_id": region_type, "bbox": (x1, y1, x2, y2),
                    "confidence": confidence, "ocr_confidence": ocr_conf,
                    "quality_score": quality_score,
                    "src_text": original_text, "tgt_text": translated_text,
                })
                continue

            processing_logs.append({
                "region_id": region_counter, "class": region_name,
                "class_id": region_type, "bbox": (x1, y1, x2, y2),
                "confidence": confidence, "ocr_confidence": ocr_conf,
                "quality_score": quality_score,
                "src_text": original_text, "tgt_text": translated_text,
            })
            # Try to match this detection to a bubble mask
            matched_bubble = _find_matching_bubble((x1, y1, x2, y2), bubble_masks)
            ready_regions.append((region_type, (x1, y1, x2, y2), translated_text, matched_bubble))

        self._log(
            "translated",
            f"   ✅ OCR+Translation done — {len(ready_regions)} regions to render",
        )

        # ── Save translations to MangaProfile rolling memory ─────────────────
        if self.manga_profile and ready_regions:
            memory_lines = []
            for _, (x1, y1, x2, y2), translated_text, _ in ready_regions:
                # Find the matching original text from ocr_pending
                for rtype, rname, bbox, conf, orig, oconf in ocr_pending:
                    if bbox == (x1, y1, x2, y2):
                        memory_lines.append({
                            "japanese": orig,
                            "english": translated_text,
                            "chapter": self.chapter_num,
                            "page": self.page_num,
                        })
                        break
            if memory_lines:
                self.manga_profile.add_translated_lines(memory_lines)
                print(f"   📚 Saved {len(memory_lines)} lines to translation memory")

        # ── Unload LLM from GPU so LaMa can use the VRAM for inpainting ──────────
        print("🧹 Unloading translation model from GPU...")
        self.translator.unload()
        # Re-check LaMa availability now that VRAM is free (lazy-loads on first call)
        if not self.inpainter.available:
            self.inpainter.try_reconnect()

        # ── PASS 2: Smart inpainting with background detection ────────────────────
        self._log(
            "inpainting",
            f"🎨 Step 3/4: Smart inpainting {len(ready_regions)} regions...",
        )

        inpaint_stats = {'white': 0, 'light': 0, 'screentone': 0, 'artwork': 0}

        for region_type, (x1, y1, x2, y2), _, matched_bubble in ready_regions:
            output_arr = np.array(output_image.convert("RGB"))

            if matched_bubble is not None:
                # Smart inpainting with background detection
                bubble_mask = matched_bubble["mask"]
                inpainted, bg_type = smart_inpaint_bubble(
                    output_arr,
                    bubble_mask,
                    (x1, y1, x2, y2),
                    lama_inpainter=self.inpainter
                )
                output_image = Image.fromarray(inpainted)
                inpaint_stats[bg_type] += 1
                self._log(
                    "inpainting",
                    f"   🫧 Smart inpaint ({x1},{y1})→({x2},{y2}): {bg_type}",
                )
            else:
                # Fallback: simple brightness-based inpainting
                region_pixels = output_arr[y1:y2, x1:x2]
                mean_brightness = region_pixels.mean()

                if mean_brightness >= 240:
                    self._log(
                        "inpainting",
                        f"   ⚡ Brightness skip ({mean_brightness:.0f}) — flat fill",
                    )
                    ImageDraw.Draw(output_image).rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
                    inpaint_stats['white'] += 1
                elif region_type in USE_LAMA_FOR_REGIONS and self.inpainter.available:
                    self._log("inpainting", "   🎨 LaMa inpainting (bbox fallback)...")
                    output_image = self.inpainter.inpaint_region(output_image, (x1, y1, x2, y2))
                    inpaint_stats['artwork'] += 1
                else:
                    background_color = find_whitest_pixel(region_pixels)
                    ImageDraw.Draw(output_image).rectangle([x1, y1, x2, y2], fill=background_color)
                    inpaint_stats['light'] += 1

        self._log(
            "inpainting",
            f"   Inpainting summary: white={inpaint_stats['white']}, "
            f"light={inpaint_stats['light']}, screentone={inpaint_stats['screentone']}, "
            f"artwork={inpaint_stats['artwork']}",
        )

        # Rebuild draw context once after all inpainting is done
        image_array = np.array(output_image)
        draw_context = ImageDraw.Draw(output_image)

        # Persist a clean (post-inpaint, pre-render) copy so the review UI
        # can re-composite this bubble's region from a pristine base when
        # the user applies a revised translation.
        if clean_save_path:
            try:
                Path(clean_save_path).parent.mkdir(parents=True, exist_ok=True)
                output_image.save(clean_save_path)
            except Exception as e:
                self._log("rendering", f"   ⚠️ Could not save clean base ({e})")

        # ── PASS 3: Render translated text ───────────────────────────────────────
        self._log(
            "rendering",
            f"✍️  Step 4/4: Rendering {len(ready_regions)} translations...",
        )

        # Separate mask-aware bubbles from bbox-only regions
        mask_bubbles = []
        mask_translations = []
        bbox_fallback_regions = []

        for _, (x1, y1, x2, y2), translated_text, matched_bubble in ready_regions:
            if not translated_text:
                continue
            if matched_bubble is not None:
                mask_bubbles.append(matched_bubble)
                mask_translations.append(translated_text)
            else:
                bbox_fallback_regions.append(((x1, y1, x2, y2), translated_text))

        # Mask-aware rendering (text flows inside bubble shape)
        if mask_bubbles:
            print(f"   🫧 Mask-aware rendering: {len(mask_bubbles)} bubbles")
            output_image = render_all_bubbles(
                output_image, mask_bubbles, mask_translations,
                text_color=text_rgb, smart_color=True,
            )

        # Bbox fallback for regions without a matching bubble mask
        if bbox_fallback_regions:
            print(f"   📦 Bbox fallback rendering: {len(bbox_fallback_regions)} regions")
            draw_context = ImageDraw.Draw(output_image)
            for (x1, y1, x2, y2), translated_text in bbox_fallback_regions:
                wrapped_text, font = fit_text_to_box(draw_context, translated_text, (x1, y1, x2, y2))
                overlay_boxes.append((x1, y1, x2, y2))
                overlay_texts.append(wrapped_text)
                overlay_font_sizes.append(font.size)
                region_arr = np.array(output_image)[y1:y2, x1:x2]
                bg_brightness = region_arr.mean()
                region_color = (255, 255, 255, 255) if bg_brightness < 160 else (*text_rgb, 255)
                overlay_colors.append(region_color)

            if overlay_boxes:
                output_image = render_text_overlay(
                    output_image, overlay_boxes, overlay_texts,
                    overlay_font_sizes, overlay_colors,
                )

        # ── Sort logs in manga RTL reading order and assign final bubble indices ──
        processing_logs = _sort_manga_reading_order(processing_logs)
        for idx, log in enumerate(processing_logs):
            log["bubble_index"] = idx

        self._log("page_done", "✅ Complete!")
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
