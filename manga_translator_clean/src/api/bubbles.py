"""Bubble review + manual annotation endpoints."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.db.models import Bubble, Chapter, Page, Project
from src.feedback.capture import CorrectionCapture

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["bubbles"])


# ── Pydantic models ─────────────────────────────────────────────────

class BubbleOut(BaseModel):
    id: int
    bubble_index: int
    bubble_type: str
    x1: int
    y1: int
    x2: int
    y2: int
    mask_points: str | None
    japanese_text: str | None
    suggested_translation: str | None
    human_translation: str | None
    status: str
    ocr_confidence: float | None
    quality_score: float | None
    edit_distance: int | None
    notes: str | None
    # Annotation fields
    is_manual: bool
    mode: str
    mask_polygon: str | None
    font_family: str | None
    font_size: int | None
    font_color: str | None
    stroke_color: str | None
    stroke_width: int | None
    text_align: str | None

    model_config = {"from_attributes": True}


class CorrectBody(BaseModel):
    human_translation: str
    notes: str | None = None


class ManualBubbleCreate(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    polygon: list[dict] | None = None  # [{x, y}, ...]
    mode: str = "translate_and_inpaint"
    bubble_type: str = "speech"
    manual_text: str | None = None
    font_family: str = "Bangers"
    font_size: int | None = None
    font_color: str = "#000000"
    stroke_color: str = "#ffffff"
    stroke_width: int = 1
    text_align: str = "center"


class PolygonUpdate(BaseModel):
    points: list[dict]  # [{x, y}, ...]


class FontUpdate(BaseModel):
    font_family: str | None = None
    font_size: int | None = None
    font_color: str | None = None
    stroke_color: str | None = None
    stroke_width: int | None = None
    text_align: str | None = None


# ── Existing review endpoints ────────────────────────────────────────

@router.get("/pages/{page_id}/bubbles", response_model=list[BubbleOut])
def get_page_bubbles(page_id: int, db: Session = Depends(get_db)):
    bubbles = (
        db.query(Bubble)
        .filter(Bubble.page_id == page_id)
        .order_by(Bubble.bubble_index)
        .all()
    )
    return [_bubble_out(b) for b in bubbles]


@router.post("/bubbles/{bubble_id}/accept", response_model=BubbleOut)
def accept_bubble(bubble_id: int, db: Session = Depends(get_db)):
    bubble, series = _get_bubble_and_series(bubble_id, db)
    cc = CorrectionCapture(db)
    cc.log_correction(
        bubble_id=bubble.id,
        series_name=series,
        action="accept",
        japanese_text=bubble.japanese_text or "",
        suggested=bubble.suggested_translation or "",
    )
    db.refresh(bubble)
    # Feed accepted translation back into MangaProfile
    _update_profile(db, bubble, series)
    return _bubble_out(bubble)


@router.post("/bubbles/{bubble_id}/correct", response_model=BubbleOut)
def correct_bubble(bubble_id: int, body: CorrectBody, db: Session = Depends(get_db)):
    bubble, series = _get_bubble_and_series(bubble_id, db)
    if body.notes is not None:
        bubble.notes = body.notes
        db.commit()
    cc = CorrectionCapture(db)
    cc.log_correction(
        bubble_id=bubble.id,
        series_name=series,
        action="correct",
        japanese_text=bubble.japanese_text or "",
        suggested=bubble.suggested_translation or "",
        human=body.human_translation,
    )
    db.refresh(bubble)
    # Feed corrected translation back into MangaProfile
    _update_profile(db, bubble, series)
    return _bubble_out(bubble)


@router.post("/bubbles/{bubble_id}/skip", response_model=BubbleOut)
def skip_bubble(bubble_id: int, db: Session = Depends(get_db)):
    bubble, series = _get_bubble_and_series(bubble_id, db)
    cc = CorrectionCapture(db)
    cc.log_correction(
        bubble_id=bubble.id,
        series_name=series,
        action="skip",
        japanese_text=bubble.japanese_text or "",
        suggested=bubble.suggested_translation or "",
    )
    db.refresh(bubble)
    return _bubble_out(bubble)


# ── Manual annotation endpoints ──────────────────────────────────────

@router.post("/pages/{page_id}/bubbles/manual", response_model=BubbleOut)
def create_manual_bubble(page_id: int, body: ManualBubbleCreate, db: Session = Depends(get_db)):
    """Create a new manually drawn bubble region."""
    page = db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")

    # Determine next bubble_index
    max_idx = (
        db.query(func.max(Bubble.bubble_index))
        .filter(Bubble.page_id == page_id)
        .scalar()
    )
    next_idx = (max_idx or 0) + 1

    bubble = Bubble(
        page_id=page_id,
        bubble_index=next_idx,
        bubble_type=body.bubble_type,
        x1=body.x1, y1=body.y1, x2=body.x2, y2=body.y2,
        mask_polygon=json.dumps(body.polygon) if body.polygon else None,
        is_manual=True,
        mode=body.mode,
        status="pending",
        font_family=body.font_family,
        font_size=body.font_size,
        font_color=body.font_color,
        stroke_color=body.stroke_color,
        stroke_width=body.stroke_width,
        text_align=body.text_align,
    )
    if body.mode == "manual_text" and body.manual_text:
        bubble.human_translation = body.manual_text

    db.add(bubble)
    db.commit()
    db.refresh(bubble)

    # Update chapter bubble count
    chapter = db.get(Chapter, page.chapter_id)
    if chapter:
        chapter.total_bubbles = (
            db.query(func.count(Bubble.id))
            .join(Page)
            .filter(Page.chapter_id == chapter.id)
            .scalar()
            or 0
        )
        db.commit()

    return _bubble_out(bubble)


@router.post("/bubbles/{bubble_id}/polygon", response_model=BubbleOut)
def update_polygon(bubble_id: int, body: PolygonUpdate, db: Session = Depends(get_db)):
    """Update the polygon shape of a bubble (vertex editing)."""
    bubble = db.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(404, "Bubble not found")

    bubble.mask_polygon = json.dumps(body.points)

    # Update bbox from polygon bounds
    if body.points:
        xs = [p["x"] for p in body.points]
        ys = [p["y"] for p in body.points]
        bubble.x1 = int(min(xs))
        bubble.y1 = int(min(ys))
        bubble.x2 = int(max(xs))
        bubble.y2 = int(max(ys))

    db.commit()
    db.refresh(bubble)
    return _bubble_out(bubble)


@router.post("/bubbles/{bubble_id}/font", response_model=BubbleOut)
def update_font(bubble_id: int, body: FontUpdate, db: Session = Depends(get_db)):
    """Update font/color settings for a bubble."""
    bubble = db.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(404, "Bubble not found")

    for field in ("font_family", "font_size", "font_color", "stroke_color", "stroke_width", "text_align"):
        val = getattr(body, field)
        if val is not None:
            setattr(bubble, field, val)

    db.commit()
    db.refresh(bubble)
    return _bubble_out(bubble)


@router.post("/bubbles/{bubble_id}/ocr")
def rerun_ocr(bubble_id: int, db: Session = Depends(get_db)):
    """Re-run OCR on a single bubble's region."""
    bubble = db.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(404, "Bubble not found")
    page = db.get(Page, bubble.page_id)
    if not page or not page.original_image_path:
        raise HTTPException(400, "No original image for this page")

    from pathlib import Path
    if not Path(page.original_image_path).exists():
        raise HTTPException(400, "Original image file missing")

    try:
        import numpy as np
        from PIL import Image
        from src.utils.ocr_smart import ocr_region_with_preprocessing
        from src.models.ocr import OCRExtractor

        image = np.array(Image.open(page.original_image_path).convert("RGB"))
        ocr = OCRExtractor()
        text, confidence = ocr_region_with_preprocessing(
            ocr.model, image,
            (bubble.x1, bubble.y1, bubble.x2, bubble.y2),
            bubble.bubble_index,
        )
        bubble.japanese_text = text
        bubble.ocr_confidence = confidence
        db.commit()
        return {"japanese_text": text, "ocr_confidence": confidence}
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")


class ApplyBody(BaseModel):
    mode: str | None = None
    human_translation: str | None = None


@router.post("/bubbles/{bubble_id}/apply")
def apply_bubble(bubble_id: int, body: ApplyBody = ApplyBody(), db: Session = Depends(get_db)):
    """
    Apply the operation for this bubble: inpaint, optionally render text.
    Accepts mode + human_translation in the request body so the frontend
    can pass current editor state without a separate save round-trip.
    """
    bubble = db.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(404, "Bubble not found")
    page = db.get(Page, bubble.page_id)
    if not page:
        raise HTTPException(400, "No page for this bubble")

    # Persist editor state sent from the frontend before processing
    if body.mode is not None:
        bubble.mode = body.mode
    if body.human_translation is not None:
        bubble.human_translation = body.human_translation
    db.commit()

    from pathlib import Path
    import shutil

    # Always work on the inpainted image — never overwrite the original.
    # If no inpainted image exists yet, create one from the original now.
    if page.inpainted_image_path and Path(page.inpainted_image_path).exists():
        base_path = page.inpainted_image_path
    elif page.original_image_path and Path(page.original_image_path).exists():
        orig = Path(page.original_image_path)
        inpainted = orig.parent / f"inpainted_{orig.name}"
        shutil.copy2(orig, inpainted)
        page.inpainted_image_path = str(inpainted)
        db.commit()
        base_path = str(inpainted)
    else:
        raise HTTPException(400, "No source image found for this page")

    from PIL import Image
    import numpy as np

    try:
        img = Image.open(base_path).convert("RGB")
    except Exception as e:
        raise HTTPException(500, f"Cannot open image: {e}")

    # Reload bubble after commit so we see the freshly persisted values
    db.refresh(bubble)
    bbox = (bubble.x1, bubble.y1, bubble.x2, bubble.y2)

    # Compute the wipe region: the full bubble area, not just the OCR text
    # bbox. On re-Apply the base image already has previously-rendered text
    # that may extend beyond the YOLO bbox (mask-aware placement can fill
    # the entire bubble interior), so inpainting only the text bbox leaves
    # stale characters visible. Union the text bbox with any stored polygon
    # bounding box and pad generously.
    def _wipe_box(b, img_w: int, img_h: int):
        x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
        for poly_field in (b.mask_polygon, b.mask_points):
            if not poly_field:
                continue
            try:
                pts = json.loads(poly_field)
            except Exception:
                continue
            if not pts:
                continue
            if isinstance(pts[0], dict):
                xs = [int(p["x"]) for p in pts]
                ys = [int(p["y"]) for p in pts]
            elif isinstance(pts[0], (list, tuple)):
                xs = [int(p[0]) for p in pts]
                ys = [int(p[1]) for p in pts]
            else:
                continue
            if xs and ys:
                x1 = min(x1, min(xs))
                y1 = min(y1, min(ys))
                x2 = max(x2, max(xs))
                y2 = max(y2, max(ys))
        pad = max(6, int(max(x2 - x1, y2 - y1) * 0.08))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_w, x2 + pad)
        y2 = min(img_h, y2 + pad)
        return (x1, y1, x2, y2)

    wipe_box = _wipe_box(bubble, img.width, img.height)

    # Step 0: If we have a clean post-inpaint base image saved by the pipeline
    # (…/chapters/{id}/clean/page_XXXX.png), copy that region's pristine pixels
    # into the working image BEFORE the per-bubble inpainting step. This wipes
    # any previously-rendered translation text from the bubble region — the
    # pipeline inpainter already removed the original Japanese text, so the
    # clean base has no text at all in this area. Without this, re-applying a
    # bubble would try to inpaint over rendered English text, which often
    # leaves ghosting because the local inpainter isn't as strong as LaMa run
    # on a full clean slate.
    try:
        inpainted_p = Path(base_path)
        clean_p = inpainted_p.parent.parent / "clean" / inpainted_p.name
        if clean_p.exists():
            clean_img = Image.open(clean_p).convert("RGB")
            if clean_img.size == img.size:
                x1, y1, x2, y2 = wipe_box
                clean_patch = clean_img.crop((x1, y1, x2, y2))
                img.paste(clean_patch, (x1, y1))
    except Exception as e:
        log.warning(f"Could not restore clean base for bubble {bubble_id}: {e}")

    # Step 1: Inpaint — always save immediately so inpainting is never lost
    if bubble.mode != "review_later":
        try:
            from src.models.inpainter import TextInpainter
            inpainter = TextInpainter()
            if inpainter.available:
                img = inpainter.inpaint_region(img, wipe_box)
            else:
                # OpenCV fallback — fill the entire bbox region with the
                # dominant background colour, then blend edges.  This is
                # more reliable than threshold-based masking which misses
                # coloured text (e.g. pipeline-rendered blue text).
                import cv2
                arr = np.array(img)  # RGB
                x1, y1, x2, y2 = wipe_box

                # Sample dominant background colour from a thin border
                # strip around the bbox (4px inside each edge).
                border = 4
                strips = []
                region = arr[y1:y2, x1:x2]
                if region.shape[0] > border * 2 and region.shape[1] > border * 2:
                    strips.append(region[:border, :])            # top
                    strips.append(region[-border:, :])           # bottom
                    strips.append(region[:, :border])            # left
                    strips.append(region[:, -border:])           # right
                    border_pixels = np.concatenate([s.reshape(-1, 3) for s in strips])
                    bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
                else:
                    bg_color = np.array([255, 255, 255], dtype=np.uint8)

                # Fill the entire bbox with the background colour
                arr[y1:y2, x1:x2] = bg_color

                # Feather the edges: Gaussian-blur a 6px border so the
                # fill blends smoothly into the surrounding artwork.
                feather = 6
                fy1 = max(0, y1 - feather)
                fy2 = min(arr.shape[0], y2 + feather)
                fx1 = max(0, x1 - feather)
                fx2 = min(arr.shape[1], x2 + feather)

                # Create a soft mask: 1 inside bbox, 0 outside, blurred edge
                mask = np.zeros((fy2 - fy1, fx2 - fx1), dtype=np.float32)
                mask[y1 - fy1:y2 - fy1, x1 - fx1:x2 - fx1] = 1.0
                mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), feather / 2)

                # Blend: filled region × mask + original × (1 - mask)
                original_patch = np.array(img)[fy1:fy2, fx1:fx2].astype(np.float32)
                filled_patch = arr[fy1:fy2, fx1:fx2].astype(np.float32)
                mask_3d = mask[:, :, np.newaxis]
                blended = (filled_patch * mask_3d + original_patch * (1 - mask_3d))
                arr[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)

                img = Image.fromarray(arr)
        except Exception as e:
            log.warning(f"Inpainting failed for bubble {bubble_id}: {e}")

        # Save after inpainting — even if text render fails later, inpainting is persisted
        try:
            img.save(base_path)
        except Exception as e:
            raise HTTPException(500, f"Cannot save image after inpainting: {e}")

    # Step 2: Render text
    text_to_render = None
    if bubble.mode == "translate_and_inpaint":
        text_to_render = bubble.human_translation or bubble.suggested_translation
    elif bubble.mode == "manual_text":
        text_to_render = bubble.human_translation

    log.info(f"apply_bubble {bubble_id}: mode={bubble.mode!r} text={text_to_render!r}")

    if text_to_render and text_to_render.strip():
        try:
            img = _render_bubble_text(img, text_to_render.strip(), bubble, bbox)
            img.save(base_path)
        except Exception as e:
            log.error(f"Text rendering failed for bubble {bubble_id}: {e}", exc_info=True)
            # Inpainting was already saved; return partial success
            return {"status": "ok", "image_url": f"/api/images/{page.id}/inpainted",
                    "warning": f"Text render failed: {e}"}

    db.commit()
    return {"status": "ok", "image_url": f"/api/images/{page.id}/inpainted"}


@router.post("/bubbles/{bubble_id}/translate")
def translate_bubble(bubble_id: int, db: Session = Depends(get_db)):
    """Translate the bubble's Japanese OCR text using the best available engine."""
    bubble = db.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(404, "Bubble not found")
    if not bubble.japanese_text or not bubble.japanese_text.strip():
        raise HTTPException(400, "No Japanese text — run OCR first")

    try:
        translated = _quick_translate(bubble.japanese_text.strip())
        bubble.suggested_translation = translated
        db.commit()
        return {"suggested_translation": translated}
    except Exception as e:
        log.exception("translate_bubble failed")
        raise HTTPException(500, f"Translation failed: {e}")


class TypeUpdate(BaseModel):
    bubble_type: str


@router.patch("/bubbles/{bubble_id}/type", response_model=BubbleOut)
def update_bubble_type(bubble_id: int, body: TypeUpdate, db: Session = Depends(get_db)):
    """Change the bubble_type label (speech / sfx / narration / thought / signs)."""
    bubble = db.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(404, "Bubble not found")
    bubble.bubble_type = body.bubble_type
    db.commit()
    db.refresh(bubble)
    return _bubble_out(bubble)


# ── Text rendering helper ─────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FONT_MAP: dict[str, str] = {
    "Bangers":     str(_PROJECT_ROOT / "assets" / "fonts" / "Bangers-Regular.ttf"),
    "Anime Ace":   str(_PROJECT_ROOT / "assets" / "fonts" / "Bangers-Regular.ttf"),
    "Wild Words":  str(_PROJECT_ROOT / "assets" / "fonts" / "Bangers-Regular.ttf"),
    "DejaVu Sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "Arial":       "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}
_FALLBACK_FONT = str(_PROJECT_ROOT / "assets" / "fonts" / "Bangers-Regular.ttf")


def _hex_to_rgb(hex_color: str, default=(0, 0, 0)) -> tuple[int, int, int]:
    try:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except Exception:
        return default


def _render_bubble_text(img, text: str, bubble, bbox: tuple) -> "Image.Image":
    """
    Render *text* inside *bbox* on *img*, respecting the bubble's font/color/align settings.
    Returns a new RGB PIL Image.
    """
    from PIL import Image, ImageDraw, ImageFont
    import textwrap

    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    # ── Font ──────────────────────────────────────────────────────────
    font_path = _FONT_MAP.get(bubble.font_family or "Bangers", _FALLBACK_FONT)
    if not Path(font_path).exists():
        font_path = _FALLBACK_FONT

    # ── Colours ───────────────────────────────────────────────────────
    text_rgb   = _hex_to_rgb(bubble.font_color   or "#000000")
    stroke_rgb = _hex_to_rgb(bubble.stroke_color or "#ffffff")
    stroke_w   = max(0, bubble.stroke_width or 1)

    # ── Alignment ─────────────────────────────────────────────────────
    align = bubble.text_align or "center"

    # ── Padding (10% of box or 6px min) ───────────────────────────────
    pad_x = max(6, box_w // 10)
    pad_y = max(6, box_h // 10)
    avail_w = max(1, box_w - 2 * pad_x)
    avail_h = max(1, box_h - 2 * pad_y)

    # ── Auto-size or use user value ───────────────────────────────────
    user_size = bubble.font_size  # None = auto
    max_size  = user_size if user_size else 28
    min_size  = 8

    def _make_font(size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            return ImageFont.load_default()

    def _wrap_and_measure(size: int):
        font = _make_font(size)
        dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        m = dummy.textbbox((0, 0), "W", font=font)
        char_w = max(1, m[2] - m[0])
        n_chars = max(1, avail_w // char_w)
        wrapped = "\n".join(textwrap.wrap(text, width=n_chars,
                                          break_long_words=True,
                                          break_on_hyphens=False))
        tb = dummy.multiline_textbbox((0, 0), wrapped, font=font,
                                      stroke_width=stroke_w)
        return wrapped, font, tb[2] - tb[0], tb[3] - tb[1]

    best_wrapped, best_font = text, _make_font(min_size)

    if user_size:
        best_wrapped, best_font, _, _ = _wrap_and_measure(user_size)
    else:
        for sz in range(max_size, min_size - 1, -2):
            wrapped, font, tw, th = _wrap_and_measure(sz)
            if tw <= avail_w and th <= avail_h:
                best_wrapped, best_font = wrapped, font
                break

    # ── Draw on transparent tile ──────────────────────────────────────
    tile = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)

    cx = box_w // 2
    cy = box_h // 2

    draw.multiline_text(
        (cx, cy),
        best_wrapped,
        font=best_font,
        fill=(*text_rgb, 255),
        anchor="mm",
        align=align,
        stroke_width=stroke_w,
        stroke_fill=(*stroke_rgb, 255),
    )

    # ── Composite tile onto page ──────────────────────────────────────
    base = img.convert("RGBA")
    base.paste(tile, (x1, y1), mask=tile)
    return base.convert("RGB")


def _quick_translate(text: str) -> str:
    """Translate Japanese text using the best available engine (Ollama → Google)."""
    # Try Ollama/Gemma (already running for the pipeline)
    try:
        from src.translators.gemma import GemmaTranslator
        tr = GemmaTranslator(source_lang="ja", target_lang="en")
        result = tr.translate(text)
        if result and result.strip():
            return result.strip()
    except Exception as e:
        log.debug(f"Gemma translation failed: {e}")

    # Try Google Translate via deep-translator
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        result = GoogleTranslator(source="ja", target="en").translate(text)
        if result and result.strip():
            return result.strip()
    except Exception as e:
        log.debug(f"Google translation failed: {e}")

    raise RuntimeError("No translation engine available (Ollama not running, deep-translator not installed)")


class NotesBody(BaseModel):
    notes: str


@router.patch("/bubbles/{bubble_id}/notes", response_model=BubbleOut)
def update_notes(bubble_id: int, body: NotesBody, db: Session = Depends(get_db)):
    """Save or update editorial notes for a bubble."""
    bubble = db.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(404, "Bubble not found")
    bubble.notes = body.notes
    db.commit()
    db.refresh(bubble)
    return _bubble_out(bubble)


@router.delete("/bubbles/{bubble_id}")
def delete_bubble(bubble_id: int, db: Session = Depends(get_db)):
    """Delete a bubble entirely."""
    bubble = db.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(404, "Bubble not found")

    page = db.get(Page, bubble.page_id)
    db.delete(bubble)
    db.commit()

    # Update chapter count
    if page:
        chapter = db.get(Chapter, page.chapter_id)
        if chapter:
            chapter.total_bubbles = (
                db.query(func.count(Bubble.id))
                .join(Page)
                .filter(Page.chapter_id == chapter.id)
                .scalar()
                or 0
            )
            db.commit()

    return {"status": "deleted"}


# ── Helpers ──────────────────────────────────────────────────────────

def _update_profile(db: Session, bubble: Bubble, series_name: str):
    """
    Feed a reviewed bubble's translation back into the MangaProfile.

    This is the core feedback loop:
      1. Add JP→EN pair to rolling translation memory
      2. Auto-extract glossary terms from the pair
      3. Save profile to disk

    Effect: next chapter of the same series gets better suggestions.
    """
    final_en = bubble.human_translation or bubble.suggested_translation
    jp = bubble.japanese_text
    if not jp or not final_en:
        return

    # Resolve chapter/page numbers
    page = db.get(Page, bubble.page_id)
    chapter = db.get(Chapter, page.chapter_id) if page else None
    chapter_num = chapter.chapter_num if chapter else 1
    page_num = page.page_num if page else 1

    try:
        profiles_dir = Path(__file__).resolve().parents[2] / "profiles"
        profiles_dir.mkdir(exist_ok=True)

        from src.translation.manga_profile import MangaProfile
        profile = MangaProfile(series_name, profiles_dir=str(profiles_dir))

        # 1. Add to rolling memory (also triggers auto_update_glossary_from_pair)
        profile.add_translated_lines([{
            "japanese": jp,
            "english": final_en,
            "chapter": chapter_num,
            "page": page_num,
        }])

        profile.save()
        log.info(f"Profile updated for '{series_name}': {jp[:20]}... → {final_en[:20]}...")
    except Exception as e:
        log.warning(f"Failed to update profile for '{series_name}': {e}")


def _get_bubble_and_series(bubble_id: int, db: Session) -> tuple[Bubble, str]:
    bubble = db.get(Bubble, bubble_id)
    if not bubble:
        raise HTTPException(404, "Bubble not found")
    page = db.get(Page, bubble.page_id)
    chapter = db.get(Chapter, page.chapter_id) if page else None
    project = db.get(Project, chapter.project_id) if chapter else None
    series = project.series_name if project else "Unknown"
    return bubble, series


def _bubble_out(b: Bubble) -> BubbleOut:
    return BubbleOut(
        id=b.id,
        bubble_index=b.bubble_index,
        bubble_type=b.bubble_type or "speech",
        x1=b.x1 or 0,
        y1=b.y1 or 0,
        x2=b.x2 or 0,
        y2=b.y2 or 0,
        mask_points=b.mask_points,
        japanese_text=b.japanese_text,
        suggested_translation=b.suggested_translation,
        human_translation=b.human_translation,
        status=b.status or "pending",
        ocr_confidence=b.ocr_confidence,
        quality_score=b.quality_score,
        edit_distance=b.edit_distance,
        notes=b.notes,
        is_manual=b.is_manual or False,
        mode=b.mode or "translate_and_inpaint",
        mask_polygon=b.mask_polygon,
        font_family=b.font_family,
        font_size=b.font_size,
        font_color=b.font_color,
        stroke_color=b.stroke_color,
        stroke_width=b.stroke_width,
        text_align=b.text_align,
    )
