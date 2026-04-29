"""Chapter upload, processing, and status endpoints."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, WebSocket
from pydantic import BaseModel
from PIL import Image
from sqlalchemy.orm import Session

from src.api.deps import get_db, gpu_sem
from src.db.models import Bubble, Chapter, Page, Project

router = APIRouter(prefix="/api/chapters", tags=["chapters"])

# In-memory processing status (chapter_id -> list of status messages)
_processing_status: dict[int, list[dict]] = {}

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class FindReplaceBody(BaseModel):
    find: str
    replace: str
    field: str = "suggested_translation"  # suggested_translation | human_translation | japanese_text
    case_sensitive: bool = False


class ChapterOut(BaseModel):
    id: int
    project_id: int
    chapter_num: int
    status: str
    total_pages: int
    total_bubbles: int
    reviewed_bubbles: int
    accepted_bubbles: int
    created_at: str | None

    model_config = {"from_attributes": True}


class PageOut(BaseModel):
    id: int
    page_num: int
    status: str
    bubble_count: int
    original_image_url: str | None
    inpainted_image_url: str | None

    model_config = {"from_attributes": True}


@router.post("/upload")
async def upload_chapter(
    files: list[UploadFile] = File(...),
    series_name: str = Form(...),
    chapter_num: int = Form(...),
    translation_engine: str = Form(default="Gemma3"),
    detection_confidence: float = Form(default=0.10),
    nms_iou_threshold: float = Form(default=0.55),
    db: Session = Depends(get_db),
):
    """Upload chapter images (or a single ZIP/CBZ) and start processing."""
    # Get or create project
    project = db.query(Project).filter(Project.series_name == series_name).first()
    if not project:
        project = Project(series_name=series_name, created_at=datetime.now(timezone.utc))
        db.add(project)
        db.commit()
        db.refresh(project)

    # Create chapter
    chapter = Chapter(
        project_id=project.id,
        chapter_num=chapter_num,
        status="processing",
        created_at=datetime.now(timezone.utc),
    )
    db.add(chapter)
    db.commit()
    db.refresh(chapter)

    # Save uploaded files to disk
    chapter_dir = DATA_DIR / "chapters" / str(chapter.id)
    originals_dir = chapter_dir / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []

    for f in files:
        content = await f.read()
        fname = f.filename or "upload"

        if fname.lower().endswith((".zip", ".cbz")):
            # Extract ZIP/CBZ
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in sorted(zf.namelist()):
                    if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        data = zf.read(name)
                        dest = originals_dir / Path(name).name
                        dest.write_bytes(data)
                        image_paths.append(dest)
        elif fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            dest = originals_dir / fname
            dest.write_bytes(content)
            image_paths.append(dest)

    image_paths.sort(key=lambda p: p.name)

    # Create page rows
    for i, img_path in enumerate(image_paths):
        page = Page(
            chapter_id=chapter.id,
            page_num=i + 1,
            original_image_path=str(img_path),
            status="pending",
        )
        db.add(page)

    chapter.total_pages = len(image_paths)
    db.commit()

    # Start background processing
    _processing_status[chapter.id] = []
    asyncio.get_event_loop().create_task(
        _process_chapter(chapter.id, translation_engine, detection_confidence, nms_iou_threshold)
    )

    return {
        "chapter_id": chapter.id,
        "total_pages": len(image_paths),
        "status": "processing",
    }


@router.get("/{chapter_id}", response_model=ChapterOut)
def get_chapter(chapter_id: int, db: Session = Depends(get_db)):
    chapter = db.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(404, "Chapter not found")
    return ChapterOut(
        id=chapter.id,
        project_id=chapter.project_id,
        chapter_num=chapter.chapter_num,
        status=chapter.status,
        total_pages=chapter.total_pages or 0,
        total_bubbles=chapter.total_bubbles or 0,
        reviewed_bubbles=chapter.reviewed_bubbles or 0,
        accepted_bubbles=chapter.accepted_bubbles or 0,
        created_at=chapter.created_at.isoformat() if chapter.created_at else None,
    )


@router.get("/{chapter_id}/pages", response_model=list[PageOut])
def get_chapter_pages(chapter_id: int, db: Session = Depends(get_db)):
    pages = (
        db.query(Page)
        .filter(Page.chapter_id == chapter_id)
        .order_by(Page.page_num)
        .all()
    )
    result = []
    for p in pages:
        bubble_count = db.query(Bubble).filter(Bubble.page_id == p.id).count()
        result.append(PageOut(
            id=p.id,
            page_num=p.page_num,
            status=p.status,
            bubble_count=bubble_count,
            original_image_url=f"/api/images/{p.id}/original" if p.original_image_path else None,
            inpainted_image_url=f"/api/images/{p.id}/inpainted" if p.inpainted_image_path else None,
        ))
    return result


@router.websocket("/{chapter_id}/status")
async def chapter_status_ws(websocket: WebSocket, chapter_id: int):
    """WebSocket endpoint for real-time processing progress."""
    await websocket.accept()
    last_idx = 0
    try:
        while True:
            msgs = _processing_status.get(chapter_id, [])
            if len(msgs) > last_idx:
                for msg in msgs[last_idx:]:
                    await websocket.send_json(msg)
                last_idx = len(msgs)
                # Check if done
                if msgs and msgs[-1].get("stage") == "done":
                    break
            await asyncio.sleep(0.3)
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/{chapter_id}/export")
def export_chapter(chapter_id: int, format: str = "cbz", db: Session = Depends(get_db)):
    """Export a reviewed chapter as CBZ or PDF."""
    chapter = db.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(404, "Chapter not found")

    pages = (
        db.query(Page)
        .filter(Page.chapter_id == chapter_id)
        .order_by(Page.page_num)
        .all()
    )

    if format == "cbz":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for p in pages:
                # Prefer final (with human corrections rendered), fall back to inpainted
                img_path = p.final_image_path or p.inpainted_image_path or p.original_image_path
                if img_path and Path(img_path).exists():
                    zf.write(img_path, f"page_{p.page_num:04d}.png")
        buf.seek(0)
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=chapter_{chapter.chapter_num}.cbz"},
        )

    elif format == "pdf":
        images = []
        for p in pages:
            img_path = p.final_image_path or p.inpainted_image_path or p.original_image_path
            if img_path and Path(img_path).exists():
                images.append(Image.open(img_path).convert("RGB"))
        if not images:
            raise HTTPException(400, "No images to export")
        buf = io.BytesIO()
        images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
        buf.seek(0)
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            buf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=chapter_{chapter.chapter_num}.pdf"},
        )

    raise HTTPException(400, f"Unsupported format: {format}")


@router.post("/{chapter_id}/find-replace")
def find_replace(chapter_id: int, body: FindReplaceBody, db: Session = Depends(get_db)):
    """
    Batch find-and-replace across all bubble text fields in a chapter.
    Returns the count of modified bubbles and their IDs.
    """
    ALLOWED_FIELDS = {"suggested_translation", "human_translation", "japanese_text"}
    if body.field not in ALLOWED_FIELDS:
        raise HTTPException(400, f"field must be one of {sorted(ALLOWED_FIELDS)}")

    chapter = db.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(404, "Chapter not found")

    pages = db.query(Page).filter(Page.chapter_id == chapter_id).all()
    page_ids = [p.id for p in pages]
    bubbles = db.query(Bubble).filter(Bubble.page_id.in_(page_ids)).all()

    find_str = body.find if body.case_sensitive else body.find.lower()
    affected_ids: list[int] = []

    for bubble in bubbles:
        current: str | None = getattr(bubble, body.field)
        if current is None:
            continue
        haystack = current if body.case_sensitive else current.lower()
        if find_str in haystack:
            # Perform the replacement preserving original case for non-case-sensitive
            if body.case_sensitive:
                setattr(bubble, body.field, current.replace(body.find, body.replace))
            else:
                import re as _re
                setattr(
                    bubble, body.field,
                    _re.sub(_re.escape(body.find), body.replace, current, flags=_re.IGNORECASE),
                )
            affected_ids.append(bubble.id)

    if affected_ids:
        db.commit()

    return {"replaced": len(affected_ids), "bubble_ids": affected_ids}


@router.get("/{chapter_id}/analytics")
def get_chapter_analytics(chapter_id: int, db: Session = Depends(get_db)):
    """
    Aggregated analytics for the Export / heatmap view:
    - Overall stats (total, reviewed, accepted, avg quality score)
    - Top correction discrepancies (suggested → human, sorted by frequency)
    - Quality score distribution buckets
    """
    chapter = db.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(404, "Chapter not found")

    pages = db.query(Page).filter(Page.chapter_id == chapter_id).all()
    page_ids = [p.id for p in pages]
    bubbles = db.query(Bubble).filter(Bubble.page_id.in_(page_ids)).all()

    total = len(bubbles)
    reviewed = sum(1 for b in bubbles if b.status != "pending")
    accepted = sum(1 for b in bubbles if b.status == "accepted")
    quality_scores = [b.quality_score for b in bubbles if b.quality_score is not None]
    avg_quality = round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else None

    # Quality score distribution: buckets 0–0.5, 0.5–0.7, 0.7–0.9, 0.9–1.0
    buckets = {"low": 0, "medium": 0, "high": 0, "excellent": 0}
    for q in quality_scores:
        if q < 0.5:
            buckets["low"] += 1
        elif q < 0.7:
            buckets["medium"] += 1
        elif q < 0.9:
            buckets["high"] += 1
        else:
            buckets["excellent"] += 1

    # Top correction discrepancies: where suggested ≠ human translation
    from collections import Counter
    discrepancies: Counter = Counter()
    for b in bubbles:
        if b.suggested_translation and b.human_translation:
            if b.suggested_translation.strip() != b.human_translation.strip():
                discrepancies[(b.suggested_translation.strip(), b.human_translation.strip())] += 1

    top_discrepancies = [
        {"suggested": k[0], "human": k[1], "frequency": v}
        for k, v in discrepancies.most_common(10)
    ]

    return {
        "chapter_id": chapter_id,
        "total_bubbles": total,
        "reviewed_bubbles": reviewed,
        "accepted_bubbles": accepted,
        "acceptance_rate": round(accepted / reviewed, 3) if reviewed else 0.0,
        "avg_quality_score": avg_quality,
        "quality_distribution": buckets,
        "top_discrepancies": top_discrepancies,
    }


# ── Background processing ──────────────────────────────────────────────

async def _process_chapter(chapter_id: int, engine: str, detection_confidence: float = 0.10, nms_iou_threshold: float = 0.55):
    """Run the ML pipeline on all pages of a chapter in the background."""
    from src.db.database import get_session, get_engine as get_db_engine

    db_engine = get_db_engine()
    session = get_session(db_engine)

    try:
        chapter = session.get(Chapter, chapter_id)
        if not chapter:
            return
        project = session.get(Project, chapter.project_id)
        series_name = project.series_name if project else "Unknown"

        # Load (or create) a MangaProfile for this series — shared across all pages
        # so glossary, character names, and translation memory carry through.
        from src.translation.manga_profile import MangaProfile
        profiles_dir = Path(__file__).resolve().parents[2] / "profiles"
        manga_profile = MangaProfile(series_name, profiles_dir=str(profiles_dir))
        print(f"📚 Profile loaded for '{series_name}': {manga_profile}")

        pages = (
            session.query(Page)
            .filter(Page.chapter_id == chapter_id)
            .order_by(Page.page_num)
            .all()
        )

        chapter_dir = DATA_DIR / "chapters" / str(chapter_id)
        inpainted_dir = chapter_dir / "inpainted"
        inpainted_dir.mkdir(parents=True, exist_ok=True)
        clean_dir = chapter_dir / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)

        total_bubbles = 0

        # Build the pipeline ONCE for the whole chapter so YOLO, OCR, and
        # segmenter models are only loaded from disk a single time.
        loop = asyncio.get_event_loop()
        _emit(chapter_id, 0, "detecting", "Loading ML models (first page only)...")
        async with gpu_sem:
            pipeline = await loop.run_in_executor(
                None,
                _build_pipeline,
                engine,
                detection_confidence,
                nms_iou_threshold,
                manga_profile,
                chapter.chapter_num,
            )

        # Route pipeline progress events into the WebSocket status stream so
        # the frontend Console_Output panel can show stage-by-stage progress
        # (region detection, OCR results, translations, inpainting) live.
        def _pipeline_progress(stage: str, message: str) -> None:
            try:
                _emit(chapter_id, getattr(pipeline, "page_num", 0), stage, message)
            except Exception:
                pass

        pipeline.progress_callback = _pipeline_progress

        for page in pages:
            page_num = page.page_num
            _emit(chapter_id, page_num, "detecting", f"Page {page_num}: Detecting text regions...")
            page.status = "processing"
            session.commit()

            if not page.original_image_path or not Path(page.original_image_path).exists():
                _emit(chapter_id, page_num, "error", f"Page {page_num}: Image not found")
                continue

            # Run the pipeline in executor (GPU-bound)
            clean_path = clean_dir / f"page_{page_num:04d}.png"
            async with gpu_sem:
                try:
                    _emit(chapter_id, page_num, "ocr", f"Page {page_num}: OCR + Translation...")
                    result_img, logs = await loop.run_in_executor(
                        None,
                        _run_pipeline_for_page,
                        page.original_image_path,
                        pipeline,
                        page_num,
                        manga_profile,
                        str(clean_path),
                    )
                except Exception as e:
                    _emit(chapter_id, page_num, "error", f"Page {page_num}: Pipeline failed: {e}")
                    page.status = "ready"  # still mark ready so user can see it
                    session.commit()
                    continue

            # Save inpainted image
            inpainted_path = inpainted_dir / f"page_{page_num:04d}.png"
            result_img.save(str(inpainted_path))
            page.inpainted_image_path = str(inpainted_path)

            # Create bubble rows from pipeline logs
            _emit(chapter_id, page_num, "saving", f"Page {page_num}: Saving {len(logs)} bubbles...")
            for i, log_entry in enumerate(logs):
                bbox = log_entry.get("bbox", (0, 0, 0, 0))
                bubble = Bubble(
                    page_id=page.id,
                    bubble_index=log_entry.get("bubble_index", i),
                    bubble_type=_infer_bubble_type(log_entry.get("class", "")),
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                    mask_points=json.dumps(log_entry.get("mask_points")) if log_entry.get("mask_points") else None,
                    japanese_text=log_entry.get("src_text", ""),
                    suggested_translation=log_entry.get("tgt_text", ""),
                    status="pending",
                    ocr_confidence=log_entry.get("ocr_confidence", 0.0),
                    quality_score=log_entry.get("quality_score"),
                )
                session.add(bubble)
                total_bubbles += 1

            page.status = "ready"
            session.commit()
            _emit(chapter_id, page_num, "done_page", f"Page {page_num}: Done ({len(logs)} bubbles)")

        chapter.total_bubbles = total_bubbles
        chapter.status = "ready"
        session.commit()
        _emit(chapter_id, 0, "done", f"Chapter ready: {chapter.total_pages} pages, {total_bubbles} bubbles")

    except Exception as e:
        _emit(chapter_id, 0, "error", f"Processing failed: {e}")
        try:
            chapter = session.get(Chapter, chapter_id)
            if chapter:
                chapter.status = "ready"  # mark ready even on error so it's accessible
                session.commit()
        except Exception:
            pass
    finally:
        session.close()


def _build_pipeline(
    engine: str,
    detection_confidence: float,
    nms_iou_threshold: float,
    manga_profile=None,
    chapter_num: int = 1,
):
    """
    Instantiate the ML pipeline once per chapter.
    All heavy models (YOLO, OCR, segmenter) are loaded here and reused
    across every page — avoids reloading from disk on each page.
    Runs in a thread executor so it doesn't block the async event loop.
    """
    from src.pipeline import MangaTranslationPipeline
    return MangaTranslationPipeline(
        source_lang="ja",
        target_lang="en",
        translation_engine=engine,
        detection_confidence=detection_confidence,
        nms_iou_threshold=nms_iou_threshold,
        chapter_num=chapter_num,
        page_num=1,
        manga_profile=manga_profile,
    )


def _run_pipeline_for_page(
    image_path: str,
    pipeline,
    page_num: int,
    manga_profile=None,
    clean_save_path: str | None = None,
) -> tuple[Image.Image, list[dict]]:
    """
    Process a single page with an already-loaded pipeline.
    Updates page_num and manga_profile on the pipeline before calling process()
    so rolling context stays current without reloading models.
    Runs in thread executor.
    """
    pipeline.page_num = page_num
    if manga_profile is not None:
        pipeline.manga_profile = manga_profile
    image = Image.open(image_path).convert("RGB")
    return pipeline.process(image, clean_save_path=clean_save_path)


def _infer_bubble_type(class_name: str) -> str:
    """Map YOLO class name to bubble_type."""
    lower = class_name.lower()
    if "sound" in lower or "sfx" in lower:
        return "sfx"
    if "narrat" in lower or "text" in lower:
        return "narration"
    if "sign" in lower:
        return "narration"
    return "speech"


def _emit(chapter_id: int, page_num: int, stage: str, message: str):
    """Append a status message for the WebSocket consumer."""
    entry = {"page": page_num, "stage": stage, "message": message, "ts": datetime.now(timezone.utc).isoformat()}
    _processing_status.setdefault(chapter_id, []).append(entry)
