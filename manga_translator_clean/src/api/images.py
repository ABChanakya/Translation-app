"""Serve page images (original, inpainted, final) by page ID."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.db.models import Page

router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("/{page_id}/original")
def get_original_image(page_id: int, db: Session = Depends(get_db)):
    page = db.get(Page, page_id)
    if not page or not page.original_image_path:
        raise HTTPException(404, "Image not found")
    path = Path(page.original_image_path)
    if not path.exists():
        raise HTTPException(404, "Image file missing from disk")
    return FileResponse(path, media_type="image/png")


@router.get("/{page_id}/inpainted")
def get_inpainted_image(page_id: int, db: Session = Depends(get_db)):
    page = db.get(Page, page_id)
    if not page or not page.inpainted_image_path:
        raise HTTPException(404, "Image not found")
    path = Path(page.inpainted_image_path)
    if not path.exists():
        raise HTTPException(404, "Image file missing from disk")
    return FileResponse(path, media_type="image/png")


@router.get("/{page_id}/final")
def get_final_image(page_id: int, db: Session = Depends(get_db)):
    page = db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")
    # Prefer final, fall back to inpainted, then original
    for attr in ("final_image_path", "inpainted_image_path", "original_image_path"):
        img_path = getattr(page, attr)
        if img_path and Path(img_path).exists():
            return FileResponse(Path(img_path), media_type="image/png")
    raise HTTPException(404, "No image available")
