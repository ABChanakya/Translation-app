"""
Async FastAPI service for the manga translation pipeline.

Runs alongside the existing Flask app on a separate port (default 8000).
Provides two translation endpoints:
  POST /translate       — Tier 1 (manga-ocr + LLM + LaMa)
  POST /translate/vlm   — Tier 2 (YOLO + Gemma 3 vision + LaMa)

asyncio.Semaphore(1) prevents concurrent GPU inference — vision models consume
nearly all VRAM and queue requests internally anyway.

Start:
    cd manga_translator_clean
    uvicorn src.fastapi_service:app --host 0.0.0.0 --port 8000

Or via launch.sh option 4.
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a module
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

app = FastAPI(
    title="Manga Translator API",
    version="2.0",
    description=(
        "Human-assisted manga translation tool. "
        "AI handles detection, OCR, and draft translations. "
        "Humans review, correct, and export."
    ),
)

# CORS — allow the React dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount review-tool API routers
from src.api.projects import router as projects_router
from src.api.chapters import router as chapters_router
from src.api.bubbles import router as bubbles_router
from src.api.images import router as images_router

app.include_router(projects_router)
app.include_router(chapters_router)
app.include_router(bubbles_router)
app.include_router(images_router)

# One GPU inference at a time — prevents VRAM contention when multiple
# requests arrive simultaneously (Ollama queues internally but the image
# encoding step still allocates GPU memory on the caller side).
_gpu_sem = asyncio.Semaphore(1)

# ── Serve built React frontend ───────────────────────────────────────────────
# The Vite build output lives in frontend/dist/. Mount static assets first,
# then add a catch-all that serves index.html for client-side routing.
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    # Serve JS/CSS/images at /assets/...
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="frontend-assets")

    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for any non-API route (SPA client routing)."""
        file_path = _FRONTEND_DIST / full_path
        if full_path and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_FRONTEND_DIST / "index.html"))


# ── sync pipeline runners (called in executor to avoid blocking the loop) ─────

def _run_tier1(img_bytes: bytes, engine: str, target_lang: str, vlm_context: bool) -> tuple:
    from src.pipeline import MangaTranslationPipeline
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pipeline = MangaTranslationPipeline(
        source_lang="ja",
        target_lang=target_lang,
        translation_engine=engine,
        vlm_context_enabled=vlm_context,
    )
    result_img, logs = pipeline.process(image)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    return buf.getvalue(), logs


def _run_tier2(img_bytes: bytes, model: str) -> tuple:
    from src.vlm_pipeline import VLMTranslationPipeline
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pipeline = VLMTranslationPipeline(model=model)
    result_img, logs = pipeline.process(image)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    return buf.getvalue(), logs


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/translate")
async def translate_tier1(
    file: UploadFile = File(...),
    engine: str = Form(default="gemma3"),
    target_lang: str = Form(default="en"),
    vlm_context: bool = Form(default=False),
):
    """
    Tier 1: manga-ocr → batch LLM translation → LaMa inpainting → text render.

    Args:
        file: Manga page image (PNG/JPG/WEBP)
        engine: Translation engine id (gemma3, translategemma, google, deepl, …)
        target_lang: ISO language code for output (en, fr, de, …)
        vlm_context: If true, sends the page to Gemma 3 vision once for scene context
    """
    img_bytes = await file.read()
    loop = asyncio.get_event_loop()
    async with _gpu_sem:
        png, logs = await loop.run_in_executor(
            None, _run_tier1, img_bytes, engine, target_lang, vlm_context
        )
    return JSONResponse({
        "image_b64": base64.b64encode(png).decode(),
        "logs": logs,
    })


@app.post("/translate/vlm")
async def translate_tier2(
    file: UploadFile = File(...),
    model: str = Form(default="gemma3:12b"),
):
    """
    Tier 2: YOLO detection → Gemma 3 vision per region (OCR+translate) → LaMa.

    Simpler but ~20-30% lower OCR accuracy than Tier 1. Useful for unusual
    or handwritten fonts, or when manga-ocr is unavailable.

    Args:
        file: Manga page image
        model: Ollama model tag (gemma3:12b, gemma3:27b, translategemma:12b)
    """
    img_bytes = await file.read()
    loop = asyncio.get_event_loop()
    async with _gpu_sem:
        png, logs = await loop.run_in_executor(None, _run_tier2, img_bytes, model)
    return JSONResponse({
        "image_b64": base64.b64encode(png).decode(),
        "logs": logs,
    })


# ── dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run("src.fastapi_service:app", host="0.0.0.0", port=8000, reload=False)
