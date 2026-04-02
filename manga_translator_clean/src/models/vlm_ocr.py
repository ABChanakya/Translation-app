"""
VLM visual context extraction for improved translation quality.

Sends the manga page image to Gemma 3 vision once per page to extract a
short scene description (character emotions, tone, situation). This context
is then appended to the translation prompt so the translator knows the mood
and scenario for every speech bubble on that page.

Architecture:
  manga-ocr (accurate OCR) + PageContextExtractor (visual context, 1 call/page)
  → translator receives text + visual context → better translations

Cost: 1 extra Ollama call per page (not per region — much cheaper than per-region VLM OCR).
"""

from __future__ import annotations

import base64
import io

from PIL import Image


class PageContextExtractor:
    """
    Extracts a short visual context description from a manga page image.

    Designed to be called once per page before the batch translation call.
    The returned string is appended to the translation context prompt so the
    LLM understands the scene (who's speaking, emotion, situation) when
    translating every region on that page.

    Usage:
        extractor = PageContextExtractor(model="gemma3:12b")
        ctx = extractor.extract_context(page_image)
        # ctx → "Two characters arguing in a hallway. The speaker on the left
        #         looks furious, pointing accusingly at the other."
    """

    # Short prompt — long prompts trigger Gemma3's infinite-repetition bug
    _PROMPT = (
        "Describe this manga page in 2 sentences: "
        "Who is speaking or acting? What is the mood or emotion? "
        "Any important visual context for translation?"
    )

    def __init__(self, model: str = "gemma3:12b"):
        self.model = model

    @staticmethod
    def _add_margin(img: Image.Image, pct: float = 0.10) -> Image.Image:
        """
        Add a white border around the image.

        Pan & Scan (Gemma 3's adaptive tiling for non-square images) can crop
        text near the edges. A 10% white margin prevents this.
        """
        w, h = img.size
        pw, ph = int(w * pct), int(h * pct)
        result = Image.new("RGB", (w + 2 * pw, h + 2 * ph), (255, 255, 255))
        result.paste(img, (pw, ph))
        return result

    def extract_context(self, page_image: Image.Image) -> str:
        """
        Return a short visual context string, or "" on failure.

        Safe to call even if Ollama is unavailable — failures are caught and
        logged so the main pipeline continues without context.
        """
        try:
            import ollama  # type: ignore

            padded = self._add_margin(page_image)
            buf = io.BytesIO()
            # JPEG is smaller than PNG — reduces transfer size to Ollama
            padded.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()

            resp = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": self._PROMPT,
                    "images": [b64],
                }],
                options={"temperature": 0.2, "num_ctx": 4096},
            )
            return resp.message.content.strip()
        except Exception as e:
            print(f"   ⚠️  VLM context extraction failed: {e}")
            return ""
