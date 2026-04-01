"""Google Translate implementation backed by ``deep-translator``."""

from __future__ import annotations

from .base import BaseTranslator


class GoogleTranslator(BaseTranslator):
    """Single-text Google translator wrapper used by the main pipeline."""

    def __init__(self, source_lang: str, target_lang: str):
        super().__init__(source_lang, target_lang)
        self.client = None
        try:
            from deep_translator import GoogleTranslator as DeepTranslatorClient
        except Exception as exc:
            raise RuntimeError("Google translator requires the 'deep-translator' package.") from exc

        self.client = DeepTranslatorClient(source=source_lang, target=target_lang)

    @property
    def name(self) -> str:
        return "Google"

    def translate(self, text: str, context_prompt: str = "", story_context: str = "") -> str:
        text = text.strip()
        if not text:
            return ""

        try:
            return self.client.translate(text)
        except Exception as exc:
            print(f"⚠️ Google translation failed: {exc}")
            raise

    def is_available(self) -> bool:
        return self.client is not None


# Backwards-compatible alias for older imports.
Google = GoogleTranslator
