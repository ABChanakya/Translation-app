"""
TranslateGemma translator via Ollama.

TranslateGemma is a Gemma 3 fine-tune optimized specifically for translation.
The 12B variant outperforms Gemma 3-27B on MetricX translation quality while
fitting in ~8 GB at Q4 quantization. It retains Gemma 3's vision capability,
so images can be sent to it exactly like gemma3.
"""

from config.settings import TRANSLATEGEMMA_MODEL
from src.translators.gemma import GemmaTranslator


class TranslateGemmaTranslator(GemmaTranslator):
    """Translation-optimized Gemma 3 fine-tune via Ollama (translategemma:12b)."""

    def __init__(self, source_lang: str, target_lang: str):
        super().__init__(source_lang, target_lang, model=TRANSLATEGEMMA_MODEL)

    @property
    def name(self) -> str:
        return "TranslateGemma"
