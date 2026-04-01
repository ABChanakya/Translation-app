"""
DeepL professional translation API.
"""

from src.translators.base import BaseTranslator
from config.settings import DEEPL_API_KEY


class DeepLTranslator(BaseTranslator):
    """Translator using DeepL API"""
    
    def __init__(self, source_lang: str, target_lang: str):
        super().__init__(source_lang, target_lang)
        self.translator = None
        if DEEPL_API_KEY:
            try:
                import deepl
                self.translator = deepl.Translator(DEEPL_API_KEY)
            except Exception:
                self.translator = None
    
    @property
    def name(self) -> str:
        return "DeepL"
    
    def translate(self, text: str, context_prompt: str = "", story_context: str = "") -> str:
        """Translate text using DeepL API"""
        text = text.strip()
        if not text:
            return ""
        
        if not self.is_available():
            raise ValueError("DeepL API key not configured")
        
        try:
            result = self.translator.translate_text(
                text,
                source_lang=self.source_lang.upper(),
                target_lang=self.target_lang.upper()
            )
            return result.text
        
        except Exception as e:
            print(f"⚠️ DeepL translation failed: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if DeepL is configured"""
        return self.translator is not None
