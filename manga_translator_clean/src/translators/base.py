"""
Base translator interface.
"""

from abc import ABC, abstractmethod

from .registry import get_engine_status


class BaseTranslator(ABC):
    """Abstract base class for all translators"""
    
    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    @abstractmethod
    def translate(self, text: str, context_prompt: str = "", story_context: str = "") -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            context_prompt: Optional context from previous pages for narrative continuity
            story_context: Optional global story context (characters, plot, glossary)
        
        Returns:
            Translated text
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this translator"""
        pass
    
    def translate_batch(self, texts: list, context_prompt: str = "", story_context: str = "") -> list:
        """
        Translate a list of texts in one call. Default: call translate() once per item.
        Subclasses that support true batching (e.g. Gemma3) override this.
        """
        return [self.translate(t, context_prompt=context_prompt, story_context=story_context) for t in texts]

    def is_available(self) -> bool:
        """Check if this translator is available/configured"""
        return True


class TranslatorFactory:
    """Factory for creating translator instances"""
    
    @staticmethod
    def create(engine: str, source_lang: str, target_lang: str):
        """
        Create a translator instance.
        
        Args:
            engine: Translation engine name
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Translator instance
        """
        engine_status = get_engine_status(engine)
        if not engine_status.enabled:
            reason = engine_status.disable_reason or "Engine is not currently available."
            raise RuntimeError(f"{engine_status.label} is unavailable. {reason}")

        engine_name = engine_status.engine_id

        if engine_name == "gemma3":
            from src.translators.gemma import GemmaTranslator
            return GemmaTranslator(source_lang, target_lang)
        
        elif engine_name == "google":
            from src.translators.google import GoogleTranslator
            return GoogleTranslator(source_lang, target_lang)
        
        elif engine_name == "deepl":
            from src.translators.deepl import DeepLTranslator
            return DeepLTranslator(source_lang, target_lang)
        
        elif engine_name in ("argos", "marianmt", "nllb"):
            from src.translators.offline import OfflineTranslator
            return OfflineTranslator(source_lang, target_lang, engine_name)
        
        else:
            raise ValueError(f"Unknown translation engine: {engine}")
