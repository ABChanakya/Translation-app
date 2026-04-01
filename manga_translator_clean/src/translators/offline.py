"""
Offline translation engines: Argos, MarianMT, NLLB.
"""

from functools import lru_cache
import os

from src.translators.base import BaseTranslator
from config.settings import (
    MARIAN_MODEL_PREFIX,
    NLLB_MODEL_ID,
    CACHE_DIR
)


class OfflineTranslator(BaseTranslator):
    """Offline translation using Argos, MarianMT, or NLLB"""
    
    def __init__(self, source_lang: str, target_lang: str, engine: str):
        super().__init__(source_lang, target_lang)
        self.engine = engine.lower()
    
    @property
    def name(self) -> str:
        return self.engine.upper()
    
    def translate(self, text: str, context_prompt: str = "", story_context: str = "") -> str:
        """Translate using the specified offline engine"""
        text = text.strip()
        if not text:
            return ""
        
        if self.engine == "argos":
            return self._translate_argos(text)
        elif self.engine == "marianmt":
            return self._translate_marian(text)
        elif self.engine == "nllb":
            return self._translate_nllb(text)
        else:
            raise ValueError(f"Unknown offline engine: {self.engine}")
    
    def _translate_argos(self, text: str) -> str:
        """Translate using Argos Translate"""
        import argostranslate.translate

        self._ensure_argos_package()
        return argostranslate.translate.translate(
            text, self.source_lang, self.target_lang
        )
    
    def _translate_marian(self, text: str) -> str:
        """Translate using MarianMT"""
        tokenizer, model = self._load_marian()
        tokens = tokenizer(text, return_tensors="pt")
        output = model.generate(**tokens, max_length=256)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    def _translate_nllb(self, text: str) -> str:
        """Translate using NLLB"""
        tokenizer, model = self._load_nllb()
        
        if self.source_lang not in tokenizer.lang_code_to_id:
            raise ValueError(f"NLLB doesn't support language: {self.source_lang}")
        
        tokens = tokenizer(text, return_tensors="pt")
        tokens["forced_bos_token_id"] = tokenizer.lang_code_to_id.get(
            self.target_lang, 0
        )
        output = model.generate(**tokens, max_length=256)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    def _ensure_argos_package(self):
        """Download and install Argos language pack if needed"""
        import requests
        import argostranslate.package

        installed = {
            (p.from_code, p.to_code)
            for p in argostranslate.package.get_installed_packages()
        }
        
        if (self.source_lang, self.target_lang) in installed:
            return
        
        # Download package
        pack_url = (
            f"https://huggingface.co/argosopentech/"
            f"argos-translate-{self.source_lang}_{self.target_lang}/"
            f"resolve/main/{self.source_lang}_{self.target_lang}.argos"
        )
        
        try:
            pack_path = os.path.join(
                CACHE_DIR,
                f"{self.source_lang}_{self.target_lang}.argos"
            )
            
            if not os.path.exists(pack_path):
                print(f"📥 Downloading Argos pack: {self.source_lang} → {self.target_lang}")
                response = requests.get(pack_url, timeout=30)
                response.raise_for_status()
                with open(pack_path, "wb") as f:
                    f.write(response.content)
            
            argostranslate.package.install_from_path(pack_path)
            print(f"✅ Installed Argos pack: {self.source_lang} → {self.target_lang}")
        
        except Exception as e:
            print(f"⚠️ Failed to install Argos pack: {e}")
            raise
    
    @lru_cache(maxsize=4)
    def _load_marian(self):
        """Load MarianMT model (cached)"""
        from transformers import MarianMTModel, MarianTokenizer

        model_name = f"{MARIAN_MODEL_PREFIX}-{self.source_lang}-{self.target_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    
    @lru_cache(maxsize=2)
    def _load_nllb(self):
        """Load NLLB model (cached)"""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID)
        return tokenizer, model

    def is_available(self) -> bool:
        if self.engine == "argos":
            try:
                import argostranslate.translate  # noqa: F401
            except Exception:
                return False
            return True

        if self.engine in {"marianmt", "nllb"}:
            try:
                import transformers  # noqa: F401
            except Exception:
                return False
            return True

        return False
