"""
Honorific-Aware Translator Wrapper
Wraps any translator with honorific preservation
"""
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from honorifics_preserver import HonorificPreserver


class HonorificAwareTranslator:
    """
    Wrapper that adds honorific preservation to any translator
    """
    
    def __init__(
        self,
        base_translator,
        preserve_honorifics: bool = True,
        keep_romanji: bool = True,
        custom_honorifics: Optional[Dict[str, str]] = None
    ):
        """
        Initialize honorific-aware translator
        
        Args:
            base_translator: Base translator instance (must have translate() method)
            preserve_honorifics: Enable honorific preservation
            keep_romanji: Keep romanji form (True) or convert to Japanese (False)
            custom_honorifics: Additional honorifics dict
        """
        self.base_translator = base_translator
        self.preserve_honorifics = preserve_honorifics
        self.keep_romanji = keep_romanji
        
        # Initialize preserver
        self.preserver = HonorificPreserver(custom_honorifics=custom_honorifics)
        
        # Track character honorifics across pages
        self.character_honorifics = {}
    
    def translate(
        self,
        text: str,
        src_lang: str = 'ja',
        tgt_lang: str = 'en',
        **kwargs
    ) -> str:
        """
        Translate text with honorific preservation
        
        Args:
            text: Text to translate
            src_lang: Source language
            tgt_lang: Target language
            **kwargs: Additional translator-specific arguments
            
        Returns:
            Translated text with honorifics preserved
        """
        # Translate using base translator
        translated = self.base_translator.translate(
            text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            **kwargs
        )
        
        # Preserve honorifics if enabled
        if self.preserve_honorifics and src_lang == 'ja':
            translated = self.preserver.preserve_in_translation(
                text,
                translated,
                keep_romanji=self.keep_romanji
            )
            
            # Apply known character honorifics
            if self.character_honorifics:
                translated = self.preserver.add_honorifics_post_translation(
                    translated,
                    self.character_honorifics
                )
        
        return translated
    
    def translate_batch(
        self,
        texts: List[str],
        src_lang: str = 'ja',
        tgt_lang: str = 'en',
        **kwargs
    ) -> List[str]:
        """
        Translate batch with honorific preservation
        
        Args:
            texts: List of texts to translate
            src_lang: Source language
            tgt_lang: Target language
            **kwargs: Additional arguments
            
        Returns:
            List of translated texts with honorifics preserved
        """
        # Check if base translator has batch method
        if hasattr(self.base_translator, 'translate_batch'):
            translations = self.base_translator.translate_batch(
                texts,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                **kwargs
            )
        else:
            # Fallback to individual translation
            translations = [
                self.base_translator.translate(t, src_lang, tgt_lang, **kwargs)
                for t in texts
            ]
        
        # Preserve honorifics if enabled
        if self.preserve_honorifics and src_lang == 'ja':
            translations = [
                self.preserver.preserve_in_translation(
                    src,
                    trans,
                    keep_romanji=self.keep_romanji
                )
                for src, trans in zip(texts, translations)
            ]
            
            # Apply known character honorifics
            if self.character_honorifics:
                translations = [
                    self.preserver.add_honorifics_post_translation(
                        trans,
                        self.character_honorifics
                    )
                    for trans in translations
                ]
        
        return translations
    
    def update_character_honorifics(
        self,
        dialogue_log: List[Dict[str, str]]
    ):
        """
        Update character-honorific mappings from dialogue
        
        Args:
            dialogue_log: List of dialogue entries with 'src_text'
        """
        extracted = self.preserver.extract_character_honorifics(dialogue_log)
        self.character_honorifics.update(extracted)
    
    def set_character_honorific(
        self,
        character_name: str,
        honorific: str
    ):
        """
        Manually set honorific for a character
        
        Args:
            character_name: Character name
            honorific: Honorific to use (e.g., 'san', 'kun', 'chan')
        """
        self.character_honorifics[character_name] = honorific
    
    def get_character_honorifics(self) -> Dict[str, str]:
        """Get current character-honorific mappings"""
        return self.character_honorifics.copy()
    
    def validate_consistency(
        self,
        dialogue_log: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        """
        Check for inconsistent honorific usage
        
        Args:
            dialogue_log: List of dialogue entries
            
        Returns:
            Dict mapping character names to list of different honorifics used
        """
        return self.preserver.validate_honorific_consistency(dialogue_log)


class HonorificConfig:
    """Configuration for honorific preservation"""
    
    def __init__(self):
        self.enabled = True
        self.keep_romanji = True
        self.custom_honorifics = {}
        self.character_mappings = {}
        self.auto_detect = True
        self.validate_consistency = True
    
    def add_custom_honorific(self, romanji: str, japanese: str):
        """Add custom honorific mapping"""
        self.custom_honorifics[romanji] = japanese
    
    def set_character_honorific(self, character: str, honorific: str):
        """Set specific character honorific"""
        self.character_mappings[character] = honorific
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'keep_romanji': self.keep_romanji,
            'custom_honorifics': self.custom_honorifics,
            'character_mappings': self.character_mappings,
            'auto_detect': self.auto_detect,
            'validate_consistency': self.validate_consistency
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'HonorificConfig':
        """Create from dictionary"""
        config = cls()
        config.enabled = data.get('enabled', True)
        config.keep_romanji = data.get('keep_romanji', True)
        config.custom_honorifics = data.get('custom_honorifics', {})
        config.character_mappings = data.get('character_mappings', {})
        config.auto_detect = data.get('auto_detect', True)
        config.validate_consistency = data.get('validate_consistency', True)
        return config


def wrap_translator_with_honorifics(
    translator,
    config: Optional[HonorificConfig] = None
) -> HonorificAwareTranslator:
    """
    Convenience function to wrap translator with honorific support
    
    Args:
        translator: Base translator instance
        config: Optional honorific configuration
        
    Returns:
        Honorific-aware translator wrapper
    """
    if config is None:
        config = HonorificConfig()
    
    wrapper = HonorificAwareTranslator(
        translator,
        preserve_honorifics=config.enabled,
        keep_romanji=config.keep_romanji,
        custom_honorifics=config.custom_honorifics
    )
    
    # Set character mappings if provided
    for character, honorific in config.character_mappings.items():
        wrapper.set_character_honorific(character, honorific)
    
    return wrapper
