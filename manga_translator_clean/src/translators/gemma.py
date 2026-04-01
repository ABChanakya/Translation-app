"""
Gemma3 LLM-based translator via Ollama.
"""

from src.translators.base import BaseTranslator
from config.settings import GEMMA_MODEL, GEMMA_KEEP_ALIVE


class GemmaTranslator(BaseTranslator):
    """Translator using Gemma3 LLM via Ollama"""

    @staticmethod
    def _get_client():
        try:
            import ollama
        except Exception as exc:
            raise RuntimeError("Gemma3 translation requires the 'ollama' package.") from exc
        return ollama
    
    @property
    def name(self) -> str:
        return "Gemma3"
    
    def translate(self, text: str, context_prompt: str = "", story_context: str = "") -> str:
        """
        Translate text using Gemma3 LLM with optional narrative and story context.
        
        Provides high-quality, context-aware translations with
        understanding of nuance and cultural context.
        
        Args:
            text: Text to translate
            context_prompt: Optional context from previous pages
            story_context: Optional global story context (characters, plot, glossary, etc.)
        """
        text = text.strip()
        if not text:
            return ""
        
        # Build system prompt with story context if available
        system_prompt = (
            f"You are a professional translator specializing in {self.source_lang} "
            f"and {self.target_lang}. Translate the following {self.source_lang} text "
            f"into natural, fluent {self.target_lang}. Preserve tone, nuance, and "
            f"cultural context. Output ONLY the translated text, nothing else."
        )
        
        # Add story context to system prompt for consistency across all pages.
        # Cap at 1200 chars so the story context never dominates the context window.
        if story_context:
            ctx = story_context[:1200] + ("…" if len(story_context) > 1200 else "")
            system_prompt += f"\n\n[Story Context for Translation Consistency]\n{ctx}"
        
        # Include narrative context if available
        context_part = ""
        if context_prompt:
            context_part = f"\n{context_prompt}"
        
        user_prompt = (
            f"=== {self.source_lang.upper()} TEXT ==={context_part}\n"
            f"{text}\n"
            f"=== END TEXT ==="
        )
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        try:
            ollama = self._get_client()
            response = ollama.chat(
                model=GEMMA_MODEL,
                messages=messages,
                keep_alive=GEMMA_KEEP_ALIVE,
                format='',  # Plain text
                options={
                    'num_ctx': 8192,   # Default is 2048 — too small for story context
                    'temperature': 1.0,
                    'min_p': 0.01,
                    'repeat_penalty': 1.0,
                    'top_k': 64,
                    'top_p': 0.95,
                }
            )
            
            translation = response.message.content
            return translation.strip()
        
        except Exception as e:
            print(f"⚠️ Gemma3 translation failed: {e}")
            raise
    
    def translate_batch(self, texts: list, context_prompt: str = "", story_context: str = "") -> list:
        """
        Translate all texts on a page in a single Ollama call.
        Sends them as a numbered list and parses the numbered response.
        Falls back to per-item translate() if parsing fails.
        """
        if not texts:
            return []
        if len(texts) == 1:
            return [self.translate(texts[0], context_prompt=context_prompt, story_context=story_context)]

        system_prompt = (
            f"You are a professional translator specializing in {self.source_lang} "
            f"and {self.target_lang}. "
            f"Translate each numbered {self.source_lang} text into natural, fluent {self.target_lang}. "
            f"Preserve tone, nuance, and cultural context. "
            f"Output ONLY the translations, using the same numbers. "
            f"Format exactly as:\n1. <translation>\n2. <translation>\netc."
        )
        if story_context:
            ctx = story_context[:1200] + ("…" if len(story_context) > 1200 else "")
            system_prompt += f"\n\n[Story Context]\n{ctx}"

        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        user_prompt = f"{context_prompt}\n{numbered}" if context_prompt else numbered

        try:
            ollama = self._get_client()
            response = ollama.chat(
                model=GEMMA_MODEL,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_prompt},
                ],
                keep_alive=GEMMA_KEEP_ALIVE,
                format='',
                options={
                    'num_ctx': 8192,
                    'temperature': 1.0,
                    'min_p': 0.01,
                    'repeat_penalty': 1.0,
                    'top_k': 64,
                    'top_p': 0.95,
                },
            )
            raw = response.message.content.strip()
            return self._parse_numbered(raw, texts)
        except Exception as e:
            print(f"⚠️ Gemma3 batch translation failed ({type(e).__name__}): {e} — falling back to per-item")
            return [self.translate(t, context_prompt=context_prompt, story_context=story_context) for t in texts]

    @staticmethod
    def _parse_numbered(raw: str, originals: list) -> list:
        """Parse '1. text\\n2. text' response back into a list aligned with originals."""
        import re
        lines = raw.splitlines()
        results = {}
        for line in lines:
            m = re.match(r'^\s*(\d+)[.)]\s*(.*)', line)
            if m:
                idx = int(m.group(1)) - 1
                results[idx] = m.group(2).strip()
        # Build output: use parsed value where available, fall back to original
        out = []
        for i, orig in enumerate(originals):
            out.append(results.get(i, orig))
        return out

    def is_available(self) -> bool:
        """Check if Ollama and Gemma3 are available"""
        try:
            ollama = self._get_client()
            ollama.list()
            return True
        except Exception:
            return False
