"""
Gemma3 LLM-based translator via Ollama.
"""

from src.translators.base import BaseTranslator
from config.settings import GEMMA_MODEL, GEMMA_KEEP_ALIVE


class GemmaTranslator(BaseTranslator):
    """Translator using Gemma3 LLM via Ollama"""

    _MAX_STORY_CONTEXT_CHARS = 1200
    _MAX_PAGE_CONTEXT_CHARS = 1200

    def __init__(self, source_lang: str, target_lang: str, model: str = None):
        super().__init__(source_lang, target_lang)
        self.model = model or GEMMA_MODEL

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

    @staticmethod
    def _trim_context(text: str, limit: int) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "…"
    
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
        
        # Add story context and page context to system prompt.
        # Keeping all context in the system message prevents Gemma from treating
        # it as part of the text to translate (which causes it to repeat context).
        if story_context:
            ctx = self._trim_context(story_context, self._MAX_STORY_CONTEXT_CHARS)
            system_prompt += (
                "\n\n[Story Context — Reference Only]"
                "\nUse this only to keep terminology/character consistency."
                "\nDo NOT copy, continue, or paraphrase this section in the output."
                f"\n<story_context>\n{ctx}\n</story_context>"
            )
        if context_prompt:
            ctx = self._trim_context(context_prompt, self._MAX_PAGE_CONTEXT_CHARS)
            system_prompt += (
                "\n\n[Previous Page Context — Reference Only]"
                "\nUse only for disambiguation (names, pronouns, tone)."
                "\nDo NOT continue this text. Do NOT include any part of it in output."
                f"\n<previous_page_context>\n{ctx}\n</previous_page_context>"
            )

        user_prompt = (
            "Translate exactly one text segment."
            "\nReturn only the translated segment, with no quotes, labels, or extra lines."
            f"\n\n=== {self.source_lang.upper()} TEXT ===\n{text}\n=== END TEXT ==="
        )
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        try:
            ollama = self._get_client()
            response = ollama.chat(
                model=self.model,
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
            ctx = self._trim_context(story_context, self._MAX_STORY_CONTEXT_CHARS)
            system_prompt += (
                "\n\n[Story Context — Reference Only]"
                "\nUse this only to keep terminology/character consistency."
                "\nDo NOT copy, continue, or paraphrase this section in output."
                f"\n<story_context>\n{ctx}\n</story_context>"
            )
        # Context goes in system prompt so Gemma treats it as background, not as
        # text to continue — putting it in the user message causes Gemma to repeat it.
        if context_prompt:
            ctx = self._trim_context(context_prompt, self._MAX_PAGE_CONTEXT_CHARS)
            system_prompt += (
                "\n\n[Previous Page Context — Reference Only]"
                "\nUse only for disambiguation (names, pronouns, tone)."
                "\nDo NOT continue this text. Do NOT include any part of it in output."
                f"\n<previous_page_context>\n{ctx}\n</previous_page_context>"
            )

        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        user_prompt = (
            "Translate each numbered source text independently."
            "\nReturn only numbered translations for the given items."
            "\nDo not add commentary and do not continue any context text."
            f"\n\n{numbered}"
        )

        try:
            ollama = self._get_client()
            response = ollama.chat(
                model=self.model,
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
            results = []
            for t in texts:
                try:
                    results.append(self.translate(t, context_prompt=context_prompt, story_context=story_context))
                except Exception as item_err:
                    print(f"   ⚠️ Per-item translation failed: {item_err}")
                    results.append(t)  # keep original on failure
            return results

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

    def unload(self) -> None:
        """Tell Ollama to evict this model from VRAM so inpainting can use the GPU."""
        try:
            ollama = self._get_client()
            # keep_alive=0 unloads the model immediately after the (empty) response
            ollama.generate(model=self.model, prompt="", keep_alive=0)
            print(f"   🧹 Unloaded {self.model} from GPU VRAM")
        except Exception as e:
            print(f"   ⚠️  Ollama unload skipped: {e}")

    def is_available(self) -> bool:
        """Check if Ollama and Gemma3 are available"""
        try:
            ollama = self._get_client()
            ollama.list()
            return True
        except Exception:
            return False
