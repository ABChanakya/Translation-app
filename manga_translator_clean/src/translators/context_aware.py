"""
Context-Aware Translator for Manga
Uses chapter context and page metadata for consistent translations
"""
from typing import Optional, Dict, List
import ollama

from src.metadata_manager import PageMeta, ChapterContext


class ContextAwareTranslator:
    """
    Enhanced translator that uses page metadata and chapter context
    for consistent character names, terms, and tone
    """
    
    def __init__(
        self,
        model: str = "gemma3:latest",
        source_lang: str = "ja",
        target_lang: str = "en"
    ):
        """
        Initialize context-aware translator
        
        Args:
            model: Ollama model name
            source_lang: Source language code
            target_lang: Target language code
        """
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def translate_with_context(
        self,
        text: str,
        page_meta: PageMeta,
        chapter_context: Optional[ChapterContext] = None,
        bubble_type: str = "dialogue"
    ) -> str:
        """
        Translate text with full context awareness
        
        Args:
            text: Original text to translate
            page_meta: Page metadata
            chapter_context: Optional chapter context for consistency
            bubble_type: Type of bubble (dialogue, sfx, sign)
            
        Returns:
            Translated text
        """
        # Build system prompt with context
        system_prompt = self._build_system_prompt(page_meta, bubble_type)
        
        # Build user prompt with context hints
        context_hint = ""
        if chapter_context:
            context_hint = self._build_context_hint(chapter_context)
        
        prompt = f"{context_hint}\nOriginal text:\n{text}\n\nTranslated text:"
        
        try:
            # Call Ollama API
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            
            translated = response["message"]["content"].strip()
            
            # Clean up any explanations or formatting
            translated = self._clean_translation(translated)
            
            return translated
            
        except Exception as e:
            print(f"⚠️  Translation error: {e}")
            return text  # Fallback to original
    
    def translate_page_batch(
        self,
        bubbles: List[Dict[str, str]],
        page_meta: PageMeta,
        chapter_context: Optional[ChapterContext] = None
    ) -> List[str]:
        """
        Translate all bubbles on a page together for better coherence
        
        Args:
            bubbles: List of bubble dicts with 'text' and 'type' keys
            page_meta: Page metadata
            chapter_context: Optional chapter context
            
        Returns:
            List of translated texts in same order
        """
        if not bubbles:
            return []
        
        # Build system prompt
        system_prompt = self._build_system_prompt(page_meta, "dialogue")
        
        # Build context hint
        context_hint = ""
        if chapter_context:
            context_hint = self._build_context_hint(chapter_context)
        
        # Format all bubbles with numbering
        bubble_text = []
        for i, bubble in enumerate(bubbles, 1):
            text = bubble.get('text', '')
            bubble_type = bubble.get('type', 'dialogue')
            bubble_text.append(f"Bubble {i} ({bubble_type}): {text}")
        
        prompt = (
            f"{context_hint}\n\n"
            "Translate the following manga page dialogue in reading order.\n"
            "Return ONLY the translated bubbles in the same numbered format.\n\n"
            + "\n".join(bubble_text)
        )
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            
            result_text = response["message"]["content"].strip()
            
            # Parse numbered responses
            translations = self._parse_numbered_response(result_text, len(bubbles))
            
            return translations
            
        except Exception as e:
            print(f"⚠️  Batch translation error: {e}")
            # Fallback to individual translation
            return [bubble.get('text', '') for bubble in bubbles]
    
    def update_chapter_context(
        self,
        chapter_context: ChapterContext,
        page_meta: PageMeta,
        page_dialogue: List[str]
    ) -> ChapterContext:
        """
        Update chapter context based on translated page dialogue
        
        Args:
            chapter_context: Current chapter context
            page_meta: Page metadata
            page_dialogue: List of translated dialogue from page
            
        Returns:
            Updated chapter context
        """
        joined_dialogue = "\n".join(page_dialogue)
        
        prompt = f"""
Current chapter summary:
{chapter_context.summary}

Current characters:
{', '.join(chapter_context.characters)}

New page (series={page_meta.series}, chapter={page_meta.chapter}, page={page_meta.page}) dialogue:
{joined_dialogue}

Tasks:
1) Update the story summary (2-3 sentences max)
2) Extract/update main character names and relationships
3) Identify stable terms (school names, attack names, locations) for glossary

Return JSON with keys: summary, characters (array), glossary (object with JP->EN mappings if found)
"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are analyzing manga chapters for translation consistency. Return valid JSON only."
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            
            result = response["message"]["content"].strip()
            
            # Try to parse JSON
            import json
            # Extract JSON from markdown code blocks if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            
            data = json.loads(result)
            
            # Update context
            if "summary" in data:
                chapter_context.summary = data["summary"]
            if "characters" in data and isinstance(data["characters"], list):
                # Merge new characters
                new_chars = [c for c in data["characters"] if c not in chapter_context.characters]
                chapter_context.characters.extend(new_chars)
            if "glossary" in data and isinstance(data["glossary"], dict):
                chapter_context.glossary.update(data["glossary"])
            
        except Exception as e:
            print(f"⚠️  Context update error: {e}")
        
        return chapter_context
    
    def _build_system_prompt(self, page_meta: PageMeta, bubble_type: str) -> str:
        """Build system prompt with page context"""
        return f"""
You are translating a manga.
Series: {page_meta.series}
Chapter: {page_meta.chapter}
Page: {page_meta.page}

Type: {bubble_type}

Guidelines:
- Use consistent character names and tone across the chapter
- For dialogue, keep it natural and conversational
- For SFX, use English onomatopoeia when appropriate
- For signs, translate clearly and concisely
- Return ONLY the translated text, no explanations
""".strip()
    
    def _build_context_hint(self, chapter_context: ChapterContext) -> str:
        """Build context hint from chapter context"""
        hints = []
        
        if chapter_context.summary:
            hints.append(f"Story so far:\n{chapter_context.summary}")
        
        if chapter_context.characters:
            char_list = ", ".join(chapter_context.characters[:10])  # Limit to 10
            hints.append(f"\nMain characters: {char_list}")
        
        if chapter_context.glossary:
            glossary_items = [f"{jp} = {en}" for jp, en in list(chapter_context.glossary.items())[:5]]
            if glossary_items:
                hints.append(f"\nTerm glossary:\n" + "\n".join(glossary_items))
        
        return "\n".join(hints) if hints else ""
    
    def _clean_translation(self, text: str) -> str:
        """Clean up translated text"""
        # Remove common prefixes
        prefixes = [
            "Translated text:",
            "Translation:",
            "Here's the translation:",
            "The translation is:",
        ]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # Remove quotes if the entire text is quoted
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        return text.strip()
    
    def _parse_numbered_response(self, response: str, expected_count: int) -> List[str]:
        """Parse numbered bubble responses"""
        import re
        
        translations = []
        lines = response.split('\n')
        
        for line in lines:
            # Look for patterns like "Bubble 1: text" or "1. text" or "1) text"
            match = re.match(r'^(?:Bubble\s+)?(\d+)[\s:.)\-]+(.+)$', line.strip(), re.IGNORECASE)
            if match:
                bubble_num = int(match.group(1))
                text = match.group(2).strip()
                
                # Remove bubble type annotation if present
                text = re.sub(r'^\([^)]+\):\s*', '', text)
                
                translations.append(text)
        
        # If we didn't get enough translations, pad with empty strings
        while len(translations) < expected_count:
            translations.append("")
        
        # If we got too many, truncate
        translations = translations[:expected_count]
        
        return translations
