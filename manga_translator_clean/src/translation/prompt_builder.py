"""
Context-aware translation prompt builder.

Assembles glossary, character info, chapter summaries, and recent dialogue
into a single structured prompt for the LLM translator.
"""

from __future__ import annotations

import re
from typing import List

from src.translation.manga_profile import MangaProfile


STYLE_INSTRUCTIONS = {
    "natural": (
        "Translate naturally into English. "
        "The translation should read smoothly to a native English speaker. "
        "Preserve the speaker's personality and tone."
    ),
    "literal": (
        "Translate closely to the original Japanese structure. "
        "Preserve sentence order where possible. "
        "Keep Japanese speech patterns."
    ),
    "localized": (
        "Translate with full western localization. "
        "Remove honorifics, adapt cultural references, "
        "make it read like it was originally written in English."
    ),
}


def build_translation_prompt(
    texts_to_translate: List[str],
    profile: MangaProfile,
    chapter_num: int = 1,
    page_num: int = 1,
) -> str:
    """
    Builds a single batched translation prompt for all bubbles on a page.

    Returns the full prompt string ready to send to Gemma3.
    """
    settings = profile.data["settings"]
    style = settings.get("translation_style", "natural")
    preserve_honorifics = settings.get("preserve_honorifics", True)

    sections: List[str] = []

    # ── System instruction ───────────────────────────────────────────
    sections.append(
        "You are a professional manga translator. "
        "Translate the Japanese dialogue below into English. "
        f"{STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS['natural'])}"
    )

    if preserve_honorifics:
        sections.append(
            "Keep Japanese honorifics (-san, -kun, -chan, -sama, -senpai) "
            "as they convey important social relationships."
        )

    # ── Glossary block ───────────────────────────────────────────────
    glossary_block = profile.get_glossary_as_prompt_block()
    if glossary_block:
        sections.append(glossary_block)

    # ── Character block ──────────────────────────────────────────────
    chars_block = profile.get_characters_as_prompt_block()
    if chars_block:
        sections.append(chars_block)

    # ── Chapter summaries ────────────────────────────────────────────
    summaries_block = profile.get_recent_chapter_summaries(n=2)
    if summaries_block:
        sections.append(summaries_block)

    # ── Recent dialogue context (relevance-based when possible) ─────
    if texts_to_translate:
        recent_block = profile.get_relevant_translations_as_prompt_block(
            current_texts=texts_to_translate, n=5,
        )
    else:
        recent_block = profile.get_recent_translations_as_prompt_block(n=5)
    if recent_block:
        sections.append(recent_block)

    # ── Translation task ─────────────────────────────────────────────
    sections.append(
        f"NOW TRANSLATE — Chapter {chapter_num}, Page {page_num}:"
    )
    sections.append(
        "Translate each numbered line. "
        "Output ONLY the translations in the same numbered format. "
        "Do not add explanations, notes, or extra text. "
        "If a line is a sound effect (onomatopoeia), translate it naturally."
    )

    # Numbered input lines
    numbered = "\n".join(
        f"{i}. {text}" for i, text in enumerate(texts_to_translate, 1)
    )
    sections.append(numbered)

    sections.append("OUTPUT (numbered translations only):")

    return "\n\n".join(sections)


def parse_translation_response(
    response: str, expected_count: int
) -> List[str]:
    """
    Parses a numbered LLM response back into a list of translations.
    Handles malformed responses gracefully.
    """
    lines = response.strip().split("\n")
    translations: dict[int, str] = {}

    for line in lines:
        match = re.match(r"^(\d+)[.):\s]+(.+)$", line.strip())
        if match:
            idx = int(match.group(1))
            text = match.group(2).strip()
            if text:
                translations[idx] = text

    result: List[str] = []
    for i in range(1, expected_count + 1):
        if i in translations:
            result.append(translations[i])
        else:
            result.append(f"[Translation {i} unavailable]")

    return result
