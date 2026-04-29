"""
Chapter summarizer — generates a 1-2 sentence summary at the end of a chapter
so subsequent chapters have narrative context.
"""

from __future__ import annotations

from typing import List

from src.translation.manga_profile import MangaProfile


def generate_chapter_summary(
    all_translations: List[str],
    chapter_num: int,
    profile: MangaProfile,
    ollama_client,
    model: str = "gemma4:latest",
) -> str:
    """
    After translating a full chapter, ask Gemma3 to summarise it
    in 1-2 sentences for use as context in future chapters.

    Args:
        all_translations: all translated text from the chapter, in order
        chapter_num: chapter number
        profile: MangaProfile to save the summary into
        ollama_client: ollama module (must have .chat())
        model: Ollama model name

    Returns:
        The summary string (also saved to the profile).
    """
    # Combine all dialogue, capped at 3000 chars
    combined = "\n".join(t for t in all_translations if t and t.strip())
    max_chars = 3000
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "..."

    prompt = (
        f"This is the dialogue from Chapter {chapter_num} of "
        f"'{profile.series_name}':\n\n"
        f"{combined}\n\n"
        f"Write a 1-2 sentence summary of what happened in this chapter. "
        f"Be specific about character names and key events. "
        f"This summary will be used to give context to the translator "
        f"for the next chapter."
    )

    response = ollama_client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3, "num_ctx": 4096},
    )

    summary = response["message"]["content"].strip()

    # Cap length
    max_len = profile.data["settings"].get("max_summary_length", 200)
    if len(summary) > max_len:
        summary = summary[:max_len].rsplit(" ", 1)[0] + "…"

    profile.add_chapter_summary(chapter_num, summary)
    return summary
