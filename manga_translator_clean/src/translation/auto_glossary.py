"""
Auto-glossary extraction — asks the LLM to identify recurring terms
that should be locked in the glossary for consistent translation.
"""

from __future__ import annotations

import json
import re
from typing import List, Dict

from src.translation.manga_profile import MangaProfile


def extract_glossary_candidates(
    japanese_texts: List[str],
    english_texts: List[str],
    profile: MangaProfile,
    ollama_client,
    model: str = "gemma4:latest",
) -> List[Dict[str, str]]:
    """
    After translating a chapter, asks Gemma3 to identify
    recurring terms that should be locked in the glossary.

    Returns list of {japanese, english, category} candidates
    for the user to review and approve.
    """
    # Build sample pairs (first 20 bubbles)
    pairs = []
    for jp, en in zip(japanese_texts[:20], english_texts[:20]):
        if jp and en:
            pairs.append(f"JP: {jp}\nEN: {en}")

    if not pairs:
        return []

    sample = "\n\n".join(pairs)

    prompt = (
        "You are analyzing manga dialogue to identify terms that should "
        "be consistently translated. Look at these translation pairs:\n\n"
        f"{sample}\n\n"
        "Identify: character names, place names, special techniques, "
        "and recurring terms that should always be translated the same way.\n\n"
        "Output as JSON array only, no other text:\n"
        '[{"japanese": "...", "english": "...", '
        '"category": "character|place|technique|general"}]'
    )

    response = ollama_client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "num_ctx": 4096},
    )

    content = response["message"]["content"].strip()

    # Extract JSON from the response
    json_match = re.search(r"\[.*\]", content, re.DOTALL)
    if not json_match:
        return []

    try:
        candidates = json.loads(json_match.group())
        # Validate shape
        valid = []
        for c in candidates:
            if (
                isinstance(c, dict)
                and "japanese" in c
                and "english" in c
            ):
                valid.append({
                    "japanese": str(c["japanese"]),
                    "english": str(c["english"]),
                    "category": str(c.get("category", "general")),
                })
        return valid
    except json.JSONDecodeError:
        return []
