"""
Persistent manga series profile for translation consistency.

Stores glossary terms, character names, chapter summaries, and recent
translations so the LLM produces consistent output across pages and chapters.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class MangaProfile:
    """
    Persistent profile for a manga series.
    Stored as a JSON file in profiles/{series_name}.json
    """

    def __init__(self, series_name: str, profiles_dir: str = "profiles"):
        self.series_name = series_name
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        # Sanitise the file name
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in series_name)
        self.profile_path = self.profiles_dir / f"{safe_name}.json"
        self.data = self._load_or_create()

    def _load_or_create(self) -> dict:
        if self.profile_path.exists():
            with open(self.profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "series_name": self.series_name,
            "created_at": datetime.now().isoformat(),
            "glossary": {},
            "chapter_summaries": {},
            "recent_translations": [],
            "character_names": {},
            "settings": {
                "max_recent_lines": 30,
                "max_summary_length": 200,
                "translation_style": "natural",
                "preserve_honorifics": True,
            },
        }

    def save(self):
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ── Glossary management ──────────────────────────────────────────────

    def add_glossary_term(
        self, japanese: str, english: str, category: str = "general"
    ):
        """
        Add a locked translation term.
        category: 'character', 'place', 'technique', 'honorific', 'general'
        """
        self.data["glossary"][japanese] = {
            "english": english,
            "category": category,
            "added_at": datetime.now().isoformat(),
        }
        self.save()

    def remove_glossary_term(self, japanese: str):
        self.data["glossary"].pop(japanese, None)
        self.save()

    def get_glossary_as_prompt_block(self) -> str:
        """Returns glossary formatted for injection into LLM prompt."""
        if not self.data["glossary"]:
            return ""

        lines = ["GLOSSARY (always use these translations exactly):"]

        # Group by category
        categories: Dict[str, List[str]] = {}
        for jp, entry in self.data["glossary"].items():
            cat = entry["category"]
            label = " (auto)" if entry.get("auto_detected") else ""
            categories.setdefault(cat, []).append(
                f"  {jp} → {entry['english']}{label}"
            )

        for cat in sorted(categories):
            lines.append(f"[{cat.upper()}]")
            lines.extend(categories[cat])

        return "\n".join(lines)

    # ── Character name management ────────────────────────────────────────

    def add_character(
        self,
        japanese_name: str,
        english_name: str,
        pronouns: str = "they/them",
        role: str = "",
        voice: str = "",
    ):
        """
        Registers a character for consistent name translation.
        voice: speaking style hint for the translator, e.g.
               'formal', 'rough/masculine', 'childlike', 'archaic', 'robotic'
        """
        self.data["character_names"][japanese_name] = {
            "english": english_name,
            "pronouns": pronouns,
            "role": role,
            "voice": voice,
        }
        # Also add to glossary automatically
        self.add_glossary_term(japanese_name, english_name, "character")

    def get_characters_as_prompt_block(self) -> str:
        if not self.data["character_names"]:
            return ""

        lines = ["CHARACTERS:"]
        for jp, info in self.data["character_names"].items():
            line = f"  {jp} = {info['english']}"
            if info.get("pronouns") and info["pronouns"] != "they/them":
                line += f" ({info['pronouns']})"
            if info.get("role"):
                line += f" — {info['role']}"
            if info.get("voice"):
                line += f" [voice: {info['voice']}]"
            lines.append(line)

        return "\n".join(lines)

    # ── Chapter context management ───────────────────────────────────────

    def add_chapter_summary(self, chapter_num: int, summary: str):
        self.data["chapter_summaries"][str(chapter_num)] = {
            "summary": summary,
            "created_at": datetime.now().isoformat(),
        }
        self.save()

    def get_recent_chapter_summaries(self, n: int = 2) -> str:
        """Returns summaries of the last *n* chapters for context."""
        summaries = self.data["chapter_summaries"]
        if not summaries:
            return ""

        sorted_keys = sorted(summaries.keys(), key=lambda x: int(x))[-n:]
        lines = ["PREVIOUS CHAPTERS:"]
        for key in sorted_keys:
            lines.append(f"  Chapter {key}: {summaries[key]['summary']}")

        return "\n".join(lines)

    # ── Auto-glossary extraction ────────────────────────────────────────

    _COMMON_ENGLISH_WORDS = frozenset({
        "The", "And", "But", "For", "You", "Are", "Was",
        "That", "This", "With", "Have", "Will", "From",
        "They", "She", "Him", "Her", "Its", "Not", "All",
        "Can", "Had", "Has", "How", "May", "New", "Now",
        "Old", "One", "Our", "Out", "Own", "Say", "Too",
        "Two", "Way", "Who", "Did", "Got", "Let", "Put",
    })

    def auto_update_glossary_from_pair(
        self, japanese: str, english: str
    ) -> List[dict]:
        """
        Given one JP→EN translation pair, extracts likely proper nouns
        (katakana names) and adds high-confidence ones to the glossary.

        Uses heuristics only — no LLM call, runs inline.
        Returns list of terms that were auto-added.
        """
        added: List[dict] = []

        # Katakana sequences of 3+ chars (names / loanwords)
        katakana_matches = re.findall(r"[ァ-ヶー]{2,}", japanese)

        # Capitalised English words (proper nouns) in the translation
        capitalized = re.findall(r"\b[A-Z][a-z]{2,}\b", english)

        if not katakana_matches or not capitalized:
            return added

        pairs_to_add: List[tuple] = []
        if len(katakana_matches) == len(capitalized):
            pairs_to_add = list(zip(katakana_matches, capitalized))
        elif len(katakana_matches) == 1 and capitalized:
            pairs_to_add = [(katakana_matches[0], capitalized[0])]

        for jp_term, en_term in pairs_to_add:
            if jp_term in self.data["glossary"]:
                continue
            if len(en_term) < 3:
                continue
            if en_term in self._COMMON_ENGLISH_WORDS:
                continue

            self.data["glossary"][jp_term] = {
                "english": en_term,
                "category": "character",
                "added_at": datetime.now().isoformat(),
                "auto_detected": True,
            }
            added.append({"japanese": jp_term, "english": en_term})

        # Don't call self.save() here — caller (add_translated_lines) saves.
        return added

    # ── Rolling translation memory ───────────────────────────────────────

    def add_translated_lines(self, lines: List[dict]):
        """
        lines: list of {japanese, english, chapter, page}
        Keeps only the last *max_recent_lines* entries.
        Auto-extracts glossary terms from each new pair.
        """
        self.data["recent_translations"].extend(lines)
        max_lines = self.data["settings"]["max_recent_lines"]
        if len(self.data["recent_translations"]) > max_lines:
            self.data["recent_translations"] = self.data[
                "recent_translations"
            ][-max_lines:]

        # Auto-extract glossary terms from each new pair
        auto_added_total: List[dict] = []
        for entry in lines:
            jp = entry.get("japanese", "")
            en = entry.get("english", "")
            if jp and en:
                auto_added = self.auto_update_glossary_from_pair(jp, en)
                auto_added_total.extend(auto_added)

        if auto_added_total:
            print(
                f"  📖 Auto-glossary: added {len(auto_added_total)} term(s): "
                + ", ".join(
                    f"{t['japanese']}→{t['english']}" for t in auto_added_total
                )
            )

        self.save()

    def get_recent_translations_as_prompt_block(self, n: int = 10) -> str:
        """Returns last *n* translations as context for the LLM."""
        recent = self.data["recent_translations"][-n:]
        if not recent:
            return ""

        lines = ["RECENT DIALOGUE (for context and consistency):"]
        for entry in recent:
            lines.append(f"  JP: {entry['japanese']}")
            lines.append(f"  EN: {entry['english']}")

        return "\n".join(lines)

    def get_relevant_translations_as_prompt_block(
        self,
        current_texts: List[str],
        n: int = 5,
    ) -> str:
        """
        Returns the *n* most relevant past translations to the current page,
        ranked by character-level Jaccard similarity.
        No external ML model needed — fast enough to run inline.
        """
        recent = self.data["recent_translations"]
        if not recent or not current_texts:
            return ""

        current_chars = set("".join(current_texts))
        if not current_chars:
            return ""

        scored = []
        for entry in recent:
            past_chars = set(entry.get("japanese", ""))
            if not past_chars:
                continue
            intersection = len(current_chars & past_chars)
            union = len(current_chars | past_chars)
            score = intersection / union if union > 0 else 0.0
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [entry for _, entry in scored[:n]]

        if not top:
            return ""

        lines = ["RELEVANT PAST DIALOGUE (for context and consistency):"]
        for entry in top:
            lines.append(f"  JP: {entry['japanese']}")
            lines.append(f"  EN: {entry['english']}")

        return "\n".join(lines)

    # ── Settings ─────────────────────────────────────────────────────────

    def set_translation_style(self, style: str):
        """
        style: 'natural' | 'literal' | 'localized'
        """
        valid = ("natural", "literal", "localized")
        if style not in valid:
            raise ValueError(f"Style must be one of {valid}")
        self.data["settings"]["translation_style"] = style
        self.save()

    def set_preserve_honorifics(self, preserve: bool):
        self.data["settings"]["preserve_honorifics"] = preserve
        self.save()

    # ── Convenience ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        g = len(self.data["glossary"])
        c = len(self.data["character_names"])
        s = len(self.data["chapter_summaries"])
        r = len(self.data["recent_translations"])
        return (
            f"MangaProfile('{self.series_name}', "
            f"glossary={g}, characters={c}, "
            f"summaries={s}, recent={r})"
        )
