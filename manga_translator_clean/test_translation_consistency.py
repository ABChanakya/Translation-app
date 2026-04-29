"""
Tests for translation consistency system:
  MangaProfile, prompt builder, response parser, rolling memory,
  chapter summarizer, and auto-glossary extraction.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from src.translation.manga_profile import MangaProfile
from src.translation.prompt_builder import (
    build_translation_prompt,
    parse_translation_response,
)


TEST_PROFILES_DIR = tempfile.mkdtemp(prefix="manga_profiles_test_")

passed = 0
failed = 0


def _result(name: str, ok: bool, reason: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {reason}")


# ──────────────────────────────────────────────────────────────────────────
# Test 1: Glossary injection
# ──────────────────────────────────────────────────────────────────────────
def test_glossary_injection():
    print("\n" + "=" * 70)
    print("TEST 1: Glossary injection")
    print("=" * 70)

    profile = MangaProfile("Berserk_Test1", profiles_dir=TEST_PROFILES_DIR)
    profile.add_glossary_term("ガッツ", "Guts", "character")
    profile.add_glossary_term("鷹の団", "Band of the Hawk", "place")
    profile.add_glossary_term("覇王の卵", "Egg of the King", "technique")

    prompt = build_translation_prompt(
        texts_to_translate=["ガッツは鷹の団を去った"],
        profile=profile,
        chapter_num=1,
        page_num=1,
    )

    # Check glossary block appears
    _result(
        "Glossary block in prompt",
        "GLOSSARY" in prompt and "ガッツ → Guts" in prompt,
        f"Missing glossary. Prompt starts: {prompt[:200]}",
    )
    _result(
        "All three terms present",
        "Band of the Hawk" in prompt and "Egg of the King" in prompt,
        "Missing terms",
    )
    _result(
        "Category headers present",
        "[CHARACTER]" in prompt and "[PLACE]" in prompt,
        "Missing category headers",
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 2: Parse translation response
# ──────────────────────────────────────────────────────────────────────────
def test_parse_response():
    print("\n" + "=" * 70)
    print("TEST 2: Parse translation response")
    print("=" * 70)

    # Well-formed response
    good_response = (
        "1. Guts left the Band of the Hawk.\n"
        "2. Griffith was furious.\n"
        "3. Casca couldn't believe it.\n"
        "4. The eclipse was near.\n"
        "5. Everything was about to change."
    )
    results = parse_translation_response(good_response, 5)
    _result("Parse 5 translations", len(results) == 5, f"Got {len(results)}")
    _result(
        "First translation correct",
        results[0] == "Guts left the Band of the Hawk.",
        f"Got: {results[0]}",
    )
    _result(
        "Last translation correct",
        results[4] == "Everything was about to change.",
        f"Got: {results[4]}",
    )

    # Malformed response (missing line 3)
    bad_response = (
        "1. Hello.\n"
        "2. Goodbye.\n"
        "4. See you.\n"
        "5. Later."
    )
    results = parse_translation_response(bad_response, 5)
    _result("Handles missing line 3", len(results) == 5, f"Got {len(results)}")
    _result(
        "Missing line uses placeholder",
        "[Translation 3 unavailable]" in results[2],
        f"Got: {results[2]}",
    )

    # Alternate format (parentheses)
    paren_response = "1) First line\n2) Second line"
    results = parse_translation_response(paren_response, 2)
    _result(
        "Parses 1) format",
        results[0] == "First line" and results[1] == "Second line",
        f"Got: {results}",
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 3: Rolling memory
# ──────────────────────────────────────────────────────────────────────────
def test_rolling_memory():
    print("\n" + "=" * 70)
    print("TEST 3: Rolling memory")
    print("=" * 70)

    profile = MangaProfile("Memory_Test", profiles_dir=TEST_PROFILES_DIR)
    profile.data["settings"]["max_recent_lines"] = 30

    # Add 35 translation pairs
    for i in range(35):
        profile.add_translated_lines([{
            "japanese": f"テスト{i}",
            "english": f"Test {i}",
            "chapter": 1,
            "page": i // 5 + 1,
        }])

    _result(
        "Caps at 30 entries",
        len(profile.data["recent_translations"]) == 30,
        f"Got {len(profile.data['recent_translations'])}",
    )
    _result(
        "Oldest entries dropped",
        profile.data["recent_translations"][0]["english"] == "Test 5",
        f"First entry: {profile.data['recent_translations'][0]['english']}",
    )
    _result(
        "Newest entry present",
        profile.data["recent_translations"][-1]["english"] == "Test 34",
        f"Last entry: {profile.data['recent_translations'][-1]['english']}",
    )

    # Prompt block shows last 10
    block = profile.get_recent_translations_as_prompt_block(n=10)
    lines_with_en = [l for l in block.split("\n") if l.strip().startswith("EN:")]
    _result(
        "Prompt block shows 10 entries",
        len(lines_with_en) == 10,
        f"Got {len(lines_with_en)} EN: lines",
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 4: Character pronouns
# ──────────────────────────────────────────────────────────────────────────
def test_character_pronouns():
    print("\n" + "=" * 70)
    print("TEST 4: Character pronouns")
    print("=" * 70)

    profile = MangaProfile("Pronoun_Test", profiles_dir=TEST_PROFILES_DIR)
    profile.add_character("キャスカ", "Casca", "she/her", "warrior")
    profile.add_character("ガッツ", "Guts", "he/him", "protagonist")

    block = profile.get_characters_as_prompt_block()
    _result("CHARACTERS header", "CHARACTERS:" in block, "Missing header")
    _result(
        "Casca she/her",
        "Casca" in block and "she/her" in block,
        f"Block: {block}",
    )
    _result(
        "Casca role",
        "warrior" in block,
        f"Block: {block}",
    )
    _result(
        "Guts he/him",
        "Guts" in block and "he/him" in block,
        f"Block: {block}",
    )


# ──────────────────────────────────────────────────────────────────────────
# Test 5: Full prompt build
# ──────────────────────────────────────────────────────────────────────────
def test_full_prompt_build():
    print("\n" + "=" * 70)
    print("TEST 5: Full prompt build")
    print("=" * 70)

    profile = MangaProfile("FullPrompt_Test", profiles_dir=TEST_PROFILES_DIR)

    # Glossary
    profile.add_glossary_term("ガッツ", "Guts", "character")
    profile.add_glossary_term("鷹の団", "Band of the Hawk", "place")

    # Characters
    profile.add_character("キャスカ", "Casca", "she/her", "warrior")

    # Chapter summary
    profile.add_chapter_summary(1, "Guts left the Band of the Hawk after defeating Griffith in a duel.")

    # Recent translations
    for i in range(5):
        profile.add_translated_lines([{
            "japanese": f"テスト台詞{i}",
            "english": f"Test dialogue {i}",
            "chapter": 1,
            "page": 5,
        }])

    # Build prompt
    texts = ["俺は行く", "待ってくれ", "さようなら"]
    prompt = build_translation_prompt(
        texts_to_translate=texts,
        profile=profile,
        chapter_num=2,
        page_num=3,
    )

    # Verify all blocks present
    _result("System instruction", "professional manga translator" in prompt)
    _result("Honorifics instruction", "-san" in prompt and "-kun" in prompt)
    _result("Glossary block", "GLOSSARY" in prompt and "ガッツ → Guts" in prompt)
    _result("Character block", "CHARACTERS:" in prompt and "Casca" in prompt)
    _result("Chapter summary", "PREVIOUS CHAPTERS:" in prompt and "Band of the Hawk" in prompt)
    _result("Recent dialogue", "RECENT DIALOGUE" in prompt and "Test dialogue" in prompt)
    _result("Chapter/page", "Chapter 2, Page 3" in prompt)
    _result("Numbered inputs", "1. 俺は行く" in prompt and "3. さようなら" in prompt)
    _result("Output instruction", "OUTPUT (numbered translations only):" in prompt)

    # Print full prompt for manual review
    print("\n--- FULL PROMPT (for manual review) ---")
    print(prompt)
    print("--- END PROMPT ---\n")


# ──────────────────────────────────────────────────────────────────────────
# Test 6: Auto-glossary extraction (mock)
# ──────────────────────────────────────────────────────────────────────────
def test_auto_glossary():
    print("\n" + "=" * 70)
    print("TEST 6: Auto-glossary extraction")
    print("=" * 70)

    from src.translation.auto_glossary import extract_glossary_candidates

    class MockOllama:
        def chat(self, **kwargs):
            # Simulate Gemma3 returning glossary candidates
            return {
                "message": {
                    "content": json.dumps([
                        {"japanese": "ガッツ", "english": "Guts", "category": "character"},
                        {"japanese": "グリフィス", "english": "Griffith", "category": "character"},
                        {"japanese": "鷹の団", "english": "Band of the Hawk", "category": "place"},
                    ])
                }
            }

    profile = MangaProfile("Glossary_Test", profiles_dir=TEST_PROFILES_DIR)
    jp_texts = ["ガッツは言った", "グリフィスが答えた", "鷹の団の仲間たち"]
    en_texts = ["Guts said", "Griffith answered", "The members of the Band of the Hawk"]

    candidates = extract_glossary_candidates(
        jp_texts, en_texts, profile, MockOllama()
    )

    _result("Returns list", isinstance(candidates, list), f"Got {type(candidates)}")
    _result("3 candidates", len(candidates) == 3, f"Got {len(candidates)}")
    if candidates:
        _result(
            "First candidate has japanese",
            "japanese" in candidates[0] and candidates[0]["japanese"] == "ガッツ",
            f"Got: {candidates[0]}",
        )
        _result(
            "Has category",
            candidates[0].get("category") == "character",
            f"Got: {candidates[0].get('category')}",
        )

    # Test with malformed response
    class BadOllama:
        def chat(self, **kwargs):
            return {"message": {"content": "Sorry, I can't do that."}}

    bad_result = extract_glossary_candidates(
        jp_texts, en_texts, profile, BadOllama()
    )
    _result("Handles malformed response", bad_result == [], f"Got: {bad_result}")

    print("\n  Candidates found:")
    for c in candidates:
        print(f"    {c['japanese']} → {c['english']} [{c['category']}]")


# ──────────────────────────────────────────────────────────────────────────
# Test 7: Chapter summary (mock)
# ──────────────────────────────────────────────────────────────────────────
def test_chapter_summary():
    print("\n" + "=" * 70)
    print("TEST 7: Chapter summary")
    print("=" * 70)

    from src.translation.chapter_summarizer import generate_chapter_summary

    class MockOllama:
        def chat(self, **kwargs):
            return {
                "message": {
                    "content": (
                        "In Chapter 5, Guts confronted Griffith about leaving "
                        "the Band of the Hawk. After a fierce duel, Guts won "
                        "and departed, leaving Griffith in despair."
                    )
                }
            }

    profile = MangaProfile("Summary_Test", profiles_dir=TEST_PROFILES_DIR)

    translations = [
        "I'm leaving the Band of the Hawk.",
        "You can't leave, Guts!",
        "Fight me, Griffith.",
        "If you win, you're free.",
        "I won. Goodbye.",
        "Guts... don't go...",
    ] + [f"Dialogue line {i}" for i in range(14)]

    summary = generate_chapter_summary(
        translations, chapter_num=5, profile=profile, ollama_client=MockOllama()
    )

    _result("Returns string", isinstance(summary, str), f"Got {type(summary)}")
    _result("Non-empty", len(summary) > 0, "Empty summary")
    _result("Mentions Guts", "Guts" in summary, f"Summary: {summary[:100]}")

    # Check it was saved to profile
    saved = profile.data["chapter_summaries"].get("5")
    _result("Saved to profile", saved is not None, "Not in chapter_summaries")
    _result(
        "Saved content matches",
        saved and saved["summary"] == summary,
        f"Mismatch",
    )

    print(f"\n  Summary: {summary}")


# ──────────────────────────────────────────────────────────────────────────
# Test 8: CLI tool
# ──────────────────────────────────────────────────────────────────────────
def test_cli_tool():
    print("\n" + "=" * 70)
    print("TEST 8: CLI tool (manage_profile.py)")
    print("=" * 70)

    import subprocess

    tool = str(Path(__file__).parent / "tools" / "manage_profile.py")
    python = ".venv/bin/python"

    # Create
    r = subprocess.run(
        [python, tool, "create", "CLITest"],
        capture_output=True, text=True,
    )
    _result("CLI create", r.returncode == 0, r.stderr[:200] if r.stderr else "")

    # Add term
    r = subprocess.run(
        [python, tool, "add-term", "CLITest", "忍術", "ninjutsu", "technique"],
        capture_output=True, text=True,
    )
    _result("CLI add-term", r.returncode == 0 and "ninjutsu" in r.stdout, r.stderr[:200])

    # Add character
    r = subprocess.run(
        [python, tool, "add-character", "CLITest", "ナルト", "Naruto", "he/him", "main character"],
        capture_output=True, text=True,
    )
    _result("CLI add-character", r.returncode == 0 and "Naruto" in r.stdout, r.stderr[:200])

    # Show
    r = subprocess.run(
        [python, tool, "show", "CLITest"],
        capture_output=True, text=True,
    )
    _result(
        "CLI show",
        r.returncode == 0 and "Naruto" in r.stdout and "ninjutsu" in r.stdout,
        r.stderr[:200],
    )

    # List
    r = subprocess.run(
        [python, tool, "list"],
        capture_output=True, text=True,
    )
    _result("CLI list", r.returncode == 0 and "CLITest" in r.stdout, r.stderr[:200])


# ──────────────────────────────────────────────────────────────────────────
# Test 9: Profile persistence
# ──────────────────────────────────────────────────────────────────────────
def test_persistence():
    print("\n" + "=" * 70)
    print("TEST 9: Profile persistence")
    print("=" * 70)

    # Create and populate
    p1 = MangaProfile("Persist_Test", profiles_dir=TEST_PROFILES_DIR)
    p1.add_glossary_term("テスト", "test", "general")
    p1.add_character("太郎", "Taro", "he/him", "hero")
    p1.add_chapter_summary(1, "Taro began his journey.")
    p1.add_translated_lines([{
        "japanese": "こんにちは", "english": "Hello",
        "chapter": 1, "page": 1,
    }])

    # Reload from disk
    p2 = MangaProfile("Persist_Test", profiles_dir=TEST_PROFILES_DIR)

    _result(
        "Glossary persisted",
        "テスト" in p2.data["glossary"],
    )
    _result(
        "Character persisted",
        "太郎" in p2.data["character_names"],
    )
    _result(
        "Chapter summary persisted",
        "1" in p2.data["chapter_summaries"],
    )
    _result(
        "Translation memory persisted",
        len(p2.data["recent_translations"]) == 1
        and p2.data["recent_translations"][0]["english"] == "Hello",
    )


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "TRANSLATION CONSISTENCY TESTS".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    test_glossary_injection()
    test_parse_response()
    test_rolling_memory()
    test_character_pronouns()
    test_full_prompt_build()
    test_auto_glossary()
    test_chapter_summary()
    test_cli_tool()
    test_persistence()

    # Cleanup
    shutil.rmtree(TEST_PROFILES_DIR, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        print(f"\n  ❌ {failed} test(s) failed!")
        return 1
    else:
        print(f"\n  ✅ All {passed} tests passed!")
        return 0


if __name__ == "__main__":
    exit(main())
