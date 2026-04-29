"""
Tests for all three translation-consistency improvements:
  Fix 1 — Relevance-based context selection
  Fix 2 — Online auto-glossary builder
  Fix 3 — Post-translation glossary compliance validator with retry

Plus a full integration test that exercises all three in sequence.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from src.translation.manga_profile import MangaProfile
from src.translation.prompt_builder import build_translation_prompt
from src.translation.validator import (
    check_glossary_compliance,
    force_inject_terms,
    log_violations,
    validate_and_retry_translations,
)


passed = 0
failed = 0
TMPDIR = tempfile.mkdtemp(prefix="manga_improve_test_")


def _r(name: str, ok: bool, reason: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {reason}")


# ═══════════════════════════════════════════════════════════════════════════
# FIX 1 — Relevance-based context selection
# ═══════════════════════════════════════════════════════════════════════════

def test_1a_relevance_scoring():
    """Fighting texts should select fighting-related past translations."""
    print("\n" + "=" * 70)
    print("TEST 1a: Relevance scoring — fight vs shopping")
    print("=" * 70)

    p = MangaProfile("Rel1a", profiles_dir=TMPDIR)

    # 10 fighting lines
    fight_lines = [
        {"japanese": "剣を抜け！", "english": "Draw your sword!", "chapter": 1, "page": 1},
        {"japanese": "攻撃だ！", "english": "Attack!", "chapter": 1, "page": 1},
        {"japanese": "戦いの時が来た", "english": "The time to fight has come", "chapter": 1, "page": 2},
        {"japanese": "強い敵だ", "english": "A strong enemy", "chapter": 1, "page": 2},
        {"japanese": "剣術の達人", "english": "A master of swordsmanship", "chapter": 1, "page": 3},
        {"japanese": "戦うしかない", "english": "We have no choice but to fight", "chapter": 1, "page": 3},
        {"japanese": "攻撃を避けろ", "english": "Dodge the attack!", "chapter": 1, "page": 4},
        {"japanese": "必殺の一撃", "english": "A deadly blow", "chapter": 1, "page": 4},
        {"japanese": "剣で斬る", "english": "Cut with the sword", "chapter": 1, "page": 5},
        {"japanese": "戦場を走れ", "english": "Run across the battlefield", "chapter": 1, "page": 5},
    ]
    # 10 shopping lines
    shop_lines = [
        {"japanese": "買い物に行こう", "english": "Let's go shopping", "chapter": 1, "page": 6},
        {"japanese": "値段はいくら？", "english": "How much is the price?", "chapter": 1, "page": 6},
        {"japanese": "店に入った", "english": "Entered the shop", "chapter": 1, "page": 7},
        {"japanese": "お金が足りない", "english": "Not enough money", "chapter": 1, "page": 7},
        {"japanese": "買うものを選ぶ", "english": "Choose what to buy", "chapter": 1, "page": 8},
        {"japanese": "値段が高い", "english": "The price is high", "chapter": 1, "page": 8},
        {"japanese": "お店の人に聞く", "english": "Ask the shopkeeper", "chapter": 1, "page": 9},
        {"japanese": "安い商品を探す", "english": "Look for cheap goods", "chapter": 1, "page": 9},
        {"japanese": "買い物袋を持つ", "english": "Carry shopping bags", "chapter": 1, "page": 10},
        {"japanese": "レジで支払う", "english": "Pay at the register", "chapter": 1, "page": 10},
    ]

    p.add_translated_lines(fight_lines + shop_lines)

    # Query with fight-related text
    current_fight_texts = ["剣を構えろ！攻撃の準備だ"]
    block = p.get_relevant_translations_as_prompt_block(current_fight_texts, n=5)

    _r("Returns non-empty block", bool(block))
    _r("Header is RELEVANT PAST DIALOGUE", "RELEVANT PAST DIALOGUE" in block)

    # Check that all 5 returned entries are from fight set
    fight_keywords = {"剣", "攻撃", "戦", "斬", "敵", "必殺", "戦場"}
    shop_keywords = {"買", "値段", "店", "金", "レジ", "支払", "商品", "安い"}

    en_lines = [l.strip() for l in block.split("\n") if l.strip().startswith("JP:")]
    fight_count = sum(1 for l in en_lines if any(k in l for k in fight_keywords))
    shop_count = sum(1 for l in en_lines if any(k in l for k in shop_keywords))

    _r(
        f"All 5 selected are fight-related (fight={fight_count}, shop={shop_count})",
        fight_count == 5 and shop_count == 0,
        f"fight={fight_count}, shop={shop_count}",
    )

    # Print scores for the summary
    current_chars = set("".join(current_fight_texts))
    scored = []
    for entry in p.data["recent_translations"]:
        past_chars = set(entry["japanese"])
        inter = len(current_chars & past_chars)
        union = len(current_chars | past_chars)
        scored.append((inter / union if union else 0, entry["japanese"][:20]))
    scored.sort(reverse=True)
    print(f"\n  Top-5 scores:")
    for score, text in scored[:5]:
        print(f"    {score:.3f}  {text}")


def test_1b_empty_recent():
    print("\n" + "=" * 70)
    print("TEST 1b: Empty recent translations")
    print("=" * 70)

    p = MangaProfile("Rel1b", profiles_dir=TMPDIR)
    block = p.get_relevant_translations_as_prompt_block(["テスト"], n=5)
    _r("Returns empty string", block == "")


def test_1c_prompt_uses_relevant_block():
    print("\n" + "=" * 70)
    print("TEST 1c: Prompt uses RELEVANT block")
    print("=" * 70)

    p = MangaProfile("Rel1c", profiles_dir=TMPDIR)
    p.add_translated_lines([
        {"japanese": "剣を抜け", "english": "Draw your sword", "chapter": 1, "page": 1},
        {"japanese": "走れ", "english": "Run", "chapter": 1, "page": 1},
    ])
    prompt = build_translation_prompt(["剣で戦う"], p, chapter_num=1, page_num=2)
    _r("RELEVANT PAST DIALOGUE in prompt", "RELEVANT PAST DIALOGUE" in prompt)
    _r("RECENT DIALOGUE NOT in prompt", "RECENT DIALOGUE (for context" not in prompt)


# ═══════════════════════════════════════════════════════════════════════════
# FIX 2 — Online auto-glossary builder
# ═══════════════════════════════════════════════════════════════════════════

def test_2a_katakana_name():
    print("\n" + "=" * 70)
    print("TEST 2a: Katakana name extraction")
    print("=" * 70)

    p = MangaProfile("Auto2a", profiles_dir=TMPDIR)
    added = p.auto_update_glossary_from_pair("ルフィは言った", "Luffy said")
    _r("One term extracted", len(added) == 1, f"Got {len(added)}")
    _r("ルフィ → Luffy", added[0]["japanese"] == "ルフィ" and added[0]["english"] == "Luffy" if added else False)
    _r("auto_detected=True", p.data["glossary"].get("ルフィ", {}).get("auto_detected") is True)


def test_2b_multiple_names():
    print("\n" + "=" * 70)
    print("TEST 2b: Multiple names in one line")
    print("=" * 70)

    p = MangaProfile("Auto2b", profiles_dir=TMPDIR)
    added = p.auto_update_glossary_from_pair("ゾロとナミが戦う", "Zoro and Nami fight")
    _r("Two terms extracted", len(added) == 2, f"Got {len(added)}: {added}")
    names = {a["japanese"] for a in added}
    _r("ゾロ extracted", "ゾロ" in p.data["glossary"])
    _r("ナミ extracted", "ナミ" in p.data["glossary"])


def test_2c_skips_common():
    print("\n" + "=" * 70)
    print("TEST 2c: Skips common English words")
    print("=" * 70)

    p = MangaProfile("Auto2c", profiles_dir=TMPDIR)
    added = p.auto_update_glossary_from_pair("バトル", "The battle")
    # "The" should be skipped, "battle" is lowercase so not picked up
    _r("No terms added for common words", len(added) == 0, f"Got {added}")
    _r("'The' not in glossary", "The" not in [e.get("english") for e in p.data["glossary"].values()])


def test_2d_skips_duplicates():
    print("\n" + "=" * 70)
    print("TEST 2d: Skips duplicates")
    print("=" * 70)

    p = MangaProfile("Auto2d", profiles_dir=TMPDIR)
    p.auto_update_glossary_from_pair("ルフィが笑った", "Luffy laughed")
    p.auto_update_glossary_from_pair("ルフィが泣いた", "Luffy cried")
    _r("Only one ルフィ entry", sum(1 for k in p.data["glossary"] if k == "ルフィ") == 1)


def test_2e_auto_label_in_prompt():
    print("\n" + "=" * 70)
    print("TEST 2e: (auto) label in glossary prompt block")
    print("=" * 70)

    p = MangaProfile("Auto2e", profiles_dir=TMPDIR)
    p.auto_update_glossary_from_pair("ルフィが走った", "Luffy ran")
    block = p.get_glossary_as_prompt_block()
    _r("(auto) label present", "(auto)" in block, f"Block: {block[:200]}")


def test_2f_inline_extraction():
    print("\n" + "=" * 70)
    print("TEST 2f: Online extraction during add_translated_lines")
    print("=" * 70)

    p = MangaProfile("Auto2f", profiles_dir=TMPDIR)
    p.add_translated_lines([
        {"japanese": "サンジが料理した", "english": "Sanji cooked", "chapter": 1, "page": 1},
    ])
    _r("サンジ auto-added to glossary", "サンジ" in p.data["glossary"])
    _r(
        "Entry says Sanji",
        p.data["glossary"].get("サンジ", {}).get("english") == "Sanji",
    )


# ═══════════════════════════════════════════════════════════════════════════
# FIX 3 — Post-translation glossary compliance validator
# ═══════════════════════════════════════════════════════════════════════════

def test_3a_violation_basic():
    print("\n" + "=" * 70)
    print("TEST 3a: Violation detection — basic")
    print("=" * 70)

    p = MangaProfile("Val3a", profiles_dir=TMPDIR)
    p.add_glossary_term("ガッツ", "Guts", "character")

    violations = check_glossary_compliance("ガッツは戦った", "The swordsman fought", p)
    _r("Violation detected", len(violations) == 1, f"Got {len(violations)}")
    _r(
        "Correct expected_english",
        violations[0]["expected_english"] == "Guts" if violations else False,
    )


def test_3b_no_violation():
    print("\n" + "=" * 70)
    print("TEST 3b: No violation — term present")
    print("=" * 70)

    p = MangaProfile("Val3b", profiles_dir=TMPDIR)
    p.add_glossary_term("ガッツ", "Guts", "character")

    violations = check_glossary_compliance("ガッツは戦った", "Guts fought bravely", p)
    _r("No violations", len(violations) == 0, f"Got {violations}")


def test_3c_auto_not_hard():
    print("\n" + "=" * 70)
    print("TEST 3c: Auto-detected terms not hard violations")
    print("=" * 70)

    p = MangaProfile("Val3c", profiles_dir=TMPDIR)
    # Add as auto-detected
    p.data["glossary"]["ルフィ"] = {
        "english": "Luffy",
        "category": "character",
        "added_at": "2026-01-01",
        "auto_detected": True,
    }
    p.save()

    violations = check_glossary_compliance("ルフィは笑った", "The boy laughed", p)
    _r("Violation detected", len(violations) == 1)
    _r("Marked as auto_detected", violations[0].get("auto_detected") is True if violations else False)

    # validate_and_retry should not treat this as a hard violation
    corrected, reports = validate_and_retry_translations(
        ["ルフィは笑った"], ["The boy laughed"], p, ollama_client=None, max_retries=0,
    )
    _r("No hard violations (auto excluded)", len(reports) == 0, f"Got {reports}")
    _r("Translation unchanged", corrected[0] == "The boy laughed")


def test_3d_retry_fixes(mock=True):
    print("\n" + "=" * 70)
    print("TEST 3d: Retry fixes violation (mock)")
    print("=" * 70)

    p = MangaProfile("Val3d", profiles_dir=TMPDIR)
    p.add_glossary_term("ガッツ", "Guts", "character")

    class MockOllama:
        def chat(self, **kwargs):
            return {"message": {"content": "Guts fought bravely."}}

    corrected, reports = validate_and_retry_translations(
        ["ガッツは戦った"],
        ["The swordsman fought"],
        p,
        ollama_client=MockOllama(),
        max_retries=1,
    )
    _r("Violation detected", len(reports) == 1)
    _r("Translation corrected", corrected[0] == "Guts fought bravely.")


def test_3e_force_inject():
    print("\n" + "=" * 70)
    print("TEST 3e: Force inject when retry fails")
    print("=" * 70)

    p = MangaProfile("Val3e", profiles_dir=TMPDIR)
    p.add_glossary_term("ガッツ", "Guts", "character")

    class BadOllama:
        def chat(self, **kwargs):
            return {"message": {"content": "The warrior fought."}}

    corrected, reports = validate_and_retry_translations(
        ["ガッツは戦った"],
        ["The swordsman fought"],
        p,
        ollama_client=BadOllama(),
        max_retries=1,
    )
    _r("Force-injected [Guts]", "[Guts]" in corrected[0], f"Got: {corrected[0]}")


def test_3f_violation_log():
    print("\n" + "=" * 70)
    print("TEST 3f: Violation logging")
    print("=" * 70)

    p = MangaProfile("Val3f", profiles_dir=TMPDIR)
    p.add_glossary_term("ガッツ", "Guts", "character")
    p.add_glossary_term("鷹の団", "Band of the Hawk", "place")

    _, reports = validate_and_retry_translations(
        ["ガッツは鷹の団を去った", "鷹の団は崩壊した"],
        ["The man left the group", "The group collapsed"],
        p,
        ollama_client=None,
        max_retries=0,
    )

    log_path = os.path.join(TMPDIR, "violations_log.jsonl")
    log_violations(reports, page_num=5, chapter_num=2, log_path=log_path)

    _r("Log file created", os.path.exists(log_path))

    with open(log_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    _r("2 entries logged", len(entries) == 2, f"Got {len(entries)}")
    if entries:
        _r("Has chapter field", entries[0].get("chapter") == 2)
        _r("Has page field", entries[0].get("page") == 5)
        _r("Has violations list", isinstance(entries[0].get("violations"), list))


def test_3g_empty_glossary():
    print("\n" + "=" * 70)
    print("TEST 3g: No profile — no validation crash")
    print("=" * 70)

    p = MangaProfile("Val3g", profiles_dir=TMPDIR)
    corrected, reports = validate_and_retry_translations(
        ["テスト"], ["Test"], p, ollama_client=None, max_retries=0,
    )
    _r("Returns original unchanged", corrected == ["Test"])
    _r("No violations", len(reports) == 0)


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST — all three fixes together
# ═══════════════════════════════════════════════════════════════════════════

def test_integration():
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: All three fixes working together")
    print("=" * 70)

    p = MangaProfile("Integration", profiles_dir=TMPDIR)

    # Step 1: Add 20 past translations (10 fight, 10 other)
    fight = [
        {"japanese": "剣を抜け！", "english": "Draw your sword!", "chapter": 1, "page": 1},
        {"japanese": "攻撃だ！", "english": "Attack!", "chapter": 1, "page": 1},
        {"japanese": "戦いの時が来た", "english": "The time to fight has come", "chapter": 1, "page": 2},
        {"japanese": "強い敵だ", "english": "A strong enemy", "chapter": 1, "page": 2},
        {"japanese": "剣術の達人", "english": "A master of swordsmanship", "chapter": 1, "page": 3},
        {"japanese": "戦うしかない", "english": "We have no choice but to fight", "chapter": 1, "page": 3},
        {"japanese": "攻撃を避けろ", "english": "Dodge the attack!", "chapter": 1, "page": 4},
        {"japanese": "必殺の一撃", "english": "A deadly blow", "chapter": 1, "page": 4},
        {"japanese": "剣で斬る", "english": "Cut with the sword", "chapter": 1, "page": 5},
        {"japanese": "戦場を走れ", "english": "Run across the battlefield", "chapter": 1, "page": 5},
    ]
    other = [
        {"japanese": "お腹すいた", "english": "I'm hungry", "chapter": 1, "page": 6},
        {"japanese": "食べよう", "english": "Let's eat", "chapter": 1, "page": 6},
        {"japanese": "天気がいいね", "english": "Nice weather", "chapter": 1, "page": 7},
        {"japanese": "散歩しよう", "english": "Let's take a walk", "chapter": 1, "page": 7},
        {"japanese": "花が咲いた", "english": "Flowers bloomed", "chapter": 1, "page": 8},
        {"japanese": "春が来た", "english": "Spring has come", "chapter": 1, "page": 8},
        {"japanese": "今日は暑い", "english": "It's hot today", "chapter": 1, "page": 9},
        {"japanese": "水を飲む", "english": "Drink water", "chapter": 1, "page": 9},
        {"japanese": "眠いな", "english": "I'm sleepy", "chapter": 1, "page": 10},
        {"japanese": "寝よう", "english": "Let's sleep", "chapter": 1, "page": 10},
    ]
    p.add_translated_lines(fight + other)
    _r("Step 1: 20 translations added", len(p.data["recent_translations"]) == 20)

    # Step 2: Add locked glossary term
    p.add_glossary_term("ガッツ", "Guts", "character")
    _r("Step 2: ガッツ→Guts locked", "ガッツ" in p.data["glossary"])

    # Step 3: Add translation with katakana name → verify auto-glossary
    p.add_translated_lines([
        {"japanese": "グリフィスが現れた", "english": "Griffith appeared", "chapter": 2, "page": 1},
    ])
    _r("Step 3: Auto-glossary extracted グリフィス", "グリフィス" in p.data["glossary"])
    _r(
        "  → English = Griffith",
        p.data["glossary"].get("グリフィス", {}).get("english") == "Griffith",
    )

    # Step 4: Build prompt for fight scene → verify relevant context
    prompt = build_translation_prompt(
        ["剣を構えろ", "攻撃準備だ"], p, chapter_num=2, page_num=2,
    )
    _r("Step 4: RELEVANT PAST DIALOGUE in prompt", "RELEVANT PAST DIALOGUE" in prompt)

    # Step 5: Simulate glossary violation
    violations = check_glossary_compliance(
        "ガッツは戦った", "The warrior fought", p,
    )
    _r("Step 5: Violation detected for ガッツ", len(violations) >= 1)

    # Step 6: Simulate retry fixing it
    class FixingOllama:
        def chat(self, **kwargs):
            return {"message": {"content": "Guts fought bravely."}}

    corrected, reports = validate_and_retry_translations(
        ["ガッツは戦った"],
        ["The warrior fought"],
        p,
        ollama_client=FixingOllama(),
        max_retries=1,
    )
    _r("Step 6: Retry corrected to include Guts", "Guts" in corrected[0])

    # Step 7: Check violations log
    log_path = os.path.join(TMPDIR, "integration_violations.jsonl")
    log_violations(reports, page_num=2, chapter_num=2, log_path=log_path)
    _r("Step 7: violations log written", os.path.exists(log_path))


# ═══════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  Fix 1 — Relevance selector:")
    print(f"    Selects fight-related context from mixed 20-entry pool")
    print(f"    Character-level Jaccard similarity scores printed above")

    print(f"\n  Fix 2 — Auto-glossary:")
    print(f"    Extracts katakana → capitalized-English pairs inline")
    print(f"    Marks auto-detected terms with (auto) in prompt block")
    print(f"    Skips common words, duplicates, short terms")

    print(f"\n  Fix 3 — Glossary compliance validator:")
    print(f"    Detects missing locked terms in translations")
    print(f"    Auto-detected terms are soft (logged, not retried)")
    print(f"    Retries via LLM, force-injects if retry fails")
    print(f"    Logs all violations to JSONL for review")

    print(f"\n  RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed:
        print(f"\n  ❌ {failed} test(s) FAILED")
    else:
        print(f"\n  ✅ All {passed} tests passed!")


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + "TRANSLATION IMPROVEMENT TESTS".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    # Fix 1
    test_1a_relevance_scoring()
    test_1b_empty_recent()
    test_1c_prompt_uses_relevant_block()

    # Fix 2
    test_2a_katakana_name()
    test_2b_multiple_names()
    test_2c_skips_common()
    test_2d_skips_duplicates()
    test_2e_auto_label_in_prompt()
    test_2f_inline_extraction()

    # Fix 3
    test_3a_violation_basic()
    test_3b_no_violation()
    test_3c_auto_not_hard()
    test_3d_retry_fixes()
    test_3e_force_inject()
    test_3f_violation_log()
    test_3g_empty_glossary()

    # Integration
    test_integration()

    # Cleanup
    shutil.rmtree(TMPDIR, ignore_errors=True)

    print_summary()
    return 1 if failed else 0


if __name__ == "__main__":
    exit(main())
