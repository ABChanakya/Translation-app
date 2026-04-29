"""
Post-translation glossary compliance validator.

Checks that locked glossary terms were respected in the translation,
and retries non-compliant translations automatically.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Tuple

from src.translation.manga_profile import MangaProfile


def check_glossary_compliance(
    japanese_text: str,
    english_text: str,
    profile: MangaProfile,
) -> List[dict]:
    """
    Checks whether a translation respects all relevant glossary terms.

    Returns a list of violations:
    [{"japanese": ..., "expected_english": ..., "category": ..., "auto_detected": ...}]

    A violation means: the JP term appears in the source text, but the
    expected English translation does NOT appear in the English output.
    """
    violations: List[dict] = []
    glossary = profile.data["glossary"]

    for jp_term, entry in glossary.items():
        if jp_term not in japanese_text:
            continue

        expected_en = entry["english"]
        if expected_en.lower() not in english_text.lower():
            violations.append({
                "japanese": jp_term,
                "expected_english": expected_en,
                "category": entry.get("category", "general"),
                "auto_detected": entry.get("auto_detected", False),
            })

    return violations


def force_inject_terms(
    english_text: str,
    violations: List[dict],
) -> str:
    """
    Last resort: if retry still fails, append the expected terms
    in brackets so they're visible and the translation is at least usable.
    """
    result = english_text
    for v in violations:
        expected = v["expected_english"]
        if expected.lower() not in result.lower():
            result = result.rstrip(".")
            result += f" [{expected}]"
    return result


def validate_and_retry_translations(
    japanese_texts: List[str],
    english_texts: List[str],
    profile: MangaProfile,
    ollama_client=None,
    model: str = "gemma4:latest",
    max_retries: int = 1,
) -> Tuple[List[str], List[dict]]:
    """
    Validates all translations against the glossary.
    Retries any that have hard violations (manually-locked terms).

    Args:
        japanese_texts: source texts
        english_texts: current translations
        profile: MangaProfile with glossary
        ollama_client: ollama module (needs .chat()); None to skip retry
        model: Ollama model name
        max_retries: how many LLM retries to attempt

    Returns:
        (corrected_translations, violation_reports)
    """
    corrected = list(english_texts)
    all_violations: List[dict] = []

    for i, (jp, en) in enumerate(zip(japanese_texts, english_texts)):
        violations = check_glossary_compliance(jp, en, profile)

        # Only retry for manually-locked terms
        hard_violations = [
            v for v in violations if not v.get("auto_detected", False)
        ]

        if not hard_violations:
            continue

        all_violations.append({
            "index": i,
            "japanese": jp,
            "english": en,
            "violations": hard_violations,
        })

        if max_retries <= 0 or ollama_client is None:
            continue

        # Build a correction prompt
        violation_list = "\n".join(
            f"  - '{v['japanese']}' must be translated as "
            f"'{v['expected_english']}', not omitted or paraphrased"
            for v in hard_violations
        )

        correction_prompt = (
            f"Retranslate this Japanese manga dialogue:\n"
            f"JP: {jp}\n\n"
            f"Your previous translation was:\n"
            f"EN: {en}\n\n"
            f"It violated these mandatory glossary terms:\n"
            f"{violation_list}\n\n"
            f"Provide ONLY the corrected English translation, "
            f"no explanation. The corrected translation must "
            f"include all the required terms above."
        )

        try:
            response = ollama_client.chat(
                model=model,
                messages=[{"role": "user", "content": correction_prompt}],
                options={"temperature": 0.1},
            )
            retry_text = response["message"]["content"].strip()

            # Verify retry fixed it
            retry_violations = check_glossary_compliance(jp, retry_text, profile)
            retry_hard = [
                v for v in retry_violations if not v.get("auto_detected", False)
            ]

            if not retry_hard:
                corrected[i] = retry_text
                print(f"  ✅ Glossary retry fixed bubble {i + 1}")
            else:
                corrected[i] = force_inject_terms(retry_text, retry_hard)
                print(
                    f"  ⚠️  Bubble {i + 1}: retry incomplete, "
                    f"force-injected terms"
                )
        except Exception as e:
            print(f"  ⚠️  Retry failed for bubble {i + 1}: {e}")

    return corrected, all_violations


def log_violations(
    violations: List[dict],
    page_num: int,
    chapter_num: int,
    log_path: str = "violations_log.jsonl",
):
    """
    Appends glossary violation reports to violations_log.jsonl
    for later review and fine-tuning data collection.
    """
    if not violations:
        return

    with open(log_path, "a", encoding="utf-8") as f:
        for v in violations:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "chapter": chapter_num,
                "page": page_num,
                "bubble_index": v["index"],
                "japanese": v["japanese"],
                "original_english": v["english"],
                "violations": v["violations"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"  📋 Logged {len(violations)} glossary violation(s) "
        f"to {log_path}"
    )
