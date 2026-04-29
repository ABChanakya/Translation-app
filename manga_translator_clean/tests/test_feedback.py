#!/usr/bin/env python3
"""Unit tests for CorrectionCapture."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.database import init_db, get_session
from src.db.models import Bubble, Chapter, Page, Project
from src.feedback.capture import CorrectionCapture, _edit_distance

PASS = 0
FAIL = 0


def _r(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  \u2705 {label}")
    else:
        FAIL += 1
        print(f"  \u274c {label}" + (f": {detail}" if detail else ""))


def _setup():
    """Create an in-memory DB with one project/chapter/page and 3 bubbles."""
    engine = init_db(":memory:")
    session = get_session(engine)

    project = Project(series_name="TestManga")
    session.add(project)
    session.flush()

    chapter = Chapter(project_id=project.id, chapter_num=1, total_bubbles=3)
    session.add(chapter)
    session.flush()

    page = Page(chapter_id=chapter.id, page_num=1, status="ready")
    session.add(page)
    session.flush()

    bubbles = []
    for i, (jp, en) in enumerate([
        ("\u304a\u524d\u3092\u5012\u3059", "I will defeat you"),
        ("\u884c\u304f\u305e", "Let's go"),
        ("\u5371\u306a\u3044\uff01", "Watch out!"),
    ]):
        b = Bubble(
            page_id=page.id,
            bubble_index=i,
            bubble_type="speech",
            x1=0, y1=i * 100, x2=200, y2=(i + 1) * 100,
            japanese_text=jp,
            suggested_translation=en,
            status="pending",
            ocr_confidence=0.95,
        )
        session.add(b)
        bubbles.append(b)
    session.flush()

    return session, project, chapter, page, bubbles


def test_edit_distance():
    print("\n" + "=" * 60)
    print("TEST: edit_distance")
    print("=" * 60)
    _r("identical strings", _edit_distance("abc", "abc") == 0)
    _r("one insertion", _edit_distance("abc", "abcd") == 1)
    _r("one deletion", _edit_distance("abcd", "abc") == 1)
    _r("one substitution", _edit_distance("abc", "axc") == 1)
    _r("empty vs non-empty", _edit_distance("", "abc") == 3)
    _r("both empty", _edit_distance("", "") == 0)


def test_log_accept():
    print("\n" + "=" * 60)
    print("TEST: log_correction \u2014 accept")
    print("=" * 60)
    session, project, chapter, page, bubbles = _setup()
    cc = CorrectionCapture(session)

    entry = cc.log_correction(
        bubble_id=bubbles[0].id,
        series_name="TestManga",
        action="accept",
        japanese_text=bubbles[0].japanese_text,
        suggested=bubbles[0].suggested_translation,
    )

    _r("Log entry created", entry.id is not None)
    _r("Action is accept", entry.action == "accept")
    _r("Edit distance 0", entry.edit_distance == 0)
    _r("Bubble status updated", bubbles[0].status == "accepted")
    _r("Bubble was_accepted True", bubbles[0].was_accepted is True)
    _r("Human translation set to suggestion", bubbles[0].human_translation == bubbles[0].suggested_translation)


def test_log_correct():
    print("\n" + "=" * 60)
    print("TEST: log_correction \u2014 correct")
    print("=" * 60)
    session, project, chapter, page, bubbles = _setup()
    cc = CorrectionCapture(session)

    entry = cc.log_correction(
        bubble_id=bubbles[1].id,
        series_name="TestManga",
        action="correct",
        japanese_text=bubbles[1].japanese_text,
        suggested=bubbles[1].suggested_translation,
        human="Let's move out",
    )

    _r("Log entry created", entry.id is not None)
    _r("Action is correct", entry.action == "correct")
    _r("Edit distance > 0", entry.edit_distance > 0)
    _r("Bubble status corrected", bubbles[1].status == "corrected")
    _r("Bubble was_accepted False", bubbles[1].was_accepted is False)
    _r("Human translation stored", bubbles[1].human_translation == "Let's move out")


def test_log_skip():
    print("\n" + "=" * 60)
    print("TEST: log_correction \u2014 skip")
    print("=" * 60)
    session, project, chapter, page, bubbles = _setup()
    cc = CorrectionCapture(session)

    entry = cc.log_correction(
        bubble_id=bubbles[2].id,
        series_name="TestManga",
        action="skip",
        japanese_text=bubbles[2].japanese_text,
        suggested=bubbles[2].suggested_translation,
    )

    _r("Log entry created", entry.id is not None)
    _r("Action is skip", entry.action == "skip")
    _r("Bubble status skipped", bubbles[2].status == "skipped")


def test_get_corrections_for_series():
    print("\n" + "=" * 60)
    print("TEST: get_corrections_for_series")
    print("=" * 60)
    session, project, chapter, page, bubbles = _setup()
    cc = CorrectionCapture(session)

    cc.log_correction(bubbles[0].id, "TestManga", "accept", "\u304a\u524d\u3092\u5012\u3059", "I will defeat you")
    cc.log_correction(bubbles[1].id, "TestManga", "correct", "\u884c\u304f\u305e", "Let's go", "Let's move out")
    cc.log_correction(bubbles[2].id, "TestManga", "skip", "\u5371\u306a\u3044\uff01", "Watch out!")

    corrections = cc.get_corrections_for_series("TestManga")
    _r("Returns 2 (accept + correct, not skip)", len(corrections) == 2)
    _r("First is accept", corrections[0]["action"] == "accept")
    _r("Second is correct", corrections[1]["action"] == "correct")
    _r("Has japanese_text", corrections[0]["japanese_text"] == "\u304a\u524d\u3092\u5012\u3059")


def test_accuracy_stats():
    print("\n" + "=" * 60)
    print("TEST: get_accuracy_stats")
    print("=" * 60)
    session, project, chapter, page, bubbles = _setup()
    cc = CorrectionCapture(session)

    cc.log_correction(bubbles[0].id, "TestManga", "accept", "\u304a\u524d\u3092\u5012\u3059", "I will defeat you")
    cc.log_correction(bubbles[1].id, "TestManga", "correct", "\u884c\u304f\u305e", "Let's go", "Let's move out")
    cc.log_correction(bubbles[2].id, "TestManga", "accept", "\u5371\u306a\u3044\uff01", "Watch out!")

    stats = cc.get_accuracy_stats("TestManga")
    _r("Total reviewed = 3", stats["total_reviewed"] == 3)
    _r("Acceptance rate = 2/3", abs(stats["acceptance_rate"] - 0.667) < 0.01)
    _r("Most corrected has entry", len(stats["most_corrected_terms"]) >= 1)
    _r("Improvement by chapter exists", 1 in stats["improvement_by_chapter"])

    # Empty series
    empty = cc.get_accuracy_stats("NonExistent")
    _r("Empty series returns 0", empty["total_reviewed"] == 0)


def test_export_training_pairs():
    print("\n" + "=" * 60)
    print("TEST: export_training_pairs")
    print("=" * 60)
    session, project, chapter, page, bubbles = _setup()
    cc = CorrectionCapture(session)

    cc.log_correction(bubbles[0].id, "TestManga", "accept", "\u304a\u524d\u3092\u5012\u3059", "I will defeat you")
    cc.log_correction(bubbles[1].id, "TestManga", "correct", "\u884c\u304f\u305e", "Let's go", "Let's go!")  # edit_distance=1
    cc.log_correction(bubbles[2].id, "TestManga", "skip", "\u5371\u306a\u3044\uff01", "Watch out!")

    pairs = cc.export_training_pairs("TestManga")
    _r("Returns 2 pairs (skip excluded)", len(pairs) == 2, f"Got {len(pairs)}")
    _r("First pair has japanese", pairs[0]["japanese"] == "\u304a\u524d\u3092\u5012\u3059")
    _r("First pair has english", pairs[0]["english"] == "I will defeat you")
    _r("Corrected pair uses human text", pairs[1]["english"] == "Let's go!")


def test_chapter_counters():
    print("\n" + "=" * 60)
    print("TEST: chapter counter updates")
    print("=" * 60)
    session, project, chapter, page, bubbles = _setup()
    cc = CorrectionCapture(session)

    cc.log_correction(bubbles[0].id, "TestManga", "accept", "\u304a\u524d\u3092\u5012\u3059", "I will defeat you")
    session.refresh(chapter)
    _r("Reviewed count = 1 after first accept", chapter.reviewed_bubbles == 1)
    _r("Accepted count = 1 after first accept", chapter.accepted_bubbles == 1)

    cc.log_correction(bubbles[1].id, "TestManga", "correct", "\u884c\u304f\u305e", "Let's go", "Let's move")
    session.refresh(chapter)
    _r("Reviewed count = 2 after correction", chapter.reviewed_bubbles == 2)
    _r("Accepted count still 1", chapter.accepted_bubbles == 1)


if __name__ == "__main__":
    test_edit_distance()
    test_log_accept()
    test_log_correct()
    test_log_skip()
    test_get_corrections_for_series()
    test_accuracy_stats()
    test_export_training_pairs()
    test_chapter_counters()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    if FAIL:
        print(f"\n\u274c {FAIL} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"\n\u2705 All {PASS} tests passed!")
