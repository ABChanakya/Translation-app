"""
Correction capture system for the human-in-the-loop translation workflow.

Every time a human accepts, corrects, or skips a translation suggestion,
the action is logged here. This data drives:
  1. Rolling accuracy stats (acceptance rate per series/chapter)
  2. MangaProfile updates (glossary, memory)
  3. Training pair export for future fine-tuning
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.db.models import Bubble, Chapter, CorrectionLog, Page, Project


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


class CorrectionCapture:
    """Records and queries human review actions."""

    def __init__(self, session: Session):
        self.session = session

    # ── Recording ────────────────────────────────────────────────────

    def log_correction(
        self,
        bubble_id: int,
        series_name: str,
        action: str,  # "accept" | "correct" | "skip"
        japanese_text: str,
        suggested: str,
        human: str | None = None,
    ) -> CorrectionLog:
        """
        Log a single human review action for a bubble.

        Updates the bubble row and appends to correction_log.
        """
        now = datetime.now(timezone.utc)

        bubble = self.session.get(Bubble, bubble_id)
        if bubble is None:
            raise ValueError(f"Bubble {bubble_id} not found")

        dist = 0
        if action == "accept":
            bubble.human_translation = suggested
            bubble.was_accepted = True
            bubble.status = "accepted"
        elif action == "correct":
            bubble.human_translation = human
            bubble.was_accepted = False
            bubble.status = "corrected"
            dist = _edit_distance(suggested, human or "")
        elif action == "skip":
            bubble.status = "skipped"
        else:
            raise ValueError(f"Unknown action: {action}")

        bubble.edit_distance = dist
        bubble.reviewed_at = now

        entry = CorrectionLog(
            bubble_id=bubble_id,
            series_name=series_name,
            action=action,
            japanese_text=japanese_text,
            suggested=suggested,
            human=human if action == "correct" else suggested,
            edit_distance=dist,
            timestamp=now,
        )
        self.session.add(entry)
        self.session.flush()  # ensure bubble status is visible to count queries

        # Update chapter counters
        page = self.session.get(Page, bubble.page_id)
        if page:
            chapter = self.session.get(Chapter, page.chapter_id)
            if chapter:
                chapter.reviewed_bubbles = (
                    self.session.query(func.count(Bubble.id))
                    .join(Page)
                    .filter(
                        Page.chapter_id == chapter.id,
                        Bubble.status.in_(["accepted", "corrected", "skipped"]),
                    )
                    .scalar()
                    or 0
                )
                chapter.accepted_bubbles = (
                    self.session.query(func.count(Bubble.id))
                    .join(Page)
                    .filter(
                        Page.chapter_id == chapter.id,
                        Bubble.status == "accepted",
                    )
                    .scalar()
                    or 0
                )

        self.session.commit()
        return entry

    # ── Queries ──────────────────────────────────────────────────────

    def get_corrections_for_series(self, series_name: str) -> list[dict[str, Any]]:
        """All human-approved translations for a series (accepted or corrected)."""
        rows = (
            self.session.query(CorrectionLog)
            .filter(
                CorrectionLog.series_name == series_name,
                CorrectionLog.action.in_(["accept", "correct"]),
            )
            .order_by(CorrectionLog.timestamp)
            .all()
        )
        return [
            {
                "id": r.id,
                "bubble_id": r.bubble_id,
                "action": r.action,
                "japanese_text": r.japanese_text,
                "suggested": r.suggested,
                "human": r.human,
                "edit_distance": r.edit_distance,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in rows
        ]

    def get_accuracy_stats(self, series_name: str) -> dict[str, Any]:
        """
        Aggregate accuracy stats for a series.

        Returns:
            total_reviewed, acceptance_rate, most_corrected_terms,
            improvement_by_chapter.
        """
        logs = (
            self.session.query(CorrectionLog)
            .filter(CorrectionLog.series_name == series_name)
            .all()
        )
        if not logs:
            return {
                "total_reviewed": 0,
                "acceptance_rate": 0.0,
                "most_corrected_terms": [],
                "improvement_by_chapter": {},
            }

        total = len(logs)
        accepted = sum(1 for l in logs if l.action == "accept")
        acceptance_rate = accepted / total if total else 0.0

        # Most corrected terms — Japanese texts that were corrected most often
        corrected_jp = [l.japanese_text for l in logs if l.action == "correct" and l.japanese_text]
        term_counts = Counter(corrected_jp).most_common(10)
        most_corrected = [{"japanese": jp, "count": c} for jp, c in term_counts]

        # Improvement by chapter — acceptance rate per chapter number
        # Need to join through bubbles → pages → chapters
        chapter_stats: dict[int, dict] = {}
        for log in logs:
            bubble = self.session.get(Bubble, log.bubble_id)
            if not bubble:
                continue
            page = self.session.get(Page, bubble.page_id)
            if not page:
                continue
            chapter = self.session.get(Chapter, page.chapter_id)
            if not chapter:
                continue
            ch_num = chapter.chapter_num
            if ch_num not in chapter_stats:
                chapter_stats[ch_num] = {"total": 0, "accepted": 0}
            chapter_stats[ch_num]["total"] += 1
            if log.action == "accept":
                chapter_stats[ch_num]["accepted"] += 1

        improvement = {
            ch: round(s["accepted"] / s["total"], 3) if s["total"] else 0.0
            for ch, s in sorted(chapter_stats.items())
        }

        return {
            "total_reviewed": total,
            "acceptance_rate": round(acceptance_rate, 3),
            "most_corrected_terms": most_corrected,
            "improvement_by_chapter": improvement,
        }

    def export_training_pairs(self, series_name: str) -> list[dict[str, str]]:
        """
        High-quality JP→EN pairs suitable for fine-tuning.

        Includes accepted suggestions and corrections with edit_distance < 5.
        """
        rows = (
            self.session.query(CorrectionLog)
            .filter(
                CorrectionLog.series_name == series_name,
                CorrectionLog.action.in_(["accept", "correct"]),
            )
            .all()
        )
        pairs = []
        for r in rows:
            if r.action == "accept" or (r.edit_distance is not None and r.edit_distance < 5):
                pairs.append({
                    "japanese": r.japanese_text or "",
                    "english": r.human or r.suggested or "",
                })
        return pairs
