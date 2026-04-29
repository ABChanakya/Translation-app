"""
SQLAlchemy ORM models for the manga translation review tool.

Tables:
    projects        — manga series
    chapters        — uploaded chapters within a project
    pages           — individual page images within a chapter
    bubbles         — detected text regions with OCR + translation
    correction_log  — every human review action (accept/correct/skip)
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    series_name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    profile_path = Column(String)  # path to MangaProfile JSON

    chapters = relationship("Chapter", back_populates="project", cascade="all, delete-orphan")


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    chapter_num = Column(Integer, nullable=False)
    status = Column(String, default="processing")  # processing / ready / complete
    total_pages = Column(Integer, default=0)
    total_bubbles = Column(Integer, default=0)
    reviewed_bubbles = Column(Integer, default=0)
    accepted_bubbles = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="chapters")
    pages = relationship("Page", back_populates="chapter", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"), nullable=False)
    page_num = Column(Integer, nullable=False)
    original_image_path = Column(String)
    inpainted_image_path = Column(String)
    final_image_path = Column(String)
    status = Column(String, default="pending")  # pending / processing / ready / reviewed

    chapter = relationship("Chapter", back_populates="pages")
    bubbles = relationship("Bubble", back_populates="page", cascade="all, delete-orphan")


class Bubble(Base):
    __tablename__ = "bubbles"

    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False)
    bubble_index = Column(Integer, nullable=False)
    bubble_type = Column(String, default="speech")  # speech / narration / sfx / thought
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)
    mask_points = Column(Text)  # JSON polygon from YOLOv8-seg
    japanese_text = Column(Text)
    suggested_translation = Column(Text)
    human_translation = Column(Text)
    status = Column(String, default="pending")  # pending / accepted / corrected / skipped
    ocr_confidence = Column(Float)
    quality_score = Column(Float)   # 0.0–1.0 composite confidence (OCR + translation)
    edit_distance = Column(Integer)
    was_accepted = Column(Boolean)
    reviewed_at = Column(DateTime, nullable=True)

    # ── Annotation / manual editing fields ──────────────────────────
    is_manual = Column(Boolean, default=False)  # True if user drew this region
    mode = Column(String, default="translate_and_inpaint")  # translate_and_inpaint / inpaint_only / manual_text / review_later
    mask_polygon = Column(Text)  # JSON array of {x, y} points (user-editable polygon)
    font_family = Column(String, default="Bangers")
    font_size = Column(Integer)  # NULL = auto-fit
    font_color = Column(String, default="#000000")
    stroke_color = Column(String, default="#ffffff")
    stroke_width = Column(Integer, default=1)
    text_align = Column(String, default="center")  # left / center / right
    notes = Column(Text, nullable=True)  # editorial notes from reviewer

    page = relationship("Page", back_populates="bubbles")
    corrections = relationship("CorrectionLog", back_populates="bubble", cascade="all, delete-orphan")


class CorrectionLog(Base):
    __tablename__ = "correction_log"

    id = Column(Integer, primary_key=True)
    bubble_id = Column(Integer, ForeignKey("bubbles.id"), nullable=False)
    series_name = Column(String)
    action = Column(String)  # accept / correct / skip
    japanese_text = Column(Text)
    suggested = Column(Text)
    human = Column(Text)
    edit_distance = Column(Integer)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    bubble = relationship("Bubble", back_populates="corrections")
