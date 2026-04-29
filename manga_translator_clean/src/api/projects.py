"""Project (manga series) endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.db.models import Chapter, Project
from src.feedback.capture import CorrectionCapture

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    series_name: str


class ProjectOut(BaseModel):
    id: int
    series_name: str
    created_at: str
    profile_path: str | None
    chapters_count: int
    acceptance_rate: float

    model_config = {"from_attributes": True}


@router.post("", response_model=ProjectOut)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    existing = db.query(Project).filter(Project.series_name == body.series_name).first()
    if existing:
        raise HTTPException(400, f"Project '{body.series_name}' already exists")

    project = Project(
        series_name=body.series_name,
        created_at=datetime.now(timezone.utc),
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _project_out(project, db)


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return [_project_out(p, db) for p in projects]


@router.get("/{series}", response_model=ProjectOut)
def get_project(series: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.series_name == series).first()
    if not project:
        raise HTTPException(404, "Project not found")
    return _project_out(project, db)


@router.get("/{series}/chapters")
def get_project_chapters(series: str, db: Session = Depends(get_db)):
    """List all chapters for a project."""
    project = db.query(Project).filter(Project.series_name == series).first()
    if not project:
        raise HTTPException(404, "Project not found")
    chapters = (
        db.query(Chapter)
        .filter(Chapter.project_id == project.id)
        .order_by(Chapter.chapter_num)
        .all()
    )
    return [
        {
            "id": ch.id,
            "project_id": ch.project_id,
            "chapter_num": ch.chapter_num,
            "status": ch.status,
            "total_pages": ch.total_pages or 0,
            "total_bubbles": ch.total_bubbles or 0,
            "reviewed_bubbles": ch.reviewed_bubbles or 0,
            "accepted_bubbles": ch.accepted_bubbles or 0,
            "created_at": ch.created_at.isoformat() if ch.created_at else None,
        }
        for ch in chapters
    ]


@router.get("/{series}/stats")
def get_project_stats(series: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.series_name == series).first()
    if not project:
        raise HTTPException(404, "Project not found")
    cc = CorrectionCapture(db)
    return cc.get_accuracy_stats(series)


def _project_out(project: Project, db: Session) -> ProjectOut:
    chapters_count = db.query(Chapter).filter(Chapter.project_id == project.id).count()
    cc = CorrectionCapture(db)
    stats = cc.get_accuracy_stats(project.series_name)
    return ProjectOut(
        id=project.id,
        series_name=project.series_name,
        created_at=project.created_at.isoformat() if project.created_at else "",
        profile_path=project.profile_path,
        chapters_count=chapters_count,
        acceptance_rate=stats["acceptance_rate"],
    )
