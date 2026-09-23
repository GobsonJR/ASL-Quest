from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import PracticeSession, User, UserProgress
from backend.security import get_current_admin
from backend.services.admin_stats import get_admin_letters_stats, get_admin_overview
from backend.services.mentor_dashboard import get_mentor_dashboard
from backend.services.words import get_admin_word_stats

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/overview")
def admin_overview(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    return get_admin_overview(db)


@router.get("/activity")
def admin_activity(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    start = datetime.now(timezone.utc) - timedelta(days=90)
    day = func.date(PracticeSession.created_at)
    rows = (
        db.query(day.label("day"), func.count(PracticeSession.id).label("attempts"))
        .filter(PracticeSession.created_at >= start)
        .group_by(day)
        .order_by(day.asc())
        .all()
    )
    return {"points": [{"date": str(row.day), "value": int(row.attempts)} for row in rows]}


@router.get("/letters")
def admin_letters(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    return get_admin_letters_stats(db)


@router.get("/words")
def admin_words(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    return get_admin_word_stats(db)


@router.get("/users")
def admin_users(
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = None,
    role: str | None = None,
):
    query = db.query(User)
    if search:
        like = f"%{search.strip()}%"
        query = query.filter((User.username.ilike(like)) | (User.email.ilike(like)))
    if role:
        query = query.filter(User.role == role)
    total = query.count()
    users = query.order_by(User.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    items = []
    for user in users:
        progress = db.query(UserProgress).filter(UserProgress.user_id == user.id).first()
        sessions = db.query(PracticeSession).filter(PracticeSession.user_id == user.id).count()
        last_activity = (
            db.query(func.max(PracticeSession.created_at)).filter(PracticeSession.user_id == user.id).scalar()
        )
        items.append(
            {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "created_at": user.created_at.isoformat(),
                "practice_sessions": sessions,
                "xp": progress.xp if progress else 0,
                "last_activity": last_activity.isoformat() if last_activity else None,
            }
        )
    return {"page": page, "page_size": page_size, "total": total, "items": items}


@router.get("/recent-activity")
def admin_recent_activity(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    rows = db.query(PracticeSession).order_by(PracticeSession.created_at.desc()).limit(20).all()
    return {
        "items": [
            {
                "user_id": row.user_id,
                "letter": row.letter,
                "correct": row.correct,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    }


@router.get("/mentor-dashboard")
def admin_mentor_dashboard(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    return get_mentor_dashboard(db)
