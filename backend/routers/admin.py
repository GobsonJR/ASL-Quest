from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import LetterProgress, PracticeSession, User, UserProgress, XpEvent
from backend.security import get_current_admin
from backend.services.progress import LETTERS
from backend.services.words import get_admin_word_stats

router = APIRouter(prefix="/admin", tags=["admin"])


def get_admin_overview(db: Session) -> dict:
    total_users = db.query(User).count()
    total_sessions = db.query(PracticeSession).count()
    total_xp = db.query(func.sum(XpEvent.amount)).scalar() or 0
    total_attempts = db.query(PracticeSession).count()
    total_correct = db.query(PracticeSession).filter(PracticeSession.correct.is_(True)).count()
    accuracy = round((total_correct / total_attempts) * 100, 1) if total_attempts else None
    active_users = db.query(PracticeSession.user_id).distinct().count()
    word_stats = get_admin_word_stats(db)
    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_practice_sessions": total_sessions,
        "total_xp_awarded": int(total_xp),
        "average_accuracy": accuracy,
        "total_letter_attempts": total_attempts,
        "total_word_practice_sessions": word_stats["total_word_practice_sessions"],
        "words_completed": word_stats["words_completed"],
        "average_word_accuracy": word_stats["average_word_accuracy"],
    }


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
    popular = []
    difficult = []
    for letter in LETTERS:
        attempts = db.query(PracticeSession).filter(PracticeSession.letter == letter).count()
        correct = db.query(PracticeSession).filter(PracticeSession.letter == letter, PracticeSession.correct.is_(True)).count()
        accuracy = round((correct / attempts) * 100, 1) if attempts else None
        item = {"letter": letter, "attempts": attempts, "accuracy": accuracy}
        popular.append(item)
    popular.sort(key=lambda item: item["attempts"], reverse=True)
    difficult = sorted([item for item in popular if item["accuracy"] is not None], key=lambda item: item["accuracy"])
    return {"popular": popular[:10], "difficult": difficult[:10]}


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
