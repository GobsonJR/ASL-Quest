"""Platform-wide (not per-user) aggregate stats shared by the admin router
and the mentor dashboard. Moved out of backend/routers/admin.py so both it
and backend/services/mentor_dashboard.py can import these without a circular
import between a router and a service.
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import PracticeSession, User, XpEvent
from backend.services.progress import LETTERS
from backend.services.words import get_admin_word_stats


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


def get_admin_letters_stats(db: Session) -> dict:
    popular = []
    for letter in LETTERS:
        attempts = db.query(PracticeSession).filter(PracticeSession.letter == letter).count()
        correct = db.query(PracticeSession).filter(PracticeSession.letter == letter, PracticeSession.correct.is_(True)).count()
        accuracy = round((correct / attempts) * 100, 1) if attempts else None
        item = {"letter": letter, "attempts": attempts, "accuracy": accuracy}
        popular.append(item)
    popular.sort(key=lambda item: item["attempts"], reverse=True)
    difficult = sorted([item for item in popular if item["accuracy"] is not None], key=lambda item: item["accuracy"])
    return {"popular": popular[:10], "difficult": difficult[:10]}
