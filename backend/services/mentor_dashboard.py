"""Read-only aggregation for the Mentor Analytics Dashboard.

Every number here is computed live from the existing tables — no new tables,
no invented data. Wherever an equivalent aggregate already exists (overview,
per-letter, per-word stats), this reuses that existing function rather than
recomputing it, per the "reuse existing services" instruction.
"""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import (
    ChatbotConversation,
    ChatbotFeedback,
    ChatbotMessage,
    NativeSign,
    NativeSignProgress,
    NativeSignSession,
    User,
    UserProgress,
)
from backend.services.admin_stats import get_admin_letters_stats, get_admin_overview
from backend.services.words import get_admin_word_stats

MASTERY_COMPLETE_THRESHOLD = 100.0  # same convention as backend/routers/native.py


def _student_rows(db: Session) -> list[dict]:
    users = db.query(User).order_by(User.created_at.asc()).all()
    rows: list[dict] = []
    for user in users:
        progress = db.query(UserProgress).filter(UserProgress.user_id == user.id).first()

        native_rows = db.query(NativeSignProgress).filter(NativeSignProgress.user_id == user.id).all()
        native_mastery = (
            round(sum(row.mastery for row in native_rows) / len(native_rows), 1) if native_rows else None
        )

        rows.append(
            {
                "id": user.id,
                "username": user.username,
                "role": user.role,
                "level": progress.level if progress else 1,
                "xp": progress.xp if progress else 0,
                "current_streak": progress.current_streak if progress else 0,
                "accuracy": (
                    round((progress.total_correct / progress.total_attempts) * 100, 1)
                    if progress and progress.total_attempts
                    else None
                ),
                "native_mastery": native_mastery,
            }
        )
    # Most active learners first — the dashboard is for mentors scanning for
    # engagement, not an alphabetical roster.
    rows.sort(key=lambda item: item["xp"], reverse=True)
    return rows


def _native_stats(db: Session) -> dict:
    signs = db.query(NativeSign).filter(NativeSign.active.is_(True)).order_by(NativeSign.gloss).all()

    attempt_counts = dict(
        db.query(NativeSignSession.native_sign_id, func.count(NativeSignSession.id))
        .group_by(NativeSignSession.native_sign_id)
        .all()
    )
    mastery_by_sign = dict(
        db.query(NativeSignProgress.native_sign_id, func.avg(NativeSignProgress.mastery))
        .group_by(NativeSignProgress.native_sign_id)
        .all()
    )

    per_sign = []
    category_attempts: dict[str, int] = {}
    for sign in signs:
        attempts = int(attempt_counts.get(sign.id, 0))
        avg_mastery = mastery_by_sign.get(sign.id)
        per_sign.append(
            {
                "sign_id": sign.id,
                "gloss": sign.gloss,
                "display_name": sign.display_name,
                "category": sign.category,
                "attempts": attempts,
                "mastery": round(float(avg_mastery), 1) if avg_mastery is not None else None,
            }
        )
        if attempts and sign.category:
            category_attempts[sign.category] = category_attempts.get(sign.category, 0) + attempts

    per_sign.sort(key=lambda item: item["attempts"], reverse=True)
    category_distribution = [
        {"category": name, "attempts": count}
        for name, count in sorted(category_attempts.items(), key=lambda item: item[1], reverse=True)
    ]

    total_attempts = db.query(NativeSignSession).count()
    mastered_pairs = db.query(NativeSignProgress).filter(NativeSignProgress.mastery >= MASTERY_COMPLETE_THRESHOLD).count()

    return {
        "total_attempts": total_attempts,
        "signs_mastered_by_someone": mastered_pairs,
        "per_sign": per_sign,
        "category_distribution": category_distribution,
    }


def _chatbot_stats(db: Session) -> dict:
    total_conversations = db.query(ChatbotConversation).count()
    total_messages = db.query(ChatbotMessage).count()
    positive = db.query(ChatbotFeedback).filter(ChatbotFeedback.rating == 1).count()
    negative = db.query(ChatbotFeedback).filter(ChatbotFeedback.rating == 0).count()
    return {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "positive_feedback": positive,
        "negative_feedback": negative,
    }


def get_mentor_dashboard(db: Session) -> dict:
    overview = get_admin_overview(db)
    overview["total_native_sign_attempts"] = db.query(NativeSignSession).count()

    return {
        "overview": overview,
        "students": _student_rows(db),
        "az": get_admin_letters_stats(db),
        "words": get_admin_word_stats(db),
        "native": _native_stats(db),
        "chatbot": _chatbot_stats(db),
    }
