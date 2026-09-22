from __future__ import annotations

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import Achievement, LetterProgress, PracticeSession, User, UserAchievement, UserProgress, XpEvent
from backend.services.analytics import get_letter_insights, get_overview
from backend.services.words import export_word_data, get_word_recommendations
from backend.services.progress import (
    LETTERS,
    MASTERED_CORRECT,
    letters_learned_count,
    letters_mastered_count,
    serialize_game_state,
)

ACHIEVEMENT_RULES = {
    "first_sign": lambda ctx: ctx["total_correct"] >= 1,
    "streak_7": lambda ctx: ctx["current_streak"] >= 7,
    "alphabet_beginner": lambda ctx: ctx["letters_learned"] >= 5,
    "alphabet_explorer": lambda ctx: ctx["letters_learned"] >= 13,
    "alphabet_master": lambda ctx: ctx["letters_learned"] >= 26,
    "speed_signer": lambda ctx: ctx["speed_completed"],
    "daily_perfect": lambda ctx: ctx["daily_perfect"],
    "fast_signer": lambda ctx: ctx["fast_signs"] >= 5,
}


def build_achievement_context(db: Session, user: User, state) -> dict:
    progress = user.progress
    fast_signs = (
        db.query(PracticeSession)
        .filter(
            PracticeSession.user_id == user.id,
            PracticeSession.correct.is_(True),
            PracticeSession.response_time.isnot(None),
            PracticeSession.response_time <= 5000,
        )
        .count()
    )
    return {
        "total_correct": progress.total_correct if progress else state.totalCorrect,
        "current_streak": progress.current_streak if progress else state.streak,
        "letters_learned": letters_learned_count(state),
        "letters_mastered": letters_mastered_count(state),
        "speed_completed": state.challenges.speedCompleted,
        "daily_perfect": state.dailyChallenge.perfectBonusAwarded,
        "fast_signs": fast_signs,
    }


def evaluate_and_persist_badges(db: Session, user: User, state) -> list[str]:
    achievements = {item.slug: item for item in db.query(Achievement).all()}
    existing = {
        link.achievement.slug
        for link in db.query(UserAchievement).join(Achievement).filter(UserAchievement.user_id == user.id).all()
    }
    context = build_achievement_context(db, user, state)
    newly_unlocked: list[str] = []
    for slug, rule in ACHIEVEMENT_RULES.items():
        if slug in existing:
            continue
        if slug not in achievements:
            continue
        if rule(context):
            db.add(UserAchievement(user_id=user.id, achievement_id=achievements[slug].id))
            newly_unlocked.append(slug)
    if newly_unlocked:
        db.commit()
    return newly_unlocked


def get_recommendations(db: Session, user: User) -> dict:
    overview = get_overview(db, user, "all")
    if not overview["has_data"]:
        word_payload = get_word_recommendations(db, user)
        return {
            "has_data": False,
            "recommended": [],
            "weakest": None,
            "strongest": None,
            "message": "Start your first practice session to receive personalized recommendations.",
            "words": word_payload,
        }

    letter_rows = db.query(LetterProgress).filter(LetterProgress.user_id == user.id).all()
    stats = {row.letter: row for row in letter_rows}
    candidates: list[tuple[int, str, str]] = []

    for letter in LETTERS:
        row = stats.get(letter)
        correct = row.correct_attempts if row else 0
        attempts = row.attempts if row else 0
        if attempts == 0:
            candidates.append((0, letter, "never_practiced"))
            continue
        mastery = min(100.0, (correct / MASTERED_CORRECT) * 100.0)
        if mastery < 100:
            accuracy = (correct / attempts) * 100 if attempts else 0
            if accuracy < 75:
                candidates.append((2, letter, "low_accuracy"))
            elif mastery < 50:
                candidates.append((1, letter, "low_mastery"))
            elif row and row.last_practiced:
                days_since = (datetime.now(row.last_practiced.tzinfo) - row.last_practiced).days
                if days_since >= 7:
                    candidates.append((3, letter, "stale"))

    candidates.sort(key=lambda item: (item[0], item[1]))
    recommended = [{"letter": letter, "reason": reason} for _, letter, reason in candidates[:5]]
    insights = get_letter_insights(db, user.id)
    weakest = insights["weakest"][0] if insights["weakest"] else None
    strongest = insights["strongest"][0] if insights["strongest"] else None
    letter_payload = {
        "has_data": True,
        "recommended": recommended,
        "weakest": weakest,
        "strongest": strongest,
        "message": None,
    }
    word_payload = get_word_recommendations(db, user)
    return {
        **letter_payload,
        "words": word_payload,
    }


def build_profile(db: Session, user: User) -> dict:
    overview = get_overview(db, user, "all")
    insights = get_letter_insights(db, user.id)
    progress = user.progress
    most_practiced = insights["most_practiced"][0]["letter"] if insights["most_practiced"] else None
    practice_dates = {
        row[0].date()
        for row in db.query(PracticeSession.created_at).filter(PracticeSession.user_id == user.id).all()
        if row[0] is not None
    }
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at.isoformat(),
        "level": progress.level if progress else 1,
        "xp": progress.xp if progress else 0,
        "current_streak": progress.current_streak if progress else 0,
        "best_streak": progress.best_streak if progress else 0,
        "letters_mastered": overview["letters_mastered"],
        "total_practice_sessions": progress.sessions if progress else 0,
        "overall_accuracy": overview["overall_accuracy"],
        "total_signs_practiced": overview["total_attempts"],
        "total_days_active": len(practice_dates),
        "most_practiced_letter": most_practiced,
        "strongest_letter": insights["strongest"][0]["letter"] if insights["strongest"] else None,
        "weakest_letter": insights["weakest"][0]["letter"] if insights["weakest"] else None,
        "avatar_initial": user.username[:1].upper(),
    }


def export_user_data(db: Session, user: User) -> dict:
    state = serialize_game_state(db, user)
    history = (
        db.query(PracticeSession)
        .filter(PracticeSession.user_id == user.id)
        .order_by(PracticeSession.created_at.desc())
        .limit(5000)
        .all()
    )
    xp_events = (
        db.query(XpEvent).filter(XpEvent.user_id == user.id).order_by(XpEvent.created_at.desc()).limit(5000).all()
    )
    achievements = (
        db.query(UserAchievement)
        .join(Achievement)
        .filter(UserAchievement.user_id == user.id)
        .order_by(UserAchievement.earned_at.desc())
        .all()
    )
    return {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "profile": build_profile(db, user),
        "progress": state.model_dump(),
        "practice_history": [
            {
                "letter": row.letter,
                "prediction": row.prediction,
                "correct": row.correct,
                "response_time_ms": row.response_time,
                "xp_earned": row.xp_earned,
                "challenge_type": row.challenge_type,
                "created_at": row.created_at.isoformat(),
            }
            for row in history
        ],
        "xp_events": [
            {"amount": row.amount, "reason": row.reason, "created_at": row.created_at.isoformat()} for row in xp_events
        ],
        "achievements": [
            {
                "slug": row.achievement.slug,
                "name": row.achievement.name,
                "earned_at": row.earned_at.isoformat(),
            }
            for row in achievements
        ],
        **export_word_data(db, user),
    }

