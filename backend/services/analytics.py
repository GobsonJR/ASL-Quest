from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Literal

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from backend.models import LetterProgress, PracticeSession, User, UserProgress, XpEvent
from backend.services.progress import MASTERED_CORRECT, ensure_user_progress, serialize_game_state
from backend.services.words import get_word_analytics

LETTERS = [chr(ord("A") + index) for index in range(26)]
RangeKey = Literal["7d", "30d", "90d", "12m", "all"]
HeatmapRange = Literal["3m", "6m", "12m"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_range(range_key: RangeKey) -> datetime | None:
    now = utcnow()
    mapping = {
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "90d": timedelta(days=90),
        "12m": timedelta(days=365),
    }
    if range_key == "all":
        return None
    return now - mapping[range_key]


def parse_heatmap_range(range_key: HeatmapRange) -> datetime:
    now = utcnow()
    mapping = {
        "3m": timedelta(days=90),
        "6m": timedelta(days=182),
        "12m": timedelta(days=365),
    }
    return now - mapping[range_key]


def analytics_letter_status(correct: int, attempts: int) -> str:
    if attempts == 0 and correct == 0:
        return "NEW"
    if correct >= MASTERED_CORRECT:
        return "MASTERED"
    if correct >= 5:
        return "PROFICIENT"
    if correct >= 3:
        return "PRACTICED"
    if correct > 0:
        return "LEARNING"
    return "NEW"


def _practice_query(db: Session, user_id: int, start: datetime | None = None):
    query = db.query(PracticeSession).filter(PracticeSession.user_id == user_id)
    if start is not None:
        query = query.filter(PracticeSession.created_at >= start)
    return query


def _accuracy(correct: int, total: int) -> float | None:
    if total == 0:
        return None
    return round((correct / total) * 100, 1)


def get_overview(db: Session, user: User, range_key: RangeKey = "all") -> dict:
    progress = ensure_user_progress(db, user)
    start = parse_range(range_key)
    attempts_q = _practice_query(db, user.id, start)
    total_attempts = attempts_q.count()
    total_correct = attempts_q.filter(PracticeSession.correct.is_(True)).count()
    letters = db.query(LetterProgress).filter(LetterProgress.user_id == user.id).all()
    letters_practiced = sum(1 for row in letters if row.attempts > 0)
    letters_mastered = sum(1 for row in letters if row.correct_attempts >= MASTERED_CORRECT)

    avg_response = (
        db.query(func.avg(PracticeSession.response_time))
        .filter(
            PracticeSession.user_id == user.id,
            PracticeSession.response_time.isnot(None),
            *( [PracticeSession.created_at >= start] if start else [] ),
        )
        .scalar()
    )

    return {
        "total_practice_sessions": progress.sessions,
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "overall_accuracy": _accuracy(total_correct, total_attempts),
        "letters_practiced": letters_practiced,
        "letters_mastered": letters_mastered,
        "current_streak": progress.current_streak,
        "best_streak": progress.best_streak,
        "total_xp": progress.xp,
        "average_response_ms": round(float(avg_response), 0) if avg_response is not None else None,
        "average_response_sec": round(float(avg_response) / 1000, 1) if avg_response is not None else None,
        "has_data": total_attempts > 0,
    }


def get_daily_activity(db: Session, user_id: int, start: datetime | None = None) -> list[dict]:
    day = func.date(PracticeSession.created_at)
    query = (
        db.query(
            day.label("day"),
            func.count(PracticeSession.id).label("attempts"),
            func.sum(case((PracticeSession.correct.is_(True), 1), else_=0)).label("correct"),
            func.sum(PracticeSession.xp_earned).label("xp"),
        )
        .filter(PracticeSession.user_id == user_id)
        .group_by(day)
        .order_by(day.asc())
    )
    if start is not None:
        query = query.filter(PracticeSession.created_at >= start)

    rows = []
    for row in query.all():
        attempts = int(row.attempts or 0)
        correct = int(row.correct or 0)
        rows.append(
            {
                "date": str(row.day),
                "attempts": attempts,
                "correct": correct,
                "accuracy": _accuracy(correct, attempts),
                "xp": int(row.xp or 0),
            }
        )
    return rows


def heatmap_intensity(attempts: int) -> int:
    if attempts <= 0:
        return 0
    if attempts <= 4:
        return 1
    if attempts <= 9:
        return 2
    if attempts <= 19:
        return 3
    return 4


def get_heatmap(db: Session, user_id: int, range_key: HeatmapRange = "12m") -> dict:
    start = parse_heatmap_range(range_key)
    daily = {item["date"]: item for item in get_daily_activity(db, user_id, start)}
    start_date = start.date()
    end_date = utcnow().date()
    cells = []
    current = start_date
    while current <= end_date:
        key = current.isoformat()
        day = daily.get(key, {"attempts": 0, "correct": 0, "accuracy": None, "xp": 0})
        cells.append(
            {
                "date": key,
                "attempts": day["attempts"],
                "correct": day.get("correct", 0),
                "accuracy": day.get("accuracy"),
                "xp": day.get("xp", 0),
                "intensity": heatmap_intensity(day["attempts"]),
            }
        )
        current += timedelta(days=1)
    return {"range": range_key, "cells": cells}


def get_activity_trend(db: Session, user_id: int, range_key: RangeKey) -> dict:
    start = parse_range(range_key)
    daily = get_daily_activity(db, user_id, start)
    return {
        "range": range_key,
        "points": [{"date": item["date"], "value": item["attempts"]} for item in daily],
        "has_data": len(daily) > 0,
    }


def get_accuracy_trend(db: Session, user_id: int, range_key: RangeKey) -> dict:
    start = parse_range(range_key)
    daily = get_daily_activity(db, user_id, start)
    points = [
        {"date": item["date"], "value": item["accuracy"]}
        for item in daily
        if item["accuracy"] is not None
    ]

    current_accuracy = None
    if daily:
        total_attempts = sum(item["attempts"] for item in daily)
        total_correct = sum(item["correct"] for item in daily)
        current_accuracy = _accuracy(total_correct, total_attempts)

    previous_accuracy = None
    if start is not None:
        previous_start = start - (utcnow() - start)
        previous_daily = get_daily_activity(db, user_id, previous_start)
        previous_daily = [item for item in previous_daily if item["date"] < start.date().isoformat()]
        if previous_daily:
            total_attempts = sum(item["attempts"] for item in previous_daily)
            total_correct = sum(item["correct"] for item in previous_daily)
            previous_accuracy = _accuracy(total_correct, total_attempts)

    delta = None
    if current_accuracy is not None and previous_accuracy is not None:
        delta = round(current_accuracy - previous_accuracy, 1)

    return {
        "range": range_key,
        "points": points,
        "current_accuracy": current_accuracy,
        "previous_accuracy": previous_accuracy,
        "delta": delta,
        "has_data": len(points) > 0,
    }


def get_letter_analytics(db: Session, user_id: int, range_key: RangeKey = "all") -> list[dict]:
    start = parse_range(range_key)
    rows = []
    for letter in LETTERS:
        session_query = db.query(PracticeSession).filter(
            PracticeSession.user_id == user_id,
            PracticeSession.letter == letter,
        )
        if start is not None:
            session_query = session_query.filter(PracticeSession.created_at >= start)
        session_attempts = session_query.count()
        session_correct = session_query.filter(PracticeSession.correct.is_(True)).count()

        letter_row = (
            db.query(LetterProgress)
            .filter(LetterProgress.user_id == user_id, LetterProgress.letter == letter)
            .first()
        )

        if start is None and letter_row and letter_row.attempts > 0:
            attempts = letter_row.attempts
            correct = letter_row.correct_attempts
            mastery_correct = letter_row.correct_attempts
            mastery_attempts = letter_row.attempts
        else:
            attempts = session_attempts
            correct = session_correct
            mastery_correct = letter_row.correct_attempts if letter_row else session_correct
            mastery_attempts = letter_row.attempts if letter_row and letter_row.attempts > 0 else session_attempts

        rows.append(
            {
                "letter": letter,
                "attempts": attempts,
                "correct": correct,
                "accuracy": _accuracy(correct, attempts),
                "mastery_percent": min(100.0, round((mastery_correct / MASTERED_CORRECT) * 100, 1)),
                "status": analytics_letter_status(mastery_correct, mastery_attempts),
                "practiced": attempts > 0,
            }
        )
    return rows


def get_letter_insights(db: Session, user_id: int) -> dict:
    letters = get_letter_analytics(db, user_id, "all")
    practiced = [item for item in letters if item["practiced"] and item["accuracy"] is not None]
    strongest = sorted(practiced, key=lambda item: item["accuracy"] or 0, reverse=True)[:3]
    weakest = sorted(practiced, key=lambda item: item["accuracy"] or 0)[:3]
    most_practiced = sorted(practiced, key=lambda item: item["attempts"], reverse=True)[:3]
    least_practiced = sorted(practiced, key=lambda item: item["attempts"])[:3]
    return {
        "strongest": strongest,
        "weakest": weakest,
        "most_practiced": most_practiced,
        "least_practiced": least_practiced,
    }


def get_response_time_analytics(db: Session, user_id: int, range_key: RangeKey = "all") -> dict:
    start = parse_range(range_key)
    query = db.query(PracticeSession).filter(
        PracticeSession.user_id == user_id,
        PracticeSession.response_time.isnot(None),
        PracticeSession.correct.is_(True),
    )
    if start is not None:
        query = query.filter(PracticeSession.created_at >= start)

    avg_ms = query.with_entities(func.avg(PracticeSession.response_time)).scalar()
    min_ms = query.with_entities(func.min(PracticeSession.response_time)).scalar()
    count = query.count()

    weekly = []
    if count > 0:
        day = func.date(PracticeSession.created_at)
        week_rows = (
            db.query(
                day.label("day"),
                func.avg(PracticeSession.response_time).label("avg_ms"),
            )
            .filter(
                PracticeSession.user_id == user_id,
                PracticeSession.response_time.isnot(None),
                PracticeSession.correct.is_(True),
                *( [PracticeSession.created_at >= start] if start else [] ),
            )
            .group_by(day)
            .order_by(day.asc())
            .all()
        )
        weekly = [
            {"date": str(row.day), "seconds": round(float(row.avg_ms) / 1000, 1)}
            for row in week_rows
        ]

    return {
        "average_ms": round(float(avg_ms), 0) if avg_ms is not None else None,
        "average_sec": round(float(avg_ms) / 1000, 1) if avg_ms is not None else None,
        "fastest_ms": int(min_ms) if min_ms is not None else None,
        "fastest_sec": round(float(min_ms) / 1000, 1) if min_ms is not None else None,
        "trend": weekly,
        "has_data": count > 0,
    }


def get_practice_history(
    db: Session,
    user_id: int,
    *,
    page: int = 1,
    page_size: int = 20,
    letter: str | None = None,
    result: str | None = None,
) -> dict:
    query = db.query(PracticeSession).filter(PracticeSession.user_id == user_id)
    if letter:
        query = query.filter(PracticeSession.letter == letter.upper())
    if result == "correct":
        query = query.filter(PracticeSession.correct.is_(True))
    elif result == "incorrect":
        query = query.filter(PracticeSession.correct.is_(False))

    total = query.count()
    rows = (
        query.order_by(PracticeSession.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": [
            {
                "id": row.id,
                "date": row.created_at.isoformat(),
                "letter": row.letter,
                "prediction": row.prediction,
                "correct": row.correct,
                "response_time_ms": row.response_time,
                "response_time_sec": round(row.response_time / 1000, 1) if row.response_time else None,
                "xp_earned": row.xp_earned,
            }
            for row in rows
        ],
    }


def get_xp_analytics(db: Session, user: User, range_key: RangeKey = "all") -> dict:
    progress = ensure_user_progress(db, user)
    start = parse_range(range_key)
    now = utcnow()
    week_start = now - timedelta(days=7)
    month_start = now - timedelta(days=30)

    def sum_xp(since: datetime | None) -> int:
        query = db.query(func.sum(XpEvent.amount)).filter(XpEvent.user_id == user.id)
        if since is not None:
            query = query.filter(XpEvent.created_at >= since)
        value = query.scalar()
        return int(value or 0)

    day = func.date(XpEvent.created_at)
    trend_query = (
        db.query(day.label("day"), func.sum(XpEvent.amount).label("xp"))
        .filter(XpEvent.user_id == user.id)
        .group_by(day)
        .order_by(day.asc())
    )
    if start is not None:
        trend_query = trend_query.filter(XpEvent.created_at >= start)

    trend = [{"date": str(row.day), "value": int(row.xp or 0)} for row in trend_query.all()]

    return {
        "total_xp": progress.xp,
        "xp_this_week": sum_xp(week_start),
        "xp_this_month": sum_xp(month_start),
        "trend": trend,
        "has_data": len(trend) > 0 or progress.xp > 0,
    }


def get_streak_analytics(db: Session, user: User) -> dict:
    progress = ensure_user_progress(db, user)
    active_days = (
        db.query(func.count(func.distinct(func.date(PracticeSession.created_at))))
        .filter(PracticeSession.user_id == user.id)
        .scalar()
    )
    daily = get_daily_activity(db, user.id, utcnow() - timedelta(days=365))
    return {
        "current_streak": progress.current_streak,
        "best_streak": progress.best_streak,
        "active_days": int(active_days or 0),
        "daily_activity": [{"date": item["date"], "attempts": item["attempts"]} for item in daily],
    }


def get_learning_funnel(db: Session, user_id: int) -> dict:
    letters = db.query(LetterProgress).filter(LetterProgress.user_id == user_id).all()
    stats = {row.letter: row for row in letters}
    counts = {"NEW": 0, "LEARNING": 0, "PRACTICED": 0, "PROFICIENT": 0, "MASTERED": 0}
    for letter in LETTERS:
        row = stats.get(letter)
        correct = row.correct_attempts if row else 0
        attempts = row.attempts if row else 0
        status = analytics_letter_status(correct, attempts)
        counts[status] += 1
    practiced = 26 - counts["NEW"]
    return {
        "total_letters": 26,
        "practiced": practiced,
        "learning": counts["LEARNING"],
        "practiced_stage": counts["PRACTICED"],
        "proficient": counts["PROFICIENT"],
        "mastered": counts["MASTERED"],
        "counts": counts,
    }


def get_weekly_summary(db: Session, user_id: int) -> dict:
    now = utcnow()
    week_start = now - timedelta(days=7)
    prev_start = now - timedelta(days=14)

    def week_metrics(start: datetime, end: datetime) -> dict:
        query = _practice_query(db, user_id, start).filter(PracticeSession.created_at < end)
        attempts = query.count()
        correct = query.filter(PracticeSession.correct.is_(True)).count()
        xp = (
            db.query(func.sum(XpEvent.amount))
            .filter(XpEvent.user_id == user_id, XpEvent.created_at >= start, XpEvent.created_at < end)
            .scalar()
        )
        avg_response = (
            query.filter(PracticeSession.response_time.isnot(None), PracticeSession.correct.is_(True))
            .with_entities(func.avg(PracticeSession.response_time))
            .scalar()
        )
        letters = (
            db.query(func.count(func.distinct(PracticeSession.letter)))
            .filter(PracticeSession.user_id == user_id, PracticeSession.created_at >= start, PracticeSession.created_at < end)
            .scalar()
        )
        return {
            "attempts": attempts,
            "accuracy": _accuracy(correct, attempts),
            "xp": int(xp or 0),
            "letters_practiced": int(letters or 0),
            "average_response_sec": round(float(avg_response) / 1000, 1) if avg_response is not None else None,
        }

    current = week_metrics(week_start, now)
    previous = week_metrics(prev_start, week_start)

    def delta(current_value: float | int | None, previous_value: float | int | None) -> float | None:
        if current_value is None or previous_value is None:
            return None
        if previous_value == 0:
            return None
        if isinstance(current_value, float) or isinstance(previous_value, float):
            return round(float(current_value) - float(previous_value), 1)
        return round(((float(current_value) - float(previous_value)) / float(previous_value)) * 100, 1)

    return {
        "current_week": current,
        "previous_week": previous,
        "delta": {
            "attempts_pct": delta(current["attempts"], previous["attempts"]),
            "accuracy_points": delta(current["accuracy"], previous["accuracy"]),
            "xp_pct": delta(current["xp"], previous["xp"]),
        },
        "has_comparison": previous["attempts"] > 0,
    }


def get_letter_detail(db: Session, user_id: int, letter: str, range_key: RangeKey = "all") -> dict:
    letter = letter.upper()
    start = parse_range(range_key)
    query = db.query(PracticeSession).filter(PracticeSession.user_id == user_id, PracticeSession.letter == letter)
    if start is not None:
        query = query.filter(PracticeSession.created_at >= start)
    attempts = query.count()
    correct = query.filter(PracticeSession.correct.is_(True)).count()
    first = query.order_by(PracticeSession.created_at.asc()).first()
    last = query.order_by(PracticeSession.created_at.desc()).first()
    avg_response = query.filter(PracticeSession.response_time.isnot(None)).with_entities(
        func.avg(PracticeSession.response_time)
    ).scalar()
    letter_row = (
        db.query(LetterProgress)
        .filter(LetterProgress.user_id == user_id, LetterProgress.letter == letter)
        .first()
    )
    mastery_correct = letter_row.correct_attempts if letter_row else 0
    trend_start = parse_range("30d")
    day = func.date(PracticeSession.created_at)
    trend_query = (
        db.query(
            day.label("day"),
            func.count(PracticeSession.id).label("attempts"),
            func.sum(case((PracticeSession.correct.is_(True), 1), else_=0)).label("correct"),
        )
        .filter(PracticeSession.user_id == user_id, PracticeSession.letter == letter)
        .group_by(day)
        .order_by(day.asc())
    )
    if trend_start is not None:
        trend_query = trend_query.filter(PracticeSession.created_at >= trend_start)
    letter_trend = [
        {
            "date": str(row.day),
            "value": _accuracy(int(row.correct or 0), int(row.attempts or 0)),
        }
        for row in trend_query.all()
        if int(row.attempts or 0) > 0
    ]
    history = get_practice_history(db, user_id, page=1, page_size=10, letter=letter)
    return {
        "letter": letter,
        "mastery_percent": min(100.0, round((mastery_correct / MASTERED_CORRECT) * 100, 1)),
        "status": analytics_letter_status(mastery_correct, letter_row.attempts if letter_row else 0),
        "accuracy": _accuracy(correct, attempts),
        "attempts": attempts,
        "correct": correct,
        "average_response_sec": round(float(avg_response) / 1000, 1) if avg_response is not None else None,
        "first_practiced": first.created_at.isoformat() if first else None,
        "last_practiced": last.created_at.isoformat() if last else None,
        "accuracy_trend": letter_trend,
        "history": history["items"],
        "practiced": attempts > 0 or (letter_row.attempts if letter_row else 0) > 0,
    }


def get_dashboard(db: Session, user: User, range_key: RangeKey = "30d", heatmap_range: HeatmapRange = "12m") -> dict:
    return {
        "overview": get_overview(db, user, range_key),
        "heatmap": get_heatmap(db, user.id, heatmap_range),
        "activity_trend": get_activity_trend(db, user.id, range_key),
        "accuracy_trend": get_accuracy_trend(db, user.id, range_key),
        "letters": get_letter_analytics(db, user.id, range_key),
        "insights": get_letter_insights(db, user.id),
        "response_time": get_response_time_analytics(db, user.id, range_key),
        "xp": get_xp_analytics(db, user, range_key),
        "streak": get_streak_analytics(db, user),
        "funnel": get_learning_funnel(db, user.id),
        "weekly_summary": get_weekly_summary(db, user.id),
        "progress": serialize_game_state(db, user),
        "words": get_word_analytics(db, user),
    }
