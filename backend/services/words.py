from __future__ import annotations

import hashlib
import json
import random
from datetime import date, datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import Achievement, User, UserAchievement, WordPracticeSession, WordProgress, XpEvent
from backend.services.progress import ensure_user_progress, level_from_xp
from backend.words.catalog import WORD_BY_ID, WORD_CATALOG, WORD_CATEGORIES, all_word_ids, get_word, list_words

WORD_MASTERED_COMPLETIONS = 3
XP_WORD_COMPLETED = 50
XP_WORD_PERFECT = 25
XP_WORD_FAST = 20
XP_WORD_CHALLENGE = 150
FAST_MS_PER_LETTER = 4000
WORD_CHALLENGE_SIZE = 5
DAILY_WORD_COUNT = 3

WORD_ACHIEVEMENT_SEED = [
    ("word_starter", "Word Starter", "Complete your first word.", "word_completions >= 1"),
    ("word_learner", "Word Learner", "Complete 5 different words.", "distinct_words_completed >= 5"),
    ("word_explorer", "Word Explorer", "Complete words from 3 categories.", "word_categories_completed >= 3"),
    ("word_master", "Word Master", "Master 10 words.", "words_mastered >= 10"),
    ("vocabulary_builder", "Vocabulary Builder", "Complete 25 words.", "word_completions >= 25"),
]


def word_mastery_percent(completions: int) -> float:
    return min(100.0, round((completions / WORD_MASTERED_COMPLETIONS) * 100.0, 1))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _accuracy(correct: int | None, total: int | None) -> float | None:
    correct_n = int(correct or 0)
    total_n = int(total or 0)
    if total_n <= 0:
        return None
    return round((correct_n / total_n) * 100.0, 1)


def serialize_word(item: dict, progress: WordProgress | None = None) -> dict:
    payload = {
        "id": item["id"],
        "word": item["word"],
        "category": item["category"],
        "difficulty": item["difficulty"],
        "description": item["description"],
        "letters": item["letters"],
        "letter_count": item["letter_count"],
        "estimated_xp": item["estimated_xp"],
        "tip": item.get("tip"),
        "progress": None,
    }
    if progress is not None:
        payload["progress"] = serialize_progress(progress, item)
    return payload


def serialize_progress(row: WordProgress, item: dict | None = None) -> dict:
    catalog = item or get_word(row.word_id) or {"word": row.word_id, "letters": [], "category": "", "difficulty": ""}
    letter_stats = json.loads(row.letter_stats or "{}")
    breakdown = []
    for letter in catalog.get("letters") or []:
        stats = letter_stats.get(letter, {"correct": 0, "attempts": 0})
        attempts = int(stats.get("attempts", 0))
        correct = int(stats.get("correct", 0))
        breakdown.append(
            {
                "letter": letter,
                "correct": correct,
                "attempts": attempts,
                "accuracy": _accuracy(correct, attempts),
                "practiced": attempts > 0,
            }
        )
    return {
        "word_id": row.word_id,
        "word": catalog.get("word", row.word_id),
        "category": catalog.get("category"),
        "difficulty": catalog.get("difficulty"),
        "attempts": row.attempts or 0,
        "completions": row.completions or 0,
        "correct_letters": row.correct_letters or 0,
        "total_letters": row.total_letters or 0,
        "accuracy": _accuracy(row.correct_letters, row.total_letters),
        "mastery": float(row.mastery or 0),
        "mastered": (row.completions or 0) >= WORD_MASTERED_COMPLETIONS,
        "best_time_ms": row.best_time_ms,
        "best_time_sec": round(row.best_time_ms / 1000, 1) if row.best_time_ms is not None else None,
        "total_time_ms": row.total_time_ms,
        "resume_index": row.resume_index,
        "last_practiced_at": row.last_practiced_at.isoformat() if row.last_practiced_at else None,
        "letter_breakdown": breakdown,
    }


def ensure_word_progress(db: Session, user_id: int, word_id: str) -> WordProgress:
    row = (
        db.query(WordProgress)
        .filter(WordProgress.user_id == user_id, WordProgress.word_id == word_id)
        .first()
    )
    if row is None:
        row = WordProgress(user_id=user_id, word_id=word_id, letter_stats="{}")
        db.add(row)
        db.flush()
    return row


def list_catalog_with_progress(db: Session, user: User, category: str | None = None) -> list[dict]:
    rows = {row.word_id: row for row in db.query(WordProgress).filter(WordProgress.user_id == user.id).all()}
    return [serialize_word(item, rows.get(item["id"])) for item in list_words(category)]


def get_all_progress(db: Session, user: User) -> list[dict]:
    rows = db.query(WordProgress).filter(WordProgress.user_id == user.id).all()
    return [serialize_progress(row) for row in rows]


def get_word_progress_detail(db: Session, user: User, word_id: str) -> dict | None:
    item = get_word(word_id)
    if item is None:
        return None
    row = (
        db.query(WordProgress)
        .filter(WordProgress.user_id == user.id, WordProgress.word_id == word_id)
        .first()
    )
    payload = serialize_word(item, row)
    if row is None:
        empty = {
            "word_id": word_id,
            "word": item["word"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "attempts": 0,
            "completions": 0,
            "correct_letters": 0,
            "total_letters": 0,
            "accuracy": None,
            "mastery": 0,
            "mastered": False,
            "best_time_ms": None,
            "best_time_sec": None,
            "total_time_ms": 0,
            "resume_index": 0,
            "last_practiced_at": None,
            "letter_breakdown": [
                {"letter": letter, "correct": 0, "attempts": 0, "accuracy": None, "practiced": False}
                for letter in item["letters"]
            ],
        }
        payload = serialize_word(item, None)
        payload["progress"] = empty
        return payload
    payload = serialize_word(item, row)
    return payload


def _merge_letter_stats(existing: dict, letters: list[dict]) -> dict:
    next_stats = dict(existing)
    for entry in letters:
        letter = str(entry.get("letter", "")).upper()
        if not letter:
            continue
        stats = next_stats.get(letter, {"correct": 0, "attempts": 0})
        attempts = stats.get("attempts", 0) + 1
        correct = stats.get("correct", 0) + (1 if entry.get("correct") else 0)
        next_stats[letter] = {"correct": correct, "attempts": attempts}
    return next_stats


def record_word_session(db: Session, user: User, payload) -> dict:
    item = get_word(payload.word_id)
    if item is None:
        raise ValueError("Unknown word.")

    letters = [entry.model_dump() if hasattr(entry, "model_dump") else dict(entry) for entry in (payload.letters or [])]
    correct_count = sum(1 for entry in letters if entry.get("correct"))
    total_count = len(letters)
    mistakes = payload.mistakes if payload.mistakes is not None else sum(1 for entry in letters if not entry.get("correct"))
    duration_ms = payload.duration_ms
    completed = bool(payload.completed)
    challenge_type = payload.challenge_type

    completion_xp = 0
    xp_reasons: list[dict] = []
    if completed:
        completion_xp += XP_WORD_COMPLETED
        xp_reasons.append({"reason": "word_completed", "amount": XP_WORD_COMPLETED})
        if mistakes == 0:
            completion_xp += XP_WORD_PERFECT
            xp_reasons.append({"reason": "word_perfect", "amount": XP_WORD_PERFECT})
        if duration_ms is not None and duration_ms <= FAST_MS_PER_LETTER * len(item["letters"]):
            completion_xp += XP_WORD_FAST
            xp_reasons.append({"reason": "word_fast", "amount": XP_WORD_FAST})
    if payload.challenge_finished:
        completion_xp += XP_WORD_CHALLENGE
        xp_reasons.append({"reason": "word_challenge_completed", "amount": XP_WORD_CHALLENGE})

    row = ensure_word_progress(db, user.id, item["id"])
    row.attempts += 1
    row.correct_letters += correct_count
    row.total_letters += total_count
    row.total_time_ms += int(duration_ms or 0)
    row.last_practiced_at = _utcnow()
    row.updated_at = _utcnow()
    row.letter_stats = json.dumps(_merge_letter_stats(json.loads(row.letter_stats or "{}"), letters))
    if completed:
        row.completions += 1
        row.resume_index = 0
        if duration_ms is not None and (row.best_time_ms is None or duration_ms < row.best_time_ms):
            row.best_time_ms = duration_ms
    else:
        row.resume_index = max(0, min(payload.resume_index or 0, len(item["letters"])))
    row.mastery = word_mastery_percent(row.completions)

    session = WordPracticeSession(
        user_id=user.id,
        word_id=item["id"],
        accuracy=_accuracy(correct_count, total_count),
        duration_ms=duration_ms,
        completed=completed,
        xp_earned=completion_xp,
        challenge_type=challenge_type,
        mistakes=mistakes,
        letter_results=json.dumps(letters),
    )
    db.add(session)

    progress = ensure_user_progress(db, user)
    if completion_xp > 0:
        progress.xp += completion_xp
        progress.level = level_from_xp(progress.xp)
        progress.updated_at = _utcnow()
        for event in xp_reasons:
            db.add(XpEvent(user_id=user.id, amount=event["amount"], reason=event["reason"]))

    db.commit()
    db.refresh(session)
    db.refresh(row)

    new_achievements = evaluate_word_achievements(db, user)

    return {
        "id": session.id,
        "word_id": session.word_id,
        "word": item["word"],
        "completed": session.completed,
        "accuracy": session.accuracy,
        "duration_ms": session.duration_ms,
        "xp_earned": session.xp_earned,
        "xp_events": xp_reasons,
        "challenge_type": session.challenge_type,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "progress": serialize_progress(row, item),
        "new_achievements": new_achievements,
    }


def word_achievement_context(db: Session, user: User) -> dict:
    rows = db.query(WordProgress).filter(WordProgress.user_id == user.id).all()
    completed_rows = [row for row in rows if row.completions > 0]
    categories = {
        (get_word(row.word_id) or {}).get("category")
        for row in completed_rows
        if (get_word(row.word_id) or {}).get("category")
    }
    return {
        "word_completions": sum(row.completions for row in rows),
        "distinct_words_completed": len(completed_rows),
        "word_categories_completed": len(categories),
        "words_mastered": sum(1 for row in rows if row.completions >= WORD_MASTERED_COMPLETIONS),
    }


def evaluate_word_achievements(db: Session, user: User) -> list[dict]:
    from backend.services.progress import seed_achievements

    seed_achievements(db)
    achievements = {item.slug: item for item in db.query(Achievement).all()}
    existing = {
        link.achievement.slug
        for link in db.query(UserAchievement).join(Achievement).filter(UserAchievement.user_id == user.id).all()
    }
    ctx = word_achievement_context(db, user)
    rules = {
        "word_starter": ctx["word_completions"] >= 1,
        "word_learner": ctx["distinct_words_completed"] >= 5,
        "word_explorer": ctx["word_categories_completed"] >= 3,
        "word_master": ctx["words_mastered"] >= 10,
        "vocabulary_builder": ctx["word_completions"] >= 25,
    }
    newly: list[dict] = []
    for slug, passed in rules.items():
        if not passed or slug in existing or slug not in achievements:
            continue
        db.add(UserAchievement(user_id=user.id, achievement_id=achievements[slug].id))
        newly.append({"id": slug, "name": achievements[slug].name})
    if newly:
        db.commit()
    return newly


def get_word_recommendations(db: Session, user: User) -> dict:
    rows = {row.word_id: row for row in db.query(WordProgress).filter(WordProgress.user_id == user.id).all()}
    letter_accuracy: dict[str, float] = {}
    from backend.models import LetterProgress

    for letter_row in db.query(LetterProgress).filter(LetterProgress.user_id == user.id).all():
        if letter_row.attempts > 0:
            letter_accuracy[letter_row.letter] = (letter_row.correct_attempts / letter_row.attempts) * 100

    candidates: list[tuple[int, str, str, str]] = []
    for item in WORD_CATALOG:
        row = rows.get(item["id"])
        if row is None or row.attempts == 0:
            candidates.append((0, item["id"], "never_practiced", f"Try spelling {item['word']} next."))
            continue
        if row.completions >= WORD_MASTERED_COMPLETIONS:
            related = next(
                (
                    other
                    for other in WORD_CATALOG
                    if other["category"] == item["category"] and other["id"] != item["id"] and other["id"] not in rows
                ),
                None,
            )
            if related:
                candidates.append(
                    (4, related["id"], "next_in_category", f"You've mastered {item['word']}. Try {related['word']} next.")
                )
            continue
        weakest_letter = None
        weakest_acc = 101.0
        for letter in item["letters"]:
            acc = letter_accuracy.get(letter)
            if acc is not None and acc < weakest_acc:
                weakest_acc = acc
                weakest_letter = letter
        if row.attempts > 0 and _accuracy(row.correct_letters, row.total_letters) is not None:
            word_acc = _accuracy(row.correct_letters, row.total_letters) or 0
            if word_acc < 75:
                reason = f"You've practiced {item['word']} often, but accuracy is still {word_acc}%."
                if weakest_letter and weakest_acc < 75:
                    reason = (
                        f"You've practiced {item['word']} often, but your {weakest_letter} accuracy is lower. "
                        f"Practice {item['word']} again."
                    )
                candidates.append((1, item["id"], "low_accuracy", reason))
                continue
        if row.completions < WORD_MASTERED_COMPLETIONS:
            candidates.append((2, item["id"], "in_progress", f"Keep going — {item['word']} is not mastered yet."))
        if row.last_practiced_at:
            days = (_utcnow() - row.last_practiced_at).days
            if days >= 7:
                candidates.append((3, item["id"], "stale", f"It's been a while since you spelled {item['word']}."))

    candidates.sort(key=lambda item: (item[0], item[1]))
    recommended = []
    seen: set[str] = set()
    for priority, word_id, reason, message in candidates:
        if word_id in seen:
            continue
        item = get_word(word_id)
        if not item:
            continue
        seen.add(word_id)
        recommended.append(
            {
                "word_id": word_id,
                "word": item["word"],
                "category": item["category"],
                "reason": reason,
                "message": message,
            }
        )
        if len(recommended) >= 5:
            break

    continue_word = None
    recent = (
        db.query(WordProgress)
        .filter(WordProgress.user_id == user.id, WordProgress.last_practiced_at.isnot(None))
        .order_by(WordProgress.last_practiced_at.desc())
        .first()
    )
    if recent:
        continue_word = serialize_progress(recent)

    return {
        "has_data": any(row.attempts > 0 for row in rows.values()),
        "recommended": recommended,
        "continue_word": continue_word,
        "message": None if recommended else "Start a word spelling session to receive recommendations.",
    }


def daily_word_ids(user_id: int, day: str | None = None) -> list[str]:
    day_key = day or date.today().isoformat()
    digest = hashlib.sha256(f"{user_id}:{day_key}:daily-words".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    ids = all_word_ids()
    return rng.sample(ids, k=min(DAILY_WORD_COUNT, len(ids)))


def challenge_word_ids(user_id: int, seed: str | None = None) -> list[str]:
    digest = hashlib.sha256(f"{user_id}:{seed or date.today().isoformat()}:word-challenge".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    ids = all_word_ids()
    return rng.sample(ids, k=min(WORD_CHALLENGE_SIZE, len(ids)))


def words_for_ids(word_ids: list[str]) -> list[dict]:
    return [serialize_word(WORD_BY_ID[word_id]) for word_id in word_ids if word_id in WORD_BY_ID]


def get_word_analytics(db: Session, user: User) -> dict:
    rows = db.query(WordProgress).filter(WordProgress.user_id == user.id).all()
    sessions = db.query(WordPracticeSession).filter(WordPracticeSession.user_id == user.id).all()
    completed_sessions = [row for row in sessions if row.completed]
    words_learned = sum(1 for row in rows if row.completions > 0)
    words_mastered = sum(1 for row in rows if row.completions >= WORD_MASTERED_COMPLETIONS)
    total_correct = sum(row.correct_letters for row in rows)
    total_letters = sum(row.total_letters for row in rows)
    total_time = sum(row.total_time_ms for row in rows)

    practiced = [serialize_progress(row) for row in rows if row.attempts > 0]
    most_practiced = sorted(practiced, key=lambda item: item["attempts"], reverse=True)[:5]
    strongest = sorted(
        [item for item in practiced if item["accuracy"] is not None],
        key=lambda item: item["accuracy"] or 0,
        reverse=True,
    )[:5]
    needing = sorted(
        [item for item in practiced if item["accuracy"] is not None and not item["mastered"]],
        key=lambda item: item["accuracy"] or 0,
    )[:5]

    category_progress = []
    for category in WORD_CATEGORIES:
        catalog_items = [item for item in WORD_CATALOG if item["category"] == category]
        completed = 0
        for item in catalog_items:
            row = next((entry for entry in rows if entry.word_id == item["id"]), None)
            if row and row.completions > 0:
                completed += 1
        total = len(catalog_items)
        percent = round((completed / total) * 100) if total else 0
        category_progress.append(
            {
                "category": category,
                "completed": completed,
                "total": total,
                "percent": percent,
            }
        )

    return {
        "words_learned": words_learned,
        "words_mastered": words_mastered,
        "words_completed": sum(row.completions for row in rows),
        "word_sessions": len(sessions),
        "word_accuracy": _accuracy(total_correct, total_letters),
        "word_practice_time_ms": total_time,
        "word_practice_time_sec": round(total_time / 1000, 1) if total_time else 0,
        "completed_sessions": len(completed_sessions),
        "most_practiced": most_practiced,
        "strongest": strongest,
        "needing_practice": needing,
        "category_progress": category_progress,
        "has_data": len(sessions) > 0 or any(row.attempts > 0 for row in rows),
    }


def get_admin_word_stats(db: Session) -> dict:
    sessions = db.query(WordPracticeSession)
    total_sessions = sessions.count()
    completed = sessions.filter(WordPracticeSession.completed.is_(True)).count()
    avg_acc = sessions.filter(WordPracticeSession.accuracy.isnot(None)).with_entities(
        func.avg(WordPracticeSession.accuracy)
    ).scalar()

    practiced = []
    hardest = []
    categories: dict[str, int] = {name: 0 for name in WORD_CATEGORIES}
    for item in WORD_CATALOG:
        count = db.query(WordPracticeSession).filter(WordPracticeSession.word_id == item["id"]).count()
        correct_letters = (
            db.query(func.sum(WordProgress.correct_letters)).filter(WordProgress.word_id == item["id"]).scalar() or 0
        )
        total_letters = (
            db.query(func.sum(WordProgress.total_letters)).filter(WordProgress.word_id == item["id"]).scalar() or 0
        )
        accuracy = _accuracy(int(correct_letters), int(total_letters))
        practiced.append({"word": item["word"], "word_id": item["id"], "sessions": count, "accuracy": accuracy})
        if count > 0:
            categories[item["category"]] += count
            if accuracy is not None:
                hardest.append({"word": item["word"], "word_id": item["id"], "accuracy": accuracy, "sessions": count})

    practiced.sort(key=lambda item: item["sessions"], reverse=True)
    hardest.sort(key=lambda item: (item["accuracy"] if item["accuracy"] is not None else 101, -item["sessions"]))
    popular_categories = [
        {"category": name, "sessions": count}
        for name, count in sorted(categories.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "total_word_practice_sessions": total_sessions,
        "words_completed": completed,
        "average_word_accuracy": round(float(avg_acc), 1) if avg_acc is not None else None,
        "most_practiced": practiced[:8],
        "hardest": hardest[:8],
        "most_popular_categories": popular_categories,
    }


def export_word_data(db: Session, user: User) -> dict:
    progress_rows = db.query(WordProgress).filter(WordProgress.user_id == user.id).all()
    session_rows = (
        db.query(WordPracticeSession)
        .filter(WordPracticeSession.user_id == user.id)
        .order_by(WordPracticeSession.created_at.desc())
        .limit(5000)
        .all()
    )
    return {
        "word_progress": [serialize_progress(row) for row in progress_rows],
        "word_practice_sessions": [
            {
                "word_id": row.word_id,
                "accuracy": row.accuracy,
                "duration_ms": row.duration_ms,
                "completed": row.completed,
                "xp_earned": row.xp_earned,
                "challenge_type": row.challenge_type,
                "mistakes": row.mistakes,
                "letter_results": json.loads(row.letter_results or "[]"),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in session_rows
        ],
    }
