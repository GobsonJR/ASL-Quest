from __future__ import annotations

import json
from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from backend.models import Achievement, ChallengeProgress, LetterProgress, User, UserAchievement, UserProgress
from backend.schemas import GameStateSchema, LetterStatsSchema

LETTERS = [chr(ord("A") + index) for index in range(26)]
MASTERED_CORRECT = 10
DAILY_TARGET = 10

ACHIEVEMENT_SEED = [
    ("first_sign", "First Sign", "Complete your first successful sign", "total_correct >= 1"),
    ("streak_7", "7 Day Streak", "Practice for 7 consecutive days", "current_streak >= 7"),
    ("alphabet_beginner", "Alphabet Beginner", "Practice 5 unique letters", "letters_learned >= 5"),
    ("alphabet_explorer", "Alphabet Explorer", "Practice 13 unique letters", "letters_learned >= 13"),
    ("alphabet_master", "Alphabet Master", "Successfully sign all 26 letters", "letters_learned >= 26"),
    ("speed_signer", "Speed Signer", "Complete a speed challenge quickly", "speed_challenge_completed"),
    ("fast_signer", "Fast Signer", "Complete 5 fast correct signs", "fast_signs >= 5"),
    ("daily_perfect", "Perfect Practice", "Complete a daily challenge without mistakes", "daily_perfect_bonus"),
    ("word_starter", "Word Starter", "Complete your first word.", "word_completions >= 1"),
    ("word_learner", "Word Learner", "Complete 5 different words.", "distinct_words_completed >= 5"),
    ("word_explorer", "Word Explorer", "Complete words from 3 categories.", "word_categories_completed >= 3"),
    ("word_master", "Word Master", "Master 10 words.", "words_mastered >= 10"),
    ("vocabulary_builder", "Vocabulary Builder", "Complete 25 words.", "word_completions >= 25"),
    ("native_first_sign", "First Native Sign", "Correctly recognize your first native ASL sign.", "native_signs_correct >= 1"),
    ("native_explorer", "Native Sign Explorer", "Correctly sign native words from 3 categories.", "native_categories_correct >= 3"),
    ("native_master", "Native Sign Master", "Master every native sign in the catalog.", "native_signs_mastered >= native_total_signs"),
]


def seed_achievements(db: Session) -> None:
    existing = {item.slug for item in db.query(Achievement).all()}
    for slug, name, description, requirement in ACHIEVEMENT_SEED:
        if slug in existing:
            continue
        db.add(Achievement(slug=slug, name=name, description=description, requirement=requirement))
    db.commit()


def mastery_percent(correct_attempts: int) -> float:
    return min(100.0, (correct_attempts / MASTERED_CORRECT) * 100.0)


def level_from_xp(xp: int) -> int:
    return max(1, xp // 1000 + 1)


def letters_learned_count(state: GameStateSchema) -> int:
    return sum(1 for letter in LETTERS if state.letterStats[letter].correct >= 1)


def letters_mastered_count(state: GameStateSchema) -> int:
    return sum(1 for letter in LETTERS if state.letterStats[letter].correct >= MASTERED_CORRECT)


def empty_letter_stats() -> dict[str, LetterStatsSchema]:
    return {letter: LetterStatsSchema() for letter in LETTERS}


def default_game_state() -> GameStateSchema:
    today = date.today().isoformat()
    return GameStateSchema(
        letterStats=empty_letter_stats(),
        dailyChallenge={
            "date": today,
            "progress": 0,
            "target": DAILY_TARGET,
            "completed": False,
            "mistakes": 0,
            "perfectBonusAwarded": False,
        },
        challenges={
            "alphabetCompleted": False,
            "speedCompleted": False,
            "speedStartedAt": None,
            "speedCount": 0,
            "wordCompleted": False,
            "wordCount": 0,
        },
    )


def ensure_user_progress(db: Session, user: User) -> UserProgress:
    if user.progress is None:
        progress = UserProgress(user_id=user.id)
        db.add(progress)
        db.commit()
        db.refresh(user)
    return user.progress


def has_meaningful_progress(db: Session, user: User) -> bool:
    progress = ensure_user_progress(db, user)
    if progress.xp > 0 or progress.total_correct > 0 or progress.sessions > 0:
        return True
    if db.query(LetterProgress).filter(LetterProgress.user_id == user.id, LetterProgress.attempts > 0).count() > 0:
        return True
    if db.query(UserAchievement).filter(UserAchievement.user_id == user.id).count() > 0:
        return True
    return False


def local_progress_is_meaningful(state: GameStateSchema) -> bool:
    if state.xp > 0 or state.totalCorrect > 0 or state.sessions > 0:
        return True
    return any(stats.attempts > 0 or stats.correct > 0 for stats in state.letterStats.values())


def serialize_game_state(db: Session, user: User) -> GameStateSchema:
    progress = ensure_user_progress(db, user)
    game_meta = json.loads(progress.game_meta or "{}")
    letter_rows = db.query(LetterProgress).filter(LetterProgress.user_id == user.id).all()
    letter_stats = empty_letter_stats()
    for row in letter_rows:
        letter_stats[row.letter] = LetterStatsSchema(correct=row.correct_attempts, attempts=row.attempts)

    challenge_rows = {row.challenge_type: row for row in user.challenge_progress}
    daily_row = challenge_rows.get("daily")
    alphabet_row = challenge_rows.get("alphabet")
    speed_row = challenge_rows.get("speed")
    speed_meta = json.loads(speed_row.meta) if speed_row else {}

    daily_meta = json.loads(daily_row.meta) if daily_row else {}
    today = date.today().isoformat()
    daily_date = daily_meta.get("date", today)
    if daily_date != today:
        daily_challenge = {
            "date": today,
            "progress": 0,
            "target": DAILY_TARGET,
            "completed": False,
            "mistakes": 0,
            "perfectBonusAwarded": False,
        }
    else:
        daily_challenge = {
            "date": daily_date,
            "progress": daily_row.progress if daily_row else 0,
            "target": daily_meta.get("target", DAILY_TARGET),
            "completed": daily_row.completed if daily_row else False,
            "mistakes": daily_meta.get("mistakes", 0),
            "perfectBonusAwarded": daily_meta.get("perfectBonusAwarded", False),
        }

    unlocked = [
        link.achievement.slug
        for link in db.query(UserAchievement)
        .join(Achievement)
        .filter(UserAchievement.user_id == user.id)
        .order_by(UserAchievement.earned_at.desc())
        .all()
    ]
    recent = [
        {
            "id": link.achievement.slug,
            "title": link.achievement.name,
            "unlockedAt": link.earned_at.isoformat(),
        }
        for link in db.query(UserAchievement)
        .join(Achievement)
        .filter(UserAchievement.user_id == user.id)
        .order_by(UserAchievement.earned_at.desc())
        .limit(8)
        .all()
    ]

    return GameStateSchema(
        xp=progress.xp,
        streak=progress.current_streak,
        longestStreak=progress.best_streak,
        lastPracticeDate=progress.last_active_date.isoformat() if progress.last_active_date else None,
        totalCorrect=progress.total_correct,
        totalAttempts=progress.total_attempts,
        sessions=progress.sessions,
        letterStats=letter_stats,
        unlockedBadges=unlocked,
        dailyChallenge=daily_challenge,
        challenges={
            "alphabetCompleted": alphabet_row.completed if alphabet_row else False,
            "speedCompleted": speed_row.completed if speed_row else False,
            "speedStartedAt": speed_meta.get("speedStartedAt"),
            "speedCount": speed_row.progress if speed_row else 0,
            "wordCompleted": challenge_rows.get("word").completed if challenge_rows.get("word") else False,
            "wordCount": challenge_rows.get("word").progress if challenge_rows.get("word") else 0,
        },
        recentAchievements=recent,
    )


def persist_game_state(db: Session, user: User, state: GameStateSchema) -> GameStateSchema:
    progress = ensure_user_progress(db, user)
    # XP has two independent writers into this same row: this endpoint (the
    # client's full-state push, e.g. after an A-Z correct sign) and Native /
    # Word Spelling's direct, atomic server-side increments
    # (backend/routers/native.py, backend/services/words.py::record_word_session).
    # A client can still be holding a GameState snapshot from *before* one of
    # those direct increments landed (most realistically: another browser
    # tab/session on the same account), and its PUT's `xp` would then be
    # stale. XP is authoritative and monotonic server-side, so a client's
    # value only ever raises it, never lowers it -- this is what stops a
    # stale snapshot from silently erasing already-earned, already-persisted
    # XP. (No current caller intentionally lowers XP through this endpoint;
    # `resetProgress` in GameContext.tsx is unused today and would need its
    # own explicit reset path rather than going through this merge-safe one.)
    new_xp = max(progress.xp, state.xp)
    progress.xp = new_xp
    progress.level = level_from_xp(new_xp)
    progress.current_streak = state.streak
    progress.best_streak = state.longestStreak
    progress.last_active_date = date.fromisoformat(state.lastPracticeDate) if state.lastPracticeDate else None
    progress.total_correct = state.totalCorrect
    progress.total_attempts = state.totalAttempts
    progress.sessions = state.sessions
    progress.game_meta = json.dumps({"source": "asl_quest"})
    progress.updated_at = datetime.now(timezone.utc)

    existing_letters = {
        row.letter: row
        for row in db.query(LetterProgress).filter(LetterProgress.user_id == user.id).all()
    }
    for letter in LETTERS:
        stats = state.letterStats.get(letter, LetterStatsSchema())
        row = existing_letters.get(letter)
        if row is None:
            row = LetterProgress(user_id=user.id, letter=letter)
            db.add(row)
        row.attempts = stats.attempts
        row.correct_attempts = stats.correct
        row.mastery = mastery_percent(stats.correct)
        if stats.attempts > 0 or stats.correct > 0:
            row.last_practiced = datetime.now(timezone.utc)
        row.updated_at = datetime.now(timezone.utc)

    challenge_map = {
        "daily": state.dailyChallenge,
        "alphabet": {
            "progress": sum(1 for stats in state.letterStats.values() if stats.correct > 0),
            "completed": state.challenges.alphabetCompleted,
            "meta": {},
        },
        "speed": {
            "progress": state.challenges.speedCount,
            "completed": state.challenges.speedCompleted,
            "meta": {"speedStartedAt": state.challenges.speedStartedAt},
        },
        "word": {
            "progress": state.challenges.wordCount,
            "completed": state.challenges.wordCompleted,
            "meta": {},
        },
    }
    existing_challenges = {
        row.challenge_type: row
        for row in db.query(ChallengeProgress).filter(ChallengeProgress.user_id == user.id).all()
    }
    for challenge_type, payload in challenge_map.items():
        row = existing_challenges.get(challenge_type)
        if row is None:
            row = ChallengeProgress(user_id=user.id, challenge_type=challenge_type)
            db.add(row)
        if challenge_type == "daily":
            row.progress = payload.progress
            row.completed = payload.completed
            row.meta = json.dumps(
                {
                    "date": payload.date,
                    "target": payload.target,
                    "mistakes": payload.mistakes,
                    "perfectBonusAwarded": payload.perfectBonusAwarded,
                }
            )
        else:
            row.progress = payload["progress"]
            row.completed = payload["completed"]
            row.meta = json.dumps(payload["meta"])
        if row.completed and row.completed_at is None:
            row.completed_at = datetime.now(timezone.utc)
        if not row.completed:
            row.completed_at = None
        row.updated_at = datetime.now(timezone.utc)

    achievements = {item.slug: item for item in db.query(Achievement).all()}
    existing_user_badges = {
        link.achievement.slug: link
        for link in db.query(UserAchievement).join(Achievement).filter(UserAchievement.user_id == user.id).all()
    }
    for slug in state.unlockedBadges:
        achievement = achievements.get(slug)
        if achievement is None or slug in existing_user_badges:
            continue
        db.add(UserAchievement(user_id=user.id, achievement_id=achievement.id))

    db.commit()
    db.refresh(user)
    from backend.services.user_features import evaluate_and_persist_badges

    evaluate_and_persist_badges(db, user, state)
    return serialize_game_state(db, user)
