from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), default="student", nullable=False, index=True)
    preferences: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    progress: Mapped[UserProgress | None] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    letter_progress: Mapped[list[LetterProgress]] = relationship(back_populates="user", cascade="all, delete-orphan")
    challenge_progress: Mapped[list[ChallengeProgress]] = relationship(back_populates="user", cascade="all, delete-orphan")
    achievements: Mapped[list[UserAchievement]] = relationship(back_populates="user", cascade="all, delete-orphan")
    practice_sessions: Mapped[list[PracticeSession]] = relationship(back_populates="user", cascade="all, delete-orphan")
    xp_events: Mapped[list["XpEvent"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    word_progress: Mapped[list["WordProgress"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    word_practice_sessions: Mapped[list["WordPracticeSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    # Phase 2 additions: tables exist via Alembic migrations; not yet wired into routers/services.
    # passive_deletes=True: rely entirely on the FKs' ondelete="CASCADE" in the database
    # rather than having the ORM SELECT these children when a User is deleted. This
    # matters because these tables may not exist yet on a database that hasn't run the
    # 0002 migration (e.g. existing test fixtures that only build the legacy schema) --
    # without passive_deletes, SQLAlchemy's cascade="all, delete-orphan" tries to load
    # every relationship on delete, which would error against a missing table.
    preference: Mapped["UserPreference | None"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )
    native_sign_progress: Mapped[list["NativeSignProgress"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    native_sign_sessions: Mapped[list["NativeSignSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    sign_predictions: Mapped[list["SignPrediction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    chatbot_conversations: Mapped[list["ChatbotConversation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class UserProgress(Base):
    __tablename__ = "user_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    xp: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    level: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    current_streak: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    best_streak: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_active_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    total_correct: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sessions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    game_meta: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="progress")


class LetterProgress(Base):
    __tablename__ = "letter_progress"
    __table_args__ = (UniqueConstraint("user_id", "letter", name="uq_user_letter"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    letter: Mapped[str] = mapped_column(String(1), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    correct_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mastery: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_practiced: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="letter_progress")


class ChallengeProgress(Base):
    __tablename__ = "challenge_progress"
    __table_args__ = (UniqueConstraint("user_id", "challenge_type", name="uq_user_challenge"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    challenge_type: Mapped[str] = mapped_column(String(32), nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    meta: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="challenge_progress")


class Achievement(Base):
    __tablename__ = "achievements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    requirement: Mapped[str] = mapped_column(Text, nullable=False)

    user_links: Mapped[list[UserAchievement]] = relationship(back_populates="achievement")


class UserAchievement(Base):
    __tablename__ = "user_achievements"
    __table_args__ = (UniqueConstraint("user_id", "achievement_id", name="uq_user_achievement"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    achievement_id: Mapped[int] = mapped_column(ForeignKey("achievements.id", ondelete="CASCADE"), nullable=False)
    earned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="achievements")
    achievement: Mapped[Achievement] = relationship(back_populates="user_links")


class PracticeSession(Base):
    __tablename__ = "practice_sessions"
    __table_args__ = (
        Index("ix_practice_sessions_user_created", "user_id", "created_at"),
        Index("ix_practice_sessions_user_letter", "user_id", "letter"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    letter: Mapped[str] = mapped_column(String(1), nullable=False, index=True)
    prediction: Mapped[str | None] = mapped_column(String(1), nullable=True)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    response_time: Mapped[int | None] = mapped_column(Integer, nullable=True)
    xp_earned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    challenge_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    user: Mapped[User] = relationship(back_populates="practice_sessions")


class XpEvent(Base):
    __tablename__ = "xp_events"
    __table_args__ = (Index("ix_xp_events_user_created", "user_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    user: Mapped[User] = relationship(back_populates="xp_events")


class WordProgress(Base):
    __tablename__ = "word_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "word_id", name="uq_user_word"),
        Index("ix_word_progress_user_word", "user_id", "word_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    word_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    correct_letters: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_letters: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mastery: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    best_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_time_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    resume_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    letter_stats: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    last_practiced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="word_progress")


class WordPracticeSession(Base):
    __tablename__ = "word_practice_sessions"
    __table_args__ = (
        Index("ix_word_practice_sessions_user_created", "user_id", "created_at"),
        Index("ix_word_practice_sessions_user_word", "user_id", "word_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    word_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    xp_earned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    challenge_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    mistakes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    letter_results: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    user: Mapped[User] = relationship(back_populates="word_practice_sessions")


# =====================================================================================
# Phase 2 additions (database foundation). These tables are created by Alembic migrations
# (see migrations/versions/). They are not yet read from or written to by any router or
# service — that wiring is deliberately deferred to later phases.
#
# `Word.id` intentionally reuses the existing string catalog IDs from
# backend/words/catalog.py (e.g. "beginner-cat") as its primary key. WordProgress.word_id
# already stores exactly these IDs and is left untouched (plain, unconstrained String, as
# it always was) so existing word_progress rows keep resolving by equality with zero
# migration of that column. No hard FK is added from word_progress to words in this phase
# to avoid a SQLite table rebuild on a table that already holds live user data.
# =====================================================================================


class UserPreference(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    theme: Mapped[str] = mapped_column(String(32), default="system", nullable=False)
    language: Mapped[str] = mapped_column(String(16), default="en", nullable=False)
    notification_preferences: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    learning_preferences: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="preference")


class Word(Base):
    """Spelling-practice word catalog (System B). Reuses catalog string IDs as the PK."""

    __tablename__ = "words"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    word: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    difficulty: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    spelling_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    native_sign_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class NativeSign(Base):
    """Native isolated ASL sign vocabulary entry (System C). Independent from Word."""

    __tablename__ = "native_signs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    gloss: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    meaning: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    difficulty: Mapped[str | None] = mapped_column(String(16), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    example_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model_available: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    references_: Mapped[list["NativeSignReference"]] = relationship(
        back_populates="native_sign", cascade="all, delete-orphan"
    )
    progress: Mapped[list["NativeSignProgress"]] = relationship(
        back_populates="native_sign", cascade="all, delete-orphan"
    )
    sessions: Mapped[list["NativeSignSession"]] = relationship(
        back_populates="native_sign", cascade="all, delete-orphan"
    )


class NativeSignReference(Base):
    """Reference media/example pointer for a native sign. Never stores redistributed dataset media."""

    __tablename__ = "native_sign_references"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    native_sign_id: Mapped[int] = mapped_column(
        ForeignKey("native_signs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    local_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    license_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    native_sign: Mapped[NativeSign] = relationship(back_populates="references_")


class NativeSignProgress(Base):
    __tablename__ = "native_sign_progress"
    __table_args__ = (UniqueConstraint("user_id", "native_sign_id", name="uq_user_native_sign"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    native_sign_id: Mapped[int] = mapped_column(
        ForeignKey("native_signs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    correct_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mastery: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    best_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_response_time: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_practiced: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="native_sign_progress")
    native_sign: Mapped[NativeSign] = relationship(back_populates="progress")


class NativeSignSession(Base):
    __tablename__ = "native_sign_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    native_sign_id: Mapped[int] = mapped_column(
        ForeignKey("native_signs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    response_time: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    user: Mapped[User] = relationship(back_populates="native_sign_sessions")
    native_sign: Mapped[NativeSign] = relationship(back_populates="sessions")
    predictions: Mapped[list["SignPrediction"]] = relationship(back_populates="session")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_type", "version", name="uq_model_type_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String(512), nullable=False)
    vocabulary_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    num_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    predictions: Mapped[list["SignPrediction"]] = relationship(back_populates="model_version_ref")


class SignPrediction(Base):
    __tablename__ = "sign_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("native_sign_sessions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    predicted_label: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    expected_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ix_sign_predictions_user_created", "user_id", "created_at"),
    )

    user: Mapped[User] = relationship(back_populates="sign_predictions")
    session: Mapped[NativeSignSession | None] = relationship(back_populates="predictions")
    model_version_ref: Mapped[ModelVersion | None] = relationship(back_populates="predictions")


class ChatbotConversation(Base):
    __tablename__ = "chatbot_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="chatbot_conversations")
    messages: Mapped[list["ChatbotMessage"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class ChatbotMessage(Base):
    __tablename__ = "chatbot_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("chatbot_conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    conversation: Mapped[ChatbotConversation] = relationship(back_populates="messages")
    feedback: Mapped[list["ChatbotFeedback"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )


class ChatbotFeedback(Base):
    __tablename__ = "chatbot_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("chatbot_messages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    message: Mapped[ChatbotMessage] = relationship(back_populates="feedback")
