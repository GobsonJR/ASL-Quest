from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from backend.settings import DATA_DIR, DATABASE_URL

Base = declarative_base()
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _table_columns(table_name: str) -> set[str]:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


# Tables owned by the legacy ad-hoc migrator below. Phase 2 added several new ORM models
# (Word, NativeSign*, ModelVersion, SignPrediction, ChatbotConversation, etc.) to the same
# Base/metadata, but those tables must be created only via Alembic migrations
# (see migrations/versions/), not by this implicit create_all. Keeping this explicit list
# means existing startup behavior for the original tables is unchanged.
_LEGACY_TABLE_NAMES = (
    "users",
    "achievements",
    "user_progress",
    "letter_progress",
    "challenge_progress",
    "user_achievements",
    "practice_sessions",
    "xp_events",
    "word_progress",
    "word_practice_sessions",
)


def migrate_schema() -> None:
    from backend import models  # noqa: F401

    legacy_tables = [
        Base.metadata.tables[name] for name in _LEGACY_TABLE_NAMES if name in Base.metadata.tables
    ]
    Base.metadata.create_all(bind=engine, tables=legacy_tables)

    practice_columns = _table_columns("practice_sessions")
    if practice_columns:
        additions = {
            "xp_earned": "INTEGER NOT NULL DEFAULT 0",
            "challenge_type": "VARCHAR(32)",
            "confidence": "FLOAT",
        }
        with engine.begin() as connection:
            for column, ddl in additions.items():
                if column not in practice_columns:
                    connection.execute(text(f"ALTER TABLE practice_sessions ADD COLUMN {column} {ddl}"))

    user_columns = _table_columns("users")
    if user_columns:
        user_additions = {
            "role": "VARCHAR(16) NOT NULL DEFAULT 'student'",
            "preferences": "TEXT NOT NULL DEFAULT '{}'",
        }
        with engine.begin() as connection:
            for column, ddl in user_additions.items():
                if column not in user_columns:
                    connection.execute(text(f"ALTER TABLE users ADD COLUMN {column} {ddl}"))

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_practice_sessions_user_created "
                "ON practice_sessions (user_id, created_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_practice_sessions_user_letter "
                "ON practice_sessions (user_id, letter)"
            )
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_xp_events_user_created ON xp_events (user_id, created_at)")
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_word_progress_user_word ON word_progress (user_id, word_id)")
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_word_practice_sessions_user_created "
                "ON word_practice_sessions (user_id, created_at)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_word_practice_sessions_user_word "
                "ON word_practice_sessions (user_id, word_id)"
            )
        )


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    migrate_schema()
