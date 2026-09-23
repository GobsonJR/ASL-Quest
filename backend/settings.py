from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_DATABASE_PATH = DATA_DIR / "asl_quest.db"

SECRET_KEY = os.getenv("ASL_QUEST_SECRET_KEY", "dev-change-me-in-production")

# DATABASE_URL is the standard/recommended env var (e.g. postgresql+psycopg://user:pass@host/db).
# ASL_QUEST_DATABASE_URL is kept as a legacy fallback for existing local setups.
# Default (neither set) is the existing local SQLite dev database — unchanged.
DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("ASL_QUEST_DATABASE_URL")
    or f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}"
)
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ASL_QUEST_TOKEN_EXPIRE_MINUTES", "10080"))


def get_bootstrap_admin_email() -> str:
    return os.getenv("ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()


BOOTSTRAP_ADMIN_EMAIL = get_bootstrap_admin_email()


# Read fresh on every call (not cached at import time like the constants above) so
# tests can toggle "configured" vs "not configured" within a single process via
# os.environ, without needing to control backend.settings' import order.
def get_chatbot_api_key() -> str:
    return os.getenv("CHATBOT_API_KEY", "").strip()


def get_chatbot_model() -> str:
    return os.getenv("CHATBOT_MODEL", "").strip() or "claude-sonnet-5"


CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ASL_QUEST_CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    ).split(",")
    if origin.strip()
]
