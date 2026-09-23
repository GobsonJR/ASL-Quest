from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_DATABASE_PATH = DATA_DIR / "asl_quest.db"

# `or` (not a getenv default) so a blank `KEY=` line in .env counts as unset.
SECRET_KEY = os.getenv("ASL_QUEST_SECRET_KEY", "").strip() or "dev-change-me-in-production"

# DATABASE_URL is the standard/recommended env var (e.g. postgresql+psycopg://user:pass@host/db).
# ASL_QUEST_DATABASE_URL is kept as a legacy fallback for existing local setups.
# Default (neither set) is the existing local SQLite dev database — unchanged.
DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("ASL_QUEST_DATABASE_URL")
    or f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}"
)
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ASL_QUEST_TOKEN_EXPIRE_MINUTES", "").strip() or "10080")


def get_bootstrap_admin_email() -> str:
    return os.getenv("ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL", "").strip().lower()


BOOTSTRAP_ADMIN_EMAIL = get_bootstrap_admin_email()


# Read fresh on every call (not cached at import time like the constants above) so
# tests can toggle "configured" vs "not configured" within a single process via
# os.environ, without needing to control backend.settings' import order.
#
# AURA's provider is OpenRouter (an OpenAI-compatible chat completions API),
# hence OPENROUTER_API_KEY rather than a generic CHATBOT_* name for the secret
# itself. The CHATBOT_* prefix is kept for the model/base-URL knobs, which stay
# provider-shaped concepts either way.
def get_chatbot_api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def get_chatbot_model() -> str:
    return os.getenv("CHATBOT_MODEL", "").strip() or "openrouter/free"


# Optional override for the provider's API endpoint (e.g. a corporate proxy or a
# different OpenAI-compatible endpoint). Defaults to OpenRouter's own chat
# completions API — most deployments never need to set this.
def get_chatbot_api_base_url() -> str:
    return (
        os.getenv("CHATBOT_API_BASE_URL", "").strip()
        or "https://openrouter.ai/api/v1/chat/completions"
    )


CORS_ORIGINS = [
    origin.strip()
    for origin in (
        os.getenv("ASL_QUEST_CORS_ORIGINS", "").strip()
        or "http://127.0.0.1:5173,http://localhost:5173"
    ).split(",")
    if origin.strip()
]
