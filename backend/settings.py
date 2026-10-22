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


# Which LLM provider AURA uses: "openrouter" (default, unchanged behavior),
# "local" (a local Ollama server -- see backend/services/chatbot_local_llm.py
# -- works fully offline, never contacts OpenRouter, never needs
# OPENROUTER_API_KEY), or "auto" (prefer local when reachable, otherwise fall
# back to OpenRouter if it's configured). Defaulting to "openrouter" when
# unset means every existing deployment/test is unaffected by this setting's
# existence.
def get_chatbot_provider() -> str:
    return os.getenv("CHATBOT_PROVIDER", "").strip().lower() or "openrouter"


# Local Ollama server base URL (no trailing slash) -- e.g. http://127.0.0.1:11434.
# Only used when CHATBOT_PROVIDER is "local" or "auto".
def get_chatbot_local_base_url() -> str:
    return (os.getenv("CHATBOT_LOCAL_BASE_URL", "").strip() or "http://127.0.0.1:11434").rstrip("/")


# Ollama model tag for local/auto mode. The model itself is never downloaded
# by this project -- see README's "Offline AURA setup": `ollama pull qwen3:4b`.
def get_chatbot_local_model() -> str:
    return os.getenv("CHATBOT_LOCAL_MODEL", "").strip() or "qwen3:4b"


def _hand_detection_confidence() -> float:
    """MediaPipe min detection/presence confidence for the live hand cropper (0.05-0.95).

    0.35 was chosen from scripts/diagnose_crop_strategies.py-style measurements: it found more hands
    than MediaPipe's 0.5 default with no false positives on hand-free frames. Raise it if the app
    starts "seeing" hands that are not there in your room.
    """
    raw = os.getenv("ASL_QUEST_HAND_DETECTION_CONFIDENCE", "").strip()
    try:
        value = float(raw) if raw else 0.35
    except ValueError:
        value = 0.35
    return min(0.95, max(0.05, value))


HAND_DETECTION_CONFIDENCE = _hand_detection_confidence()

# Development-only: save the exact classifier input of live predictions to a local folder (see
# backend/services/live_diagnostics.py). OFF unless ASL_QUEST_DEV_DIAGNOSTICS is 1/true/yes.
DEV_DIAGNOSTICS = os.getenv("ASL_QUEST_DEV_DIAGNOSTICS", "").strip().lower() in {"1", "true", "yes"}
DEV_DIAGNOSTICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "diagnostics" / "live_requests"

# Trailing slashes are stripped: browsers send the Origin header without one, so
# "https://app.vercel.app/" would never match "https://app.vercel.app".
# Used when ASL_QUEST_CORS_ORIGINS is unset/blank: local dev plus the deployed Vercel frontend
# (exact origin, never a wildcard -- credentials are enabled). Setting the env var replaces this list.
DEFAULT_CORS_ORIGINS = "http://127.0.0.1:5173,http://localhost:5173,https://asl-quest-1.vercel.app"
CORS_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in (os.getenv("ASL_QUEST_CORS_ORIGINS", "").strip() or DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

# Development-only labelled webcam data collection (see backend/services/webcam_dataset.py and routers/collect.py).
# OFF unless ASL_QUEST_DEV_COLLECT is 1/true/yes. dataset/ is git-ignored: collected frames contain your face and room.
# Which recognizer answers /predict: "resnet" (the image classifier, default) or "landmarks" (the MediaPipe-landmark
# classifier, models/landmark_classifier.joblib). Switch only after scripts/compare_approaches.py shows it is better on unseen rooms.
RECOGNIZER = os.getenv("ASL_QUEST_RECOGNIZER", "resnet").strip().lower() or "resnet"
LANDMARK_MODEL_PATH = Path(os.getenv("ASL_QUEST_LANDMARK_MODEL", "").strip() or Path(__file__).resolve().parent.parent / "models" / "landmark_classifier.joblib")

# EXPERIMENTAL side-by-side comparison: a landmark model (a .joblib under models/) that clients may request per call with
# /predict?recognizer=landmarks while "resnet" stays the default for everyone. Empty (default) = feature off, nothing is loaded.
EXPERIMENTAL_LANDMARK_MODEL = os.getenv("ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL", "").strip()

DEV_COLLECT = os.getenv("ASL_QUEST_DEV_COLLECT", "").strip().lower() in {"1", "true", "yes"}
WEBCAM_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset" / "webcam"
