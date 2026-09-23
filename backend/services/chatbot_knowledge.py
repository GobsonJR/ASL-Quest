"""Project-only knowledge base and scope guard for AURA, the ASL-Quest
project assistant (display name: "AURA — ASL-Quest Project Assistant").

Everything in PROJECT_KNOWLEDGE is a fact verified against this repository
(README.md, DATABASE.md, NATIVE_10_PHASE_7_REPORT.md, backend/models.py,
backend/services/progress.py, src/i3d_transfer/predict.py, src/inference/*,
frontend/src/game/constants.ts) at the time this file was written — nothing
here is invented. If a fact isn't in this file, the system prompt instructs
the model to say it doesn't have that information rather than guess.

Keeping this as a dedicated module (not a string embedded in a React
component or scattered across the router) is what Phase 2 of the chatbot
task asked for: a single, auditable source of project knowledge.
"""

from __future__ import annotations

import re

PROJECT_KNOWLEDGE_SECTIONS: dict[str, str] = {
    "overview": (
        "ASL-Quest is a gamified American Sign Language learning platform with three "
        "independent learning systems: A-Z (static alphabet handshape recognition), "
        "Word Spelling (spelling whole words letter-by-letter using A-Z recognition), "
        "and Native Signs (isolated recognition of a complete native ASL sign, not "
        "spelled out). It tracks XP, levels, streaks, and achievements across all three."
    ),
    "a_to_z": (
        "The A-Z system recognizes static handshapes for the 26 letters A-Z from a "
        "single image (space/nothing/del are not included classes). The model is a "
        "ResNet18 classifier (src/models/resnet18.py) fine-tuned for this task. Before "
        "classification, MediaPipe (src/inference/hands.py) crops the hand region from "
        "the frame; if no hand is detected the UI shows 'No hand detected' instead of "
        "guessing. Live webcam predictions apply a 0.70 confidence threshold and a "
        "7-frame majority-vote temporal smoother (src/utils/smoothing.py) to reduce "
        "flicker between frames. Dynamic letters J and Z are NOT truly supported — the "
        "model has no temporal motion modeling, so it may still output a static guess "
        "for them. The current production checkpoint (models/best_model.pth) is a "
        "retrained/recovered checkpoint (the original was lost); on the project's "
        "11,700-image controlled sequential holdout it scores 90.91% overall accuracy, "
        "with its weakest classes being S (27.78%), X (35.33%), V, and Y. An earlier, "
        "now-lost checkpoint had scored 97.91% on the same holdout — that number "
        "describes a checkpoint that no longer exists on disk, not the current one."
    ),
    "word_spelling": (
        "Word Spelling is NOT direct word-level sign recognition. It teaches a whole "
        "word by having the learner sign it letter-by-letter using the same A-Z static "
        "handshape recognition described above, in sequence. This is different from "
        "Native Signs, where a single whole gesture is recognized as one complete sign."
    ),
    "native_signs": (
        "Native Signs is isolated recognition of a complete, native ASL sign performed "
        "as one gesture (not spelled letter-by-letter). The current vocabulary is 10 "
        "signs: BOOK, EAT1, HELLO, HELP, MOTHER, NO, PLEASE, THANKYOU, WATER, YES. The "
        "model is the official Microsoft ASL Citizen-pretrained I3D video backbone, "
        "used FROZEN (not fine-tuned), feeding a small trained linear head "
        "(Dropout(0.5) -> Linear(1024, num_classes)) on top of the backbone's "
        "temporally mean-pooled features (src/i3d_transfer/predict.py, pytorch_i3d.py, "
        "preprocessing.py). I3D was chosen after earlier ResNet18-based per-frame "
        "appearance approaches struggled to generalize to unseen signers; the frozen "
        "I3D features (already sign-discriminative from large-scale ASL Citizen "
        "pretraining) reached 96.90% top-1 / 99.22% top-5 test accuracy on the "
        "native-10 holdout with only a linear head trained on top. A camera clip is "
        "recorded client-side (about 3 seconds), sent to POST /native/predict, run "
        "through the model, and never stored server-side beyond that single request. "
        "The practice UI follows a LEARN -> WATCH -> PRACTICE -> CHECK -> RESULT flow. "
        "ASL Citizen's own demonstration videos are never redistributed or embedded in "
        "the app for licensing reasons; a sign's reference panel only shows a video "
        "when an approved external reference URL has been added to the "
        "native_sign_references table, otherwise it shows an honest placeholder."
    ),
    "gamification": (
        "XP, levels, streaks, and achievements are shared across A-Z, Word Spelling, "
        "and Native Signs. A correct native sign practice attempt awards +20 XP "
        "(NATIVE_XP_CORRECT in backend/routers/native.py), matching the base XP for one "
        "correct A-Z practice attempt. Level is computed as floor(xp / 1000) + 1 (1000 "
        "XP per level), both on the frontend (frontend/src/game/gamification.ts) and "
        "backend (backend/services/progress.py::level_from_xp). Native mastery per sign "
        "is simply the user's running accuracy for that sign (correct attempts / total "
        "attempts), stored on native_sign_progress.mastery. Achievements are seeded rows "
        "evaluated against real progress — for Native Signs: 'First Native Sign' "
        "(native_first_sign, first correct sign), 'Native Sign Explorer' "
        "(native_explorer, correct signs from 3+ categories), and 'Native Sign Master' "
        "(native_master, every active native sign mastered). XP and correctness for "
        "native practice are always determined server-side in POST /native/predict, "
        "never trusted from the client."
    ),
    "database": (
        "The database uses SQLAlchemy models (backend/models.py) and Alembic "
        "migrations (migrations/versions/). Two engines are supported via the "
        "DATABASE_URL env var: SQLite at data/asl_quest.db for local development "
        "(the default, no setup required), and PostgreSQL (via the psycopg v3 driver) "
        "recommended for production. Baseline tables include users, user_progress, "
        "letter_progress, challenge_progress, achievements, user_achievements, "
        "practice_sessions, xp_events, word_progress, and word_practice_sessions. A "
        "later migration added words, native_signs, native_sign_references, "
        "native_sign_progress, native_sign_sessions, model_versions, sign_predictions, "
        "user_preferences, and the chatbot tables (chatbot_conversations, "
        "chatbot_messages, chatbot_feedback) that this ASL-Quest Assistant uses. "
        "native_sign_progress tracks attempts, correct_attempts, accuracy, mastery, "
        "best_confidence, and best_response_time per user per sign. "
        "native_sign_sessions logs each individual practice attempt (result, "
        "confidence, response time). sign_predictions is a shared raw prediction log "
        "used by both the A-Z and Native Signs models."
    ),
    "backend": (
        "The backend is a FastAPI application (backend/main.py) composed of feature "
        "routers under backend/routers/: auth, progress, challenges, achievements, "
        "practice, analytics, users, recommendations, admin, words, native, and "
        "chatbot. Authentication is JWT bearer tokens (backend/security.py), signed "
        "with the ASL_QUEST_SECRET_KEY env var, required on almost every endpoint via "
        "a get_current_user dependency. Business logic lives in backend/services/ "
        "(e.g. progress.py, words.py), kept separate from the routers themselves. The "
        "native prediction endpoint resolves the expected sign and determines "
        "correctness entirely server-side."
    ),
    "frontend": (
        "The frontend is React + TypeScript, built with Vite. It's organized by "
        "feature: frontend/src/native/ holds the Native Signs catalog page, the "
        "practice page (LEARN -> WATCH -> PRACTICE -> CHECK -> RESULT), the camera "
        "capture component, and the native API client; frontend/src/pages/ holds "
        "pages like Progress and Achievements; frontend/src/game/ holds the shared "
        "gamification state (GameContext) used by every learning system. The Progress "
        "page shows a dedicated native-signs section (signs started/mastered, overall "
        "mastery) once a user has native progress. The Achievements page lists "
        "unlocked and locked badges, including the native-specific ones."
    ),
}

# Deliberately broad but bounded allow-list, matched as whole-word/whole-phrase
# substrings against the normalized message. If a message matches none of these, the
# backend treats it as out of scope WITHOUT calling the LLM provider at all — this is
# the actual enforcement point (Phase 4/9 require the backend, not just a system
# prompt, to own the scope boundary).
PROJECT_KEYWORDS: tuple[str, ...] = (
    "asl", "sign language", "asl-quest", "asl quest", "this app", "this project",
    "this platform", "this website", "the app", "the platform",
    "a-z", "a to z", "alphabet", "resnet", "handshape",
    "mediapipe", "hand detect", "temporal smooth", "smoothing",
    "word spelling", "word practice", "spelling", "letter-by-letter", "letter by letter",
    "native sign", "native-sign", "isolated sign", "asl citizen", "i3d", "gloss",
    "vocabulary", "backbone", "linear head", "frozen",
    "database", "sqlite", "postgres", "sqlalchemy", "alembic", "migration", "schema",
    "fastapi", "backend", "endpoint", "router", "api",
    "react", "typescript", "frontend", "vite", "component", "page",
    "authentication", "auth", "login", "register", "jwt", "bearer token",
    "xp", "experience point", "level", "streak", "achievement",
    "badge", "progress", "mastery", "mastered",
    "analytics", "dataset", "training", "evaluation", "accuracy", "checkpoint",
    "model", "practice", "camera", "webcam", "prediction", "predict", "confidence",
    "chatbot", "assistant", "how do i", "how does", "how to use", "feature",
    "class_to_idx", "resnet18", "learning platform", "signer",
)

# Very short conversational continuations that only make sense as a follow-up to an
# already-in-scope exchange (e.g. "why?" after an on-topic answer). Matched only as
# a WHOLE normalized message, never as a substring, and only honored when the
# conversation already has at least one prior in-scope assistant reply — this keeps
# "What is the capital of France?" (which is also short) from slipping through.
FOLLOWUP_PHRASES: frozenset[str] = frozenset(
    {
        "why", "why not", "why is that", "why that",
        "how", "how come", "how so",
        "more", "tell me more", "explain", "explain more", "elaborate",
        "continue", "go on", "and", "what about that", "what else",
    }
)

OFF_TOPIC_REPLY = "I can only answer questions related to the ASL-Quest project."

NO_INFO_REPLY = "I don't have enough information about that part of ASL-Quest yet."

SYSTEM_INSTRUCTION = (
    "You are AURA, the ASL-Quest Project Assistant.\n\n"
    "You may answer only questions directly related to the ASL-Quest project.\n\n"
    "Your knowledge is limited to the structured ASL-Quest project context supplied "
    "to you below.\n\n"
    "Relevant topics include: A-Z alphabet recognition, ResNet18, MediaPipe, Word "
    "Spelling, Native ASL Signs, ASL Citizen, I3D, native sign practice, FastAPI, "
    "React, TypeScript, Vite, SQLite, SQLAlchemy, authentication, APIs, progress, "
    "XP, levels, streaks, achievements, analytics, project datasets, project "
    "limitations, and project architecture.\n\n"
    "Do not answer unrelated general-knowledge, political, entertainment, personal, "
    "or arbitrary questions.\n\n"
    f"If the user asks something unrelated, say: \"{OFF_TOPIC_REPLY}\"\n\n"
    "Do not invent ASL-Quest project facts.\n\n"
    "If the supplied project context does not contain enough information, say: "
    f"\"{NO_INFO_REPLY}\"\n\n"
    "Answer only using the ASL-QUEST PROJECT KNOWLEDGE below and, when given, the "
    "USER CONTEXT. Do not use outside/general knowledge about ASL, machine learning, "
    "or software engineering beyond what is stated here, even if you happen to know "
    "more — if it isn't in the knowledge below, say you don't have that information "
    "yet rather than fill the gap.\n\n"
    "Be concise, accurate, and helpful."
)


def _normalize(text: str) -> str:
    # Underscores are treated as word separators (not \w-preserved) so identifiers
    # like "native_sign_progress" or "class_to_idx" still line up with the
    # space-separated keywords below.
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.strip().lower())).strip()


# Keywords are normalized the same way as messages (so "A-Z" and "letter-by-letter"
# line up with a message's "a z" / "letter by letter"), then matched with \b word
# boundaries — plain substring matching let short keywords like "api" false-match
# inside unrelated words (e.g. "capital" contains "api").
_KEYWORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted({re.escape(_normalize(kw)) for kw in PROJECT_KEYWORDS if _normalize(kw)}, key=len, reverse=True)) + r")\b"
)


def is_in_scope(message: str, has_prior_in_scope_turn: bool = False) -> bool:
    """Backend-owned scope gate. Returns False (out of scope) for anything that
    doesn't match the project keyword allow-list, independent of the LLM."""
    normalized = _normalize(message)
    if not normalized:
        return False
    if _KEYWORD_PATTERN.search(normalized):
        return True
    return has_prior_in_scope_turn and normalized in FOLLOWUP_PHRASES


def build_project_knowledge() -> str:
    return "\n\n".join(
        f"## {title.replace('_', ' ').title()}\n{body}" for title, body in PROJECT_KNOWLEDGE_SECTIONS.items()
    )


def build_system_prompt(user_context: str | None = None) -> str:
    parts = [SYSTEM_INSTRUCTION, "\nASL-QUEST PROJECT KNOWLEDGE:\n" + build_project_knowledge()]
    if user_context:
        parts.append("\nUSER CONTEXT (this learner's own progress — never share other users' data):\n" + user_context)
    return "\n".join(parts)
