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
# substrings against the normalized message. If a message matches none of these AND
# it isn't a trusted follow-up (see is_in_scope below), the backend treats it as out
# of scope WITHOUT calling the LLM provider at all — this is the actual enforcement
# point (Phase 4/9 require the backend, not just a system prompt, to own the scope
# boundary).
PROJECT_KEYWORDS: tuple[str, ...] = (
    "asl", "sign", "sign language", "asl-quest", "asl quest", "this app", "this project",
    "this platform", "this website", "the app", "the platform",
    "a-z", "a to z", "alphabet", "resnet", "handshape", "recognition", "recognize",
    "mediapipe", "hand detect", "temporal smooth", "smoothing",
    "word spelling", "word practice", "spelling", "letter-by-letter", "letter by letter",
    "native sign", "native-sign", "isolated sign", "asl citizen", "i3d", "gloss",
    "vocabulary", "backbone", "linear head", "frozen",
    "database", "sqlite", "postgres", "sqlalchemy", "alembic", "migration", "schema",
    "fastapi", "backend", "endpoint", "router", "api", "architecture",
    "react", "typescript", "frontend", "vite", "component", "page",
    "authentication", "auth", "login", "register", "jwt", "bearer token",
    "xp", "experience point", "level", "streak", "achievement",
    "badge", "progress", "mastery", "mastered", "wrong", "correct", "mistake", "feedback",
    "analytics", "dataset", "training", "evaluation", "accuracy", "checkpoint",
    "model", "practice", "camera", "webcam", "prediction", "predict", "confidence",
    "chatbot", "assistant", "how do i", "how does", "how to use", "feature",
    "class_to_idx", "resnet18", "learning platform", "signer",
    "limitation", "limitations", "roadmap", "alternative", "alternatives",
    "compare", "comparison", "different", "difference",
)

# A short conversational follow-up ("why?", "how is that different?", "tell me
# more") almost never repeats a keyword of the question it's following up on —
# that's what makes it a follow-up. A *content-bearing* follow-up like "What are
# the alternatives?" is already handled above: "alternative"/"alternatives" (and
# "different"/"difference"/"compare"/"comparison") are themselves in
# PROJECT_KEYWORDS, so a message like that matches directly, on its own, with no
# help from conversation state at all.
#
# What's left for THIS mechanism is the genuinely content-free case: a short
# continuation with no topic words of its own to match against anything. For that
# narrow case, an earlier version of this function tried a NEGATIVE deny-list of
# "obviously generic trivia" shapes (capital-of, weather-in, etc.) to keep such
# messages from slipping through — that approach was tried and reverted after a
# live test caught a real hole: "What is the tallest mountain in the world?",
# asked right after an in-scope reply, matched none of the deny-list's fixed
# patterns and reached the provider anyway. A deny-list can never enumerate every
# way to phrase an unrelated question, which is exactly the class of bug this
# whole rewrite exists to fix elsewhere — reintroducing it here, just inverted,
# would be a regression, not a fix.
#
# The correct-shaped tool for a scope BOUNDARY is a POSITIVE allow-list (safe
# default: reject unless recognized), not a deny-list (unsafe default: accept
# unless recognized). So: a short, low-content message is trusted as a follow-up
# only if it *starts with* one of a small set of genuine continuation openers —
# words that only make sense referring back to something already said, and that
# no fresh, self-contained question would naturally open with. Matched as a
# prefix (not requiring the whole message to be an exact fixed string, unlike the
# old FOLLOWUP_PHRASES this replaces) so natural variations ("why though?", "how
# come?", "tell me more about that") still work. Bounded by word count (a long
# message "riding along" as a follow-up could smuggle in an unrelated topic) and
# by only trusting the conversation's MOST RECENT assistant turn (not "any turn,
# ever, this conversation has had") so it can't be used to permanently unlock an
# unrelated tangent partway through a long conversation.
_MAX_FOLLOWUP_WORDS = 12

_FOLLOWUP_OPENER_PATTERN = re.compile(
    r"^(?:"
    r"why(?:\s+not)?|why is that|why that|"
    r"how(?:\s+come|\s+so)?|"
    r"more|tell me more|"
    r"explain(?:\s+more)?|elaborate|"
    r"continue|go on|"
    r"and|"
    r"what about(?:\s+that)?|what else"
    r")\b"
)


def _is_trusted_followup_opener(normalized: str) -> bool:
    return bool(_FOLLOWUP_OPENER_PATTERN.match(normalized))

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


def is_in_scope(message: str, last_turn_in_scope: bool = False) -> bool:
    """Backend-owned scope gate — this, not the LLM or the system prompt, is what
    actually keeps an off-topic message from ever reaching the provider.

    A message is in scope if either:
    1. It matches the project keyword allow-list (works for any message, first-turn
       or not) — this also covers most content-bearing follow-ups directly, e.g.
       "What are the alternatives?" matches on "alternatives" alone, or
    2. It's short, opens with a recognized content-free continuation word ("why?",
       "how come?", "tell me more", "what about that?"), and immediately follows an
       in-scope assistant reply — see _FOLLOWUP_OPENER_PATTERN.

    `last_turn_in_scope` must reflect only the conversation's most recent assistant
    turn, not "was any turn ever in scope" — see
    backend/routers/chatbot.py::_last_assistant_turn_in_scope.
    """
    normalized = _normalize(message)
    if not normalized:
        return False
    if _KEYWORD_PATTERN.search(normalized):
        return True
    if not last_turn_in_scope:
        return False
    return len(normalized.split()) <= _MAX_FOLLOWUP_WORDS and _is_trusted_followup_opener(normalized)


# Keyword groups used ONLY for offline fallback retrieval (see fallback_answer
# below) -- deliberately more specific/curated than PROJECT_KEYWORDS (which
# exists to gate scope broadly), so that each group points at the single
# PROJECT_KNOWLEDGE_SECTIONS entry it actually describes.
SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "overview": (
        "asl quest", "asl-quest", "this app", "this project", "this platform",
        "this website", "the app", "the platform", "overview", "learning platform",
        "three learning systems",
    ),
    "a_to_z": (
        "a-z", "a to z", "alphabet", "resnet", "resnet18", "handshape", "mediapipe",
        "hand detect", "temporal smooth", "smoothing", "confidence threshold",
        "class_to_idx", "best_model",
    ),
    "word_spelling": (
        "word spelling", "word practice", "spelling", "letter-by-letter", "letter by letter",
    ),
    "native_signs": (
        "native sign", "native-sign", "isolated sign", "asl citizen", "i3d", "gloss",
        "vocabulary", "backbone", "linear head", "frozen", "native",
    ),
    "gamification": (
        "xp", "experience point", "level", "streak", "achievement", "badge",
        "mastery", "mastered", "gamification", "challenge",
    ),
    "database": (
        "database", "sqlite", "postgres", "sqlalchemy", "alembic", "migration", "schema",
    ),
    "backend": (
        "fastapi", "backend", "endpoint", "router", "architecture", "authentication",
        "auth", "jwt", "bearer token", "login", "register",
    ),
    "frontend": (
        "react", "typescript", "frontend", "vite", "component", "page",
    ),
}

_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    section: re.compile(
        r"\b(?:"
        + "|".join(sorted({re.escape(_normalize(kw)) for kw in keywords if _normalize(kw)}, key=len, reverse=True))
        + r")\b"
    )
    for section, keywords in SECTION_KEYWORDS.items()
}

FALLBACK_PREFIX = (
    "AURA's AI model isn't reachable right now, so this answer comes from ASL-Quest's "
    "offline knowledge base instead:\n\n"
)


def fallback_answer(message: str) -> str:
    """Deterministic, LLM-free answer used whenever the configured provider is
    unavailable (not configured, unreachable, rate-limited, or erroring) --
    this is what keeps AURA answering ASL-Quest questions even with the local
    Ollama server stopped and no OPENROUTER_API_KEY set (e.g. during a demo
    with no internet and no local model running).

    Scores each knowledge section by how many of its curated keywords appear
    in the message and returns the content of the best-matching section(s)
    (top 2, in case of a tie) prefixed with a clear "this is the offline
    knowledge base, not the AI" disclaimer. Never invents anything not
    already in PROJECT_KNOWLEDGE_SECTIONS; if nothing matches, says so via
    NO_INFO_REPLY rather than guessing.
    """
    normalized = _normalize(message)
    if not normalized:
        return NO_INFO_REPLY

    scores = {
        section: len(pattern.findall(normalized)) for section, pattern in _SECTION_PATTERNS.items()
    }
    scores = {section: count for section, count in scores.items() if count > 0}
    if not scores:
        return NO_INFO_REPLY

    best = max(scores.values())
    top_sections = sorted(section for section, count in scores.items() if count == best)
    body = "\n\n".join(PROJECT_KNOWLEDGE_SECTIONS[section] for section in top_sections[:2])
    return FALLBACK_PREFIX + body


def build_project_knowledge() -> str:
    return "\n\n".join(
        f"## {title.replace('_', ' ').title()}\n{body}" for title, body in PROJECT_KNOWLEDGE_SECTIONS.items()
    )


def build_system_prompt(user_context: str | None = None) -> str:
    parts = [SYSTEM_INSTRUCTION, "\nASL-QUEST PROJECT KNOWLEDGE:\n" + build_project_knowledge()]
    if user_context:
        parts.append("\nUSER CONTEXT (this learner's own progress — never share other users' data):\n" + user_context)
    return "\n".join(parts)
