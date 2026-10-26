"""Offline FAQ for AURA, the ASL-Quest project assistant.

Used by chatbot_knowledge.fallback_answer when no LLM is reachable (no API key, rate limit, no
internet, local Ollama stopped). Each entry is a focused, hand-written answer to a family of
questions about the project, so the offline assistant can answer the specific thing that was asked
instead of dumping a whole documentation section.

Every fact here comes from this repository (README.md, DATABASE.md, backend/services/progress.py,
backend/words/catalog.py, frontend/src/game/constants.ts, the native-sign seed data and the
NATIVE_10_* reports). Nothing is invented: when a topic is not covered, retrieval returns no match
and the caller says so instead of guessing.

Retrieval is deliberately simple and deterministic (no network, no ML): the message is normalized,
each entry scores the keywords/phrases it contains (longer phrases score higher), and the best
entries win.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def normalize(text: str) -> str:
    """Lower-case, treat underscores as separators, drop punctuation, collapse whitespace."""
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.strip().lower())).strip()


@dataclass(frozen=True)
class FaqEntry:
    id: str
    title: str
    keywords: tuple[str, ...]
    answer: str
    related: tuple[str, ...] = ()


# Whole-word signs shipped in the Native Signs track (see the native_list entry); "how to sign HELLO" is answered with the
# Native Signs steps rather than the letter cues.
_NATIVE_GLOSSES = ("hello", "thank you", "thankyou", "please", "yes", "no", "mother", "water", "book", "help", "eat")

FAQ_ENTRIES: tuple[FaqEntry, ...] = (
    FaqEntry(
        id="what_is",
        title="What is ASL Quest?",
        keywords=(
            "what is asl quest", "about asl quest", "about this app", "about this project", "what is this app",
            "what is this project", "what does this app do", "what is this", "overview", "project summary",
            "tell me about", "purpose", "describe the project", "explain the project", "introduction",
        ),
        answer=(
            "ASL Quest is a gamified web platform for learning American Sign Language with webcam feedback. "
            "It has three learning tracks: A-Z (show a static letter to the camera), Word Spelling (spell a word "
            "one letter at a time) and Native Signs (record a short video of a whole-word sign such as HELLO or "
            "BOOK). On top of that it adds XP, levels, streaks, badges and challenges, plus analytics, an admin / "
            "mentor dashboard and AURA, this project assistant. It is built with React (frontend), FastAPI "
            "(backend), PyTorch + MediaPipe (recognition) and PostgreSQL or SQLite (database)."
        ),
        related=("features_pages", "getting_started", "tech_stack"),
    ),
    FaqEntry(
        id="capabilities",
        title="What can AURA do?",
        keywords=(
            "what can you do", "who are you", "what are you", "help me", "your name", "what do you know",
            "how can you help", "what can i ask", "what questions", "can you help",
        ),
        answer=(
            "I'm AURA, the ASL-Quest project assistant. I answer questions about this project only: how the "
            "A-Z, Word Spelling and Native Signs lessons work, XP / streaks / badges, the camera tips, the models "
            "and their accuracy, accounts and privacy, the database, the tech stack, testing and deployment. "
            "Try: \"How is XP awarded?\", \"Why isn't my sign recognized?\" or \"What database does it use?\"."
        ),
        related=("getting_started", "camera_tips", "xp_levels"),
    ),
    FaqEntry(
        id="getting_started",
        title="Where should I start / what should I learn next?",
        keywords=(
            "learn next", "should i learn", "what should i", "where do i start", "where to start",
            "get started", "getting started", "how to start", "how do i start", "start learning", "beginner",
            "beginners", "learning path", "next letter", "next sign", "first lesson", "what next", "learn first",
            "recommend", "recommendation", "recommendations", "suggest",
        ),
        answer=(
            "A good order: (1) Learn - study the A-Z alphabet, starting with the beginner letters C, D, F, H and "
            "O; (2) Practice - show each handshape to your webcam until it is confirmed (a letter is 'learning' "
            "after 3 correct attempts and 'mastered' after 10); (3) Words - spell whole words letter by letter; "
            "(4) Native Signs - record complete signs like HELLO, WATER or BOOK. Keep the daily goal (10 signs) "
            "to build a streak, and check the recommendations on your dashboard - they suggest letters based on "
            "your own practice results."
        ),
        related=("features_pages", "challenges", "mastery"),
    ),
    FaqEntry(
        id="features_pages",
        title="What pages and features does the app have?",
        keywords=(
            "pages", "features", "menu", "navigation", "sidebar", "what can i do", "sections", "tabs",
            "list of features", "main features", "app features", "what features", "page does",
        ),
        answer=(
            "The sidebar has: Learn (study the letters and your learning path), Practice (live webcam letter "
            "recognition), Alphabet (the A-Z overview), Words (spelling practice), Badges (achievements), Profile, "
            "and under More: Dashboard (home), Native Signs (record whole-word signs), Challenges (daily, speed, "
            "alphabet and word challenges), Progress (analytics), AURA (this assistant) and Settings. Admin and "
            "Mentor pages appear only for admin accounts. A small companion character, Signo, reacts to what you do."
        ),
        related=("challenges", "progress_analytics", "admin_mentor"),
    ),
    FaqEntry(
        id="a_to_z_how",
        title="How does A-Z recognition work?",
        keywords=(
            "a z", "how does recognition work", "recognition work", "resnet", "resnet18", "handshape",
            "handshapes", "how does it recognize", "how does it recognise", "how does it detect", "mediapipe",
            "how does mediapipe work", "hand crop", "classifier", "how are letters recognized", "letter recognition",
            "recognition", "recognize", "recognise", "image classifier", "landmark", "landmarks", "landmark recognizer",
            "landmark model", "hand landmarks", "21 points", "how does the camera know",
        ),
        answer=(
            "A-Z recognition works on your live camera: (1) your browser runs MediaPipe Hand Landmarker and finds "
            "21 landmark points on your hand (if no hand is found the app says so instead of guessing); (2) only "
            "those points - not your video - are sent to the server, where a small trained landmark classifier turns "
            "the hand shape into a letter A-Z with a confidence score; (3) a 0.70 confidence threshold and a vote over "
            "the last 7 readings (at least 3 must agree) smooth out flicker, so hold the sign steadily until it is "
            "confirmed. If the landmark model is not available on the server, Practice falls back to the older "
            "image classifier (a ResNet18 on a cropped photo of your hand). Only static handshapes are recognised; "
            "J and Z are practised as held shapes."
        ),
        related=("accuracy", "camera_tips", "limitations"),
    ),
    FaqEntry(
        id="accuracy",
        title="How accurate is it?",
        keywords=(
            "accuracy", "how accurate", "accurate", "precision", "recall", "f1", "how good", "performance",
            "metrics", "evaluation", "holdout", "test accuracy", "percent", "how well",
        ),
        answer=(
            "A-Z (older ResNet image model): the production checkpoint (a webcam-robustness fine-tune) scores 94.08% overall accuracy "
            "on an 11,700-image held-out set (macro precision 95.87%, macro F1 93.49%). The weakest letter is X "
            "(about 18% recall), then T, M and S. These numbers come from dataset images recorded in one setup, so "
            "they do not prove live-webcam accuracy - lighting, background and hand size matter. Native Signs: the "
            "frozen I3D model with a linear head reached 96.9% top-1 and 99.22% top-5 accuracy on the 10-sign test set. "
            "The newer landmark model that Practice uses was trained on about 9,900 webcam samples from several "
            "rooms; I do not have a published accuracy figure for it, so treat live results as the real test."
        ),
        related=("camera_tips", "native_model", "datasets_training"),
    ),
    FaqEntry(
        id="jz",
        title="Why are J and Z not supported?",
        keywords=(
            "j and z", "letter j", "letter z", "j z", "motion", "dynamic letters", "moving letters",
            "dynamic", "movement", "why not j", "why not z",
        ),
        answer=(
            "J and Z are drawn with a moving hand in real ASL, but the A-Z model looks at a single still image and "
            "has no temporal (motion) modelling. It may output a static guess for them, but they are not truly "
            "supported. Motion signs are what the Native Signs track (short videos) is for."
        ),
        related=("a_to_z_how", "native_how", "limitations"),
    ),
    FaqEntry(
        id="camera_tips",
        title="Camera or recognition problems",
        keywords=(
            "camera not working", "camera problem", "camera permission", "webcam not working", "no hand detected",
            "hand not detected", "not detecting", "not recognizing", "not recognising", "not recognized",
            "wrong prediction", "wrong letter", "lighting", "background", "tips", "improve", "low confidence",
            "why is it wrong", "wont recognize", "cant detect", "camera", "webcam", "permission", "blocked",
            "doesnt work", "not working", "black screen", "mirrored", "sign not recognized", "sign isnt recognized",
            "why is my sign not recognized", "my sign is not recognized", "not accepting my sign",
        ),
        answer=(
            "Camera / recognition checklist: (1) allow camera access when the browser asks (it only works on "
            "HTTPS or localhost) and close other apps using the camera; (2) use good, even lighting and a plain "
            "background; (3) keep your whole hand in frame, centred and not too far away; (4) hold the sign "
            "steady - a letter is confirmed only when the recent frames agree and confidence is at least 0.70; "
            "(5) compare with the reference hint on the Learn page. The model was trained with mirroring and "
            "lighting variations, but real rooms differ from the training photos, so results vary."
        ),
        related=("a_to_z_how", "confidence", "accuracy"),
    ),
    FaqEntry(
        id="confidence",
        title="Confidence threshold and smoothing",
        keywords=(
            "confidence", "threshold", "smoothing", "stable", "hold steady", "majority vote", "temporal smoothing",
            "why does it take long", "slow to confirm", "flicker", "0 70",
        ),
        answer=(
            "Live A-Z predictions need a confidence of at least 0.70, and a 7-frame majority-vote smoother "
            "requires the recent frames to agree before a letter is accepted. That prevents flicker and accidental "
            "matches, but it also means you must hold the sign steadily for a moment."
        ),
        related=("camera_tips", "a_to_z_how"),
    ),
    FaqEntry(
        id="word_spelling",
        title="How does Word Spelling work?",
        keywords=(
            "word spelling", "spell a word", "spelling", "how do words work", "word practice", "words",
            "word list", "word challenge", "which words", "spell words", "fingerspell", "fingerspelling",
            "letter by letter", "word categories",
        ),
        answer=(
            "Word Spelling is not whole-word sign recognition: you spell a word one letter at a time with the same "
            "A-Z recognition, and the app advances only when the current letter is recognized correctly. Words are "
            "grouped into Beginner, Greetings, Family, Everyday and School. XP: 50 for completing a word, +25 for a "
            "perfect word, +20 for a fast one, and 150 for finishing the 5-word Word Challenge (there are also daily "
            "spelling words)."
        ),
        related=("xp_levels", "challenges", "native_how"),
    ),
    FaqEntry(
        id="native_how",
        title="How do I practice a Native Sign?",
        keywords=(
            "native sign", "native signs", "how to practice native", "record", "recording", "video", "clip",
            "10 seconds", "countdown", "practice a sign", "practice native", "whole word sign", "native practice",
            "record a sign", "how long", "recording time", "stop recording", "native",
            *(f"how to sign {gloss}" for gloss in _NATIVE_GLOSSES), *(f"how do i sign {gloss}" for gloss in _NATIVE_GLOSSES),
        ),
        answer=(
            "Open Native Signs, pick a sign, watch the reference video, then press 'I'm ready'. Press Start "
            "recording: your camera preview appears with a 3-2-1 get-ready countdown, then it records for up to "
            "10 seconds (you can press Stop now as soon as you finish). The clip is analysed by the I3D model; "
            "an attempt counts as correct if the target sign is the top prediction or in the top 3, which awards "
            "+20 XP. Mastery for a sign is your running accuracy on it. The video is not stored after analysis."
        ),
        related=("native_list", "native_trouble", "native_model"),
    ),
    FaqEntry(
        id="native_list",
        title="Which Native Signs are available?",
        keywords=(
            "which signs", "list of signs", "what signs", "native vocabulary", "how many signs", "sign list",
            "available signs", "native categories", "vocabulary", "which native signs", "native signs available",
            "signs available", "available", "native", "signs are there", "signs can i", "all the signs",
        ),
        answer=(
            "There are 9 native signs: HELLO (Greetings), THANKYOU and PLEASE (Courtesy), YES and NO "
            "(Responses), MOTHER (Family), WATER and BOOK (Everyday), and HELP (Actions). Each has a "
            "short tip and a reference video to watch before you record. (EAT is in the model but not offered yet.)"
        ),
        related=("native_how", "native_model"),
    ),
    FaqEntry(
        id="native_model",
        title="How does Native Sign recognition work?",
        keywords=(
            "native model", "i3d", "backbone", "linear head", "asl citizen", "pretrained", "how does native",
            "native recognition", "native sign recognition", "native sign model", "video model", "frozen",
            "transfer learning", "native",
        ),
        answer=(
            "Native Signs uses Microsoft's ASL Citizen-pretrained I3D video network, kept frozen, with a small "
            "trained linear head (Dropout 0.5 then Linear 1024 to the 10 classes) on the temporally pooled "
            "features. Pretraining on a large ASL dataset made it generalise far better to unseen signers than "
            "earlier ResNet-based approaches: 96.9% top-1 / 99.22% top-5 on the native-10 test set. The clip is "
            "sent to POST /native/predict and correctness and XP are decided on the server."
        ),
        related=("native_how", "accuracy", "datasets_training"),
    ),
    FaqEntry(
        id="native_trouble",
        title="I can't pass a Native Sign lesson",
        keywords=(
            "cant complete", "cannot complete", "unable to complete", "lesson not completing", "not passing",
            "keep failing", "native sign not recognized", "native sign is not recognized", "native sign isnt recognized",
            "native sign not working", "not quite", "cant pass", "cannot pass", "failing",
            "doesnt recognize", "wont complete", "stuck", "book", "complete a lesson", "complete lesson",
            "native not working", "why wrong",
        ),
        answer=(
            "If a native sign keeps failing: (1) press 'Watch reference' and copy the exact motion - for BOOK, two "
            "flat hands open like covers; (2) keep both hands and your arms fully in view; (3) start the sign "
            "right after the countdown and finish before the 10 seconds end (or press Stop now); (4) use bright, "
            "even light and a plain background; (5) try again - an attempt passes when the target is the model's "
            "top prediction or in its top 3. The 'Top predictions' list on the result shows what the model saw."
        ),
        related=("native_how", "camera_tips", "native_list"),
    ),
    FaqEntry(
        id="xp_levels",
        title="How is XP awarded and how do levels work?",
        keywords=(
            "xp", "experience", "experience points", "level", "levels", "points", "how much xp", "earn xp",
            "level up", "how is xp", "xp values", "reward", "rewards",
        ),
        answer=(
            "XP: 20 per correct letter (30 if answered within 5 seconds), 20 per correct native sign, 100 for "
            "completing the daily challenge (150 for a perfect one), 150 for the speed challenge, 500 for the "
            "alphabet challenge, and for words 50 for completion, +25 perfect, +20 fast and 150 for the Word "
            "Challenge. Level = floor(XP / 1000) + 1, i.e. 1000 XP per level."
        ),
        related=("challenges", "streaks", "achievements"),
    ),
    FaqEntry(
        id="streaks",
        title="How do streaks work?",
        keywords=(
            "streak", "streaks", "consecutive", "daily streak", "lose streak", "7 day", "best streak",
            "streak reset", "break my streak", "day streak",
        ),
        answer=(
            "Your streak counts consecutive days on which you practice. Practising on the day after your last "
            "practice day adds 1; missing a day resets it to 1. The app also keeps your best (longest) streak, and "
            "a 7-day streak unlocks the '7 Day Streak' badge."
        ),
        related=("achievements", "xp_levels"),
    ),
    FaqEntry(
        id="mastery",
        title="What do 'learning' and 'mastered' mean?",
        keywords=(
            "mastery", "mastered", "master", "learning state", "how many correct", "mastered letter",
            "letters mastered", "how to master", "what is mastery",
        ),
        answer=(
            "For letters: a letter counts as 'learning' after 3 correct attempts and as 'mastered' after 10 "
            "(mastery percent = correct attempts / 10, capped at 100%). For Native Signs, mastery is simply your "
            "running accuracy on that sign (correct attempts / total attempts), and 100% counts as mastered."
        ),
        related=("xp_levels", "achievements", "native_how"),
    ),
    FaqEntry(
        id="achievements",
        title="Which badges can I earn?",
        keywords=(
            "achievement", "achievements", "badge", "badges", "unlock", "unlocked", "trophy", "which badges",
            "how to get badges", "list of badges", "alphabet master", "master badge", "get the badge",
        ),
        answer=(
            "There are 16 badges. Letters: First Sign, Alphabet Beginner (5 letters), Alphabet Explorer (13), "
            "Alphabet Master (all 26). Practice: 7 Day Streak, Speed Signer, Fast Signer (5 fast correct signs), "
            "Perfect Practice (daily challenge with no mistakes). Words: Word Starter, Word Learner (5 words), "
            "Word Explorer (3 categories), Word Master (10 mastered), Vocabulary Builder (25 words). Native: First "
            "Native Sign, Native Sign Explorer (3 categories) and Native Sign Master (every native sign mastered). "
            "A badge can only be earned once."
        ),
        related=("xp_levels", "streaks", "challenges"),
    ),
    FaqEntry(
        id="challenges",
        title="What challenges are there?",
        keywords=(
            "challenge", "challenges", "daily goal", "daily challenge", "speed challenge", "alphabet challenge",
            "word challenge", "daily", "game modes", "bonus xp", "daily target",
        ),
        answer=(
            "Challenges give bonus XP: the Daily Challenge (a goal of 10 correct signs, 100 XP, 150 XP when done "
            "with no mistakes), the Speed Challenge (5 signs within 45 seconds, 150 XP), the Alphabet Challenge "
            "(500 XP) and the Word Challenge (5 spelling words, 150 XP). Find them on the Challenges page."
        ),
        related=("xp_levels", "achievements", "word_spelling"),
    ),
    FaqEntry(
        id="progress_analytics",
        title="Progress page, analytics and recommendations",
        keywords=(
            "progress page", "analytics", "statistics", "stats", "charts", "heatmap", "dashboard", "insights",
            "weekly summary", "strongest letter", "weakest letter", "my progress", "progress", "history",
            "response time", "accuracy trend", "activity",
        ),
        answer=(
            "The Progress page and dashboard are built from your real practice data (never made-up numbers): "
            "overview, activity and accuracy trends, an activity heatmap, per-letter stats, response time, XP and "
            "streak analytics, strongest / weakest letters, a weekly summary, word analytics and a native-signs "
            "section once you have native progress. Recommendations suggest which letters to practice next."
        ),
        related=("getting_started", "streaks", "mastery"),
    ),
    FaqEntry(
        id="account",
        title="Accounts, login and passwords",
        keywords=(
            "register", "sign up", "signup", "create account", "login", "log in", "logout", "log out", "password",
            "forgot password", "reset password", "account", "username", "email", "cant login", "cannot login",
            "login failed", "invalid credentials", "sign in",
        ),
        answer=(
            "Create an account with a username, email and password; log in with your username or email. "
            "Passwords are stored as bcrypt hashes and sessions use JWT tokens that last 7 days. You can change "
            "your password in Settings. I don't have information about an emailed password-reset flow. If login "
            "shows 'Invalid username/email or password' re-check the details; 'Can't reach the server' means the "
            "backend may be waking up (free hosting sleeps) - wait about a minute and retry."
        ),
        related=("profile_settings", "privacy_security", "deployment"),
    ),
    FaqEntry(
        id="profile_settings",
        title="Profile and settings",
        keywords=(
            "profile", "settings", "preferences", "reduced motion", "export data", "export my data", "delete account",
            "download my data", "change password", "remove account", "delete my account",
        ),
        answer=(
            "Profile shows your level, XP, streaks, mastered letters, accuracy and strongest / weakest letters. "
            "Settings lets you set preferences (for example reduced motion), change your password, export your "
            "learning data as JSON (it excludes credentials) and delete your account (this needs your password and "
            "typing DELETE)."
        ),
        related=("account", "privacy_security"),
    ),
    FaqEntry(
        id="admin_mentor",
        title="Admin and Mentor dashboards",
        keywords=(
            "admin", "mentor", "mentor dashboard", "teacher", "role", "roles", "bootstrap admin", "administrator",
            "student role", "who is admin",
        ),
        answer=(
            "Users are 'student' by default. Admin accounts see extra pages: Admin (system totals, user list, "
            "activity, popular and difficult letters / words) and the Mentor Dashboard (learner analytics). The "
            "first admin is created by setting ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL to the email used at registration; "
            "admin routes are enforced on the server."
        ),
        related=("progress_analytics", "env_config"),
    ),
    FaqEntry(
        id="privacy_security",
        title="Privacy and security",
        keywords=(
            "privacy", "security", "safe", "secure", "data stored", "are frames stored", "is my video saved",
            "camera data", "bcrypt", "jwt", "password hashing", "cors", "is my data safe", "stored", "saved",
            "recorded", "do you store", "video stored", "video saved", "clip stored", "clip saved", "videos stored",
            "frames stored", "photos stored", "do you keep", "delete my video", "is it recorded", "who can see",
        ),
        answer=(
            "Webcam frames and native-sign clips are not stored: a clip is written to a temporary file for "
            "analysis and deleted straight away, and only the result (correct / incorrect, confidence, timing) is "
            "saved. Passwords are bcrypt-hashed, API access uses JWT bearer tokens, users can only read their own "
            "data, admin routes are checked on the server, and exports exclude credentials."
        ),
        related=("account", "profile_settings"),
    ),
    FaqEntry(
        id="tech_stack",
        title="Technology stack",
        keywords=(
            "tech stack", "technology", "technologies", "built with", "framework", "react", "fastapi", "pytorch",
            "tailwind", "vite", "python", "stack", "typescript", "libraries", "what is it made with",
        ),
        answer=(
            "Frontend: React 19 + TypeScript, Vite, Tailwind CSS and Recharts. Backend: FastAPI with SQLAlchemy and "
            "Alembic, JWT auth and bcrypt. Machine learning: PyTorch (ResNet18 for letters, I3D for native signs) "
            "and MediaPipe for hand detection. Database: SQLite locally, PostgreSQL in production. Tests: pytest "
            "and Vitest. Hosting: Vercel (frontend) and Render (backend and database)."
        ),
        related=("architecture", "database", "deployment"),
    ),
    FaqEntry(
        id="architecture",
        title="How is the system structured?",
        keywords=(
            "architecture", "how is it structured", "how does the app work", "frontend backend", "system design",
            "components", "folder", "structure", "project structure", "flow", "how it works", "data flow",
            "backend", "frontend",
        ),
        answer=(
            "The React frontend talks to a FastAPI backend over HTTPS with a JWT bearer token. The backend has "
            "routers (auth, progress, challenges, achievements, practice, analytics, users, recommendations, admin, "
            "words, native, chatbot) backed by service modules, SQLAlchemy models and the ML code in src/ "
            "(inference for A-Z, i3d_transfer for native signs). The server database is the source of truth for "
            "progress; the browser only keeps an offline cache."
        ),
        related=("api_endpoints", "database", "tech_stack"),
    ),
    FaqEntry(
        id="api_endpoints",
        title="API endpoints",
        keywords=(
            "api", "endpoint", "endpoints", "routes", "swagger", "docs", "openapi", "predict", "rest", "http",
            "api docs", "route list",
        ),
        answer=(
            "Main endpoints: /auth (register, login, me), /progress, /challenges, /achievements, /practice/session, "
            "/words, /native (signs, progress, predict), /analytics/*, /users/me/* (profile, preferences, export), "
            "/recommendations, /admin/*, /chatbot/* and the recognition endpoints /health, /predict and /history. "
            "Interactive documentation is served by the backend at /docs."
        ),
        related=("architecture", "account"),
    ),
    FaqEntry(
        id="database",
        title="Database",
        keywords=(
            "database", "postgres", "postgresql", "sqlite", "tables", "schema", "migration", "migrations",
            "alembic", "sqlalchemy", "where is data stored", "db", "orm", "data model", "database store",
            "store my progress", "save my progress", "progress saved", "where is my progress", "which database",
            "what database",
        ),
        answer=(
            "The app uses SQLAlchemy models and Alembic migrations. Locally it defaults to SQLite (data/asl_quest.db, "
            "no setup); in production set DATABASE_URL to PostgreSQL (postgresql+psycopg://...). Tables include "
            "users, user_progress, letter_progress, challenge_progress, achievements, practice_sessions, xp_events, "
            "words, word_progress, native_signs, native_sign_progress, native_sign_sessions and the chatbot tables. "
            "Run 'alembic upgrade head' to create or update the schema."
        ),
        related=("deployment", "env_config", "architecture"),
    ),
    FaqEntry(
        id="deployment",
        title="Deployment (Vercel, Render, PostgreSQL)",
        keywords=(
            "deploy", "deployed", "deployment", "vercel", "render", "hosting", "production", "live site", "host",
            "online", "go live", "publish", "free tier", "sleep", "slow first request", "cold start",
        ),
        answer=(
            "Deployment: the frontend is a Vite build on Vercel (root directory 'frontend', variable "
            "VITE_API_BASE_URL set to the backend URL); the backend runs on Render as 'uvicorn backend.main:app' "
            "with a Render PostgreSQL database. The backend needs DATABASE_URL, ASL_QUEST_SECRET_KEY, "
            "ASL_QUEST_CORS_ORIGINS (the frontend's exact origin) and optionally OPENROUTER_API_KEY. On free plans "
            "the backend sleeps when idle (the first request can take about a minute) and the model files are not "
            "in Git, so letter and sign recognition need them added on the server and more memory than the smallest plan."
        ),
        related=("env_config", "model_files", "database"),
    ),
    FaqEntry(
        id="env_config",
        title="Configuration and environment variables",
        keywords=(
            "environment variable", "environment variables", "env", "configuration", "config", "secret key",
            "api key", "openrouter key", "cors origins", "settings file", "dotenv", "env file",
        ),
        answer=(
            "Key settings (set as environment variables, never committed): DATABASE_URL (database), "
            "ASL_QUEST_SECRET_KEY (JWT signing - set a long random value in production), ASL_QUEST_CORS_ORIGINS "
            "(allowed frontend origins), ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL (first admin), OPENROUTER_API_KEY and "
            "CHATBOT_MODEL (AURA online mode), CHATBOT_PROVIDER = openrouter / local / auto, and for the "
            "frontend VITE_API_BASE_URL (build time)."
        ),
        related=("deployment", "chatbot_aura"),
    ),
    FaqEntry(
        id="chatbot_aura",
        title="About AURA (the chatbot)",
        keywords=(
            "aura", "chatbot", "assistant", "ollama", "openrouter", "offline mode", "offline ai", "knowledge base",
            "why offline", "offline knowledge", "ai model", "llm", "bot", "chat",
        ),
        answer=(
            "AURA only answers questions about ASL Quest: a scope check on the server rejects anything unrelated "
            "before any AI model is called. With an OpenRouter API key it uses an online AI model; it can also use a "
            "local Ollama model (CHATBOT_PROVIDER=local or auto); and when no model is reachable it answers from "
            "this built-in offline knowledge base - the badge under the title says which one answered."
        ),
        related=("env_config", "capabilities"),
    ),
    FaqEntry(
        id="limitations",
        title="Limitations and future work",
        keywords=(
            "limitation", "limitations", "limits", "cannot do", "not supported", "future", "future work", "roadmap ideas",
            "weakness", "weaknesses", "known issues", "what doesnt work", "drawbacks", "improvements", "constraints",
        ),
        answer=(
            "Known limitations: A-Z handles static letters only (J and Z involve motion); Word Spelling is letter "
            "spelling, not word-level recognition; Native Signs covers just 10 signs; the A-Z model was validated "
            "on dataset images, so real-webcam accuracy varies with lighting, background and hand size (X is the "
            "weakest letter). Future work: more native signs and dynamic letters, retraining on real webcam data, "
            "and automated model delivery for deployment."
        ),
        related=("accuracy", "jz", "datasets_training"),
    ),
    FaqEntry(
        id="testing",
        title="Testing",
        keywords=(
            "test", "tests", "testing", "pytest", "vitest", "unit tests", "how is it tested", "quality",
            "test suite", "run tests", "ci",
        ),
        answer=(
            "Backend tests run with pytest (auth and database, progress sync, analytics, words, native signs, "
            "chatbot, live pipeline, stability, landmarks and more); frontend tests run with Vitest and Testing "
            "Library (pages, camera states, dialogs, auth requests). 'npm run build' also type-checks the frontend "
            "with TypeScript."
        ),
        related=("setup_run", "tech_stack"),
    ),
    FaqEntry(
        id="datasets_training",
        title="Datasets and training",
        keywords=(
            "dataset", "datasets", "training", "train", "trained", "images", "asl alphabet", "asl citizen dataset",
            "how was the model trained", "augmentation", "training data", "data", "fine tune", "finetune",
        ),
        answer=(
            "A-Z: trained on the ASL Alphabet image set - 26 folders of 3,000 images (78,000 images) - and later "
            "fine-tuned for 3 epochs with robust augmentation (mirroring, scale / rotation, lighting, blur, noise, "
            "occlusion) to cope with real webcams. Native Signs: uses the ASL Citizen video dataset (licensed by "
            "Microsoft Research, never stored in this repository) only to train the small linear head on a frozen "
            "pretrained I3D. Training scripts are in train.py and scripts/."
        ),
        related=("accuracy", "native_model", "model_files"),
    ),
    FaqEntry(
        id="setup_run",
        title="How do I run the project locally?",
        keywords=(
            "run locally", "run it locally", "run this locally", "locally", "setup", "install", "how to run",
            "start the app", "localhost", "npm run dev", "uvicorn",
            "windows", "run the project", "run the app", "start the backend", "start the frontend", "get it running",
        ),
        answer=(
            "Locally: (1) create the Python 3.11 environment and install requirements.txt (scripts/windows/setup.ps1 "
            "does this on Windows); (2) copy the model files into place; (3) run 'alembic upgrade head' and seed the "
            "native signs; (4) start the backend with 'uvicorn backend.main:app --reload' (docs at /docs); (5) in "
            "frontend/ run 'npm install' then 'npm run dev' and open http://localhost:5173."
        ),
        related=("model_files", "database", "testing"),
    ),
    FaqEntry(
        id="model_files",
        title="Model files",
        keywords=(
            "model files", "best model", "pth", "weights", "checkpoint", "hand landmarker", "download model",
            "missing model", "model not loaded", "no trained model", "model not found", "degraded",
        ),
        answer=(
            "Recognition needs four files that are deliberately not in Git: models/best_model.pth (A-Z ResNet18, "
            "~43 MB), models/hand_landmarker.task (MediaPipe, ~7.5 MB), the ASL Citizen I3D weights (~58 MB) and "
            "the trained native head head_best.pt. Without them the backend still starts (health shows 'degraded', "
            "model_loaded false) but letter and sign recognition are unavailable. See deployment/README.md."
        ),
        related=("setup_run", "deployment", "datasets_training"),
    ),
    FaqEntry(
        id="signo",
        title="Who is Signo?",
        keywords=("signo", "mascot", "companion", "character", "little character", "green character"),
        answer=(
            "Signo is the small on-screen companion in ASL Quest. It reacts to what you do - greeting you, "
            "cheering correct signs and celebrating completed words - to keep practice friendly. It is separate "
            "from AURA, which answers project questions."
        ),
        related=("capabilities", "features_pages"),
    ),
    FaqEntry(
        id="reference_videos",
        title="Reference videos",
        keywords=(
            "reference video", "reference videos", "watch", "demonstration", "tutorial video", "how to sign",
            "show me the sign", "learn the sign", "video reference", "reference coming soon",
        ),
        answer=(
            "Each native sign page shows a reference demonstration to watch before you copy the sign, when an "
            "approved external source has been added (the ASL Citizen videos themselves are never redistributed). "
            "Otherwise it shows a 'Reference video coming soon' placeholder with the sign's meaning and description. "
            "For letters, the Learn page gives a written hint for each handshape."
        ),
        related=("native_how", "native_trouble"),
    ),
    FaqEntry(
        id="offline_use",
        title="Using the app offline",
        keywords=(
            "offline", "without internet", "no internet", "offline use", "work offline", "sync", "syncing",
            "local storage", "cache",
        ),
        answer=(
            "Progress lives in the server database; the browser keeps an offline cache, and word progress made "
            "while offline is saved locally and synced when you reconnect. Recognition needs the backend, so it "
            "does not work without a connection to it. AURA can run fully offline with a local Ollama model or "
            "from its built-in knowledge base."
        ),
        related=("chatbot_aura", "architecture"),
    ),
    FaqEntry(
        id="practice_how",
        title="How does Practice work?",
        keywords=(
            "how do i practice", "how does practice work", "how to practice", "practice page", "practice mode",
            "start practice", "practice a letter", "practice letters", "how practice works", "practice session",
            "how do i use practice", "how do i use the camera", "how do i play", "how do i use this",
        ),
        answer=(
            "Open Practice from the sidebar. It shows a target letter with its reference handshape. Allow camera "
            "access, put your hand in view and make the sign; the app reads your hand live and, once the same letter "
            "is held steadily (confidence of at least 0.70 and at least 3 of the last 7 readings agreeing), it "
            "confirms it. A correct letter earns 20 XP (30 if you answer within 5 seconds) and updates your mastery; "
            "a wrong one tells you which letter it saw so you can adjust. Use the Learn page first if you have not "
            "seen a letter yet."
        ),
        related=("camera_tips", "xp_levels", "mastery"),
    ),
    FaqEntry(
        id="alphabet_overview",
        title="How many letters are there, and how do I learn the alphabet?",
        keywords=(
            "alphabet", "how many letters", "number of letters", "teach me the alphabet", "learn the alphabet",
            "all letters", "a to z letters", "full alphabet", "list of letters", "fingerspelling", "fingerspell",
        ),
        answer=(
            "The A-Z track covers all 26 letters. The Alphabet page shows every handshape with a reference picture "
            "and a one-line cue, the Learn page walks you through them (beginner letters C, D, F, H and O first), "
            "and Practice checks your sign on the camera. A letter counts as 'learning' after 3 correct attempts and "
            "'mastered' after 10. J and Z are drawn with movement in real ASL; here they are practised as held shapes."
        ),
        related=("getting_started", "jz", "mastery"),
    ),
    FaqEntry(
        id="hard_letters",
        title="Which letters are the hardest?",
        keywords=(
            "which letters are hard", "hardest letters", "hardest letter", "difficult letters", "hard letters",
            "which letters are difficult", "tricky letters", "letters i struggle", "most confused", "confused letters",
            "similar letters", "look alike", "look similar",
        ),
        answer=(
            "The fist-like letters are the easiest to mix up: A, E, M, N, S and T differ only by where the thumb "
            "sits or how many fingers cross it (for example M has three fingers over the thumb, N has two). In the "
            "image-model evaluation the weakest letters were X, T, M and S. J and Z are drawn with movement in real "
            "ASL, so they are practised as held shapes here. Your own weakest letters are listed on the Progress and "
            "Profile pages, and the Learn page recommends letters to revisit."
        ),
        related=("camera_tips", "progress_analytics", "jz"),
    ),
    FaqEntry(
        id="improve_accuracy",
        title="How can I improve my accuracy?",
        keywords=(
            "improve my accuracy", "improve accuracy", "increase accuracy", "better accuracy", "get better",
            "how can i improve", "how do i improve", "how to improve", "do better", "my accuracy is low",
            "accuracy is low", "low accuracy", "raise my accuracy",
        ),
        answer=(
            "To raise your accuracy: (1) match the reference handshape shown on the Practice or Alphabet page "
            "exactly; (2) use even front lighting, a plain background, and keep your whole hand in view at about an "
            "arm's length; (3) hold the sign still for a moment until it is confirmed; (4) repeat your weakest "
            "letters - they are listed on the Progress and Profile pages, and the Learn page recommends them; "
            "(5) practise a little every day so your streak and mastery build up."
        ),
        related=("camera_tips", "hard_letters", "progress_analytics"),
    ),
    FaqEntry(
        id="asl_basics",
        title="What is ASL?",
        keywords=(
            "what is asl", "what does asl mean", "asl mean", "american sign language", "is asl universal",
            "asl vs bsl", "asl and bsl", "same as bsl", "bsl", "sign language the same", "is sign language universal",
            "what is sign language", "deaf",
        ),
        answer=(
            "ASL is American Sign Language, a complete natural language with its own grammar, used mainly by Deaf "
            "communities in the United States and parts of Canada. It is not signed English, and sign language is not "
            "universal: British Sign Language (BSL), for example, is a different language from ASL, with a different "
            "fingerspelling alphabet. ASL Quest teaches the ASL fingerspelling alphabet, spelled words and a small "
            "set of whole-word signs; it is a practice tool, not a substitute for learning from Deaf signers."
        ),
        related=("what_is", "getting_started", "limitations"),
    ),
    FaqEntry(
        id="device_support",
        title="Can I use it on my phone or another device?",
        keywords=(
            "on my phone", "use it on my phone", "on mobile", "mobile phone", "on a phone", "on my tablet",
            "which browser", "supported browser", "supported browsers", "does it work on", "what devices",
            "laptop camera", "external camera",
        ),
        answer=(
            "You need a device with a camera and a modern browser that allows camera access, on HTTPS or "
            "localhost. Hand tracking runs inside the browser (MediaPipe), so a reasonably recent phone, tablet or "
            "laptop should work, but I do not have phone-specific test results - older or low-power devices may "
            "track more slowly. Allow the camera permission when asked and keep your hand fully in view."
        ),
        related=("camera_tips", "a_to_z_how", "offline_use"),
    ),
)

# One-line handshape cues, the same wording the Learn and Alphabet pages show (frontend/src/pages/alphabetFeedback.ts).
LETTER_HINTS: dict[str, str] = {
    "A": "Make a fist with thumb alongside.",
    "B": "Four fingers up, thumb tucked in.",
    "C": "Curve your hand like the letter C.",
    "D": "Index up, other fingers curled to thumb.",
    "E": "All fingertips touch thumb.",
    "F": "Index and thumb form a circle, other fingers up.",
    "G": "Index and thumb point sideways.",
    "H": "Index and middle finger point sideways.",
    "I": "Pinky up, other fingers in a fist.",
    "J": "Pinky up - hold the J shape.",
    "K": "Index and middle in a V, thumb between them.",
    "L": "Thumb and index form an L.",
    "M": "Three fingers over thumb in fist.",
    "N": "Two fingers over thumb in fist.",
    "O": "Fingers and thumb form a circle.",
    "P": "Like K but pointing down.",
    "Q": "Like G but pointing down.",
    "R": "Index and middle crossed.",
    "S": "Closed fist with thumb over fingers.",
    "T": "Thumb tucked between index and middle.",
    "U": "Index and middle up together.",
    "V": "Index and middle spread in a V.",
    "W": "Three fingers up in a W shape.",
    "X": "Index finger bent like a hook.",
    "Y": "Thumb and pinky extended.",
    "Z": "Index points - hold the Z shape.",
}

# A single letter is only read as a letter in these shapes, so the article "a" in "what is a streak" is never mistaken for A.
_LETTER = r"(?:the )?(?:letter )?([a-z])"
_SINGLE_LETTER_PATTERNS = tuple(
    re.compile(p)
    for p in (
        rf"\bletter ([a-z])\b",
        rf"\bsign {_LETTER}$",
        rf"\bsigning {_LETTER}$",
        rf"\bwhat does {_LETTER} look like\b",
        rf"\bhow (?:do i|to|can i|would i) (?:make|form|show|do|sign|fingerspell) {_LETTER}$",
        rf"\bhow (?:do i|to|can i) (?:make|form|show|do|sign) {_LETTER} (?:in asl|with my hand)$",
    )
)
_PAIR_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\bdifference between (?:the )?(?:letters? )?([a-z]) and (?:the )?(?:letters? )?([a-z])\b",
        r"\b([a-z]) (?:vs|versus|or|and) ([a-z])\b",
    )
)


def letter_answer(normalized_message: str) -> str | None:
    """Answer 'how do I sign X', 'what does X look like' or 'M vs N' from the app's own per-letter cues; None if not such a question.

    Deterministic and exact: the cue text is the same one the Learn and Alphabet pages show, so the assistant never
    describes a handshape differently from the app. Pair patterns are limited to short messages so an ordinary sentence
    containing 'a and b' is not misread as a letter comparison.
    """
    words = normalized_message.split()
    if len(words) <= 9:
        for pattern in _PAIR_PATTERNS:
            found = pattern.search(normalized_message)
            if found and found.group(1) != found.group(2) and (len(words) <= 4 or "between" in normalized_message):
                first, second = (found.group(1).upper(), found.group(2).upper())
                return (
                    f"{first}: {LETTER_HINTS[first]}\n{second}: {LETTER_HINTS[second]}\n\n"
                    f"The Alphabet page shows both reference pictures side by side, which is the quickest way to see the difference."
                )
    for pattern in _SINGLE_LETTER_PATTERNS:
        found = pattern.search(normalized_message)
        if found:
            letter = found.group(1).upper()
            text = f"{letter}: {LETTER_HINTS[letter]} The Alphabet and Learn pages show the reference picture for {letter}, and Practice checks it on your camera."
            if letter in ("J", "Z"):
                text += f" In real ASL {letter} is drawn with a movement; here it is practised as a held shape."
            return text
    return None

_BY_ID: dict[str, FaqEntry] = {entry.id: entry for entry in FAQ_ENTRIES}


def _compile(entry: FaqEntry) -> list[tuple[re.Pattern[str], int]]:
    compiled: list[tuple[re.Pattern[str], int]] = []
    for keyword in entry.keywords:
        norm = normalize(keyword)
        if not norm:
            continue
        # Single words weigh 2, each extra word adds 1: phrases are stronger evidence than lone words.
        weight = len(norm.split()) + 1
        compiled.append((re.compile(r"\b" + re.escape(norm) + r"\b"), weight))
    return compiled


_PATTERNS: dict[str, list[tuple[re.Pattern[str], int]]] = {entry.id: _compile(entry) for entry in FAQ_ENTRIES}

MIN_SCORE = 2


def score_entries(normalized_message: str) -> list[tuple[int, FaqEntry]]:
    """All entries with a positive score, best first (ties keep the file's order)."""
    scored: list[tuple[int, int, FaqEntry]] = []
    for order, entry in enumerate(FAQ_ENTRIES):
        total = sum(weight for pattern, weight in _PATTERNS[entry.id] if pattern.search(normalized_message))
        if total > 0:
            scored.append((total, order, entry))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(score, entry) for score, _, entry in scored]


def best_matches(normalized_message: str, limit: int = 2) -> list[FaqEntry]:
    """The best answer, plus a second one only when it is almost as relevant."""
    ranked = score_entries(normalized_message)
    if not ranked or ranked[0][0] < MIN_SCORE:
        return []
    top_score = ranked[0][0]
    chosen = [ranked[0][1]]
    for score, entry in ranked[1:limit]:
        if score >= max(MIN_SCORE + 1, top_score - 1):
            chosen.append(entry)
    return chosen


def related_titles(entry: FaqEntry, exclude: set[str]) -> list[str]:
    return [_BY_ID[rid].title for rid in entry.related if rid in _BY_ID and rid not in exclude][:3]


def topic_suggestions(count: int = 6) -> list[str]:
    preferred = (
        "getting_started", "camera_tips", "xp_levels", "native_how", "achievements", "database", "deployment",
    )
    return [_BY_ID[eid].title for eid in preferred[:count]]
