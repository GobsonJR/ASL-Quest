# ASL Quest — Complete Project Documentation

> **Purpose of this file.** Anyone who reads this document end to end should know what this project is, how every part
> works, how it was built, what was measured, what is known to be weak, and how to run, test and extend it. It was written
> from the actual repository (code read, commands run, numbers measured) at the state of branch `david-dev`, last
> commit `869e187`, plus the uncommitted work described in [section 26](#26-what-changed-in-the-latest-working-sessions-uncommitted).
>
> **Honesty convention used throughout.** Statements marked **MEASURED** come from a script or test you can rerun.
> **BY DESIGN** means it is a deliberate rule in the code. **UNVERIFIED** means it was not tested here. **KNOWN ISSUE** is a
> real weakness that exists today. Nothing below is aspirational; if something is planned but not built, it says so.

---

## Table of contents

1. [What ASL Quest is](#1-what-asl-quest-is)
2. [The three learning modes at a glance](#2-the-three-learning-modes-at-a-glance)
3. [Technology stack](#3-technology-stack)
4. [Repository layout](#4-repository-layout)
5. [Architecture and data flow](#5-architecture-and-data-flow)
6. [Backend (FastAPI)](#6-backend-fastapi)
7. [Database](#7-database)
8. [Authentication, roles and security](#8-authentication-roles-and-security)
9. [Frontend (React + Vite)](#9-frontend-react--vite)
10. [Gamification: XP, levels, streaks, badges, challenges](#10-gamification-xp-levels-streaks-badges-challenges)
11. [Alphabet (A–Z) recognition in depth](#11-alphabet-az-recognition-in-depth)
12. [Landmark recognizer in depth (browser MediaPipe + classifier)](#12-landmark-recognizer-in-depth-browser-mediapipe--classifier)
13. [Word Spelling](#13-word-spelling)
14. [Native Signs (whole-word video recognition)](#14-native-signs-whole-word-video-recognition)
15. [AURA chatbot and SIGNO companion](#15-aura-chatbot-and-signo-companion)
16. [Admin, mentor and analytics](#16-admin-mentor-and-analytics)
17. [Data collection and datasets](#17-data-collection-and-datasets)
18. [Model training and the experiment history](#18-model-training-and-the-experiment-history)
19. [Measured results and what they mean](#19-measured-results-and-what-they-mean)
20. [Investigations and findings (M/T/Z, handedness, detection)](#20-investigations-and-findings-mtz-handedness-detection)
21. [Configuration reference (environment variables)](#21-configuration-reference-environment-variables)
22. [Setup and run commands](#22-setup-and-run-commands)
23. [Testing](#23-testing)
24. [Deployment](#24-deployment)
25. [Known issues, limitations and open decisions](#25-known-issues-limitations-and-open-decisions)
26. [What changed in the latest working sessions (uncommitted)](#26-what-changed-in-the-latest-working-sessions-uncommitted)
27. [Operating rules and machine constraints](#27-operating-rules-and-machine-constraints)
28. [Glossary](#28-glossary)
29. [Where to look for what (cheat sheet)](#29-where-to-look-for-what-cheat-sheet)

---

## 1. What ASL Quest is

ASL Quest is a gamified web application for learning American Sign Language using the learner's webcam. It is a
Duolingo-style experience: a learner picks something to practice, a reference is shown, the learner signs in front of the
camera, the app recognizes the sign, and the learner gets XP, streaks, badges, a level and visible mastery progress.

It is a full-stack system, not a notebook: a React single-page app, a FastAPI backend, a relational database (SQLite locally,
PostgreSQL in production), computer-vision models (PyTorch ResNet18, MediaPipe hand landmarks, a landmark classifier built
with scikit-learn, and a frozen I3D video network with a trained linear head), an offline-capable AI assistant (AURA), a
small in-app companion (SIGNO), admin and mentor dashboards, and a development-only webcam data-collection tool.

Who it is for: learners practicing ASL fingerspelling and a small set of whole-word signs; mentors/admins who want to see
how learners are doing; and developers who want to evaluate recognition models on real webcam data.

What it recognizes today:

| Mode | Input | Recognizer | Vocabulary |
|---|---|---|---|
| Alphabet | single hand pose from the live camera | MediaPipe hand landmarks in the **browser** + a landmark classifier on the server (ResNet18 image classifier is the fallback) | A–Z static handshapes |
| Words | same, letter by letter | same as Alphabet | 24 words spelled with letters |
| Native Signs | a 10-second recorded video clip | frozen I3D video backbone + trained linear head | 10 whole-word signs |

**Not supported by design (and stated honestly in the code):** J and Z involve motion in real ASL. The alphabet recognizer
looks at one still pose, so it cannot truly recognize their motion; they are practiced as held shapes.

---

## 2. The three learning modes at a glance

1. **Alphabet (Learn / Alphabet / Practice pages).** A Duolingo-style *learning path* of 26 letters. Five beginner letters
   (C, D, F, H, O) are unlocked first. The learner opens a letter, sees a lesson intro and a reference image, then practices with
   the camera. The app confirms a letter only after several consistent, confident readings (a vote window), then awards XP.
2. **Word Spelling (Words page).** The learner spells a word one letter at a time using the same camera recognizer. Words are
   grouped in five categories, with tips, bonus XP for perfect/fast spelling, mastery after 3 completions, a word challenge and
   word achievements.
3. **Native Signs (Native Signs page).** Whole ASL signs rather than letters. Flow: **Watch** the reference video → **Copy**
   (a 3-2-1 countdown, then a 10-second recording) → **Check** (the server classifies the clip with a video model). A live MediaPipe
   hand overlay is drawn on the preview while recording. Correct answers earn XP and increase per-sign mastery.

Cross-cutting features: profile, settings, progress analytics, achievements, daily/alphabet/speed/word challenges, the AURA
assistant, the SIGNO companion, and the admin/mentor area.

---

## 3. Technology stack

### 3.1 Backend and ML (Python)

| Layer | Technology | Where / why |
|---|---|---|
| Web framework | **FastAPI** ≥0.110, **uvicorn[standard]** ≥0.27 | `backend/main.py` — async API, automatic OpenAPI docs at `/docs` |
| Validation | **Pydantic** (via FastAPI) | request/response models in `backend/schemas.py`, `LandmarkPayload` in `ml.py` |
| ORM / migrations | **SQLAlchemy** ≥2.0, **Alembic** ≥1.13 | `backend/models.py`, `migrations/versions/*` |
| Databases | **SQLite** (local default), **PostgreSQL** via `psycopg[binary]` ≥3.1 | selected by `DATABASE_URL` |
| Auth | **bcrypt** (password hashing), **python-jose** (JWT HS256) | `backend/security.py` |
| HTTP client | **httpx** | AURA calls to LLM providers |
| Email validation | **email-validator** | registration |
| Uploads | **python-multipart** | image/video upload endpoints |
| Deep learning | **PyTorch** ≥2.1, **torchvision** ≥0.16 | ResNet18 (alphabet), I3D (native signs) |
| Hand tracking | **MediaPipe** ≥0.10 (Python, Hand Landmarker Tasks API) | server-side hand detection for crops and image-path landmarks |
| Classical ML | **scikit-learn** ≥1.4, **joblib** | landmark classifier (LogReg / MLP / HistGradientBoosting pipelines) |
| Imaging | **Pillow**, **opencv-python**, **numpy**, **matplotlib**, **tqdm** | preprocessing, scripts, plots |
| Tests | **pytest**, FastAPI `TestClient`, `unittest` classes | `tests/` |

Python version in use: **3.11** (virtual environment `venv311`). A second venv `venv` also exists in the folder.

### 3.2 Frontend (TypeScript)

| Layer | Technology |
|---|---|
| UI | **React 19** + **TypeScript 5.8** |
| Build / dev server | **Vite 7** (`@vitejs/plugin-react`), dev proxy `/api → http://127.0.0.1:8000` |
| Styling | **Tailwind CSS 4** (`@tailwindcss/vite`), design tokens as CSS variables in `src/index.css` |
| Charts | **Recharts 3** |
| Browser hand tracking | **`@mediapipe/tasks-vision` 1.1.1** (Apache-2.0) — `HandLandmarker`, VIDEO mode |
| Embedded video | **`@vimeo/player` 2.30.4** (MIT) — readiness/error detection for Vimeo embeds |
| Tests | **Vitest 5**, **@testing-library/react 16**, **jest-dom**, **jsdom** |

### 3.3 External services and assets

| Asset / service | Role | Notes |
|---|---|---|
| `models/best_model.pth` | production ResNet18 alphabet checkpoint (~43 MB) | git-ignored, never overwritten |
| `models/hand_landmarker.task` | MediaPipe Hand Landmarker bundle (~7.5 MB, SHA-256 `fbc2a300…cde1`) | git-ignored; **licence of the model file itself is not stated by Google's public pages** — see §25 |
| `models/pretrained/asl_citizen_i3d/ASL_citizen_I3D_weights.pt` | frozen I3D backbone (~58 MB), MIT, Microsoft Research | provenance in its `SOURCE.md` |
| `outputs/asl_citizen_native_10/experiments/i3d_frozen/checkpoints/head_best.pt` | trained linear head for 10 native signs (~43 KB) | git-ignored |
| `models/experiments/landmark_L1_s0.joblib`, `landmark_L2_s0.joblib` | experimental landmark classifiers | **L2 is what the app uses when configured** |
| OpenRouter API | optional LLM for AURA (`OPENROUTER_API_KEY`) | OpenAI-compatible chat completions |
| Ollama (local) | optional offline LLM for AURA (`qwen3:4b`) | `CHATBOT_PROVIDER=local/auto` |
| jsDelivr CDN | MediaPipe WASM runtime for the browser | pinned `@mediapipe/tasks-vision@1.1.1/wasm`, overridable |
| Google storage | default MediaPipe model URL for the browser | `…/hand_landmarker/float16/latest/hand_landmarker.task`, overridable |
| Vimeo | reference sign videos (9 of 10 native signs) | unlisted videos with privacy hash |
| ASL Citizen dataset | native sign training videos | **licensed separately; never stored in or redistributed by the repo** |
| Kaggle ASL Alphabet dataset | ResNet18 training photos | `dataset/asl_alphabet_train/` (200×200 close-ups, one signer/setup) |
| Vercel / Render | deployment targets | frontend on Vercel, backend + Postgres on Render (per config) |

---

## 4. Repository layout

```
D:\SIGN LANGUAGE\
├── backend/                  FastAPI application
│   ├── main.py               app factory, CORS, router registration, startup (DB init, seed, model load)
│   ├── settings.py           all environment-driven configuration
│   ├── database.py           engine/session, init_db
│   ├── models.py             SQLAlchemy ORM tables (see §7)
│   ├── schemas.py            Pydantic schemas (GameState etc.)
│   ├── security.py           bcrypt, JWT, get_current_user / get_current_admin
│   ├── routers/              ml, auth, progress, challenges, achievements, practice, analytics,
│   │                         users, recommendations, admin, admin_ops, words, native, chatbot, collect
│   ├── services/             progress, words, analytics, admin_stats, mentor_dashboard, user_features,
│   │                         chatbot_{llm,local_llm,faq,knowledge,exceptions}, experimental_recognizer,
│   │                         live_diagnostics, webcam_dataset, native_references
│   └── words/catalog.py      static word catalog (24 words)
├── src/                      ML code (importable as `src.*`)
│   ├── config.py             paths and ML constants
│   ├── data/                 dataset, preprocessing/augmentation, webcam export
│   ├── models/resnet18.py    build_resnet18
│   ├── training/             train.py, evaluate.py
│   ├── inference/            engine.py (Predictor), hands.py (HandDetector), stability.py, predict.py, webcam.py
│   ├── landmarks/            features.py (canonicalize + 87 features), classifier.py (train/select/run)
│   ├── i3d_transfer/         native-sign inference (frozen I3D + linear head), preprocessing, dataset
│   ├── asl_citizen/          ASL Citizen manifest/training/eval code (research)
│   ├── word_model/           word-model research code
│   └── utils/                device, checkpoint, smoothing
├── frontend/                 React app (see §9)
│   ├── src/…                 pages, components, game logic, native, admin, collect, debug, shared, services
│   ├── landmarks-debug.html  standalone live-landmark diagnostic page (dev)
│   └── vite.config.ts, vitest.config.ts, package.json
├── migrations/versions/      Alembic: 0001 baseline, 0002 native/chatbot tables, 0003 seed words, 0004 admin audit
├── scripts/                  ~70 research/diagnostic/seed/evaluation scripts (see §29)
├── tests/                    ~30 Python test modules (461 tests pass at last full run)
├── data/                     asl_quest.db (SQLite), backups/, asl_citizen_* manifests, native_reference_urls.example.json
├── dataset/                  asl_alphabet_train/, webcam/ (collected), webcam_export_* (exports), asl_citizen videos
├── models/                   checkpoints (git-ignored): best_model.pth, hand_landmarker.task, experiments/, backup/, pretrained/
├── outputs/                  all reports, metrics, diagnostics (git-ignored/untracked)
├── deployment/               staging area + README for assets that live outside Git
├── *.md                      README, PROJECT_OVERVIEW, TECHNICAL_GUIDE, DATABASE, SETUP_WINDOWS, TEAM_HANDOFF,
│                             WEBCAM_EVAL_PROTOCOL, PROJECT_PROGRESS_SUMMARY, and this file
├── train.py / evaluate.py / predict.py / webcam.py   thin entry points into src/
└── requirements.txt, alembic.ini
```

---

## 5. Architecture and data flow

```
                           ┌───────────────────────── Browser (React 19 SPA) ──────────────────────────┐
                           │  Pages: Learn/Alphabet/Practice/Words/Native/Challenges/Progress/Profile…  │
                           │  Game state (XP, letters, badges) in React + localStorage + server sync     │
                           │                                                                              │
   Webcam ──getUserMedia──►│  HandLandmarker (MediaPipe WASM, VIDEO mode, requestAnimationFrame)         │
                           │      │ 21 landmarks + raw handedness   (no image leaves the browser)         │
                           │      ▼                                                                       │
                           │  POST /predict_landmarks (JSON)  ─────────────┐                              │
                           │  StableLetterTracker: vote window → "stable letter" → onCorrect/onIncorrect │
                           └──────────────────────────────────────────────┼──────────────────────────────┘
                                                                          ▼
                              ┌────────────────────── FastAPI backend ──────────────────────────┐
                              │ /predict_landmarks: validate → mirror/handedness → canonicalize  │
                              │      → 87 features → sklearn pipeline (L2) → top-k + confidence │
                              │ /predict (image): MediaPipe crop → ResNet18  (fallback / legacy) │
                              │ /native/predict (video): frozen I3D → linear head → top-k        │
                              │ /progress, /words, /native, /analytics, /admin, /chatbot …       │
                              └──────────────┬───────────────────────────────┬──────────────────┘
                                             ▼                               ▼
                                  SQLite (local) / PostgreSQL        models/*.pth, *.joblib, *.pt
```

Key design decisions and why:

* **Landmarks are detected in the browser.** A persistent MediaPipe `HandLandmarker` in `VIDEO` mode tracks the hand
  between frames. Only 21 numbers triples go to the server (≈1 KB JSON instead of a JPEG), so there is no image upload and
  no per-frame server-side detector initialization. Detection never blocks the camera preview (it runs in a
  `requestAnimationFrame` loop and never awaits the network).
* **One request in flight.** The recognizer sends only the newest landmarks; stale frames are dropped, never queued.
* **The server owns correctness for Native Signs and Words** (XP and mastery are computed server-side atomically); the
  **A–Z flow is client-authoritative** for its game state, which the client pushes with `PUT /progress`.
* **XP on the server is monotonic:** `persist_game_state` keeps `max(server_xp, client_xp)` so a stale browser tab can never
  erase XP that Native/Words awarded directly on the server.
* **ResNet18 remains in the code as the production fallback**, selected automatically when no landmark model is loaded.

---

## 6. Backend (FastAPI)

### 6.1 Application startup (`backend/main.py`)

`FastAPI(title="ASL Quest", version="5.0.0")` with CORS (`allow_credentials=True`, exact origins, never `*`). Routers are
registered for `ml, auth, progress, challenges, achievements, practice, analytics, users, recommendations, admin,
admin_ops, words, native, chatbot, collect`.

On startup:

1. `init_db()` creates the legacy tables; `seed_achievements()` inserts the 16 achievement rows if missing.
2. Loads the production alphabet model: `models/best_model.pth` (fallback `asl_resnet18_best.pth`) into a `Predictor`, and builds a
   `HandDetector(static_image_mode=True, detection_confidence=HAND_DETECTION_CONFIDENCE)` (default 0.35).
   Errors are kept as `load_error` so `/health` reports `degraded` instead of crashing.
3. If `ASL_QUEST_RECOGNIZER=landmarks`, loads the server-wide landmark classifier (`ASL_QUEST_LANDMARK_MODEL`).
4. If `ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL` is set, loads that `.joblib` **only if it lives under `models/`** (pickles can run
   code) and its feature count matches the live pipeline. A failure only disables the feature; the ResNet is never affected.

`python -m backend.main` binds to `127.0.0.1:8000`, or `0.0.0.0:$PORT` when `PORT` is set (hosting platforms).

### 6.2 Complete endpoint reference

All paths are relative to the API root (the frontend reaches it as `/api/...` in development through the Vite proxy).

**Recognition (`routers/ml.py`)**

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, model path, classes, device/GPU info, hand-detector availability, experimental landmark model info |
| GET | `/history` | last 50 recognitions and per-letter counts (in-memory ring buffer, 200 max) |
| POST | `/predict` | classify one still image (multipart). `require_hand`, `debug`, `top_k`, `recognizer=resnet\|landmarks`, dev capture params |
| POST | `/predict_landmarks` | **classify 21 landmarks** (JSON). Validated; see §12 |

**Auth, users**: `POST /auth/register`, `POST /auth/login`, `GET /auth/me`; `GET /users/me/profile`,
`GET|PUT /users/me/preferences`, `POST /users/me/change-password`, `DELETE /users/me`, `GET /users/me/export` and `/export.json`.

**Progress and gamification**: `GET|PUT /progress`, `GET /progress/letters`, `PUT /progress/letters/{letter}`,
`POST /progress/migrate-local`, `GET /progress/migration-offer`; `GET /challenges`, `POST /challenges/{type}/progress`;
`GET /achievements`; `POST /practice/session`; `GET /recommendations`.

**Words**: `GET /words`, `/words/progress`, `/words/recommendations`, `/words/analytics`, `/words/challenge`, `/words/{id}`,
`/words/{id}/progress`; `POST /words/practice/session`.

**Native Signs**: `GET /native/signs`, `GET /native/progress`, `GET /native/progress/{sign_id}`, `POST /native/predict`.

**Analytics**: `GET /analytics/{dashboard, overview, activity, accuracy, letters, letters/{letter}, response-time, history,
heatmap, streak, xp, insights, funnel, weekly-summary}`.

**Admin**: `GET /admin/{overview, activity, letters, words, users, recent-activity, mentor-dashboard}`;
management: `GET|PATCH|DELETE /admin/users/{id}`, `POST /admin/users/{id}/reset-progress`, `GET|PATCH /admin/native-signs`,
`GET|PATCH /admin/achievements`, `GET /admin/models`, `GET /admin/predictions`, `GET /admin/chatbot/feedback`,
`GET|DELETE /admin/dataset/sessions`, `GET /admin/audit-log`. All admin management actions are written to `admin_audit_log`.

**Chatbot (AURA)**: `GET /chatbot/status`, `POST|GET /chatbot/conversations`, `GET /chatbot/conversations/{id}`,
`POST /chatbot/conversations/{id}/messages`, `POST /chatbot/messages/{id}/feedback`.

**Development-only collection** (`ASL_QUEST_DEV_COLLECT=1`, otherwise 404): `/collect/status`, `/collect/sessions`,
`/collect/analyze`, `/collect/samples`, `/collect/negatives` and deletes.

### 6.3 `/predict` (image path)

1. Reject empty uploads and undecodable images (400). `recognizer` must be `resnet` or `landmarks` (422 otherwise).
2. `landmarks` path → `_predict_landmarks`: server-side MediaPipe detection → `LandmarkClassifier.predict_observation`. A missing model is 503; a missing/crashing
   detector or no hand is a structured `hand_detector_unavailable` / `no_hand_detected` reply (never a guess).
3. `resnet` path: with `require_hand=true` the hand is detected and cropped to a **padded square** (padding 0.30, never a stretched rectangle);
   the crop goes through `Predictor.predict_pil` (224×224, ImageNet normalization). Response includes `status`, `prediction`, `confidence`,
   `top_predictions`, `low_confidence` (confidence < 0.70), `hand_detected`, `hand_box`, and optional `diagnostics`.
4. The result is appended to the in-memory history.

### 6.4 `/predict_landmarks` (the live path)

Request body (`LandmarkPayload`): `landmarks` (21 × `[x, y, z]`), `handedness` (`"Left"`, `"Right"` or omitted — MediaPipe's **raw** label for
the image it ran on), `image_mirrored` (bool), `top_k` (1–10).

Validation (all failures are `422`, never a 500): exactly 21 points; each a triple of **finite** numbers; `-0.5 ≤ x,y ≤ 1.5` and `-5 ≤ z ≤ 5`;
handedness must be `Left`/`Right`/null; a degenerate hand (zero palm length) is rejected. No model loaded → `503`.

Logic: if `image_mirrored`, every `x → 1 − x` and the label is swapped (so the classifier always sees an unmirrored-frame convention);
build a `HandObservation`; `LandmarkClassifier.predict_observation` returns top-k with confidence; the response adds `actual_hand`
and `timings_ms {classify, server_total}`. It uses the experimental model if configured, else the server-wide one. It **never** uses the ResNet.

### 6.5 Services (summary)

| File | Responsibility |
|---|---|
| `services/progress.py` | achievement seed list, `level_from_xp`, `serialize_game_state`, `persist_game_state` (XP merge-max), migration offer logic |
| `services/words.py` | word progress, XP rules (50/25/20/150), mastery (3 completions), challenge, analytics, word achievements |
| `services/analytics.py`, `admin_stats.py`, `mentor_dashboard.py`, `user_features.py` | learner analytics, admin overview, mentor summaries, profile/preferences/export |
| `services/chatbot_*.py` | AURA: scope gate, offline FAQ (38 verified topics), knowledge base, LLM providers (§15) |
| `services/experimental_recognizer.py` | safe loading of a landmark `.joblib` (path under `models/`, `.joblib` only, feature count checked) |
| `services/live_diagnostics.py` | dev-only saving of exact classifier inputs of live requests |
| `services/webcam_dataset.py` | dev-only labelled data collection logic (§17) |
| `services/native_references.py` | validated attach/update of reference-video links for native signs |

---

## 7. Database

Local default: SQLite file `data/asl_quest.db`. Production: PostgreSQL through `DATABASE_URL` (`postgresql+psycopg://…`). Schema is
managed by Alembic (`alembic upgrade head`); `init_db()` additionally creates the legacy ten tables for old setups.
Migrations: `0001_baseline_existing_schema`, `0002_add_native_sign_chatbot_catalog_tables`, `0003_seed_words_catalog`,
`0004_admin_audit_and_user_active` (current head).

| Table | Purpose |
|---|---|
| `users` | account, email, bcrypt hash, `role` (`student` / `admin`), active flag |
| `user_progress` | one row per user: `xp`, `level`, streaks, `total_correct`, `total_attempts`, `sessions`, `game_meta` |
| `letter_progress` | per user per letter: attempts, correct_attempts, mastery, last_practiced |
| `challenge_progress` | daily / alphabet / speed / word challenge state |
| `achievements`, `user_achievements` | 16 seeded achievements and who earned what, when |
| `practice_sessions` | individual A–Z practice attempts (letter, predicted, correct, confidence, response time) |
| `xp_events` | **ledger** of XP awards: amount + reason (e.g. `native_sign_correct`, `word_completed`, "Quick and clear!") |
| `words`, `word_progress`, `word_practice_sessions` | word catalog, per-user word progress, per-attempt records |
| `native_signs` | catalog of whole-word signs (gloss, display name, meaning, category, difficulty, active, dataset/model flags) |
| `native_sign_references` | **reference media pointers** (never hosted media): `source_type` (`vimeo`/`youtube`/`url`/`external_url`), `source_identifier`, licence note, `is_primary` |
| `native_sign_progress` | per user per sign: attempts, correct_attempts, accuracy, mastery, best_confidence, best_response_time |
| `native_sign_sessions` | one row per native attempt: result (`correct`/`incorrect`), confidence, response time, model version |
| `model_versions`, `sign_predictions` | prediction audit: predicted/expected label, confidence, latency, correctness, model type/version |
| `user_preferences` | UI preferences (sound, motion, SIGNO, etc.) |
| `chatbot_conversations`, `chatbot_messages`, `chatbot_feedback` | AURA history and thumbs feedback |
| `admin_audit_log` | who did which admin action to what |

Relationships worth knowing: `native_sign_sessions.native_sign_id → native_signs.id`; `sign_predictions.session_id →
native_sign_sessions.id`; each correct native attempt also writes an `xp_events` row and increments `user_progress.xp`.

Local database state at the time of writing (MEASURED, read-only): 3 users (two admins and one learner on the developer machine), 10 native signs,
9 native references, 6 native sessions. Backups live in `data/backups/` (a fresh timestamped backup was taken before every change this session).

---

## 8. Authentication, roles and security

* Registration hashes the password with **bcrypt**; login returns a **JWT (HS256)** signed with `ASL_QUEST_SECRET_KEY` (default dev value
  `dev-change-me-in-production` — **must be set in production**), valid `ASL_QUEST_TOKEN_EXPIRE_MINUTES` (default 10080 = 7 days).
* The frontend stores the token in `localStorage` (`asl-quest-token`) and sends `Authorization: Bearer …`.
* `get_current_user` resolves the token; `get_current_admin` requires `role == "admin"` (403 otherwise). A bootstrap admin can be defined via
  `ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL`.
* CORS uses an explicit origin list (default: `http://127.0.0.1:5173`, `http://localhost:5173`, `https://asl-quest-1.vercel.app`) because credentials are enabled.
* Uploaded native-sign videos are written to a private temp file, processed, and **always deleted**; clips are never persisted. Size cap 100 MB;
  allowed extensions `.mp4 .mov .avi .mkv .webm`.
* Pickle safety: only `.joblib` files under `models/` may be loaded as landmark models.
* The AURA scope gate refuses off-topic questions and provider errors never leak response bodies.
* Dataset licensing: ASL Citizen media is never stored or redistributed; native reference links must be approved external sources with a licence note.
* The Vimeo/MediaPipe browser dependencies load third-party assets at runtime (jsDelivr, Google storage, Vimeo); self-hosting is supported via env overrides.

---

## 9. Frontend (React + Vite)

### 9.1 Structure (`frontend/src`)

| Folder / file | Contents |
|---|---|
| `App.tsx` | provider tree (Auth → Toast → Game → Signo), page switch, health check, passes landmark-model status to Practice and Words |
| `auth/` | `AuthContext` (user, login/logout, migration offer, sync status), `api.ts`, `token.ts` |
| `game/` | `constants.ts` (XP values, badges, challenges), `gamification.ts` (all pure rules), `GameContext.tsx`, `storage.ts`, `sync.ts`, `types.ts` |
| `pages/` | Home, Learn, Alphabet, Practice, Words, WordPractice, Challenges, Progress, Achievements, Profile, Settings, Admin, Auth pages |
| `components/` | `AppShell` (sidebar, mobile bottom nav, header, Explore popup), `CameraPractice` (ResNet/image path), `LandmarkPractice` (landmark path), `AlphabetLearningPath`, `AlphabetReferencePanel`, `LetterProgressStrip`, `MissionCard`, `CelebrationOverlay`, `Dialog`, `ExplorePopup`, `Toast`, `Icon`, `RecognizerSwitch` (unused UI), `signo/*` |
| `native/` | `NativeSignsPage`, `NativeSignPracticePage`, `NativeVideoCapture`, `useVimeoReady`, `api.ts` |
| `words/` | `catalog.ts`, `api.ts` |
| `shared/` | `recognition.ts` (tracker + tips), `handLandmarker.ts`, `useHandOverlay.ts`, `cameraError.ts`, `httpError.ts` |
| `debug/` | `LandmarkDebugView`, `landmarkStats.ts` (RollingStat, FpsMeter, jitter, payload builder) |
| `collect/` | dev-only webcam data collection UI |
| `admin/` | UsersTab, ContentTab, ModelsTab, DatasetTab, AuditTab, MentorDashboard, shared UI |
| `chatbot/` | AURA chat page and API |
| `analytics/` | analytics API and chart components |
| `services/signoService.ts` | SIGNO → AURA bridge |

### 9.2 Navigation

`NavPage` = `home, learn, alphabet, words, practice, native, native-practice, challenges, progress, achievements, assistant, profile, settings, admin,
mentor-dashboard, collect (dev only)`. The desktop sidebar lists primary pages, an **"Explore paths"** button (opens a centered popup with Alphabet /
Words / Native Signs cards, showing "N / 26 letters mastered"), and a "More" group. Mobile uses a bottom bar plus a "More" sheet. Admin pages show only for `role=admin`.

### 9.3 State, persistence and sync

* Game state (`GameState`: xp, streak, letterStats, badges, daily/challenge state) lives in `GameContext`, is cached in `localStorage` (`asl-learning-progress-v1`),
  and is pushed to the server with debounced `PUT /progress` (`scheduleProgressSync`). On login the server state is loaded; a **migration offer** lets a first login
  import local progress.
* Native and Word XP are awarded **by the server**; the client applies the returned `xp_earned`/badges to its local copy (`recordNativeSignAttempt`,
  `applyWordCompletionXp`).
* Because the server keeps `max(server_xp, client_xp)`, lowering server XP requires also correcting the browser's saved state (relevant to repairs, §25).

### 9.4 Design system

Tailwind 4 with CSS variables (`--color-primary` green `#58cc02`-family, `--color-native` blue `#1cb0f6`, XP amber, success/streak colours, `--radius-panel`,
`--radius-control`). Duolingo-inspired: rounded cards, soft shadows, pop animations (disabled under `reduce-motion`), accessible roles/labels on interactive pieces.
The learning-path node for a completed letter shows a tick over a faint **watermark of the letter** so the letter is still identifiable.

### 9.5 The Practice flow (Alphabet)

1. `PracticePage` picks the target letter (`nextPracticeLetter` or the letter opened from Learn/Alphabet). A lesson intro (reference image, hint, progress) is shown first
   when entered from the learning path.
2. `PracticePage` selects the recognizer: **landmarks** if the backend reports the experimental landmark model as loaded, otherwise **ResNet** (`CameraPractice`). There is no
   user-facing recognizer switch any more (removed on request).
3. The recognizer component reports `onCorrect(elapsedMs, confidence)` once the target letter is *stable* and `onIncorrect(letter)` once a different letter is stable.
4. `handleCorrect` calls `markCorrect` (the only place that awards A–Z XP), shows the celebration, XP toast, level-up overlay, badge toasts.
5. Wrong stable letter → `markIncorrect` and a specific "Almost! You're showing X…" feedback with hints.

### 9.6 `LandmarkPractice` (the landmark recognizer component)

Same props as `CameraPractice`. On mount (when ready) it creates **one** VIDEO-mode `HandLandmarker` (GPU delegate, falling back to CPU), kept for the component's lifetime and
closed on unmount. While `active` it: opens the camera (960×720 requested, unmirrored like the backend expects), runs a `requestAnimationFrame` loop processing only
genuinely new video frames with strictly increasing timestamps, draws all 21 points and the skeleton on an overlay canvas, and sends **only the newest** landmarks to
`/predict_landmarks` at most every 100 ms with **one request in flight** (an `AbortController` cancels pending requests on cleanup). Replies from an older attempt epoch or for
a hand that left the frame are discarded. A missed hand is counted at most once per request interval so the tracker's "frame" unit is the same with or without a hand.
After a confirmed correct sign it stops itself; a new target letter or "practice again" resets votes and **restarts** the loop (this restart bug was found and fixed when Words reused it).
It shows hand-detected / reading / unsure chips, status, tips, camera errors, hand-tracker load errors and a "Technical details" panel (median detection and request times).

### 9.7 `StableLetterTracker` (shared by both recognizers; `shared/recognition.ts`)

| Constant | Value | Meaning |
|---|---|---|
| `VOTE_WINDOW` | 7 | recent confident readings considered |
| `MIN_STABLE_VOTES` | 3 | minimum agreeing votes (and a majority) to call a letter stable |
| `NO_HAND_GRACE_FRAMES` | 2 | missed-hand frames tolerated before votes are cleared |
| `LOW_CONFIDENCE_RESET_FRAMES` | 6 | weak frames tolerated before votes are cleared |
| `HAND_MOVED_IOU` | 0.35 | if the hand box overlaps its previous box less than this, old votes are dropped |
| `MIN_FRAME_INTERVAL_MS` | 120 | request pacing of the image path |
| `TIP_AFTER_FRAMES` | 12 | persistent-problem troubleshooting tips |

Low-confidence frames neither add votes nor erase good ones immediately. `src/inference/stability.py` is a Python mirror used to *measure* confirmation latency and false confirms offline.

---

## 10. Gamification: XP, levels, streaks, badges, challenges

### 10.1 A–Z (client rules, `game/gamification.ts`, constants in `game/constants.ts`)

| Item | Value |
|---|---|
| Correct sign | **+20 XP** (`XP_CORRECT`); **+30** if answered within 5 s (`XP_FAST_CORRECT`, "Quick and clear!") |
| Level | `floor(xp / 1000) + 1` (`XP_PER_LEVEL = 1000`) |
| Daily challenge | 10 correct signs → **+100 XP**; perfect (no mistakes) → **+150 XP** bonus |
| Alphabet challenge | every letter at least once → **+500 XP** |
| Speed challenge | 5 correct within 45 s → **+150 XP** |
| Letter mastery | `correct / 10`, capped at 100% (`MASTERED_CORRECT = 10`); "Learning" tier at 3 (`LEARNING_CORRECT`) |
| Streak | consecutive practice days (`updateStreak`) |
| Learning path | 26 nodes: `completed` (mastered), `current`, `available` (beginner letters C, D, F, H, O), `locked`; after all beginners are mastered the next letter becomes current |

The five beginner letters were chosen from a benchmark of the production ResNet (C, D, F, H, O had perfect precision/recall and are static, distinct shapes; A was excluded
because its precision was only 0.55).

### 10.2 Words (server rules, `services/words.py`)

Completion **+50**, perfect **+25**, fast **+20** (≤ 4 s per letter), word challenge **+150** (5 words); mastery = `completions / 3`; `DAILY_WORD_COUNT = 3`.

### 10.3 Native Signs (server rules, `routers/native.py`)

Correct attempt **+20 XP** (`NATIVE_XP_CORRECT`), written as an `xp_events` row with reason `native_sign_correct`. **Correctness is strict top-1** (see §14).

### 10.4 Achievements (16, seeded server-side and mirrored in `BADGES`)

`first_sign`, `streak_7`, `alphabet_beginner` (5 letters), `alphabet_explorer` (13), `alphabet_master` (26), `speed_signer`, `fast_signer` (5 fast signs),
`daily_perfect`, `word_starter`, `word_learner` (5 words), `word_explorer` (3 categories), `word_master` (10 mastered), `vocabulary_builder` (25 completions),
`native_first_sign`, `native_explorer` (3 categories), `native_master` (every native sign mastered).

---

## 11. Alphabet (A–Z) recognition in depth

### 11.1 The production ResNet18 (still in the repo, still the fallback)

* **Architecture:** torchvision ResNet18 (ImageNet-pretrained) with the final layer replaced by `Linear(512, 26)` (`src/models/resnet18.py`).
* **Input:** 224×224 RGB, ImageNet mean/std (`src/config.py`).
* **Training data:** Kaggle "ASL Alphabet" photos in `dataset/asl_alphabet_train/` (26 folders). Split 70/15/15 by **numeric filename order** (last 15% of each class is test), indices stored in `outputs/splits.json`.
* **Training recipe (defaults):** batch 64, 8 epochs, AdamW-style LR `1e-4`, weight decay `1e-4`, seed 42, early-stop patience 3, augmentations that **do not mirror** (`Resize+RandomCrop`, `RandomRotation(12)`, `ColorJitter`), plus optional `RandomJpeg` and Gaussian noise
  for webcam-like robustness. Entry: `train.py` → `src/training/train.py`.
* **Inference (`src/inference/engine.py::Predictor`):** loads the checkpoint (`class_names`, `image_size`, state dict), eval transforms, softmax top-k, `low_confidence = confidence < 0.70`
  (`CONFIDENCE_THRESHOLD`), diagnostics (tensor shape, device, inference ms).
* **Hand cropping (`src/inference/hands.py::HandDetector`):** supports the legacy MediaPipe `solutions.Hands` and the Tasks `HandLandmarker`; the crop is a **square, padded (0.30), edge-replicated** window
  around the landmark box — measured to give 92.9% vs 69.9% for a stretched rectangle on the same held-out images (`scripts/diagnose_crop_strategies.py`).

### 11.2 Why the landmark recognizer became the live default

MEASURED: the ResNet reads Kaggle-style close-ups well (94.1% on the whole-frame Kaggle test) but **drops sharply on real webcam frames** (52.1% end to end on a 94-frame pilot). Fine-tuning the ResNet on the
small webcam set made it *worse* (experiments `az_robust*`, `az_webcam*`). The landmark classifier throws away appearance (skin, face, room) and keeps hand geometry, and scored better on that pilot
(L2: 84.0%). That pilot is small and not a promotion-grade evaluation (§19, §25), which is why ResNet stays as the automatic fallback and nothing was "promoted" in the checkpoint sense.

---

## 12. Landmark recognizer in depth (browser MediaPipe + classifier)

### 12.1 MediaPipe in the browser (`frontend/src/shared/handLandmarker.ts`)

`FilesetResolver.forVisionTasks(WASM_BASE)` then `HandLandmarker.createFromOptions` with `runningMode: "VIDEO"`, `numHands` (1 for letters, 2 for native-sign overlay),
`minHandDetectionConfidence = minHandPresenceConfidence = minTrackingConfidence = 0.5`, `delegate: "GPU"` with automatic CPU fallback.
`WASM_BASE` and the model URL default to the pinned jsDelivr runtime and Google's official model URL and can be overridden with `VITE_MEDIAPIPE_WASM_BASE` and
`VITE_HAND_LANDMARKER_MODEL_URL`. The `.task` file is **not bundled** (git-ignored; redistribution terms not stated).

### 12.2 Feature extraction (`src/landmarks/features.py`) — **87 features**

1. `canonicalize`: if the hand is `left`, mirror x; subtract the wrist (landmark 0); divide by the **palm length** (wrist → middle-finger MCP, image-plane distance); optionally rotate so the palm points up (`rotate`;
   **both L1 and L2 were fitted with `rotate=False`**, i.e. they expect the studio orientation).
2. `feature_vector`: 60 normalized coordinates (landmarks 1–20 × x,y,z) + 10 tip-to-tip distances + 5 tip-to-wrist distances + 4 thumb-tip-to-finger-MCP distances + 4 thumb-tip-to-finger-PIP distances +
   8 finger flexion cosines (PIP and DIP of four fingers) + 2 thumb flexion cosines = **87**. `FEATURE_DIM` is computed from the code and checked when a model loads.
3. Features include **z (relative depth)**, which matters: zeroing z drops M from 98% to 22% in a sensitivity test.

### 12.3 Classifier (`src/landmarks/classifier.py`)

* Class order is fixed `A…Z` (a model whose stored class list differs is refused at load).
* `select_and_train` chooses among LogisticRegression, an MLP and HistGradientBoosting, each with and without rotation, by validation macro-F1 on an unseen-room set or leave-room-out CV; it **refuses**
  to select on data from a single room. Training adds augmented copies (small rotation, scale, noise).
* `LandmarkClassifier.predict_points` → `predict_proba` → top-k with `low_confidence` (< 0.70).
* **L2** (`landmark_L2_s0.joblib`): `StandardScaler + LogisticRegression(C=2.0)`, `rotate=False`, trained on 9,878 rows = 9,458 landmark sets extracted from the **Kaggle photos** (in 5 "blocks" of the same signer/setup) + 420 real webcam frames
  covering only A, B, C, F, U, V, W, X, Y. **M, N, T, Z and 13 other letters have no webcam training rows.** **L1** is an MLP trained without the webcam rows.

### 12.4 Handedness and mirroring (the part that confuses everyone)

* MediaPipe labels a hand as if the image were a **mirrored selfie**. The app sends **unmirrored** camera frames, so `HandObservation.actual_hand` maps raw `Left → "right"` and `Right → "left"`.
  Training (`with_deployed_hand`) and live use the same table, so the model is internally consistent.
* MEASURED: on this project's real webcam frames the person's own labels agree with the **raw** MediaPipe label (257/259) and disagree with the mapped name (2/259); i.e. the *names* in the table are probably swapped, but because training and inference apply the
  same table, the behaviour is consistent. Correcting it would require retraining every landmark model.
* MEASURED (`outputs/mtz_investigation/`): the hand label is used for exactly one thing — mirroring x — and it matters a lot for some letters. Flipping the label drops T from 100% to 2.4% (232 of 382 become Z) and similarly collapses S, I, G, R, L, K.
  The label disagrees with the hand's own geometry (sign of index-MCP × pinky-MCP) on 15.8% of held-out Kaggle frames (33% for T, 75% for D).
* `/predict_landmarks` accepts `image_mirrored` so a client that analysed a mirrored frame can say so (the server flips x and swaps the label).

### 12.5 Timing (MEASURED on the dev machine, medians)

MediaPipe detection ≈ 30–50 ms; ResNet forward ≈ 12–22 ms; **landmark features + classifier ≈ 1 ms**; so detection dominates both pipelines. In the browser the page shows median/p95 detection latency, FPS, landmark jitter,
API round trip, server time, classifier time and capture-to-prediction time (`/landmarks-debug.html`).

### 12.6 Standalone debug page

`frontend/landmarks-debug.html` + `src/debug/*`: live video with the 21-point skeleton, hand status, FPS, detection latency (median/p95), jitter, raw handedness, and an optional "send to `/predict_landmarks`" checkbox.
It loads the repo's local `models/hand_landmarker.task` (via `server.fs.allow: [".."]` in Vite) so it is **development only**; it is not in the production build.

---

## 13. Word Spelling

* **Catalog:** 24 static words (`backend/words/catalog.py`, seeded by migration `0003`): categories **Beginner (7), Everyday (5), Greetings (4), Family (4), School (4)**, each with id, difficulty, description, tip and `estimated_xp`.
  It is explicitly *not* native ASL signs — it is fingerspelling.
* **Flow (`WordPracticePage`):** the learner spells the word letter by letter with the same recognizer as Practice (landmarks when available, ResNet otherwise). Each correct letter calls the word logic; on completion the page posts
  `POST /words/practice/session` with the per-letter log (letter, prediction, correctness, response time), mistakes and total time.
* **Server awards** (atomic, ledgered in `xp_events`): +50 completion, +25 perfect, +20 fast, +150 for completing a 5-word challenge; mastery = completions / 3; five word achievements; daily words (3) and recommendations based on weak letters.
* The word page reuses `LandmarkPractice`; because the component keeps the camera active between letters, the loop-restart fix (§9.6) is required for words to work beyond the first letter.

---

## 14. Native Signs (whole-word video recognition)

### 14.1 Vocabulary and data

Ten signs (class order in `data/asl_citizen_native_10/class_to_idx.json`): `BOOK, EAT1, HELLO, HELP, MOTHER, NO, PLEASE, THANKYOU, WATER, YES`. Training/eval clips come from the **ASL Citizen** dataset
(Microsoft Research; licensed separately and never redistributed). Splits (`data/asl_citizen_native_10_i3d/`): train 149 rows, val 39, test 129 (CSV rows minus header).
The catalog rows live in `native_signs`; categories (e.g. Greetings, Courtesy, Responses, Actions, Everyday, Family) are written by `scripts/seed_native_signs_native_10.py`.

### 14.2 Model: frozen I3D + linear head (`src/i3d_transfer/`)

* **Backbone:** Inception-I3D (`pytorch_i3d.py`), Microsoft's ASL Citizen checkpoint (MIT), **frozen**.
* **Preprocessing (`preprocessing.py`):** exact port of the official loader — BGR frames, values scaled to `[-1, 1]`, up to **64 frames** with adaptive frame-skip (1 / 2 / 3 for <96 / ≥96 / ≥160 frames), short side ≥ 226 and long side ≤ 256 (aspect preserved), short clips padded, **center crop 224**.
* **Features:** temporal mean-pooled 1024-d vector.
* **Head:** `Dropout(0.5) → Linear(1024, 10)` trained with LR 1e-3, weight decay 1e-3, early stopping (best epoch 12 of 27).
* **Inference:** `src/i3d_transfer/predict.py::predict_video(path, top_k)` → softmax over 10 classes → `prediction`, `confidence`, `top_k` list. Isolated from the A–Z path by construction.
* **MEASURED (dataset test split, 129 clips):** top-1 **96.9%**, top-5 99.2%, macro-F1 0.969. **These are dataset clips, not webcam clips.** On live webcam clips the head's output is often nearly flat (e.g. 20% / 11% / 10% …), the softmax is not a calibrated probability, and
  real-world webcam accuracy has **not** been measured (§25).

### 14.3 Server flow (`POST /native/predict`)

1. Validate extension (`.mp4/.mov/.avi/.mkv/.webm`), non-empty, ≤100 MB; resolve `native_sign_id` against the DB **before** inference (404 if unknown/inactive).
2. Write the clip to a private temp file, run `predict_video`, always delete the temp file (503 if the model is missing; 422 if the video can't be read).
3. **Correctness (strict top-1, changed this session):** `correct = predicted_gloss.strip().upper() == expected_gloss.strip().upper()`. The target appearing in the top 2/3 does **not** count.
   (An earlier lenient top-3 rule credited random wrong attempts roughly 30% of the time with 10 classes — see §25 for the historical rows it created.)
4. For a practice request (`native_sign_id` given): insert a `native_sign_sessions` row, update `native_sign_progress` (attempts, correct_attempts, accuracy, **mastery = accuracy**, best confidence, best response time),
   and, only if correct, award **+20 XP** (`user_progress.xp`, level recomputation, `xp_events` row), then evaluate `native_*` achievements. A `sign_predictions` row is always written.
5. The response returns `prediction`, `confidence`, `top_k` (still returned for any consumer), `model_type/version`, `latency_ms`, `native_sign_id`, `correct`, `expected_sign`, `session_id`, `response_time`, `xp_earned`, `new_achievements`.

### 14.4 Frontend flow (`NativeSignPracticePage`)

* **WATCH → COPY → CHECK** step tracker. Recording is gated behind watching the reference and pressing "I'm ready".
* **Reference panel:** if the sign has an available reference with `source_type` `vimeo` or `youtube`, it is embedded as an iframe (`player.vimeo.com/video/<id>?h=<hash>`; the privacy hash is mandatory for unlisted videos). Other sources render a link-out card.
  With no reference the panel shows a positive placeholder ("Demo video on its way").
* **Reliable failure detection (`useVimeoReady`):** an iframe's `load` event also fires on Vimeo's "video does not exist" page, so readiness is determined with Vimeo's official Player SDK — `ready()` resolving means a real player,
  rejecting (`NotFoundError`, `PrivacyError`) or an `error` event means failure, and a **15-second timeout** (`EMBED_READY_TIMEOUT_MS`) covers blocked embeds without condemning slow ones (a late `ready()` clears the fallback). On failure the existing
  "This reference couldn't load" panel shows with an "Open source instead ↗" link to the original page. Listeners/timers are released on unmount or sign change, and results are tagged with the URL they belong to.
* **Recording (`NativeVideoCapture`):** `getUserMedia` → live preview → **3-2-1 countdown** → `MediaRecorder` for **10 s** (WebM/MP4, picks a supported MIME) with a "Stop now" button. A **preview-only MediaPipe overlay** (`useHandOverlay`, up to 2 hands, 21 points each) and a chip
  ("2 hands detected" / "No hand detected" / "Loading hand tracker…" / "Hand overlay unavailable") are drawn on a separate canvas; the **recorded stream is the raw camera stream**, so the overlay is never in the clip and the I3D model sees unmodified frames. If MediaPipe fails to load, recording still works.
* **Result card:** "Correct!" / "Not quite", confidence (%), expected sign, predicted sign, response time, mastery; "+N XP" only for correct attempts; badges. The "Top predictions" list was removed from the UI (the API still returns it).

### 14.5 Reference videos

The 9 restored references are unlisted Vimeo videos by *Learn How to Sign* (learnhowtosign.com), recorded as `source_type="vimeo"` with `https://vimeo.com/<id>/<hash>`:

| Sign | Vimeo id / hash |
|---|---|
| HELLO | 899575446 / 766377de37 |
| THANKYOU | 966259802 / c3cf18c8dd |
| PLEASE | 953394002 / 651e2c3d28 |
| YES | 1040044390 / aca803a079 |
| NO | 951761278 / 1c671e552d |
| MOTHER | 951748939 / 8bd9733aad |
| WATER | 1040043378 / 305ff4723b |
| HELP | 899575546 / f782b25b02 |
| BOOK | 897045923 / ca27cc3775 |
| EAT1 | none (the only candidate video had the wrong title, so none was attached) |

Origin: `scripts/seed_native_sign_references.py` (page links) then `scripts/update_native_sign_references_embed.py` (upgrade to Vimeo embeds), both from commit `93bdf3e`. The references table is **not** in Git — it lives only in whichever database the scripts were run against.
`backend/services/native_references.py` + `scripts/set_native_references.py` provide a validated way to add other approved https links (requires a licence note).
Embedding permission was *inferred* from Vimeo's privacy behaviour, not from a written statement by the creator.

---

## 15. AURA chatbot and SIGNO companion

### 15.1 AURA (project assistant)

* **Scope:** answers questions about ASL-Quest only (gated). Conversations, messages and thumbs feedback are stored.
* **Layers (in order):** a **scope gate** (off-topic refused; greetings handled), an **offline FAQ** (`chatbot_faq.py`, 38 verified topics scored by weighted regex patterns, `MIN_SCORE = 2`, related-topic suggestions),
  a **knowledge base** (`chatbot_knowledge.py`: verified facts about the project's database, architecture, modes, XP rules, limitations), and an **LLM provider** (`chatbot_llm.py`) used when configured.
* **Providers (`CHATBOT_PROVIDER`):** `openrouter` (default; `OPENROUTER_API_KEY`, model `openrouter/free`), `local` (Ollama at `127.0.0.1:11434`, model `qwen3:4b`, fully offline), or `auto` (prefer local, fall back to OpenRouter).
  Timeouts 30 s, `MAX_TOKENS = 1024`, history limited to 10 messages; rate-limit/timeout/provider errors map to typed exceptions and friendly replies. If no provider is available AURA still answers from the FAQ/knowledge base.
* **Progress-aware answers:** for XP/streak questions the backend adds the user's real numbers to the prompt (`_build_user_context`).
* **Note:** AURA's server-side knowledge still states the project's honest limitations (e.g. J/Z, experimental status) — these were deliberately **not** rewritten when the on-screen negative wording was removed, so the assistant stays truthful.

### 15.2 SIGNO (in-app companion)

A small animated character (`components/signo/*`) with speech bubbles reacting to real events (lesson opened, practice started, correct/incorrect, hints). Local reactions are deterministic and clearly labelled; free-form questions go through
AURA's API (`services/signoService.ts`), so there is a single chat stack and no keys in the frontend. Preferences (show/hide, etc.) persist; conversations do not.

---

## 16. Admin, mentor and analytics

* **Admin page** (`role=admin`): Users (view/edit/deactivate/reset/delete), Content (native signs and achievements), Models (model versions/predictions), Dataset (collected webcam sessions), Audit log; every management action is audited.
* **Mentor dashboard:** per-learner summary (levels, XP, accuracy, native mastery averages and mastered counts) from `services/mentor_dashboard.py`.
* **Analytics:** accuracy over time, heatmap, response time, streak, XP history, funnel, insights and a weekly summary for the learner (`ProgressPage`) and aggregates for admins.
* **Recommendations:** weakest/strongest letters and next-letter suggestions based on per-letter stats.

---

## 17. Data collection and datasets

* **Collect page / `/collect/*`** (development only, `ASL_QUEST_DEV_COLLECT=1`): record labelled webcam sessions (room, lighting, background, distance, hand used, face visible, glasses), single frames, short "hold" sequences and negative (no-sign) frames.
  `dataset/webcam/` is git-ignored because frames contain faces and rooms.
* **Sessions on disk (MEASURED):** `s20261019_222243_plain-wall` (A, C, F), `s20261019_224255_bedroom` (empty), `s20261020_093606_bedroom` (A, B, C, F, U, V, W, X, Y).
  Exports (`dataset/webcam_export_*`) hold manifests with crops, landmarks, MediaPipe raw handedness and split labels (`trainall`: 136 rows; `holdout`: 214; `alldiag`: 259). **No M, N, T or Z webcam samples exist.**
* **Protocol:** `WEBCAM_EVAL_PROTOCOL.md` defines what counts as held-out (session *and* person unseen), the coverage required, and the comparison of ResNet vs L1 vs L2. Status: **incomplete** — 94 eligible frames, 9 letters, one room/lighting, no recorded person.
* **Other datasets:** Kaggle ASL Alphabet (training, `dataset/asl_alphabet_train`); ASL Citizen subsets (`data/asl_citizen_100`, `data/asl_citizen_native_10`, `data/asl_citizen_native_10_i3d`).
* **Extracted landmarks:** `outputs/diagnostics/landmarks/dataset_landmarks.npz` (9,458 sets from the Kaggle train split) and `outputs/mtz_investigation/test_landmarks.npz` (4,054 held-out detected frames).

---

## 18. Model training and the experiment history

Scripts (all under `scripts/`, entries at repo root): `train.py`, `evaluate.py`, `predict.py`, `webcam.py`; landmark training `train_landmark_classifier.py`; native training/evaluation family
`train_native_10*.py`, `evaluate_native_10*.py`, `extract_i3d_features_native_10.py`; checkpoint management `promote_checkpoint.py`, `upgrade_checkpoint.py`, `compare_checkpoints.py`.

Experiment chronology (what was tried, per the repo's reports; none promoted automatically):

1. **Production ResNet18** trained on Kaggle photos (`models/best_model.pth`; backups in `models/backup/`).
2. **Robustness variants** (`az_robust*`, `az_seed43*`, `az_webcamall_s42*`, `az_webcamho_s42*` in `models/experiments/`): augmentation with JPEG/noise, and fine-tuning on the small webcam set. **Fine-tuning on the small webcam set hurt performance**; the checkpoints are kept as experiments only.
3. **Landmark classifiers L1/L2** (§12.3) — better on the pilot webcam set; L2 is the one wired into the app.
4. **ASL Citizen native-10** research ladder: appearance-augmentation, layer-4 fine-tune, full I3D fine-tune and finally the **frozen backbone + linear head** (best epoch 12), which is what the app uses.
5. **Video baseline / inspection / detection / handedness investigations** — see §20 and `outputs/*`.

Rules the project follows: never overwrite `best_model.pth`; experimental checkpoints are kept separate; promotion needs an explicit decision and evidence on unseen rooms/people; backups are taken before database changes.

---

## 19. Measured results and what they mean

All numbers come from `outputs/compare_resnet_landmarks/REPORT.md`, `outputs/asl_citizen_native_10/.../test_evaluation_report.json` and the investigation scripts unless stated.

**Alphabet, Kaggle test split (11,700 images, 450 per class)**

| System | End-to-end accuracy | Notes |
|---|---|---|
| `resnet_full` (whole frame, no hand detection) | **94.1%** | how the checkpoint was validated |
| `resnet_crop` (MediaPipe crop → ResNet) | 32.3% | MediaPipe finds a hand in only **34.6%** of these close-up photos (detection failure 65.4%); accuracy *when it answered*: 93.3% |
| L1 / L2 landmark classifiers | 30.7% / 30.6% | same detection ceiling; when answered 88.7% / 88.3% |

The Kaggle images are not representative of webcam use (single signer, 200×200 close-ups); they cap every detector-dependent system.

**Alphabet, real webcam pilot (94 frames, 9 letters, no J/Z, one session)**

| System | End-to-end | Accepted (conf ≥ 0.70) accuracy |
|---|---|---|
| ResNet crop | 52.1% | 78.8% (33/94 accepted) |
| Landmark L1 | 69.1% | 71.9% |
| Landmark L2 | **84.0%** | 94.5% |

Weak spots: C scored 0% on both landmark models and V 42% / 8%; this is **a small pilot, not a verdict** (confidence intervals are wide and frames within a class are near-duplicates).

**Native Signs (129 ASL Citizen test clips):** top-1 96.9%, macro-F1 0.969. **No webcam measurement exists.**

**Latency:** see §12.5.

**Confidence calibration (L2, held-out Kaggle frames, MEASURED):** reported confidence is a softmax score, not a probability the answer is right — 0.80–0.90 → 66% actual accuracy; 0.90–0.95 → 78%; ≥0.99 → 97.9%. Expected calibration error 0.056 even on the favourable same-signer data.

---

## 20. Investigations and findings (M/T/Z, handedness, detection)

### 20.1 M and T read as Z (landmark recognizer)

Reproduce: `venv311\Scripts\python.exe scripts\investigate_mtz.py` and `scripts\investigate_mtz_handedness.py extract|report`; outputs in `outputs/mtz_investigation/`.

* **Training coverage:** M (287 rows), T (382), Z (387) exist only as Kaggle landmark sets (one signer); **zero webcam rows**. The model-selection score (0.94) came from "rooms" that are 5 chunks of the same signer.
* **Held-out confusion (Kaggle, detected frames):** M recall 96.6% (precision 47.7%, M is a catch-all answer for other fist letters — 92 other-letter frames were called M), **T recall 61.4% with 13 of 70 read as Z (19%)**, Z recall 96.6%.
* **Handedness:** M barely changes with the label (never Z); **T collapses when the label is flipped** (100% → 2.4%, 232/382 → Z). Geometry-based mirroring gave T 68.6% vs 61.4%, overall 87.6% vs 88.3% — a small, mixed effect.
* **Rotation/scale/depth:** no rotation or aspect change tested turns M into Z; M depends strongly on z (22% without it). Normalizing a 16:9 frame instead of square hurts (T 2.9%, overall 82.3%); 4:3 is mild (overall 86.8%).
* **Conclusion:** the live M→Z at ~81% is **not explained** by handedness, rotation, aspect or depth on held-out data. M and Z are far apart in feature space, so the user's real M landmarks are probably unlike any Kaggle M (e.g. MediaPipe misplacing fingers on fist-like poses).
  The decisive missing evidence is real webcam M/T/Z data. **Nothing was retrained or changed.** Proposed next step: collect ~30 frames each of M, T, Z in two rooms with the Collect page, rerun the scripts, and only then consider geometry-based mirroring, x-aspect correction and a stricter confidence policy.

### 20.2 Left-vs-Right accuracy gap (`outputs/recognition_investigation/REPORT.md`, `handedness.md`)

On Kaggle frames, raw "Left" accuracy is 48.7% vs 94.4% for "Right". It is **not** a mirroring bug, tilt, size, confidence or letter-mix effect; the raw label does not track geometry for these images. The hypothesis (not proven) is that MediaPipe's landmark layout itself is poor on dark backlit silhouettes.

### 20.3 Detection failures

MediaPipe fails to find a hand on 65.4% of Kaggle test images (worst: P, X, B, C, T, I, H, N); on webcam frames detection was 100% (94/94). See `outputs/recognition_investigation/detection.md`.

---

## 21. Configuration reference (environment variables)

**Backend (`backend/settings.py`)**

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` / `ASL_QUEST_DATABASE_URL` | `sqlite:///data/asl_quest.db` | database |
| `ASL_QUEST_SECRET_KEY` | `dev-change-me-in-production` | JWT signing key (**set in production**) |
| `ASL_QUEST_TOKEN_EXPIRE_MINUTES` | 10080 | token lifetime |
| `ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL` | empty | email promoted to admin |
| `ASL_QUEST_CORS_ORIGINS` | localhost:5173, 127.0.0.1:5173, `https://asl-quest-1.vercel.app` | comma-separated exact origins |
| `ASL_QUEST_HAND_DETECTION_CONFIDENCE` | 0.35 (clamped 0.05–0.95) | server MediaPipe confidence |
| `ASL_QUEST_RECOGNIZER` | `resnet` | server default for `/predict` (`resnet` or `landmarks`) |
| `ASL_QUEST_LANDMARK_MODEL` | `models/landmark_classifier.joblib` | server-wide landmark model |
| `ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL` | empty (off) | landmark `.joblib` under `models/`; **enables `/predict_landmarks` and the landmark recognizer in the UI** |
| `ASL_QUEST_DEV_DIAGNOSTICS` | off | save exact classifier inputs of live requests |
| `ASL_QUEST_DEV_COLLECT` | off | enable the `/collect/*` API |
| `OPENROUTER_API_KEY`, `CHATBOT_MODEL` (`openrouter/free`), `CHATBOT_API_BASE_URL` | – | AURA via OpenRouter |
| `CHATBOT_PROVIDER` (`openrouter`/`local`/`auto`), `CHATBOT_LOCAL_BASE_URL`, `CHATBOT_LOCAL_MODEL` (`qwen3:4b`) | – | AURA provider selection / Ollama |
| `PORT` | unset | bind `0.0.0.0:$PORT` for `python -m backend.main` |

**Frontend (Vite `import.meta.env`)**

| Variable | Meaning |
|---|---|
| `VITE_API_BASE_URL` | backend URL in production; unset in dev → `/api` proxy |
| `VITE_DEMO_MODE=true` | skip login, frontend-only demo (backend features report unavailable) |
| `VITE_MEDIAPIPE_WASM_BASE` | self-hosted MediaPipe WASM directory |
| `VITE_HAND_LANDMARKER_MODEL_URL` | self-hosted `hand_landmarker.task` URL |

---

## 22. Setup and run commands

Run from `D:\SIGN LANGUAGE` unless stated. Windows helpers exist in `scripts/windows/` (`setup.ps1`, `start_backend.ps1`, `start_frontend.ps1`, `validate_setup.ps1`, `install_deployment_assets.ps1`); see `SETUP_WINDOWS.md`.

```powershell
# Backend (recommended venv: venv311). Loads .env, uses CUDA when available.
venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --env-file .env
#   to use the landmark recognizer in the app, set in .env:
#   ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL=models/experiments/landmark_L2_s0.joblib

# Database schema (if using Alembic explicitly)
venv311\Scripts\python.exe -m alembic upgrade head

# Frontend
cd frontend
npm install
npm run dev            # http://localhost:5173  (proxies /api to :8000)
#   http://localhost:5173/landmarks-debug.html   live landmark diagnostic page

# Seeds (idempotent)
venv311\Scripts\python.exe scripts\seed_native_signs_native_10.py
venv311\Scripts\python.exe scripts\seed_native_sign_references.py
venv311\Scripts\python.exe scripts\update_native_sign_references_embed.py
```

Required assets outside Git (copy from a machine that has them; see `deployment/README.md`): `models/best_model.pth`, `models/hand_landmarker.task`,
`models/pretrained/asl_citizen_i3d/ASL_citizen_I3D_weights.pt`, `outputs/asl_citizen_native_10/experiments/i3d_frozen/checkpoints/head_best.pt`.

---

## 23. Testing

| Suite | Command | Last result |
|---|---|---|
| Backend (pytest, 30 modules) | `venv311\Scripts\python.exe -m pytest tests -q` | **461 passed**, 11 subtests, 5 FastAPI-deprecation warnings (≈3 min) |
| Frontend unit/integration (Vitest) | `cd frontend; npx vitest run` | **27 files, 212 tests passed** |
| Types | `cd frontend; npx tsc --noEmit` | clean |
| Production build | `cd frontend; npm run build` | succeeds (main chunk ≈1 MB; Vite warns about chunk size) |

Notable test modules: `test_native.py` (strict top-1, XP exactly-once, progress), `test_predict_landmarks.py` (validation, mirroring equivalence, class order, feature dimension), `test_native_references.py`,
`test_landmark_conventions.py`, `test_experimental_recognizer.py`, `test_stability.py` (tracker parity), `test_progress_sync.py`, `test_admin_ops.py`, `test_chatbot*.py`, `test_webcam_*`.
Frontend: `LandmarkPractice.test.tsx` (single onCorrect, one request in flight, low-confidence, cleanup, resume), `useVimeoReady.test.tsx`, `useHandOverlay.test.tsx`, `NativeSignPracticePage.test.tsx`, `ExplorePopup.test.tsx`, `AlphabetLearningPath.test.tsx`.
Browser-level checks were done with headless Chrome (Playwright) against the running dev app with the **API mocked** (see §26); there is no committed end-to-end suite and nothing was tested against the real backend with a login.

---

## 24. Deployment

Per configuration in the repo: **frontend on Vercel** (`VITE_API_BASE_URL` points at the API; `VITE_DEMO_MODE` is for a frontend-only build), **backend on Render** started with `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` (or `python -m backend.main`),
**PostgreSQL on Render** via `DATABASE_URL` (the production URL appears only as a commented line in the local `.env`; the code fails fast when Postgres is unreachable). Large assets (`*.pth`, `*.pt`, `*.task`, the SQLite DB) are **not in Git**; they must be provisioned on the host.
Set a strong `ASL_QUEST_SECRET_KEY`, correct `ASL_QUEST_CORS_ORIGINS`, and (to enable landmark recognition) `ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL` with the `.joblib` present under `models/`.
**Nothing in this session was deployed or pushed.** The deployed database has its own rows; the native catalog and reference scripts must be run against it separately if needed.

---

## 25. Known issues, limitations and open decisions

**Recognition quality**
* Landmark L2's 84% is from a 94-frame, 9-letter, single-session pilot. The full held-out protocol (`WEBCAM_EVAL_PROTOCOL.md`) is **not complete**. C and V are weak; **M, N, T, Z have never been evaluated on real webcam frames**.
* J and Z are motion letters; a still-pose model cannot recognize them truthfully.
* The landmark model depends on MediaPipe's handedness label to decide mirroring; the label is unreliable on some frames and strongly affects T, S, I, G, R, L, K.
* Landmark confidence is not calibrated; the 0.70 threshold is a guess, not a measured operating point.

**Native Signs**
* **Mastery formula is too generous (KNOWN ISSUE, change proposed not implemented):** mastery = running accuracy, so **one correct attempt = 100% = "mastered"**. Proposed rule: `min(100, correct_attempts / 5 × 100)` by correct count (never decreasing), with the code/test/FAQ changes listed in the proposal (`native.py`, mentor dashboard numbers shift, AURA text, tests, one-off recalculation). Awaiting a decision (5 vs 3).
* **Historical data written by the old lenient top-3 rule still exists (KNOWN ISSUE, repair plan awaiting approval).** Two HELLO attempts by one local learner account were stored as correct although the top prediction was PLEASE and NO:
  `native_sign_sessions` 1 and 3, `sign_predictions` 1 and 3, `xp_events` 143 and 145 (+20 each), `user_progress.xp` 4470 (should be 4430), `total_correct` 144 (should be 142), and `native_sign_progress` for HELLO (3/3, 100% → should be 1/3). Achievements remain valid.
  Server XP is `max(server, client)`, so a server-only repair would be undone by the browser's saved state — the repair must also reset that browser state.
* The head's webcam confidence is typically low/flat; webcam accuracy is unmeasured.
* The Vimeo player intermittently answers "this video does not exist" for a working video (observed on MOTHER and HELP in 2 of ~16 loads); the fallback now appears with a working link, but there is no automatic retry. Embedding permission was inferred, not granted in writing.
* EAT1 has no reference video.

**Process / data**
* The MediaPipe `hand_landmarker.task` licence is not stated on Google's public pages; the app therefore fetches it from Google at runtime rather than bundling it.
* The first browser load of the landmark recognizer depends on jsDelivr and Google being reachable.
* Production bundle is ≈1 MB in the main chunk (MediaPipe + Vimeo SDK added); no code-splitting was done.
* `RecognizerSwitch.tsx` and `CameraPractice`'s `recognizer="landmarks"` image path remain in the code but are no longer reachable from the UI by a user choice.
* The mobile bottom bar does not include the "Explore paths" popup (desktop sidebar only).
* The AURA FAQ/knowledge still describe limitations honestly (e.g. a "Why are J and Z not supported?" entry).
* `npm audit` reported 5 vulnerabilities (2 moderate, 3 high) and npm warned about an unapproved `esbuild` install script; neither was acted on (no `npm audit fix`, no script approval).

**Open decisions (need the owner's call):** mastery target (5 or 3); XP repair as delete vs offset events and when; whether to capture real M/T/Z webcam data next; whether to keep or remove the unused experimental-recognizer UI code.

---

## 26. What changed in the latest working sessions (uncommitted)

Branch `david-dev`, base commit `869e187`. Uncommitted work, in the order it was done (each item was verified with the checks in §23 unless marked otherwise):

1. **Live landmark debug page** (`frontend/landmarks-debug.html`, `src/debug/*`) + dependency `@mediapipe/tasks-vision@1.1.1`.
2. **`POST /predict_landmarks`** with strict validation and tests (`tests/test_predict_landmarks.py`); `HandObservation` reuse; mirrored-frame handling.
3. **`LandmarkPractice`** wired into Practice as the recognizer when the landmark model is loaded; ResNet `CameraPractice` is the automatic fallback; the recognizer switch UI was removed from the page.
4. **Words page** also uses `LandmarkPractice`; fixed the loop-restart bug for multi-letter words.
5. **Native Signs catalog restored locally** (10 signs seeded; 9 Vimeo references restored from commit `93bdf3e` scripts) with timestamped DB backups before each change.
6. **MediaPipe overlay** on the Native Signs recording preview (`useHandOverlay`), never part of the recorded clip.
7. **Reliable Vimeo failure detection** with `@vimeo/player` (`useVimeoReady`), 15-second timeout, correct fallback link, cleanup tests. Verified in real headless Chrome with the API mocked: 9/9 references reach the right state; intentional failures (non-existent video, blocked Vimeo) show the fallback.
8. **Strict top-1 correctness for Native Signs** + regression tests; "Top predictions" UI removed; "Correct!"/XP only for genuinely correct attempts.
9. **Learning-path watermark** of the letter behind the completed tick.
10. **Positive wording** for descriptive copy (Learn/Alphabet descriptions, J/Z hints, Native placeholder, camera tips, AURA blurb) and the **Explore paths popup** in the sidebar.
11. **Investigations (read-only):** M/T/Z (`scripts/investigate_mtz*.py`, `outputs/mtz_investigation/`), historical-data audit, mastery rule review, Vimeo playback matrix.
12. **Supporting helpers:** `backend/services/native_references.py`, `scripts/set_native_references.py`, `data/native_reference_urls.example.json`, `tests/test_native_references.py`.

13. **Roles, header menu, heatmap-first Progress and admin dataset collection** (details in `DATASET_COLLECTION.md`):
    avatar account menu (Profile / Settings / Admin workspace / Log out) in the header; Progress opens with a GitHub-style activity
    heatmap (`GET /analytics/heatmap?range=&tz=`, counts letters + words + native signs by the learner's local day, with summary);
    registration can never create an admin (bootstrap e-mail only works while no admin exists); `POST /admin/users` lets an admin create
    an admin; `/collect/*` is now admin-only (it previously had no authentication, only the dev flag); collection gained consent,
    native-sign clips, gallery, upload validation and a labelled zip export; Admin workspace has a **Collect data** tab.
    Backend tests: 511 passed. Frontend: 34 files / 271 tests passed.

Not done / not changed in these sessions: no model was trained, replaced or promoted; `models/best_model.pth` and all existing datasets/backups are untouched; no production system was changed; nothing was deployed or pushed.

Browser testing caveat: the Playwright runs used the real dev frontend and real Vimeo network but **mocked `/api` responses** (the sign list was exported read-only from the real database). Practice/Words/landmark recognition was exercised through unit tests and by the owner's manual use, not by an automated live-camera test.

---

## 27. Operating rules and machine constraints

* **Memory:** the development laptop (Windows 11, RTX 4050 6 GB, 16 GB RAM) is often near its commit limit. A backend Python process alone commits ~3.4 GB (torch + CUDA). Run **one heavy job at a time** (torch + MediaPipe + backend + Vitest together can fail with
  `WinError 1455` / CUDA out-of-memory). For experiments, send requests to the already-running backend instead of loading torch again; use `--num-workers 0` for training/evaluation and small batches.
* **Never** kill `python.exe` or `chrome.exe` by image name — it ends the user's running backend/Chrome. Stop only the specific PID you started.
* The user normally runs the backend on port 8000 and Vite on 5173.
* Shell quirks (Git Bash tool on Windows): backslash escapes inside heredocs/`node -e` get mangled; write files with a file tool instead of shell quoting; `/tmp` for bash is not the same path Node sees.
* **Safety rules followed:** back up `data/asl_quest.db` with a timestamp and verify it before any DB write; never delete datasets/backups; never promote or overwrite checkpoints without explicit approval; ask before repairing data or changing production.

---

## 28. Glossary

| Term | Meaning |
|---|---|
| ASL | American Sign Language |
| Fingerspelling | spelling words with the 26 letter handshapes (A–Z) |
| Native sign | a whole-word ASL sign (as opposed to spelling it) |
| Gloss | the written label of a sign, e.g. `THANKYOU` |
| MediaPipe Hand Landmarker | Google's hand tracking model: 21 landmarks per hand + handedness |
| Landmark | one of 21 points on a hand (0 = wrist, 4/8/12/16/20 = fingertips, 1/5/9/13/17 = MCP joints) |
| Canonicalization | normalizing landmarks (wrist origin, palm length 1, mirror to one hand) so a letter looks the same anywhere |
| Handedness | MediaPipe's Left/Right label; labelled as if the image were a mirrored selfie |
| ResNet18 | the 18-layer CNN used as the image classifier for letters |
| I3D | Inflated 3D ConvNet, a video classification network; here frozen and used as a feature extractor |
| Linear probe / head | a single linear layer trained on frozen features |
| Top-1 / top-k | the single highest-scoring class / the k highest |
| Softmax confidence | a normalized score; **not** a calibrated probability |
| ECE | expected calibration error |
| Held-out | data never used for training or model selection (here also a *different person and session*) |
| Epoch (vote) | the tracker's recent-readings window used to confirm a stable letter |
| XP ledger | `xp_events`, the audit trail of XP awards |
| AURA / SIGNO | the project assistant chatbot / the in-app companion character |

---

## 29. Where to look for what (cheat sheet)

| I want to… | Look at |
|---|---|
| change XP values or badges | `frontend/src/game/constants.ts` (A–Z), `backend/services/words.py` (words), `backend/routers/native.py` (native) |
| change when a letter counts as confirmed | `frontend/src/shared/recognition.ts` (constants at top) and mirror in `src/inference/stability.py` |
| change landmark features | `src/landmarks/features.py` (this changes `FEATURE_DIM`; every saved model must be retrained) |
| change how Native correctness works | `backend/routers/native.py` (the `correct = …` line), tests in `tests/test_native.py` |
| change native mastery | `backend/routers/native.py` (`progress.mastery = …`), `services/mentor_dashboard.py`, AURA texts, tests |
| add/replace a native reference video | `scripts/set_native_references.py` + `backend/services/native_references.py` (https + licence note required) |
| add a native sign | `native_signs` row + a trained class in the I3D head (needs retraining) — see `scripts/seed_native_signs_native_10.py` |
| switch the alphabet recognizer | `ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL` (landmarks) or leave unset (ResNet) |
| add a word | `backend/words/catalog.py` + a new Alembic seed migration |
| see model/detection investigations | `outputs/recognition_investigation/`, `outputs/mtz_investigation/`, `outputs/compare_resnet_landmarks/REPORT.md` |
| evaluate on new webcam data | `WEBCAM_EVAL_PROTOCOL.md`, `scripts/evaluate_webcam_heldout.py` |
| understand the DB | this file §7, `DATABASE.md`, `backend/models.py` |
| set up a new machine | `SETUP_WINDOWS.md`, `deployment/README.md`, `scripts/windows/*` |
| debug live landmark behaviour | `http://localhost:5173/landmarks-debug.html`, `/predict_landmarks` timings |
| older overviews | `README.md`, `PROJECT_OVERVIEW.md`, `PROJECT_TECHNICAL_GUIDE.md`, `TEAM_HANDOFF.md`, `PROJECT_PROGRESS_SUMMARY.md` |

---

*End of document. If you change behaviour, update the relevant section and keep the MEASURED / BY DESIGN / UNVERIFIED / KNOWN ISSUE labels honest.*
