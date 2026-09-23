# ASL Alphabet Recognition

Static **A–Z** (26 classes) American Sign Language letter recognition. The project reuses a trained ResNet18 checkpoint and serves it through a Python CLI, FastAPI backend, and React frontend.

This is **not** a dynamic/sign-language translator. Each prediction is one still-image letter. **J and Z motion is not supported.**

## Current Architecture

ASL Quest is a **gamified multi-user learning platform** with persistent accounts and progress.

```
React frontend (ASL Quest UI)
        ↓  JWT Bearer auth + /api proxy
FastAPI backend
        ├── ML routes: /health, /predict, /history
        └── App routes: /auth/*, /progress/*, /challenges/*, /achievements, /practice/*
        ↓
SQLAlchemy ORM
        ↓
SQLite database: data/asl_quest.db
```

- **Authentication:** Register/login with bcrypt password hashing and JWT bearer tokens.
- **Progress source of truth:** Server database (SQLite). Browser `localStorage` is used only as an offline cache.
- **ML inference:** Unchanged ResNet18 + MediaPipe pipeline in `src/inference/`.
- **Gamification engine:** Frontend `frontend/src/game/` computes XP/streaks/badges optimistically; significant events sync to the API (debounced, not per webcam frame).

### Database location & initialization

- Local dev default: SQLite at `data/asl_quest.db` (created automatically on backend startup, no setup required).
- Recommended for production: PostgreSQL, configured via `DATABASE_URL`. Both engines run the same schema and code — see **[DATABASE.md](DATABASE.md)** for full setup, Alembic commands, backup procedure, and the SQLite → PostgreSQL import script.
- Schema for the original ten tables (users, progress, achievements, practice sessions, word progress, etc.) is still created/updated automatically on backend startup, same as before. Newer tables (native sign vocabulary, chatbot, model versioning — added as database foundation for future phases, not yet used by any route) are created only by running `alembic upgrade head`; see DATABASE.md.
- The database file is gitignored; do not commit user data.

Set optional environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `ASL_QUEST_SECRET_KEY` | JWT signing secret | dev-only placeholder |
| `DATABASE_URL` | SQLAlchemy URL (e.g. `postgresql+psycopg://user:pass@host/db`) | unset |
| `ASL_QUEST_DATABASE_URL` | Legacy alias for `DATABASE_URL`, kept for backward compatibility | unset |
| `ASL_QUEST_CORS_ORIGINS` | Allowed frontend origins | `http://127.0.0.1:5173,http://localhost:5173` |
| `ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL` | Email that receives `admin` role on registration | unset |

If neither `DATABASE_URL` nor `ASL_QUEST_DATABASE_URL` is set, the app falls back to `sqlite:///data/asl_quest.db`.

### Authentication flow

1. User registers or logs in via `/auth/register` or `/auth/login`.
2. Backend returns a JWT access token.
3. Frontend stores the token in `localStorage` (`asl-quest-token`).
4. Protected requests send `Authorization: Bearer <token>`.
5. `/auth/me` returns the current user profile.

### API endpoints (app layer)

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/auth/register` | No | Create account |
| POST | `/auth/login` | No | Login |
| GET | `/auth/me` | Yes | Current user |
| GET | `/progress` | Yes | Full game state |
| PUT | `/progress` | Yes | Save full game state |
| GET | `/progress/letters` | Yes | Letter stats |
| PUT | `/progress/letters/{letter}` | Yes | Update one letter |
| POST | `/progress/migrate-local` | Yes | Import local progress if server is empty |
| GET | `/progress/migration-offer` | Yes | Whether import can be offered |
| GET | `/challenges` | Yes | Challenge progress |
| POST | `/challenges/{type}/progress` | Yes | Update challenge |
| GET | `/achievements` | Yes | Badge list + earned state |
| POST | `/practice/session` | Yes | Record a practice attempt |

ML endpoints (`/health`, `/predict`, `/history`) remain **unchanged** and do not require authentication.

### Learning analytics (Phase 3)

Authenticated analytics endpoints under `/analytics/*` aggregate real practice session and XP event data:

| Method | Path | Description |
| --- | --- | --- |
| GET | `/analytics/dashboard` | Consolidated dashboard payload |
| GET | `/analytics/overview` | Summary stats |
| GET | `/analytics/activity` | Practice activity trend |
| GET | `/analytics/accuracy` | Accuracy trend |
| GET | `/analytics/letters` | A–Z letter stats |
| GET | `/analytics/letters/{letter}` | Letter detail |
| GET | `/analytics/heatmap` | GitHub-style activity heatmap |
| GET | `/analytics/history` | Paginated practice history |
| GET | `/analytics/xp` | XP analytics |
| GET | `/analytics/streak` | Streak analytics |
| GET | `/analytics/insights` | Strongest/weakest letters |
| GET | `/analytics/funnel` | Learning funnel |
| GET | `/analytics/weekly-summary` | Week-over-week summary |

Practice attempts are stored in `practice_sessions`; XP history is stored in `xp_events`. Analytics never use fabricated or localStorage data.

### Full-stack platform (Phase 4)

Phase 4 completes ASL Quest as a multi-user educational web application while preserving the ML pipeline unchanged.

**Current AI scope:** The model recognizes **static ASL alphabet gestures A–Z only**. It is not a full ASL translator.

| Area | Description |
| --- | --- |
| Profile | `/users/me/profile` — username, XP, streaks, mastery, ASL journey stats from the database |
| Settings | Preferences, change password, logout, export, delete account |
| Recommendations | `/recommendations` — deterministic letter suggestions from real practice data |
| Achievements | Server-side badge evaluation on progress sync; badges cannot be earned twice |
| Admin | Role-based admin dashboard at `/admin/*` with aggregate system analytics |
| Export | `/users/me/export` — JSON export of the requesting user's learning data only |
| Notifications | Lightweight toast feedback for XP, sync, achievements, and errors |

**Roles:** Users default to `student`. Set `ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL` to the email used at registration to create the first admin account.

**New API endpoints (Phase 4):**

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/users/me/profile` | Yes | Full profile + journey stats |
| GET/PUT | `/users/me/preferences` | Yes | User preferences (e.g. reduced motion) |
| POST | `/users/me/change-password` | Yes | Change password |
| DELETE | `/users/me` | Yes | Delete account (requires password + `DELETE` confirmation) |
| GET | `/users/me/export` | Yes | Export learning data (JSON) |
| GET | `/recommendations` | Yes | Personalized letter recommendations |
| GET | `/admin/overview` | Admin | System aggregate stats |
| GET | `/admin/users` | Admin | Paginated user management |
| GET | `/admin/activity` | Admin | Practice activity over time |
| GET | `/admin/letters` | Admin | Popular / difficult letters |
| GET | `/admin/recent-activity` | Admin | Recent learner activity |

**Frontend routes (logged in):** Home, Learn, Practice, Challenges, Progress, Achievements, Profile, Settings, Admin (admin only).

**Security:** Passwords are bcrypt-hashed; JWT bearer auth protects user routes; admin routes verify `role=admin` server-side; users can only access their own data; exports exclude authentication secrets.

**OpenAPI docs:** `http://127.0.0.1:8000/docs` — endpoints grouped by Authentication, Users, Practice, Progress, Challenges, Achievements, Analytics, Admin, Recommendations.

**Database additions:** `users.role`, `users.preferences`; indexes on practice sessions and XP events. Existing data is migrated in place — the database is not wiped.

**Practice sessions (Phase 4J):** One meaningful sign evaluation = one `practice_sessions` row. Webcam frames are never stored.

**Learning session concept (Phase 4K):** Deferred as future work — the current architecture records individual attempts cleanly without a separate session entity.

### localStorage migration

On first login, if the browser has existing ASL Quest progress (`asl-learning-progress-v1`) **and** the server account has no meaningful progress, the app offers to import it. If server progress already exists, server data wins and local progress is not overwritten automatically.

## Word learning (Phase 5)

The current ML model recognizes static A–Z alphabet signs. Word practice is implemented as sequential alphabet-sign spelling rather than direct word-level sign recognition.

Learners study curated vocabulary on **Learn → Words**, then spell each word one letter at a time in **Word Practice**. Predictions still come from the existing ResNet18 A–Z classifier (0.70 confidence + majority-vote smoothing). The app validates the current target letter, advances only on a correct letter, and assembles the word when the sequence is complete.

| Area | Description |
| --- | --- |
| Catalog | Static beginner vocabulary (Greetings, Family, Everyday, Beginner, School) |
| Persistence | `word_progress` and `word_practice_sessions` tables, scoped to the authenticated user |
| XP | Letter XP uses existing formulas (`word_letter_correct`); completion uses `word_completed`, `word_perfect`, `word_fast`, `word_challenge_completed` |
| Challenges | Word Challenge (5 words) plus deterministic daily spelling words |
| Analytics | Words learned/completed, accuracy, practice time, category bars, letter breakdown |

**Word API endpoints:**

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/words` | Yes | Word catalog + user progress |
| GET | `/words/{word_id}` | Yes | Word detail |
| GET | `/words/progress` | Yes | All word progress for the current user |
| GET | `/words/{word_id}/progress` | Yes | One word's progress |
| POST | `/words/practice/session` | Yes | Record a spelling session (completion XP only) |
| GET | `/words/recommendations` | Yes | Deterministic word suggestions |
| GET | `/words/analytics` | Yes | Word analytics |
| GET | `/words/challenge` | Yes | Daily or word-challenge word lists |
| GET | `/admin/words` | Admin | Aggregate word statistics |

Database additions are created in place (`create_all` + indexes). Existing users and letter practice data are preserved.

## Architecture (ML)

- **Model:** ResNet18, 224×224 ImageNet-normalized RGB, 26-way classifier.
- **Training data:** `dataset/asl_alphabet_train` (26 folders A–Z, 3,000 images each, 78,000 total).
- **App checkpoint:** `models/best_model.pth` (metadata-upgraded copy of the legacy weights).
- **Legacy checkpoint (untouched originals):** `asl_resnet18_best.pth` and `models/legacy_asl_resnet18_best.pth`.
- **Inference:** `src/inference/engine.py` (`Predictor`) used by CLI, webcam, and FastAPI. The backend loads the model once at startup.
- **Live cropping:** MediaPipe Hand Landmarker (`models/hand_landmarker.task`) crops a hand region before classification when available.
- **Frontend:** React + Vite + Tailwind in `frontend/`, proxied to FastAPI at `/api`.

## Setup

Use the existing Python 3.11 environment:

```bat
venv311\Scripts\python.exe --version
```

Frontend (once):

```bat
cd frontend
npm install
```

Do not use `dataset/asl_alphabet_test` as a benchmark. It has only 28 images; `A_test.jpg` and `B_test.jpg` are duplicates; some filenames do not match the visible letter. The folder is left unmodified.

## How to run

### Backend

From the project root:

```bat
venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Health check: `http://127.0.0.1:8000/health`

### Frontend

```bat
cd frontend
npm run dev
```

Open `http://localhost:5173`. Sign in or create an account to save progress. Live practice sends frames with `require_hand=true` so the backend crops the hand, returns `no_hand_detected` when none is found, and the UI applies confidence thresholding plus majority-vote smoothing.

### Image inference

```bat
venv311\Scripts\python.exe predict.py --image path\to\image.jpg
```

### Webcam

```bat
venv311\Scripts\python.exe webcam.py
```

Preview is mirrored for comfort; classification uses the unmirrored camera frame. Press `Q` to quit. `--no-hands` classifies the full frame if the hand detector is unavailable.

### Evaluation (reproducible holdout)

```bat
venv311\Scripts\python.exe evaluate.py
```

This writes:

- `outputs/evaluation/metrics.json`
- `outputs/evaluation/classification_report.txt`
- `outputs/evaluation/per_class_metrics.csv`
- `outputs/evaluation/confusion_matrix.png`
- `outputs/evaluation/splits.json`
- `outputs/splits.json` (same full sequential split)

The split is **70 / 15 / 15 per class by numeric filename order** (54,600 / 11,700 / 11,700). It does not shuffle, does not modify the dataset, and does not use horizontal flipping. A previous 26-image smoke split is ignored.

### Training (optional, writes v2 weights only)

```bat
venv311\Scripts\python.exe train.py
```

New weights go to `models/asl_resnet18_v2_best.pth`. Legacy and `best_model.pth` are not overwritten.

### Tests

```bat
venv311\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm run build
```

## Classes

A B C D E F G H I J K L M N O P Q R S T U V W X Y Z

Space / nothing / del are not included. Those class folders are not in the training set.

## Evaluation metrics — historical (original checkpoint, verified 14 Sep 2026)

Checkpoint evaluated: `models/best_model.pth` on NVIDIA GeForce RTX 4050 Laptop GPU. **This original checkpoint file was later lost and is no longer present at this path.** See "Evaluation metrics — current production checkpoint" below for what `models/best_model.pth` refers to today.

### Legacy random-split number (not a holdout)

The checkpoint stored `accuracy` / `val_accuracy` = **98.22%**. That value came from the original training run’s **80/20 random validation split** and can leak near-duplicate frames. It is **not** the controlled test result below, and it describes a checkpoint that no longer exists on disk.

### Controlled sequential holdout (historical)

| Metric | Value |
| --- | --- |
| Test images | 11,700 (450 per class) |
| Overall accuracy | **97.91%** |
| Macro precision | **98.12%** |
| Macro recall | **97.91%** |
| Macro F1 | **97.90%** |

Weakest per-class recall / accuracy on this historical holdout:

- J: 80.67% (363 / 450)
- P: 89.56% (403 / 450)
- N: 90.89% (409 / 450)
- B: 93.56% (421 / 450)

These historical numbers were produced by `evaluate.py` running inference on the holdout images at the time. They are not hand-entered, but they describe the original checkpoint above, not the checkpoint currently at `models/best_model.pth`.

## Evaluation metrics — current production checkpoint

`models/best_model.pth` currently refers to a **recovered/retrained** production checkpoint — the original checkpoint above was lost and this one was retrained from `dataset/asl_alphabet_train/` using the same architecture, preprocessing, and evaluation procedure. Its controlled test result, on the same 11,700-image sequential holdout:

| Metric | Value |
| --- | --- |
| Test images | 11,700 (450 per class) |
| Overall accuracy | **90.91%** |
| Macro precision | **93.65%** |
| Macro recall | **90.91%** |
| Macro F1 | **90.22%** |

Weakest per-class recall / accuracy on this holdout:

- S: 27.78%
- X: 35.33%
- V: 69.78%
- Y: 79.11%

These current numbers were produced by `evaluate.py` running inference on the same holdout images and are recorded in `outputs/evaluation/metrics.json`. They are not hand-entered.

## Limitations

- The current ML model recognizes static A–Z alphabet signs. Word practice is implemented as sequential alphabet-sign spelling rather than direct word-level sign recognition.
- Static letters only; no native ASL word signs, sentences, or continuous translation.
- **Dynamic J and Z are not supported.** The model may still output “J” or “Z” from a still pose; it does not use temporal motion.
- Live MediaPipe cropping can miss hands (including some already-cropped dataset photos). The UI then shows **No hand detected** instead of guessing.
- Webcam/live predictions use a 0.70 confidence threshold and a 7-frame majority vote to reduce flicker. This does not add a fixed sleep delay; the next frame is sent after the previous response.
- The deployed classifier was trained with `RandomHorizontalFlip` in the legacy pipeline. Current training code no longer uses that augmentation; a v2 retrain has not been run.
- `dataset/asl_alphabet_test` is not a valid benchmark.
- Exact-duplicate leakage across the sequential split has not been fully hashed in this pass (`scripts/check_split_leakage.py` can do that; it is slow on 78,000 files).
