# ASL Quest — Complete Project Documentation

A gamified, multi-user web platform for learning American Sign Language (ASL) with real-time webcam feedback.

- **Repository:** https://github.com/GobsonJR/ASL-Quest (working branch: `david-dev`, stable: `main`)
- **Stack:** React 19 + Vite + Tailwind (frontend) · FastAPI + SQLAlchemy (backend) · PyTorch + MediaPipe (ML) · SQLite (dev) / PostgreSQL (production)

---

## 1. What the project does

ASL Quest teaches ASL through three learning tracks, all wrapped in a game layer (XP, levels, streaks, badges, challenges):

| Track | What the learner does | How it is recognised |
|---|---|---|
| **A–Z Alphabet** | Shows each static letter to the webcam | ResNet18 image classifier + MediaPipe hand crop |
| **Word Spelling** | Spells a word (e.g. CAT, HELLO) one letter at a time | The same A–Z classifier, applied letter by letter |
| **Native Signs** | Performs a real whole-word ASL sign (e.g. BOOK, HELLO, WATER) as a short video | Pretrained I3D video network + a trained linear head |

A built-in chatbot (**AURA**) answers questions about the project, and a mentor/admin dashboard shows learner analytics.

### Honest scope and limitations

- A–Z recognition is **static images only**. **J and Z (which involve motion) are not supported** by the alphabet model.
- Word Spelling is **not** word-level recognition; it is sequential letter recognition.
- Native Signs covers a **10-sign vocabulary** only: HELLO, THANKYOU, PLEASE, YES, NO, MOTHER, WATER, EAT1, HELP, BOOK.
- The A–Z model was validated on dataset images, not extensively on real webcam frames, so accuracy in a new room/lighting may be lower than the numbers below.

---

## 2. Architecture

```
React frontend (Vite, Tailwind, Recharts)
        │   JWT bearer auth, /api proxy (dev) or VITE_API_BASE_URL (prod)
        ▼
FastAPI backend (backend/)
        ├── ML routes      /health /predict /history        (A–Z, unauthenticated)
        ├── Native routes  /native/signs /native/predict ...
        ├── App routes     /auth /progress /challenges /achievements /practice
        │                  /users /recommendations /words /analytics /admin /chatbot
        └── Dev-only       /collect (labelled webcam data collection)
        │
        ├── SQLAlchemy ORM ──► SQLite (data/asl_quest.db)  or  PostgreSQL (DATABASE_URL)
        ├── src/inference/    ResNet18 + MediaPipe pipeline (loaded once at startup)
        └── src/i3d_transfer/ I3D video inference for Native Signs
```

Key design rules:
- **Server database is the source of truth** for progress. Browser `localStorage` is only an offline cache.
- Webcam frames/clips are **never stored**; only the result of an attempt is saved.
- XP and correctness for native practice are decided **server-side**, never trusted from the client.
- The gamification engine (`frontend/src/game/`) computes XP/streaks optimistically; significant events sync to the API (debounced, not per frame).

---

## 3. Repository layout

```
backend/            FastAPI app
  main.py             app + CORS + router registration
  settings.py         env-driven configuration
  database.py, models.py, schemas.py, security.py
  routers/            achievements, admin, analytics, auth, challenges, chatbot,
                      collect, ml, native, practice, progress, recommendations, users, words
  services/           analytics, admin_stats, mentor_dashboard, progress, words,
                      user_features, chatbot_llm / chatbot_local_llm / chatbot_knowledge,
                      experimental_recognizer, live_diagnostics, webcam_dataset
  words/catalog.py    spelling-word catalogue
frontend/           React app
  src/pages/          Home, Learn, Alphabet, Practice, Words, WordPractice, Challenges,
                      Progress, Achievements, Profile, Settings, Admin, Auth
  src/native/         Native Signs catalogue, practice page, video capture
  src/components/     AppShell, CameraPractice, overlays, Dialog, Signo companion, ...
  src/game/           XP / streak / badge engine and constants
  src/chatbot/, src/admin/, src/analytics/, src/collect/
src/                ML code
  inference/          engine.py (Predictor), hands.py (MediaPipe), stability.py (smoothing)
  i3d_transfer/       pytorch_i3d.py, preprocessing.py, predict.py
  word_model/, landmarks/, asl_citizen/, training/, data/, models/
models/             checkpoints (git-ignored): best_model.pth, hand_landmarker.task, pretrained/
migrations/         Alembic migrations
scripts/            training, evaluation, diagnostics, seeding, Postgres migration
tests/              pytest suite (auth, analytics, native, words, chatbot, live pipeline, ...)
deployment/         staging area + manifest for large model files
```

---

## 4. Machine-learning components

### 4.1 A–Z alphabet recogniser
- **Model:** ResNet18 fine-tuned on 26 classes (`models/best_model.pth`, ~43 MB).
- **Data:** `dataset/asl_alphabet_train` — 26 folders × 3,000 images = 78,000 images.
- **Pipeline:** MediaPipe Hand Landmarker (`models/hand_landmarker.task`) crops the hand → ResNet18 classifies.
- **Live behaviour:** confidence threshold **0.70** with **majority-vote smoothing** (`src/inference/stability.py`) so a letter must be stable across frames before it counts.
- **Measured accuracy** (11,700-image held-out set, current webcam-robustness fine-tune): overall **94.08%**, macro precision 95.87%, macro F1 93.49% (up from 90.91% before the fine-tune). Weakest letter: X (~18% recall), then T, M, S. The original 97.91% checkpoint was lost, so that figure no longer describes the model on disk. These numbers come from dataset images recorded in one setup and do not prove live-webcam accuracy.
- Alternative recogniser: a MediaPipe-landmark classifier (`ASL_QUEST_RECOGNIZER=landmarks`), kept experimental until it beats ResNet on unseen rooms.

### 4.2 Native Signs recogniser (I3D)
- **Backbone:** Microsoft ASL Citizen pretrained I3D (MIT licence), **frozen** (`ASL_citizen_I3D_weights.pt`, ~58 MB).
- **Head:** a trained linear classifier on the 10 native signs (`head_best.pt`, ~43 KB).
- **Result:** frozen I3D features reach **96.9% test top-1 accuracy** on the native-10 set (see `NATIVE_10_PHASE_7_REPORT.md`).
- **Data:** ASL Citizen videos (licensed separately by Microsoft Research; never stored in this repo).
- Earlier experiments (from-scratch, appearance augmentation, layer-4 fine-tune, 100-class) are documented in the `NATIVE_10_PHASE_*` reports and `NATIVE_ASL_STRATEGY_REVIEW.md`.

### 4.3 Native practice grading
A learner records a clip; the backend runs `predict_video`, then compares against the target sign from the database. An attempt counts as **correct if the target is the model's top prediction or within its top 3** (relaxed because webcam clips differ from the training footage). Correct attempts award +20 XP; mastery is the running accuracy.

---

## 5. Features in detail

### 5.1 Learning and practice
- **Learn:** per-letter guidance and hints, an alphabet learning path, beginner letters (C, D, F, H, O).
- **Practice:** live webcam recognition with per-letter feedback and a reference/feedback panel for wrong guesses.
- **Word Spelling:** curated words in categories *Beginner, Greetings, Family, Everyday, School*; advances only on the correct letter; word completion, perfect and fast bonuses; a 5-word Word Challenge and deterministic daily words.
- **Native Signs:** catalogue of 10 signs with embedded reference videos and tips; a **Watch → Copy → Check** flow. Recording now has a **3-2-1 get-ready countdown and a 10-second window** (with a "Stop now" button); results show prediction, confidence, top-k, response time and mastery.

### 5.2 Gamification
| Constant | Value |
|---|---|
| XP per level | 1000 |
| Correct letter / fast correct (< 5 s) | 20 / 30 XP |
| Daily complete / perfect daily | 100 / 150 XP |
| Alphabet challenge / speed challenge | 500 / 150 XP |
| Word completed / perfect / fast / challenge | 50 / 25 / 20 / 150 XP |
| Daily target / speed target | 10 / 5 letters |
| Letter "learning" / "mastered" | 3 / 10 correct |

Plus streaks, badges (server-side, cannot be earned twice), challenges, celebration overlays and a "Signo" companion.

### 5.3 Analytics and recommendations
Server-backed (never fabricated): dashboard, overview, activity and accuracy trends, per-letter stats and detail, response time, heatmap, history, XP, streak, insights (strongest/weakest letters), funnel, weekly summary, word analytics, and deterministic letter/word recommendations.

### 5.4 Accounts, profile and admin
- Register/login with **bcrypt** passwords and **JWT** tokens (default expiry 7 days).
- Profile, preferences (e.g. reduced motion), change password, export own data as JSON, delete account (requires password + typed `DELETE`).
- Roles: `student` (default) and `admin`. The first admin is created by setting `ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL` before registering that email.
- Admin and **Mentor Dashboard**: aggregate system stats, user management, activity, popular/difficult letters and words.

### 5.5 AURA chatbot
Answers only questions about ASL Quest; a deterministic scope guard runs before any LLM call. Providers: **OpenRouter** (online, needs `OPENROUTER_API_KEY`), **local Ollama** (`qwen3:4b`, fully offline) or **auto**. If no provider is available it falls back to a static knowledge base, and the reply is tagged `source: "llm"` or `"knowledge_base"`.

### 5.6 Dev-only tooling
`/collect` (enabled by `ASL_QUEST_DEV_COLLECT`) saves labelled webcam frames for retraining; `ASL_QUEST_DEV_DIAGNOSTICS` saves exact classifier inputs for debugging. Both are off by default; collected data is git-ignored because it contains faces and rooms.

---

## 6. Database

SQLAlchemy ORM; same schema on SQLite and PostgreSQL. Alembic manages migrations (`alembic upgrade head`). Core tables are also auto-created at startup; newer ones need Alembic.

| Group | Tables |
|---|---|
| Users | `users` (with `role`, `preferences`), `user_preferences` |
| Progress | `user_progress`, `letter_progress`, `challenge_progress` |
| Achievements | `achievements`, `user_achievements` |
| Practice | `practice_sessions`, `xp_events` |
| Words | `words`, `word_progress`, `word_practice_sessions` |
| Native signs | `native_signs`, `native_sign_references`, `native_sign_progress`, `native_sign_sessions` |
| ML bookkeeping | `model_versions`, `sign_predictions` |
| Chatbot | `chatbot_conversations`, `chatbot_messages`, `chatbot_feedback` |

Details, backup procedure and the SQLite→Postgres import script (`scripts/migrate_sqlite_to_postgres.py`) are in `DATABASE.md`.

---

## 7. API reference (summary)

Interactive docs: `/docs` (OpenAPI). All routes except `/health`, `/predict`, `/history` and `/auth/register|login` need `Authorization: Bearer <token>`.

| Area | Endpoints |
|---|---|
| Auth | `POST /auth/register`, `POST /auth/login`, `GET /auth/me` |
| ML (A–Z) | `GET /health`, `POST /predict`, `GET /history` |
| Progress | `GET/PUT /progress`, `GET /progress/letters`, `PUT /progress/letters/{letter}`, `POST /progress/migrate-local`, `GET /progress/migration-offer` |
| Challenges / badges | `GET /challenges`, `POST /challenges/{type}/progress`, `GET /achievements` |
| Practice | `POST /practice/session` |
| Words | `GET /words`, `/words/{id}`, `/words/progress`, `/words/{id}/progress`, `/words/recommendations`, `/words/analytics`, `/words/challenge`, `POST /words/practice/session` |
| Native | `GET /native/signs`, `GET /native/progress`, `GET /native/progress/{id}`, `POST /native/predict` |
| Analytics | `/analytics/dashboard, overview, activity, accuracy, letters, letters/{l}, response-time, history, heatmap, streak, xp, insights, funnel, weekly-summary` |
| Users | `/users/me/profile`, `/users/me/preferences`, `POST /users/me/change-password`, `DELETE /users/me`, `/users/me/export` |
| Recommendations | `GET /recommendations` |
| Admin | `/admin/overview, activity, letters, words, users, recent-activity, mentor-dashboard` |
| Chatbot | `/chatbot/status`, conversations CRUD, `POST .../messages`, `POST /messages/{id}/feedback` |
| Dev | `/collect/status, sessions, analyze, samples` |

---

## 8. Configuration (environment variables)

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` / `ASL_QUEST_DATABASE_URL` | SQLAlchemy URL, e.g. `postgresql+psycopg://user:pass@host/db` | local SQLite |
| `ASL_QUEST_SECRET_KEY` | JWT signing secret — **must be set in production** | insecure dev placeholder |
| `ASL_QUEST_TOKEN_EXPIRE_MINUTES` | Token lifetime | 10080 (7 days) |
| `ASL_QUEST_CORS_ORIGINS` | Comma-separated allowed frontend origins | localhost:5173 |
| `ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL` | Email that becomes admin on registration | unset |
| `ASL_QUEST_RECOGNIZER` | `resnet` or `landmarks` | `resnet` |
| `OPENROUTER_API_KEY`, `CHATBOT_MODEL` | AURA online provider | unset / `openrouter/free` |
| `CHATBOT_PROVIDER`, `CHATBOT_LOCAL_BASE_URL`, `CHATBOT_LOCAL_MODEL` | `openrouter` / `local` / `auto`; Ollama URL and model | `openrouter`, `127.0.0.1:11434`, `qwen3:4b` |
| `ASL_QUEST_DEV_COLLECT`, `ASL_QUEST_DEV_DIAGNOSTICS` | Dev tooling toggles | off |
| `VITE_API_BASE_URL` (frontend, build time) | Public backend URL in production | dev proxy `/api` |
| `VITE_DEMO_MODE` (frontend, build time) | Demo mode | false |

---

## 9. Running locally (Windows)

1. Clone and set up: `scripts\windows\setup.ps1` (checks Python/Node, creates `venv311`, installs dependencies).
2. Copy the four model files (section 10) into `deployment\models\` and run `scripts\windows\install_deployment_assets.ps1`.
3. Create the database: `venv311\Scripts\python.exe -m alembic upgrade head`, then `scripts\seed_native_signs_native_10.py`.
4. Backend: `venv311\Scripts\python.exe -m uvicorn backend.main:app --reload` → http://127.0.0.1:8000/docs
5. Frontend: `cd frontend && npm install && npm run dev` → http://localhost:5173

More detail: `SETUP_WINDOWS.md`, `TEAM_HANDOFF.md`, `README.md`.

---

## 10. Required model files (not in Git)

| File | Destination | Size | Used for |
|---|---|---|---|
| `best_model.pth` | `models/` | ~43 MB | A–Z ResNet18 |
| `hand_landmarker.task` | `models/` | ~7.5 MB | MediaPipe hand crop |
| `ASL_citizen_I3D_weights.pt` | `models/pretrained/asl_citizen_i3d/` | ~58 MB | Native I3D backbone |
| `head_best.pt` | `outputs/asl_citizen_native_10/experiments/i3d_frozen/checkpoints/` | ~43 KB | Native linear head |

---

## 11. Testing

- **Backend:** `pytest` in `tests/` — auth/DB, progress sync, analytics, phase 4/5 features, words, native signs, chatbot, mentor dashboard, live pipeline, stability, landmarks, webcam collection/training, checkpoint promotion.
- **Frontend:** `npm test` (Vitest + Testing Library) — Learn, Practice, Alphabet, WordPractice pages, CameraPractice states, AppShell, Dialog, recogniser switch.
- **Type-check/build:** `npm run build` runs `tsc --noEmit` then `vite build`.
- Utility validation scripts live in `scripts/` (`smoke_test.py`, `validate_app.py`, `validate_native_i3d_predict.py`, ...).

---

## 12. Deployment plan (Vercel + Render + PostgreSQL)

1. **PostgreSQL (Render):** create the database; use its Internal URL with the `postgresql+psycopg://` prefix as `DATABASE_URL`.
2. **Backend (Render Web Service, branch `david-dev`):**
   - Build: `pip install -r requirements.txt` (prefer `opencv-python-headless`)
   - Pre-deploy: `alembic upgrade head`
   - Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Env: `DATABASE_URL`, `ASL_QUEST_SECRET_KEY`, `ASL_QUEST_CORS_ORIGINS`, `OPENROUTER_API_KEY`
   - Needs ~1.5–2 GB RAM (torch) → Starter plan or larger; model files must be placed on a persistent disk or downloaded at build time because they are git-ignored.
3. **Frontend (Vercel):** root directory `frontend`, env `VITE_API_BASE_URL=<Render URL>`; `frontend/vercel.json` already sets the Vite build and a camera permissions header.
4. Add the final Vercel URL to `ASL_QUEST_CORS_ORIGINS` and redeploy the backend.

Status: not yet deployed.

---

## 13. Security and privacy
- bcrypt password hashing; JWT-protected routes; admin role enforced server-side; users can only read/write their own data.
- Exports exclude credentials; account deletion removes the user's data.
- No webcam frames or native clips are persisted (clips go to a temp file and are deleted immediately).
- Large/licensed assets, `.env`, databases and collected webcam data are git-ignored.

---

## 14. Recent changes (this branch)
- Native sign practice: added a 3-2-1 countdown with live preview, extended recording from 3 s to **10 s**, and relaxed grading to **top-3 matching** so lessons such as BOOK are completable.
- Redesigned learning experience, offline AURA fallback, free-form AURA conversations, embedded native sign reference videos, improved camera-error handling.

## 15. Known limitations and future work
- Add word-level/dynamic recognition (J, Z and more native signs beyond the current 10).
- Validate and retrain the A–Z model on real webcam data from varied rooms (use the `/collect` tooling).
- Add an SPA rewrite rule in `vercel.json` if deep-link refreshes 404.
- Automate model-file delivery for deployment (`render.yaml` + download script).

## 16. Related documents
`README.md` · `DATABASE.md` · `SETUP_WINDOWS.md` · `TEAM_HANDOFF.md` · `PROJECT_PROGRESS_SUMMARY.md` · `deployment/README.md` · `NATIVE_10_PHASE_4B/6/7_REPORT.md` · `NATIVE_ASL_STRATEGY_REVIEW.md`
