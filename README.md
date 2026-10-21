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

## AURA (project chatbot)

AURA answers questions about how ASL-Quest works (A-Z recognition, ResNet18, MediaPipe, Word Spelling, Native Signs, I3D, ASL Citizen, the learning path, XP/streaks/mastery/achievements, the database, FastAPI/React architecture, testing, limitations) and rejects anything unrelated to the project — a deterministic backend scope guard runs before any LLM is ever called, for both providers below. Configured via `CHATBOT_PROVIDER` in `.env` (see `.env.example` for all variables); leaving it unset keeps the original ONLINE-only behavior.

### ONLINE mode (OpenRouter)

Default when `CHATBOT_PROVIDER` is unset or `openrouter`. Requires `OPENROUTER_API_KEY` (OpenRouter's free-tier models work — see `.env.example`). Leave the key unset to disable AURA gracefully (it replies with a setup message instead of crashing the app).

### OFFLINE AURA SETUP (local Ollama)

Runs AURA with **no internet connection**, using a local [Ollama](https://ollama.com/download/windows) server and the `qwen3:4b` model. Same scope guard, same project knowledge, same conversation history as ONLINE mode — only the LLM transport changes, and OpenRouter is never contacted while this mode is active.

1. Install Ollama: https://ollama.com/download/windows
2. Pull the model once (this step requires internet; the model itself is never downloaded by this project's own code — Ollama manages it):
   ```bat
   ollama pull qwen3:4b
   ```
3. Make sure Ollama is running:
   ```bat
   ollama run qwen3:4b
   ```
4. Set in `.env`:
   ```
   CHATBOT_PROVIDER=local
   CHATBOT_LOCAL_BASE_URL=http://127.0.0.1:11434
   CHATBOT_LOCAL_MODEL=qwen3:4b
   ```
5. Start the backend and frontend as usual (see "How to run" below). AURA now answers entirely offline.

`CHATBOT_PROVIDER=auto` prefers the local model whenever Ollama is reachable and falls back to OpenRouter only if it's configured.

### Offline knowledge-base fallback (no LLM required at all)

If neither a reachable local Ollama server nor a configured OpenRouter key is available (or the active provider errors, times out, or is rate-limited), AURA does **not** just apologize — it answers from `backend/services/chatbot_knowledge.py`'s static project knowledge base instead, matching the question's keywords to the most relevant knowledge section (A-Z recognition, Word Spelling, Native Signs, gamification, database, backend, frontend, or the project overview). This is what keeps AURA useful during a fully offline demo even with Ollama stopped and no API key set. Each reply's `source` field (`"llm"` or `"knowledge_base"`) tells the frontend which path answered it, shown as an "Offline knowledge base" badge. If a question is in scope but matches no known section, AURA says so honestly ("I don't have enough information about that part of ASL-Quest yet.") rather than guessing — it never invents facts either way.

## How to run

### Backend

From the project root:

```bat
venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

`--env-file .env` loads your local `.env` (e.g. `CHATBOT_API_KEY`); drop it if you have no `.env` file. `scripts\windows\start_backend.ps1` adds it automatically when `.env` exists.

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

`models/best_model.pth` is a **webcam-robustness fine-tune (19 Oct 2026)** of the recovered/retrained checkpoint that preceded it. The original 97.91% checkpoint was lost; its replacement (below, "previous") was retrained from `dataset/asl_alphabet_train/`. The previous checkpoint was fine-tuned for 3 epochs (seed 42, lr 5e-5) with `--augment robust` (mirroring, scale/rotation, lighting, blur, JPEG artifacts, noise, occlusion; training images only) because measurements showed it failed on mirrored hands (C: 0% correct, mean over all letters 37%) and on small hands (see "Diagnosing live webcam recognition"). Same 11,700-image sequential holdout, same images for both:

| Metric | Previous checkpoint | Current (`best_model.pth`) |
| --- | --- | --- |
| Test images | 11,700 (450 per class) | 11,700 (450 per class) |
| Overall accuracy | 90.91% | **94.08%** |
| Macro precision | 93.65% | **95.87%** |
| Macro recall | 90.91% | **94.08%** |
| Macro F1 | 90.22% | **93.49%** |
| C precision / recall / F1 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| Robustness, mean over letters: mirrored / hand at 0.4x size / noise | 37% / 32% / 92% | **93% / 74% / 92%** |

Weakest per-class recall on this holdout (current): X 18.44%, T 82.89%, M 84.89%, S 91.56%, J 92.89%, Z 94.00%.
Known regression: X recall fell from 35.33% (read mostly as A) to 18.44% (now read mostly as I), which also lowers I's precision to 0.55; the held-out X frames appear to differ from the training X frames, so X needs more varied training data. S rose from 27.78% to 91.56% and V from 69.78% to 99.33%.

The previous checkpoint is kept at `models/backup/best_model_before_robust_20261019.pth` (git-ignored, like all weights). To roll back: copy it over `models/best_model.pth` and restart the backend. Its evaluation files are in `outputs/evaluation_before_robust/`; `outputs/diagnostics/checkpoint_comparison_v2.json` has the full head-to-head, and `scripts/compare_checkpoints.py` reproduces it.

These numbers were produced by `evaluate.py` (and cross-checked by `scripts/compare_checkpoints.py`) on the same holdout images and are recorded in `outputs/evaluation/metrics.json`. They are not hand-entered. **They do not prove live-webcam accuracy**: the holdout comes from the same recording setup as the training photos.

## Diagnosing live webcam recognition

The held-out test set comes from the same recording setup as the training photos (one room, one
camera angle, the forearm visible, the hand shown side-on), so a letter can score 100% there and
still fail on a real webcam. These tools separate "the model is wrong" from "the live input is
different" (measured results for C are in the table below):

```powershell
# 1. Per-letter metrics, confusion row/column, label/duplicate/leakage checks (CPU is fine)
venv311\Scripts\python.exe scripts\diagnose_class.py --letter C --device cpu

# 2. Same-image experiments: standard vs live crop, similar letters, mirror/rotation/size/lighting
#    (stage 1 finds hands, stage 2 classifies; --server reuses a running backend, no torch needed)
venv311\Scripts\python.exe scripts\diagnose_c_experiments.py --stage prepare --letter C
venv311\Scripts\python.exe scripts\diagnose_c_experiments.py --stage classify --letter C --server http://127.0.0.1:8000

# 3. Your own webcam frame: run the frontend in dev mode (npm run dev), open Practice, hold the sign,
#    open "Technical details" and press "Save last frame". Then:
venv311\Scripts\python.exe scripts\diagnose_frame.py path\to\frame_C_<time>.jpg
```

#### Capturing a REAL failing prediction (development only)

To see exactly what the classifier received for a live webcam prediction, start the backend with the capture flag and use the app normally:

```powershell
$env:ASL_QUEST_DEV_DIAGNOSTICS = "1"
venv311Scriptspython.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --env-file .env
cd frontend; npm run dev          # in dev mode every request asks the server to capture itself
```

Hold the sign in Practice for a few seconds. For each (rate-limited, newest 60 kept) prediction the server writes
`outputs/diagnostics/live_requests/<id>/` containing the original frame bytes, the hand box overlay, the exact crop given to
ResNet18, the normalized model input converted back to an image, and `result.json` (top-5, boxes, sizes, whether the preview was
mirrored, checkpoint path and hash). All of it comes from the same request. Then run the four-way comparison (original frame /
classifier crop / mirrored crop / dataset images of the target letter):

```powershell
venv311Scriptspython.exe scriptsanalyze_live_requests.py --target C --last 12
```

The capture folder is git-ignored and contains your camera frames; it is never exposed through an endpoint. Without the environment
variable nothing is written.

#### Finding from a real webcam session (19 Oct 2026, target C)

60 captured requests; in the ~30 where the hand really formed a C, the app sent a crop that contained the hand **and the signer's face, glasses and a patterned couch**. With the current checkpoint (and identical preprocessing) the recorded crops were read as C in 1/30, while the same frames with everything except the hand replaced by a plain wall colour were read as C in 27/30 (`scripts/diagnose_background_dependence.py`). Controls: the same masking on held-out dataset hands of other letters predicts C in only 4-5/100, and on the non-C poses in the session in 1/30. The classifier therefore recognises the handshape but is misled by the scene around it (every training photo is one room with a plain wall). The same crop is classified identically every time (direct inference = server record, 30/30); nudging the crop window by 10% flips the label on real frames (2.9 distinct labels per frame) but never on dataset C (1.0). Dataset C is still recognised 30/30 (mean score 0.98), so a perfect dataset score says nothing about cluttered webcam scenes.

Not fixed by this finding (needs data, not code): a controlled experiment would record labelled webcam frames of every letter from your own camera against varied backgrounds (face visible, different rooms/lighting, left and right hand), keep whole sessions out of validation/test, fine-tune with those added, and compare old vs new with `scripts/compare_checkpoints.py` on a held-out webcam session. Masking hands at inference is NOT a fix: it lowers accuracy on other letters (about 94% to 65-69% on masked dataset hands).

`diagnose_frame.py` reports the hand box (and whether it touches the frame edge), then the top-5
predictions for the exact live pipeline, the whole frame, and square crops with several paddings,
each also mirrored, and saves those crops so you can look at what the model was given.

Measured with the deployed checkpoint on held-out dataset images (`scripts/diagnose_c_experiments.py`):

| Input | C accuracy | mean over all letters |
|---|---|---|
| standard pipeline / live square crop (pad 0.3, 0.6, 1.0) | 100% | 91% |
| mirrored (left-handed signer) | **7%** (read as G or Q) | **37%** |
| rotated ±20° / +45° | 100% / 100% | ~79% / ~24% |
| hand 0.6× / 0.4× its size in the frame | 100% / 60% | ~73% / ~31% |
| JPEG q72, darker/brighter, low contrast, noise | 100% | 89–93% |

`python -m src.training.train --augment robust` trains with mirroring, scale/rotation, lighting and
occlusion augmentation (training images only; validation/test are never augmented). Compare any two
checkpoints on identical images with `scripts/compare_checkpoints.py`. Use `--eval-workers 0` on
machines with little free memory.

## Making recognition work in ordinary rooms (real webcam data workflow)

Why: the classifier was trained on photos from one room, and measurements show it is misled by faces, glasses and furniture around the hand
(see "Finding from a real webcam session"). The fix needs labelled frames from YOUR camera in several rooms. Everything below is repeatable;
nothing is labelled by the model, and whole rooms are kept out of training so the test measures unseen rooms.

### 1. Collect (development only)

```powershell
$env:ASL_QUEST_DEV_COLLECT = "1"
venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --env-file .env
cd frontend; npm run dev        # open the app > More > Collect data
```

Per room: start a session (room name, lighting, background, face/glasses). Choose a letter, make the sign, press Capture, check the crop and the
quality verdict, then press "Yes, this is X, save". Blurry, too-small, too-dark, no-hand and near-duplicate frames are rejected for you (the
sharpness threshold was calibrated on real captured crops). Use "Capture hold" once per letter per room for a 10-frame hold sequence, which is
used to measure how stable a held sign is. Frames are saved under `dataset/webcam/` (git-ignored: they contain your face and room).

Suggested protocol: at least **5-6 different rooms or backgrounds** (shelves, curtains, patterned wall, furniture, window light, a person behind
you; some with glasses, some not), 15-30 varied single frames per letter per room (vary distance, angle, hand position, both hands) and 1-2 hold
sequences per letter per room. That reaches roughly 100-200 usable frames per letter across training rooms. Keep **at least one whole room only
for testing** (and one more for validation if you can). Many near-identical frames are worth far less than a few genuinely different ones.

### 2. Export (room-separated, same crop as the live app)

```powershell
venv311\Scripts\python.exe scripts\export_webcam_dataset.py --auto
# or choose: --test-rooms kitchen --val-rooms hall
```
The report lists per-split letter coverage, skipped frames, leakage checks, and whether the hand you declared agrees with MediaPipe's handedness.
With only one room it refuses to pretend it can test generalisation.

### 3. Baseline, then two approaches, then compare on the SAME unseen rooms

```powershell
# the current model on your unseen rooms
venv311\Scripts\python.exe scripts\evaluate_webcam.py --approach resnet --checkpoint models\best_model.pth --name resnet_current

# Approach A: fine-tune the image model with the original dataset + your webcam crops
venv311\Scripts\python.exe -m src.training.train --augment robust --init-from models\best_model.pth --webcam-export dataset\webcam_export `
  --epochs 4 --lr 5e-5 --batch-size 32 --num-workers 0 --eval-workers 0 `
  --best-model models\experiments\az_webcam_best.pth --last-model models\experiments\az_webcam_last.pth `
  --history outputs\experiments\az_webcam_history.json --curves outputs\experiments\az_webcam_curves.png
venv311\Scripts\python.exe scripts\evaluate_webcam.py --approach resnet --checkpoint models\experiments\az_webcam_best.pth --name resnet_webcam

# Approach B: MediaPipe landmark classifier (no images, no torch)
venv311\Scripts\python.exe scripts\train_landmark_classifier.py
venv311\Scripts\python.exe scripts\evaluate_webcam.py --approach landmarks --name landmarks

venv311\Scripts\python.exe scripts\compare_approaches.py outputs\evaluation_webcam\resnet_current.json outputs\evaluation_webcam\resnet_webcam.json outputs\evaluation_webcam\landmarks.json
```
`evaluate_webcam.py` reports accuracy, macro-F1, per-letter precision/recall/F1, the confusion matrix, the share rejected as uncertain, accuracy among
accepted predictions, accuracy per room / background / hand / distance / lighting, latency, and, on hold sequences, how soon the app's own
confirmation rule fires and how often it confirms a WRONG letter. Read the wrong confirmations: an approach that confidently confirms wrong letters
is worse than one that abstains. Do not choose on a single number.

### 3b. Try a landmark model live, side by side with the ResNet (experimental, per request)

```powershell
$env:ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL = "models/experiments/landmark_L2_s0.joblib"   # must be a .joblib under modelsvenv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --env-file .env
```
ResNet stays the default for every request. When the model loads, `/health` reports `experimental_landmark` and the Practice page shows a
"Recognizer · developer comparison" box with **ResNet (default)** and **Hand landmarks (experimental)**; it starts on ResNet on every page load and
**Back to ResNet** is one click. `POST /predict?recognizer=landmarks` asks for the experimental engine for that one call; without the parameter nothing
changes. A missing or incompatible file only hides the box (the reason is in `/health`), and a hand the detector cannot find, unusable landmarks or a
model error return a normal "no hand" answer or a 503 for that frame instead of crashing the request. Practice results still count toward progress while
the experimental engine is selected. Frames sent by the camera are JPEG quality 0.72, slightly lower than the 0.92 of collected frames; in a measured run
about 5% of frames lost hand detection at 0.72, so expect a little more "no hand" than the offline evaluation shows.

### 4. Promote only with evidence; roll back in one command

```powershell
# Approach A winner: the gate refuses unless it beats the current model on the same rooms and confirms no more wrong letters
venv311\Scripts\python.exe scripts\promote_checkpoint.py --candidate models\experiments\az_webcam_best.pth `
  --eval-candidate outputs\evaluation_webcam\resnet_webcam.json --eval-current outputs\evaluation_webcam\resnet_current.json
venv311\Scripts\python.exe scripts\promote_checkpoint.py --rollback         # undo (the current model is backed up every time)

# Approach B winner: switch the API instead (default is the image model)
$env:ASL_QUEST_RECOGNIZER = "landmarks"     # uses models\landmark_classifier.joblib; fails loudly (503) if it cannot load
```
Restart the backend after either. `/health` reports `recognizer`, `landmark_model_loaded`, and the checkpoint path. The previous production model is
also kept at `models/backup/best_model_before_robust_20261019.pth`.

### 5. Demo checklist (ordinary room)

1. Start the backend and open http://localhost:5173; confirm `/health` shows `model_loaded: true` and `hand_detection_available: true`.
2. Practice > a letter: the chips should show "Hand detected" and a live "Reading".
3. Good conditions help every recognizer: even front lighting, hand about an arm's length away filling roughly a third of the frame, fingers visible.
4. If the sign is uncertain the app says so (it never accepts below 0.70 confidence); hold steady for about a second.
5. Try letters you collected plus a couple you did not, with a face and furniture in view; keep the Technical details panel open in dev to show
   the "no hand / uncertain / recognized" rates.
6. Know the limits: J and Z need motion and are only recognised as held shapes; X, T and M have been weak in measurements.

## Limitations

- The current ML model recognizes static A–Z alphabet signs. Word practice is implemented as sequential alphabet-sign spelling rather than direct word-level sign recognition.
- Static letters only; no native ASL word signs, sentences, or continuous translation.
- **Dynamic J and Z are not supported.** The model may still output “J” or “Z” from a still pose; it does not use temporal motion.
- Live MediaPipe cropping can miss hands (including some already-cropped dataset photos). The UI then shows **No hand detected** instead of guessing.
- Webcam/live predictions use a 0.70 confidence threshold and a 7-frame majority vote to reduce flicker. This does not add a fixed sleep delay; the next frame is sent after the previous response.
- The default training policy (`--augment default`) has no horizontal flip. The deployed checkpoint is a fine-tune with `--augment robust`, which adds mirroring and other webcam-style augmentation (see the production-checkpoint section). It was validated only on dataset images and perturbed copies of them, **not on real webcam frames**: collect labelled frames from your own camera (dev "Save last frame" button) before trusting it for a new environment.
- `dataset/asl_alphabet_test` is not a valid benchmark.
- Exact-duplicate leakage across the sequential split has not been fully hashed in this pass (`scripts/check_split_leakage.py` can do that; it is slow on 78,000 files).
