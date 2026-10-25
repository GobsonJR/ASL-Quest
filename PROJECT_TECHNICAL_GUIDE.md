# ASL Quest: Full Technical Guide

This guide describes the application as it exists in this repository. It is intended to give a new developer enough context to understand the product, architecture, technology choices, data flow, storage, machine-learning components, and local development workflow without first reverse-engineering every folder.

> **Scope reminder:** ASL Quest is an ASL learning/practice platform with several recognition modes. Its A–Z classifier recognizes static alphabet handshapes; Word Practice spells words by recognizing one alphabet letter at a time; Native Signs recognizes short video clips for a limited vocabulary. It is not a general-purpose ASL sentence translator.

## 1. Project at a glance

ASL Quest is a multi-user web application for learning American Sign Language. Learners can study letter forms, practice alphabet signs with a webcam, spell curated words, practice a small vocabulary of whole-word signs, complete challenges, earn XP and achievements, and review learning analytics. An in-scope assistant called AURA answers questions about the project. Admin and mentor views provide system and learner summaries.

### Main technology stack

| Layer | Technology | Role |
|---|---|---|
| Browser UI | React 19, TypeScript 5, Vite 7 | Single-page application and development/build toolchain |
| Styling | Tailwind CSS 4 through the Vite plugin, plus project CSS variables | Layout, responsive styles, theme tokens and component styling |
| Charts | Recharts 3 | Analytics and dashboard charts |
| Browser tests | Vitest 5, Testing Library, jsdom | Component and behavior tests |
| HTTP/API | Fetch API, JSON, multipart `FormData` | Browser-to-FastAPI communication and image/video upload |
| Backend API | Python, FastAPI, Uvicorn | HTTP endpoints, dependency injection, validation, OpenAPI docs |
| Request/response validation | Pydantic (provided through FastAPI) | API payload parsing and response schemas |
| Relational database | SQLite locally; PostgreSQL supported | Persistent user accounts, practice history, progress, catalog and assistant data |
| ORM and schema changes | SQLAlchemy 2, Alembic | Database models, queries, relationships and migrations |
| Authentication | bcrypt, JWT (`python-jose`, HS256), bearer tokens | Password hashing and stateless API authentication |
| Alphabet ML | PyTorch, Torchvision ResNet18, Pillow, NumPy | Static A–Z image classification |
| Hand detection | MediaPipe Hands / Hand Landmarker Tasks | Optional hand presence and square hand crop before alphabet classification |
| Video ML | PyTorch I3D implementation, OpenCV, NumPy | Clip decoding and whole-sign video classification |
| Optional assistant providers | OpenRouter-compatible chat API or local Ollama | AURA language model answers; static local knowledge fallback exists |
| Deployment targets/configuration | Vercel frontend configuration; documented Render/PostgreSQL backend plan | Static frontend hosting and a separately hosted Python API/database |

The exact dependency ranges and scripts live in [frontend/package.json](frontend/package.json) and [requirements.txt](requirements.txt). They use minimum/range constraints rather than pinning every Python dependency to an exact version.

## 2. Product features and what they mean technically

### Learning tracks

1. **Alphabet learning:** reference material and a guided learning path for A–Z.
2. **Alphabet practice:** the browser opens the webcam, captures still frames, and sends them to `POST /predict`. Each response contains a top prediction, confidence, top candidates and optional hand information. The browser's stability tracker requires repeated consistent predictions before treating a letter as confirmed.
3. **Word spelling:** a word such as `CAT` is practiced as `C`, then `A`, then `T`. Each step uses the alphabet recognition endpoint. This is sequential fingerspelling practice, not direct whole-word recognition.
4. **Native signs:** the browser records a short webcam clip (with a countdown and a maximum 10-second capture window), then posts it to `POST /native/predict`. The server runs video classification against the configured native vocabulary.
5. **Gamification:** XP, streaks, daily goals, challenges, word completion rewards and achievement badges make practice progress visible. Browser interactions are optimistic for responsiveness, while persistent records and native-sign grading are handled by the backend.
6. **Analytics and recommendations:** backend queries aggregate stored practice and XP data; recommendations are deterministic rules based on recorded learning progress.
7. **AURA:** an authenticated project chatbot with a deterministic project-scope check, optional remote/local LLM generation and a static project knowledge fallback.
8. **Admin / mentor tools:** role-protected user, dataset/content/model operations and aggregate learning dashboards. `/collect` is a development-oriented webcam-data tool and is disabled unless explicitly enabled in backend configuration.

### Important model scope and limits

- The alphabet model produces one of the 26 static letters. Dynamic **J** and **Z** require motion and are outside this image model's capability.
- Letter spellings in Word Practice are recognized one pose at a time.
- Native Signs uses an I3D video model and the classes in the loaded native head checkpoint. The currently documented native-10 vocabulary is `HELLO`, `THANKYOU`, `PLEASE`, `YES`, `NO`, `MOTHER`, `WATER`, `EAT1`, `HELP`, and `BOOK`; the loaded class map/checkpoint is authoritative at runtime.
- A model confidence score is a model output, not a guarantee that an interpretation is correct. Dataset evaluation is not proof of robust performance in all webcam environments.
- Production deployment is documented/planned in repository docs, but the project guide should not be read as proof that an active production environment is deployed.

## 3. System architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│ Browser: React + TypeScript + Tailwind                             │
│ Auth context | Game context | pages | camera capture | local cache  │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ Fetch: JSON / multipart upload
                               │ JWT Bearer for protected app endpoints
                      /api proxy in Vite (local)
                      VITE_API_BASE_URL (deployed)
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│ FastAPI / Uvicorn                                                   │
│ routers → services → SQLAlchemy models/session                     │
│   ├─ alphabet still image inference: ResNet18 (+ optional MediaPipe)│
│   ├─ native video inference: I3D backbone + trained class head     │
│   ├─ auth/progress/words/challenges/analytics/admin/chatbot         │
└──────────────┬───────────────────────┬─────────────────────────────┘
               │                       │
        SQLite (local)          PostgreSQL (supported)
```

The frontend is a single-page React application. The main screen selection is held in the game/application state rather than using a route library. The Vite development server listens on port 5173 and proxies `/api/*` to `http://127.0.0.1:8000`, removing the `/api` prefix. In a deployed frontend build, `VITE_API_BASE_URL` points directly to the backend origin.

The backend creates a FastAPI application in `backend/main.py`, installs CORS middleware, registers routers, initializes the database and seeds achievements at startup, then loads the A–Z model and hand detector once. Inference objects are kept in process memory for reuse. Native I3D predictors are cached per checkpoint pair on first use.

The database is the persistent source for authenticated progress, learner attempt records, catalog and reporting. The browser uses `localStorage` as a local cache and offline fallback. It can offer a one-time import of meaningful local progress when a newly logged-in server account has no meaningful server data.

## 4. Repository map

```text
backend/
  main.py                 FastAPI application, middleware, routers and startup loading
  settings.py             Environment configuration and defaults
  database.py             Engine/session, DB dependency, compatibility schema setup
  models.py               SQLAlchemy table definitions and relationships
  schemas.py              Pydantic request/response shapes
  security.py             bcrypt, JWT encoding/decoding, current-user/admin dependencies
  routers/                HTTP API handlers, grouped by product area
  services/               Business logic, analytics, chatbot providers, diagnostics, etc.
  words/catalog.py        Curated spelling vocabulary source
src/
  config.py               Shared ML paths and training/inference constants
  models/resnet18.py      Torchvision ResNet18 factory
  data/                   Dataset and image preprocessing for alphabet model
  inference/              A–Z Predictor, MediaPipe detector, smoothing/stability logic
  training/               A–Z training/evaluation entry points
  landmarks/              Experimental hand-landmark feature/classifier approach
  i3d_transfer/           Native-sign I3D backbone, video preprocessing and inference
  asl_citizen/             ASL Citizen dataset tooling, manifests and experiments
  word_model/              Separate experimental word-model code
frontend/
  src/main.tsx, App.tsx   React entry point and top-level application selection
  src/auth/               Auth API, context and token helpers
  src/game/               Client game state, XP rules, local storage and sync
  src/pages/               Main learner pages
  src/native/              Whole-word signs UI, API and video capture
  src/components/          Shared layouts, camera UI, feedback and dialogs
  src/analytics/            Analytics API and charts
  src/admin/                Admin/mentor UI and API
  src/chatbot/              AURA user interface and API
  src/words/                Word catalog UI support and API
migrations/                Alembic environment and versioned DB migrations
scripts/                    Training, dataset audit, diagnostics, seed and migration utilities
tests/                      Python backend/ML tests
outputs/                    Experiment reports, metrics, diagnostic artifacts and logs
models/                      Model/pretrained asset area (many large model assets are untracked)
dataset/                    Training/collected dataset area (not a runtime source of user data)
```

## 5. Frontend architecture

### Entry and state providers

- `frontend/src/main.tsx` mounts `<App />` under React `StrictMode` and imports the global stylesheet.
- `frontend/src/App.tsx` composes `AuthProvider`, then selects login/registration or the authenticated app. Authenticated screens are wrapped in toast, game and Signo companion providers.
- `AuthContext` restores the saved token/session, manages registration/login/logout and offers a local-progress migration when appropriate.
- `GameContext` owns the current selected screen and game progress actions. `frontend/src/game/gamification.ts` contains game state updates and rewards.
- Navigation is represented by `NavPage` values in `frontend/src/game/constants.ts`; `App.tsx` renders the selected page. This is not a URL-router architecture.

### Frontend areas

| Folder | Responsibility |
|---|---|
| `pages/` | Home, Learn, Alphabet, Practice, Words, Word Practice, Challenges, Progress, Achievements, Profile, Settings and Admin pages |
| `native/` | Native Sign catalog/practice and `MediaRecorder` webcam clip capture |
| `components/` | App shell, A–Z `CameraPractice`, reference and mastery panels, toast/dialog/celebration, migration prompt and Signo companion |
| `auth/` | Login/register API and context, bearer-token handling |
| `game/` | Game/progress types, XP rules, achievements, local persistence and server synchronization |
| `analytics/` | Analytics endpoint client and Recharts components |
| `admin/` | Admin and mentor dashboard interfaces and related API calls |
| `chatbot/` | AURA conversation interface and endpoint client |
| `words/` | Word catalog API/types and catalog support |
| `shared/` | Recognition stability, camera errors and HTTP error helpers |
| `collect/` | Development data-collection UI |

### Browser camera and recognition loop

For alphabet practice, the browser obtains camera permission through `navigator.mediaDevices.getUserMedia`. Camera code periodically captures a frame, typically encodes a JPEG, and calls `predictImage()` in `frontend/src/api.ts`. The request is multipart and can include `require_hand=true`; the API client does not send a JWT for `/predict`, which is a public inference endpoint. The app can display hand boxes, errors and confidence feedback.

`frontend/src/shared/recognition.ts` implements stable-letter acceptance. It uses a rolling vote window and safeguards for no-hand frames, weak confidence and substantial hand movement. The Python counterpart in `src/inference/stability.py` mirrors these constants for repeatable offline evaluation; confirmation is performed in the browser for the live learner interaction.

For Native Signs, `NativeVideoCapture.tsx` requests camera video only (no audio), previews it, waits through a 3-second countdown, then records up to 10 seconds using `MediaRecorder`. Supported WebM/MP4 MIME types are selected from the browser's advertised support. The resulting Blob is posted to the authenticated native prediction endpoint. Camera tracks are stopped after recording or reset.

### Browser state, storage and synchronization

- `localStorage` key: `asl-learning-progress-v1` for cached game/progress state.
- `localStorage` key: `asl-quest-token` for the JWT token (see `frontend/src/auth/token.ts`).
- State is written locally first, then full progress sync is debounced (default 900 ms) to avoid a request for every frame or small action.
- Practice attempts can also be logged individually to `/practice/session`. If logging fails offline, the current UI continues; aggregate state sync is a separate path.
- The server's progress response can replace/update the browser cache after a successful sync.
- Native practice grading is computed server-side from the model output and actual target sign row, not from a client-submitted `correct` flag.
- Demo mode is a build-time UI mode (`VITE_DEMO_MODE=true`); it is a local demo profile and does not create a server account or make backend-dependent features real.

### Frontend build and tooling

- `npm run dev`: Vite development server on port 5173.
- `npm run build`: `tsc --noEmit` then `vite build`.
- `npm run preview`: preview the generated static bundle.
- `npm test`: Vitest test run.
- Tailwind is wired through `@tailwindcss/vite`; the project also defines theme/design tokens in `src/index.css`.
- Vite plugin React performs JSX/React integration. TypeScript is configured in `tsconfig.json`.

## 6. Backend architecture

### FastAPI application lifecycle

`backend/main.py` creates the app, sets title/version and CORS origins, and includes routers. At startup it:

1. Creates the data directory and runs the compatibility schema initializer.
2. Seeds achievement definitions.
3. Loads the best A–Z checkpoint (`models/best_model.pth`, falling back to the legacy root checkpoint) and initializes a hand detector.
4. Optionally loads an explicitly selected landmark recognizer or a side-by-side experimental landmark checkpoint.
5. Stores the runtime objects in the ML router module so each request reuses them.

The common database dependency `get_db()` creates a SQLAlchemy session for a request and closes it in `finally`. Router functions use FastAPI dependencies for the session and current user. `schemas.py` describes validated HTTP payloads; `models.py` describes persistence entities.

### Backend module responsibilities

| Module | Responsibility |
|---|---|
| `routers/auth.py` | Registration, login and current-user API |
| `routers/ml.py` | Health, A–Z image prediction, recent in-memory prediction history and optional diagnostics |
| `routers/practice.py` | Persist a letter practice attempt and optional XP event |
| `routers/progress.py` | Read/write account progress, letter progress and local migration flows |
| `routers/challenges.py`, `achievements.py` | Challenge progress and achievement listing/evaluation support |
| `routers/words.py` | Catalog, spelling progress/session logging, recommendations and word analytics |
| `routers/native.py` | Native-sign catalog/progress and video inference/practice outcome persistence |
| `routers/analytics.py` | Learner-level aggregates over practice and XP records |
| `routers/recommendations.py` | Deterministic letter recommendations |
| `routers/users.py` | Profile/preferences, password changes, account export/delete |
| `routers/admin.py`, `admin_ops.py` | Admin and mentor aggregate operations, content/model/user/dataset actions and audit trail |
| `routers/chatbot.py` | AURA status, conversation CRUD, message generation and feedback |
| `routers/collect.py` | Development-only webcam collection endpoints; disabled by default |
| `services/analytics.py` | Analytics query/aggregation functions |
| `services/progress.py` | Server progress rules, XP/level and achievement data helpers |
| `services/words.py` | Word progress, completion and recommendation calculations |
| `services/chatbot_*.py` | AURA scope/knowledge, OpenRouter and local Ollama integrations |
| `services/webcam_dataset.py`, `live_diagnostics.py` | Opt-in developer sample/diagnostic file handling |

### HTTP and API behavior

FastAPI exposes interactive OpenAPI documentation at `/docs` and schema at `/openapi.json` when running locally. The principal endpoint families are:

| Prefix / endpoint | What it serves | Authentication |
|---|---|---|
| `GET /health` | Model, detector, compute-device and service health | Public |
| `POST /predict` | One still image classified as an alphabet letter | Public |
| `GET /history` | Bounded recent in-process alphabet prediction history | Public |
| `/auth/*` | Register, login, current user | Register/login public; current user protected |
| `/progress/*` | Per-account overall and per-letter learning progress, migration | Bearer token |
| `/practice/session` | One A–Z practice attempt and XP event | Bearer token |
| `/challenges/*`, `/achievements/*` | Challenge state/reward and badge state | Bearer token |
| `/words/*` | Spelling catalog, per-user progress, attempts, analytics and recommendations | Bearer token |
| `/native/*` | Whole-sign catalog, practice, progress and video predictions | Bearer token |
| `/analytics/*` | User dashboards, trends, history and learning insights | Bearer token |
| `/recommendations` | Letter-practice suggestions | Bearer token |
| `/users/me/*` | Profile, preferences, export, password change and deletion | Bearer token |
| `/admin/*` | Aggregate/admin operations | Admin bearer token |
| `/chatbot/*` | AURA conversations, messages, status and feedback | Bearer token |
| `/collect/*` | Local labelled webcam collection and inspection | Feature flag enabled; dev use |

Exact methods, response fields and query options should be confirmed from the current OpenAPI document or the relevant router. `/predict` accepts multipart image data and optional query fields such as `require_hand`, `debug`, `top_k`, `save_diagnostics`, `target`, `preview_mirrored`, `video_size` and `recognizer`. The normal frontend request uses the default ResNet route; landmark classification is optional/experimental.

### CORS and API URL

`backend/settings.py` sets allowed exact origins. When `ASL_QUEST_CORS_ORIGINS` is set, it replaces the defaults. Credentials are enabled, so the configured origins are explicit rather than wildcard `*`. Locally Vite's `/api` proxy avoids browser cross-origin configuration for normal frontend requests. Production browser builds use `VITE_API_BASE_URL` directly, and the backend must allow the deployed frontend origin.

## 7. Authentication and security model

1. Registration stores a bcrypt password hash, never the plaintext password.
2. Login verifies the supplied password against the stored bcrypt hash.
3. The backend issues an HS256 JWT with the numeric user ID in `sub` and expiry in `exp`.
4. The browser stores the access token and sends `Authorization: Bearer <token>` on protected app requests.
5. `get_current_user` decodes and validates the token, loads the user and rejects missing/inactive accounts.
6. `get_current_admin` additionally checks the persisted role. Admin authorization is enforced in the backend, not only by hiding UI controls.
7. Database rows for user progress are scoped to the authenticated user. Related account records generally use foreign keys with cascading deletion.

The token expiration default is 10,080 minutes (7 days). `ASL_QUEST_SECRET_KEY` must be set to a private random value for deployment; the source's fallback is only suitable for local development. Browser `VITE_*` variables are bundled into public frontend code and must never contain secrets. Password/export/account deletion behavior is implemented in the user endpoints. Native uploaded clips are written temporarily for video decoding and removed in a `finally` block; normal webcam frames/clips are not retained as learner media. Development collection and diagnostic modes intentionally write files when explicitly enabled and should be treated as sensitive because they may contain a person's hand, face, room or background.

## 8. Database and data model

### Engine and schema lifecycle

- SQLAlchemy 2 maps Python classes to relational tables.
- `DATABASE_URL` selects the database; `ASL_QUEST_DATABASE_URL` is a legacy alias. With neither set, the local default is `sqlite:///data/asl_quest.db`.
- SQLite connections use `check_same_thread=False`; PostgreSQL connections use a 15-second connect timeout.
- Alembic migrations live in `migrations/versions/` (`0001` through `0004` currently).
- On application startup, `backend/database.py` also ensures a defined set of legacy tables and compatibility columns/indexes exist. Newer schema is intended to be managed by Alembic. Use `alembic upgrade head` for a fully migrated database.
- `scripts/migrate_sqlite_to_postgres.py` is provided for copying a local database to PostgreSQL.

### Main entities

| Tables | Purpose and key relationships |
|---|---|
| `users` | Account identity, unique username/email, password hash, role, JSON-like preferences text, active flag and creation time |
| `user_progress` | One overall progress row per user: XP, level, streak, last active day, totals and serialized game metadata |
| `letter_progress` | Per-user letter attempts/correct counts/mastery, unique on `(user_id, letter)` |
| `practice_sessions` | Individual A–Z practice attempt: target, prediction, correctness, response time, XP, challenge, confidence and timestamp |
| `xp_events` | Append-style reason and amount records used for XP history/analytics |
| `challenge_progress` | Per-user challenge progress/completion and metadata |
| `achievements`, `user_achievements` | Shared achievement definitions and unique user awards |
| `words` | Spelling catalog database table, seeded from `backend/words/catalog.py` |
| `word_progress` | Per-user aggregate spelling progress and per-letter summary text, unique on `(user_id, word_id)` |
| `word_practice_sessions` | Word attempt results, duration, completion, mistakes and serialized letter results |
| `native_signs` | Native whole-word gloss catalog, category/difficulty/model availability and active status |
| `native_sign_references` | Reference pointers, source type and license note; can reference external links without redistributing source videos |
| `native_sign_progress` | Per-user attempts, accuracy, mastery and best measurements per native sign |
| `native_sign_sessions` | Individual video practice result and inference latency |
| `model_versions`, `sign_predictions` | Schema for model metadata/prediction tracking; existence of these tables does not mean every model is registered or every inference is written there |
| `user_preferences` | Separate theme/language/notification/learning preference structure |
| `chatbot_conversations`, `chatbot_messages`, `chatbot_feedback` | Per-user AURA conversation history and feedback |
| `admin_audit_log` | Administrative action records, actor and target metadata |

Most structured fields have normal relational types and foreign keys. Some flexible metadata and lists are stored as JSON-encoded strings in `Text` columns rather than native JSON fields. The word-progress ID fields intentionally match string IDs in the code catalog. Account deletion uses cascades for user-owned records, with audit actor links allowed to become null.

### Distinguish the stores

- **Relational DB:** accounts, practice/session records, persistent game totals, catalog entries and reports.
- **Browser localStorage:** auth token and local cache of game progress; not used as the analytics source of truth.
- **Model files:** local filesystem assets loaded by the backend; not stored as DB rows or served from a client-side model.
- **Dataset files:** training/evaluation material and optional developer-collected samples; not ordinary per-user practice media.
- **In-memory prediction history:** `/history` is a bounded process-local deque, not durable database history. It resets on server process restart and is not a substitute for user attempt records.

## 9. Machine-learning system

### A. Static alphabet image classifier

**Model:** Torchvision ResNet18, with the final fully connected layer replaced by a 26-output linear layer. The model returns logits, which are converted to softmax probabilities and mapped to class labels. At runtime `Predictor` loads the checkpoint, sets `eval()` mode, moves it to the selected CPU/CUDA device and wraps prediction in `torch.no_grad()`.

**Image preparation:** RGB input is resized to 224 × 224, converted to a tensor, and normalized with ImageNet channel means `(0.485, 0.456, 0.406)` and standard deviations `(0.229, 0.224, 0.225)`. `src/data/preprocessing.py` defines both standard and opt-in robust training transforms. Evaluation uses deterministic resize/tensor/normalization.

**Dataset/training:** the documented primary training source is `dataset/asl_alphabet_train`, with 26 class directories and 3,000 images per class (78,000 total). Training and evaluation tooling is in `src/training/`, `src/data/`, and `scripts/`. Dataset splits and experiment artifacts are in `outputs/`. Standard training augmentations avoid horizontal flipping; the opt-in `robust` policy includes flips, affine transformations, lighting/color changes, blur, grayscale, JPEG artifacts, noise and erasing. The repository contains many training experiments; the checkpoint that actually exists at the configured path is the model used by runtime, not every model in experiment outputs.

**Hand localization:** by default, frontend live prediction sends `require_hand=true` in practice. The backend optionally runs MediaPipe on the RGB frame, extracts the detected hand landmark bounding box, pads and squares the crop while preserving image geometry, fills out-of-frame regions by edge replication and classifies that crop. If hand detection is unavailable or no hand is found, the API returns an explicit status and no prediction. Without `require_hand`, the image is classified as received. MediaPipe compatibility code supports both legacy `solutions.hands` and the Tasks API, using `models/hand_landmarker.task` for the latter.

**Confidence:** the model uses a 0.70 threshold for its `low_confidence` response flag. This is a feedback/acceptance signal, not calibrated certainty. Live acceptance adds temporal stability logic in the frontend (7-vote window, minimum stable votes, no-hand grace, confidence reset and box-overlap movement handling; see `frontend/src/shared/recognition.ts`).

**Alternative:** a MediaPipe landmark feature/classifier path exists in `src/landmarks/` and backend service wiring. It is not the default. `ASL_QUEST_RECOGNIZER=landmarks` selects a configured model globally; `ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL` enables optional per-request comparison. Treat it as experimental unless evaluation indicates it should replace the image model.

### B. Native whole-sign video model

**Inference stages:** the native endpoint accepts a video upload, checks extension/size, writes bytes to a temporary file, and calls `src/i3d_transfer/predict.py`. OpenCV-backed preprocessing decodes the clip and prepares an I3D input tensor (documented pipeline: 64 sampled frames, BGR convention, pixel range scaled to `[-1, 1]`, center crop to 224). A pretrained Microsoft ASL Citizen I3D backbone extracts spatiotemporal features; these are mean-pooled across time to a 1,024-dimensional feature vector; a trained linear head maps features to gloss classes. Softmax provides top-k scores. Backbone/head objects are cached after first load. The temporary upload is deleted even when inference raises an error.

**Training provenance:** experiment code and manifests are under `src/asl_citizen/`, `src/i3d_transfer/`, `scripts/` and `outputs/`. The pretrained ASL Citizen model and dataset have their own source/licensing terms. The repository holds a license note/source file but the licensed dataset videos are not included or redistributed here. The native endpoint depends on local checkpoint assets; a catalog entry alone does not make an inference class available in the loaded checkpoint.

**Practice grading:** when `native_sign_id` is supplied, backend resolves the active catalog target first, runs inference, and marks the attempt correct if the target gloss is top-1 or appears among the first three candidates. It records the attempt and updates per-sign progress from this actual prediction. A correct native attempt is worth 20 XP in the current implementation. The recorded clip is not kept. This top-3 rule is a learner-facing relaxed grading policy, not a top-1 model accuracy measure.

### C. Other ML folders

- `src/word_model/` is an additional word-model implementation/training area; the app's current Word Practice flow uses A–Z sequential letters rather than this as a direct whole-word model.
- `src/landmarks/` builds features/classifiers from hand keypoints and is an alternative/experimental A–Z recognizer.
- `src/asl_citizen/` includes dataset preparation, manifests, metrics and training/evaluation helpers for the native-sign experiments.
- Scripts and `outputs/` include exploratory model comparisons, diagnostic reports, benchmark records, confusion matrices and training histories. These are research/engineering artifacts; they are not all loaded by the web application.

### Device selection and inference cost

`src/utils/device.py` chooses CUDA when PyTorch reports it available, otherwise CPU. `/health` exposes device, CUDA/GPU details, checkpoint loading and hand-detector availability. CPU can run the app but video inference may be slower. ResNet is loaded at backend startup; native I3D is heavier and is lazily loaded/cached on first native prediction.

## 10. Learning rules and gamification

Current user-facing constants are in `frontend/src/game/constants.ts`. Key values include:

| Rule | Value |
|---|---:|
| XP per level | 1,000 |
| Correct alphabet attempt | 20 XP |
| Fast correct alphabet attempt (under 5 seconds) | 30 XP |
| Daily goal | 10 correct letters |
| Speed challenge target | 5 fast correct letters |
| Letter learning marker | 3 correct attempts |
| Letter mastery marker | 10 correct attempts |
| Word completion | 50 XP, with perfect/fast/challenge bonuses defined separately |
| Native correct sign practice | 20 XP (backend constant) |

The client game engine updates a responsive local view and schedules syncs. Backend services and endpoints persist progress and verify some outcomes (most notably native video target grading). XP events retain the reason for earned points so analytics can report the history. The achievement catalog is seeded server-side and awards are unique per user/achievement pair. Exact challenge behavior and the current reward values can be traced to `frontend/src/game/gamification.ts`, `backend/services/progress.py`, `backend/services/words.py`, and the native router.

## 11. AURA chatbot

The chatbot is deliberately scoped to ASL Quest project knowledge. The backend has:

1. A scope/intent guard and static curated knowledge/FAQ path (`chatbot_knowledge.py`, `chatbot_faq.py`).
2. An OpenRouter chat-completions client (OpenAI-compatible API) configured with `OPENROUTER_API_KEY`, `CHATBOT_MODEL` and optional `CHATBOT_API_BASE_URL`.
3. A local Ollama client configured by `CHATBOT_LOCAL_BASE_URL` and `CHATBOT_LOCAL_MODEL`.
4. Provider selection `openrouter`, `local`, or `auto`; unset defaults to OpenRouter behavior.
5. Conversation, message and feedback persistence in the database; the response metadata reports whether an LLM or knowledge base answered.

If the configured LLM path is unavailable or fails, the backend has a static project knowledge fallback. Local Ollama mode does not call OpenRouter. The model weights are managed separately by Ollama and are not downloaded by this application. AURA's scope check should not be confused with a general safety filter or a source of guaranteed factual correctness; its strongest source is the project's own maintained knowledge content.

## 12. Configuration reference

Backend variables are read in `backend/settings.py`; frontend variables are compile-time Vite variables in `frontend/src/config.ts`.

| Variable | Used by | Purpose/default |
|---|---|---|
| `DATABASE_URL` | Backend | SQLAlchemy database URL; preferred variable; defaults to local SQLite when absent |
| `ASL_QUEST_DATABASE_URL` | Backend | Legacy database URL fallback |
| `ASL_QUEST_SECRET_KEY` | Backend | JWT signing secret; development placeholder if unset; set securely in deployment |
| `ASL_QUEST_TOKEN_EXPIRE_MINUTES` | Backend | JWT lifetime; default 10,080 minutes (7 days) |
| `ASL_QUEST_CORS_ORIGINS` | Backend | Comma-separated allowed browser origins; setting replaces defaults |
| `ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL` | Backend | Email promoted to admin during registration logic |
| `ASL_QUEST_HAND_DETECTION_CONFIDENCE` | Backend | MediaPipe confidence threshold; default 0.35, clamped to 0.05–0.95 |
| `ASL_QUEST_RECOGNIZER` | Backend | Default recognizer `resnet`; optional `landmarks` |
| `ASL_QUEST_LANDMARK_MODEL` | Backend | Landmark model path override |
| `ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL` | Backend | Optional extra landmark model for comparison requests |
| `ASL_QUEST_DEV_COLLECT` | Backend | Enables development webcam collection (`1`, `true` or `yes`) |
| `ASL_QUEST_DEV_DIAGNOSTICS` | Backend | Enables writing request diagnostic images/artifacts |
| `OPENROUTER_API_KEY` | Backend | Secret key for remote AURA provider |
| `CHATBOT_PROVIDER` | Backend | `openrouter` by default, `local`, or `auto` |
| `CHATBOT_MODEL` | Backend | OpenRouter model; default `openrouter/free` |
| `CHATBOT_API_BASE_URL` | Backend | Optional OpenAI-compatible endpoint override |
| `CHATBOT_LOCAL_BASE_URL` | Backend | Ollama URL; default `http://127.0.0.1:11434` |
| `CHATBOT_LOCAL_MODEL` | Backend | Ollama model tag; default `qwen3:4b` |
| `VITE_API_BASE_URL` | Frontend build | Public backend origin; unset uses `/api` (dev proxy in development) |
| `VITE_DEMO_MODE` | Frontend build | `true` enables local demo profile behavior |
| `PORT` | Backend process | Hosting platform port used by `python -m backend.main`; documented Uvicorn production command also consumes it |

`.env.example` is the template. `.env` and `frontend/.env.local` can contain local secrets/settings and should stay untracked. Never put server-side secrets in `VITE_*` values.

## 13. Run locally on Windows

The repository provides Windows setup scripts and a detailed `SETUP_WINDOWS.md`. A typical flow:

1. Use the project virtual environment (the repo's documented setup uses Python 3.11) and install Python dependencies with `pip install -r requirements.txt` if the environment is not already provisioned.
2. Copy `.env.example` to `.env` and set a local secret key if desired. Leave `DATABASE_URL` empty to use SQLite.
3. Ensure required model files are placed at the documented paths (see below). Alphabet inference needs the A–Z checkpoint; hand crop support also needs the MediaPipe task model when the installed MediaPipe API requires it. Native inference additionally needs both the I3D backbone and native head checkpoints.
4. Apply schema migrations: `venv311\Scripts\python.exe -m alembic upgrade head`.
5. Start the API from the repository root: `venv311\Scripts\python.exe -m uvicorn backend.main:app --reload` (API docs at `http://127.0.0.1:8000/docs`).
6. In a second terminal, go to `frontend`, run `npm install` if needed, then `npm run dev` (app at `http://localhost:5173`).

The project also includes scripts in `scripts/windows/` for setup/startup and `SETUP_WINDOWS.md` documents model provisioning. Model files are large and some are not tracked by Git. Their presence and license must be checked before expecting each recognition mode to work.

### Model asset paths

| Asset | Expected role/path |
|---|---|
| A–Z checkpoint | `models/best_model.pth`; runtime falls back to repository-root `asl_resnet18_best.pth` |
| MediaPipe Tasks landmark model | `models/hand_landmarker.task` |
| I3D pretrained backbone | `models/pretrained/asl_citizen_i3d/ASL_citizen_I3D_weights.pt` |
| Native class head | `outputs/asl_citizen_native_10/experiments/i3d_frozen/checkpoints/head_best.pt` |
| Optional landmark classifier | `models/landmark_classifier.joblib` by default or configured path |

Run `GET /health` after startup to see whether the A–Z model and hand detector loaded. Native model loading is lazy, so native asset errors may only appear when a native clip is submitted. See `PROJECT_OVERVIEW.md`, `TEAM_HANDOFF.md`, `SETUP_WINDOWS.md` and `deployment/README.md` for current provisioning details.

## 14. Testing and quality checks

The repository has Python tests in `tests/` spanning auth/database, progress sync, analytics, admin, words, native video, chatbot, webcam collection/training, stability and recognition. Frontend tests use Vitest and Testing Library for pages, shared recognition behavior, camera states, native capture, chatbot and admin interactions.

Commands defined by the project:

```powershell
# Frontend production type-check + bundle
cd frontend
npm run build

# Frontend tests
npm test

# Backend tests (from repository root / activated project environment)
pytest
```

Utility validation, training and diagnostics are under `scripts/`. Do not treat exploratory diagnostic scripts as a replacement for the normal automated test suite. This guide was written from source inspection; commands above are available, but this document update does not assert that they were run.

## 15. Deployment shape and operational considerations

The documented intended split is a static frontend on Vercel, FastAPI/Uvicorn backend on a Python-capable host such as Render, and managed PostgreSQL. The frontend needs `VITE_API_BASE_URL` at build time. The backend needs `DATABASE_URL`, a strong `ASL_QUEST_SECRET_KEY`, and the deployed frontend origin in `ASL_QUEST_CORS_ORIGINS`. A deployment must also provision model checkpoints outside normal Git-tracked source if they are ignored/too large, and the backend must have enough memory for PyTorch and the video model.

Typical backend operations:

- Migrate schema with `alembic upgrade head` before serving a new version.
- Run `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` on a host that provides `PORT`.
- Keep database backups; `DATABASE.md` covers SQLite/PostgreSQL procedures and migration notes.
- Confirm `/health`, CORS, model assets and a representative inference after deployment.
- Keep data collection and diagnostic file capture off except for controlled development/debugging.

The repository's deployment plan is not an assertion of current production status. Check `deployment/README.md` and actual hosting configuration for live deployment details.

## 16. Core concepts for a new contributor

- **Frontend/backend boundary:** React handles presentation, webcam capture and responsive game state; FastAPI owns persistence and protected decisions. Model inference happens in Python on the backend, not in the browser.
- **Router/service/model:** routers parse HTTP and apply dependencies; services hold reusable business rules/aggregation; SQLAlchemy models map to tables.
- **ORM/session:** SQLAlchemy sessions perform transactions. A request obtains a session and closes it after completion.
- **Migration:** Alembic records incremental changes to a database schema. Model code alone does not upgrade an existing production database.
- **JWT:** a signed bearer token identifies the account on protected requests. It is not encryption and must be signed with a private secret.
- **Inference vs. learning progress:** a raw classifier prediction is different from a confirmed learner attempt. Frontend smoothing stabilizes letter predictions; an attempt record is created when the UI accepts a meaningful attempt.
- **Top-k:** a model may return several candidate labels. Alphabet UI generally uses the top prediction plus confidence; native grading currently allows a target in top three.
- **Training vs. inference preprocessing:** training-time augmentations create varied input examples; evaluation and inference should use the stable deterministic transform expected by the saved checkpoint.
- **Checkpoint vs. source code:** model architecture and checkpoint weights work together. A checkpoint is an external binary asset, often ignored by Git; changing architecture or labels requires compatible metadata/weights.
- **Catalog vs. model vocabulary:** a native sign can exist in the database catalog without being recognized by the currently loaded model head.
- **Local cache vs. source of truth:** localStorage improves responsiveness/offline continuity; authenticated persistent records and backend analytics come from the database.
- **Privacy boundary:** normal learner inference is transient; opt-in dataset/diagnostic paths are designed to write local files and require deliberate use.

## 17. Further reading

- [README.md](README.md): overview, setup and feature notes
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md): previous project overview
- [DATABASE.md](DATABASE.md): database configuration, schema and backup guidance
- [SETUP_WINDOWS.md](SETUP_WINDOWS.md): Windows environment and model setup
- [TEAM_HANDOFF.md](TEAM_HANDOFF.md): development handoff notes
- [WEBCAM_EVAL_PROTOCOL.md](WEBCAM_EVAL_PROTOCOL.md): held-out webcam evaluation protocol
- [deployment/README.md](deployment/README.md): deployment assets and current deployment notes
- [backend/main.py](backend/main.py), [backend/models.py](backend/models.py), [backend/routers/](backend/routers/), [src/inference/](src/inference/) and [frontend/src/](frontend/src/): implementation source

---

# Extended implementation reference

The sections below go beyond the architectural summary. They document the concrete screens, request families, database fields, and processing sequence that are useful when implementing or debugging a feature. The Python routers and Pydantic schemas remain the final authority when an API detail changes.

## 18. Frontend screen inventory and navigation

The app selects screens using `GameContext`'s `page` state and `App.tsx`'s conditional rendering. These names are UI state, not public URL paths.

| Page state | Component | Main responsibility / data interaction |
|---|---|---|
| `home` | `HomePage` | Learner landing screen, continue practice and overview of current activity |
| `learn` | `LearnPage` | Learning entry point, guided path and links into alphabet/word learning |
| `alphabet` | `AlphabetPage` | A–Z reference content and guided learning path; reads existing per-letter game state |
| `practice` | `PracticePage` | Live webcam A–Z practice via `CameraPractice`, recognition feedback, outcome and XP handling |
| `words` | `WordsPage` | Curated word catalog, categories, progress and suggested words |
| `practice` + `practiceKind === "word"` | `WordPracticePage` | Sequential spelling practice; each letter uses the same A–Z prediction flow |
| `native` | `NativeSignsPage` | Whole-sign catalog, reference media and per-sign progress |
| `native-practice` | `NativeSignPracticePage` | Watch/copy/check flow, recording, native upload request and result/mastery display |
| `challenges` | `ChallengesPage` | Daily, alphabet, speed and spelling challenge progress |
| `progress` | `ProgressPage` | Learning analytics and history charts |
| `achievements` | `AchievementsPage` | Badge catalog, earned state and progress toward awards |
| `assistant` | `ChatbotPage` | AURA conversations and feedback |
| `profile` | `ProfilePage` | Account profile and derived learner summary |
| `settings` | `SettingsPage` | Preferences, security actions, export/delete, logout |
| `admin` | `AdminPage` | Admin dashboard/management sections; protected server calls require admin role |
| `mentor-dashboard` | `MentorDashboard` | Aggregate teacher/mentor view |
| `collect` | `CollectPage` | Developer webcam collection interface; rendered only in a Vite development build |

The authentication shell is separate: no current user means login or registration UI; while auth is being restored, a loading view is shown. `SignoProvider`/`SignoCompanion` implement the assistant mascot UI and are independent of the AURA chatbot service.

### Main browser state objects

`GameState` contains:

- `xp`, `streak`, `longestStreak`, `lastPracticeDate`;
- `totalCorrect`, `totalAttempts`, and `sessions`;
- `letterStats`, keyed by the letter with per-letter correct and attempts counts;
- `unlockedBadges` and a short `recentAchievements` list;
- the date-scoped `dailyChallenge` state;
- `challenges`, including alphabet/speed/word challenge state.

Learning-path nodes (`completed`, `current`, `available`, `locked`) are derived from `letterStats`, not stored as a second mastery ledger. Current beginner letters are C, D, F, H and O; the path highlights the first unfinished beginner letter, and unlocks the remaining alphabet sequence after those are mastered. Open/free practice selection can still select other letters.

## 19. End-to-end request flows

### 19.1 Login and protected API request

```text
Login form
  → frontend auth API POST /auth/login {login, password}
  → FastAPI validates payload and finds username/email
  → bcrypt check against users.password_hash
  → backend signs HS256 JWT with sub=user id and exp=expiry
  → frontend stores token in localStorage
  → subsequent protected request adds Authorization: Bearer <token>
  → get_current_user decodes token, reloads User row and checks is_active
  → route executes for that account
```

Tokens are not refresh tokens: the client restores a session by calling `/auth/me` with the saved token. On a failed/expired token, protected calls fail and the UI's auth handling can return the user to the login flow.

### 19.2 A–Z live practice

```text
getUserMedia(video)
  → CameraPractice samples a still frame and encodes JPEG
  → POST /predict (multipart file, require_hand=true)
  → optional MediaPipe detection + padded square crop
  → ResNet18 preprocessing: RGB, resize 224×224, tensor, ImageNet normalization
  → PyTorch logits → softmax → top predictions + confidence threshold flag
  → browser StableLetterTracker combines frame votes and hand-box continuity
  → stable letter is compared with current target
  → GameContext computes local outcome (XP/streak/challenge/badges)
  → individual attempt POST /practice/session and debounced PUT /progress
  → backend saves practice_sessions/xp_events and overall/per-letter progress
```

No request per video frame is written to the database: individual prediction requests are transient, while the accepted learner event is logged. The application can still call inference frequently while the webcam view is open, so camera capture cadence and network latency affect interaction smoothness.

### 19.3 Word Practice

```text
GET /words[?category=...]
  → learner chooses word
  → target position i maps to word[i]
  → each captured image goes through POST /predict
  → stable correct letter advances to i+1; incorrect result does not advance
  → finish submits POST /words/practice/session with letter outcomes/duration/mistakes
  → service derives accuracy, mastery/progress, XP, challenge completion and badges
  → word_progress + word_practice_sessions (and associated events) are committed
```

The request payload shape is represented by `WordPracticeSessionCreate` and nested `WordLetterResult` in `backend/schemas.py`. The server does not infer the word directly from a continuous motion clip; the browser sends a summary of the sequence of letter attempts.

### 19.4 Native Sign practice

```text
GET /native/signs and GET /native/progress
  → learner selects catalog row (native_sign_id)
  → browser preview + 3-second countdown + MediaRecorder (up to 10 seconds)
  → multipart POST /native/predict?native_sign_id=<id>&top_k=5
  → server resolves active target from DB before inference
  → validate extension and 100 MB byte limit
  → write private temp video; decode/prepare 64-frame I3D clip
  → frozen I3D feature extraction → 1024-D temporal mean → native linear head
  → compare catalog gloss to top-1/top-3; store session/progress/XP/earned badges
  → delete temp file in finally; return prediction, top-k and practice result
```

The client can call prediction without `native_sign_id` for model output only; the actual practice UI should use the ID form so the backend owns target resolution and outcome calculation. A native catalog sign that has no matching class in the checkpoint may never be top-k, even if it is active in the catalog.

### 19.5 Local progress migration

After login, the frontend checks `/progress/migration-offer`. If the backend says the account has no meaningful persisted activity and the browser cache has meaningful activity, it prompts to import. `POST /progress/migrate-local` imports through the backend's progress service. Existing server progress is not overwritten by automatically treating local cache as authoritative.

### 19.6 AURA message

```text
POST /chatbot/conversations/{id}/messages (authenticated)
  → ownership check: conversation must belong to current user
  → save user message
  → scope check using message and, for follow-ups, conversation context
  → off-topic response or construct project context/history
  → configured provider (OpenRouter / local Ollama / auto) if available
  → static project knowledge fallback on unavailable/error path
  → save assistant response and source/provider metadata
  → return user and assistant message objects
```

Conversation history and provider metadata are persisted. API key material is read only by the backend and never returned by the frontend API.

## 20. Complete API inventory

All protected rows below require a valid bearer token unless they say “admin”. Admin endpoints require a token whose user row has `role == "admin"`. `/health`, `/history`, `/predict`, login and registration are public. `/collect` has a feature-flag gate; it is a local developer tool, not a learner account API.

### Authentication and account

| Method | Path | Access | Purpose |
|---|---|---|---|
| `POST` | `/auth/register` | Public | Create user and return bearer token |
| `POST` | `/auth/login` | Public | Login by username or email and return bearer token |
| `GET` | `/auth/me` | User | Current user identity/role |
| `GET` | `/users/me/profile` | User | Derived profile and journey summary |
| `GET` | `/users/me/preferences` | User | Read preferences |
| `PUT` | `/users/me/preferences` | User | Update preferences |
| `POST` | `/users/me/change-password` | User | Verify current password and change password |
| `GET` | `/users/me/export` | User | Return account learning data as JSON |
| `GET` | `/users/me/export.json` | User | Downloadable JSON export response |
| `DELETE` | `/users/me` | User | Delete own account after password/confirmation checks |

### Alphabet inference and learner progress

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/health` | Public | Backend/model/device status |
| `GET` | `/history` | Public | Last bounded process-local predictions/counts |
| `POST` | `/predict` | Public | Classify multipart image; optional hand requirement and debug/model options |
| `GET` | `/progress` | User | Read full persisted `GameState` shape |
| `PUT` | `/progress` | User | Synchronize overall client game state |
| `GET` | `/progress/letters` | User | Read per-letter progress rows |
| `PUT` | `/progress/letters/{letter}` | User | Update one letter aggregate |
| `GET` | `/progress/migration-offer` | User | Check whether local progress import should be offered |
| `POST` | `/progress/migrate-local` | User | Submit existing local state for import |
| `POST` | `/practice/session` | User | Record one A–Z practice event and optional XP event |
| `GET` | `/challenges` | User | Read challenge state/catalog |
| `POST` | `/challenges/{challenge_type}/progress` | User | Submit progress for a challenge |
| `GET` | `/achievements` | User | Read achievement definitions and earned state |
| `GET` | `/recommendations` | User | Read deterministic learner letter recommendations |

### Words and analytics

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/words` | User | List vocabulary, optionally filtered by category |
| `GET` | `/words/progress` | User | List current user's word progress |
| `GET` | `/words/recommendations` | User | Word suggestions based on recorded progress |
| `GET` | `/words/analytics` | User | Word-level aggregate summary |
| `GET` | `/words/challenge` | User | Deterministic daily/challenge word set |
| `GET` | `/words/{word_id}` | User | Fetch one word |
| `GET` | `/words/{word_id}/progress` | User | Fetch user's progress for a word |
| `POST` | `/words/practice/session` | User | Store one sequential spelling session |
| `GET` | `/analytics/dashboard` | User | Composite analytics payload (query accepts trend/heatmap ranges) |
| `GET` | `/analytics/overview` | User | Overall metrics |
| `GET` | `/analytics/activity` | User | Attempt activity trend |
| `GET` | `/analytics/accuracy` | User | Accuracy trend and comparison |
| `GET` | `/analytics/letters` | User | Per-letter aggregate stats |
| `GET` | `/analytics/letters/{letter}` | User | Detail, trend and recent attempts for a letter |
| `GET` | `/analytics/response-time` | User | Average/fastest response and trend |
| `GET` | `/analytics/history` | User | Paginated/filterable learner attempt history |
| `GET` | `/analytics/heatmap` | User | Date-bucketed practice activity cells |
| `GET` | `/analytics/streak` | User | Current/best streak and daily activity |
| `GET` | `/analytics/xp` | User | XP history and date series |
| `GET` | `/analytics/insights` | User | Strong/weak and practiced letter summaries |
| `GET` | `/analytics/funnel` | User | Learning progression distribution |
| `GET` | `/analytics/weekly-summary` | User | Current vs previous week summary |

### Native signs and chatbot

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/native/signs` | User | Active native-sign catalog and safe reference URLs |
| `GET` | `/native/progress` | User | Per-sign and overall mastery summary |
| `GET` | `/native/progress/{sign_id}` | User | One sign's progress |
| `POST` | `/native/predict` | User | Predict clip; optional target ID enables graded practice persistence |
| `GET` | `/chatbot/status` | User | Configured provider, model availability and knowledge fallback status |
| `POST` | `/chatbot/conversations` | User | Create conversation |
| `GET` | `/chatbot/conversations` | User | List owned conversations |
| `GET` | `/chatbot/conversations/{conversation_id}` | User | Read owned conversation/messages |
| `POST` | `/chatbot/conversations/{conversation_id}/messages` | User | Send message and receive assistant response |
| `POST` | `/chatbot/messages/{message_id}/feedback` | User | Rate an assistant message and optionally add comment |

### Admin, mentor and developer endpoints

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/admin/overview` | Admin | System aggregates |
| `GET` | `/admin/activity` | Admin | Aggregate activity series |
| `GET` | `/admin/letters` | Admin | Popular/difficult letter stats |
| `GET` | `/admin/words` | Admin | Word usage aggregates |
| `GET` | `/admin/users` | Admin | Paginated user listing/search |
| `GET` | `/admin/recent-activity` | Admin | Recent learner activity |
| `GET` | `/admin/mentor-dashboard` | Admin | Combined learner, A–Z, word, native and chatbot dashboard |
| `GET` | `/admin/users/{user_id}` | Admin | Account detail and recent activity |
| `PATCH` | `/admin/users/{user_id}` | Admin | Change role/active state |
| `POST` | `/admin/users/{user_id}/reset-progress` | Admin | Reset progress aggregates/history as implemented by management route |
| `DELETE` | `/admin/users/{user_id}` | Admin | Delete target account with confirmation |
| `GET` | `/admin/native-signs` | Admin | List native catalog rows |
| `PATCH` | `/admin/native-signs/{sign_id}` | Admin | Edit selected catalog metadata/status |
| `GET` | `/admin/achievements` | Admin | List badges and award counts |
| `PATCH` | `/admin/achievements/{achievement_id}` | Admin | Edit badge display fields |
| `GET` | `/admin/models` | Admin | Model registry and recorded prediction summary |
| `GET` | `/admin/predictions` | Admin | Stored prediction records (where present) |
| `GET` | `/admin/chatbot/feedback` | Admin | Feedback summary and message context |
| `GET` | `/admin/dataset/sessions` | Admin | Developer webcam dataset session inventory |
| `DELETE` | `/admin/dataset/sessions/{session_id}` | Admin | Delete a collected dataset session |
| `GET` | `/admin/audit-log` | Admin | Paginated administrative audit records |
| `GET` | `/collect/status` | Dev flag | Collection availability/status |
| `POST` | `/collect/sessions` | Dev flag | Create capture session with metadata |
| `GET` | `/collect/sessions` | Dev flag | List capture sessions |
| `GET` | `/collect/sessions/{session_id}` | Dev flag | Session/sample details |
| `POST` | `/collect/analyze` | Dev flag | Analyze uploaded frame quality/hand region |
| `POST` | `/collect/samples` | Dev flag | Save labelled sample |
| `POST` | `/collect/negatives` | Dev flag | Save a negative/background sample |
| `DELETE` | `/collect/sessions/{session_id}/negatives/{sample_id}` | Dev flag | Remove negative sample |
| `DELETE` | `/collect/sessions/{session_id}/samples/{sample_id}` | Dev flag | Remove labelled sample |

## 21. Database field reference

The following is a practical entity guide rather than a generated DDL listing. For exact nullable/default/index details, see the corresponding class in `backend/models.py` and migration history.

### Identity and user-level progress

| Entity | Important fields |
|---|---|
| `User` | `id`, `username`, `email`, `password_hash`, `role`, `preferences` text, `is_active`, `created_at` |
| `UserProgress` | `user_id` (one-to-one), `xp`, `level`, `current_streak`, `best_streak`, `last_active_date`, `total_correct`, `total_attempts`, `sessions`, `game_meta`, timestamps |
| `LetterProgress` | `user_id`, `letter`, `attempts`, `correct_attempts`, `mastery`, `last_practiced`, `updated_at`; unique user+letter |
| `ChallengeProgress` | `user_id`, `challenge_type`, `progress`, `completed`, `completed_at`, `meta`, `updated_at`; unique user+type |
| `Achievement` | shared `slug`, display `name`, `description`, serialized `requirement` |
| `UserAchievement` | `user_id`, `achievement_id`, `earned_at`; unique user+achievement |
| `PracticeSession` | `user_id`, target `letter`, optional `prediction`, `correct`, optional `response_time`, `xp_earned`, optional `challenge_type`/`confidence`, `created_at` |
| `XpEvent` | `user_id`, signed/integer `amount`, `reason`, `created_at` |

### Word learning

| Entity | Important fields |
|---|---|
| `Word` | string `id`, `word`, `category`, `difficulty`, optional `description`, `spelling_enabled`, `native_sign_enabled`, `created_at` |
| `WordProgress` | `user_id`, `word_id`, `attempts`, `completions`, `correct_letters`, `total_letters`, `mastery`, `best_time_ms`, `total_time_ms`, `resume_index`, JSON text `letter_stats`, `last_practiced_at`, timestamps |
| `WordPracticeSession` | `user_id`, `word_id`, optional `accuracy`/`duration_ms`, `completed`, `xp_earned`, optional `challenge_type`, `mistakes`, JSON text `letter_results`, `created_at` |

### Native sign learning

| Entity | Important fields |
|---|---|
| `NativeSign` | integer `id`, unique `gloss`, `display_name`, optional `meaning`, `category`, `difficulty`, `description`, `example_text`, `dataset_available`, `model_available`, `active`, timestamps |
| `NativeSignReference` | `native_sign_id`, `source_type`, `source_identifier`, optional local/thumbnail paths and `license_note`, `is_primary`, `created_at` |
| `NativeSignProgress` | `user_id`, `native_sign_id`, `attempts`, `correct_attempts`, `accuracy`, `mastery`, best confidence/latency, last-practiced/update times; unique user+sign |
| `NativeSignSession` | `user_id`, `native_sign_id`, `model_version`, `started_at`, `completed_at`, result, confidence, response time, `created_at` |

### Operations, model and chatbot records

| Entity | Important fields |
|---|---|
| `ModelVersion` | `model_type`, `version`, checkpoint path, optional vocabulary version/class count, serialized metrics, `active`, creation time; unique model type+version |
| `SignPrediction` | user/session/model references, model type/version, predicted/expected labels, confidence, latency, optional correctness, timestamp |
| `ChatbotConversation` | `user_id`, optional title, created/updated times |
| `ChatbotMessage` | conversation, role, content, serialized metadata, created time |
| `ChatbotFeedback` | message, optional rating/comment, creation time |
| `AdminAuditLog` | optional admin FK, denormalized admin username, action, target type/id, serialized detail, creation time |
| `UserPreference` | user primary key/FK, theme, language, serialized notification/learning preferences and updated time |

### Important database behavior

- Several relationships use `ON DELETE CASCADE`, so deleting a user removes user-owned progress/session/message data. `AdminAuditLog.admin_id` uses `SET NULL` so an audit row can remain after the actor account is removed.
- Unique constraints prevent duplicate user+letter progress, user+challenge, user+achievement, user+word progress and user+native-sign progress rows.
- Timestamp helpers use timezone-aware UTC datetimes at the Python level. SQLite has limited native timezone semantics; comparisons/serialization should normalize carefully.
- `game_meta`, `letter_stats`, `letter_results`, chatbot metadata, preferences and audit details are text fields containing serialized structures. Code must decode/encode them consistently.
- Some phase-two schema tables were introduced before being wired to every feature. A table's presence alone does not prove that request handlers write to it. For example, `ModelVersion`/`SignPrediction` support admin reporting where records exist, but model runtime does not automatically register every checkpoint or persist every raw frame prediction.

## 22. Core gamification rules in more detail

The frontend game engine applies immediate state transitions. These summaries explain the current behavior; implementation is in `frontend/src/game/gamification.ts` and persistence/award services remain separate.

- **Level:** `floor(xp / 1000) + 1`; progress bar reports distance through the current 1,000-XP band.
- **Letter tiers:** fewer than 3 correct is Beginner; 3 through 9 is Learning; 10 or more is Mastered. Per-letter accuracy is `correct / attempts`; the mastery bar is based on correct count toward 10.
- **Daily challenge:** 10 correct letters on the current UTC date. Correct attempts increase progress; completion adds 100 XP. If no mistake was recorded, a one-time 150-XP perfect bonus is added.
- **Alphabet challenge:** practice all 26 letters; completion adds 500 XP.
- **Speed challenge:** 5 correct signs inside the 45-second challenge window; completion adds 150 XP.
- **A–Z correct attempt:** under/equal to 5 seconds awards 30 XP; otherwise 20 XP.
- **Word challenge:** a five-word spelling sequence; word completion and fast/perfect/challenge rewards are separate from letter-level XP.
- **Streak:** successful daily practice updates current/best streak. Dates are generated from `toISOString()` in the frontend, so daily client behavior is UTC-date keyed.
- **Badges:** some UI previews compute progress optimistically. Persistent earned state is stored server-side and should be fetched from `/achievements`; unique constraints prevent duplicate award links.
- **Native practice:** the backend records a single attempt and grants the fixed native XP only when the server computes a correct result.

If changing a reward formula, review both client-side display/calculation and server-side validation/persistence so reported XP and stored XP events agree.

## 23. Alphabet model mechanics and API response interpretation

### `POST /predict` input contract

- Multipart field name is `file`.
- Supported image formats are documented as JPG/PNG/WEBP; the backend attempts to decode with Pillow and returns a 400 for invalid data.
- `top_k` is bounded in the handler (default 3; capped to supported range).
- `require_hand=true` asks for detector availability and an actual hand crop. If MediaPipe is absent or no hand is found, status differs and prediction is null.
- `recognizer=resnet|landmarks` can select a configured alternative per request; default path is ResNet.
- `debug=true` asks for image-free technical metadata; `save_diagnostics=true` only writes files when backend diagnostics are enabled.
- `target`, `preview_mirrored`, and `video_size` are contextual fields for diagnostics and do not change the model label.

Typical response fields include:

| Field | Meaning |
|---|---|
| `status` | `ok`, `no_hand_detected`, `hand_detector_unavailable`, or an error-level HTTP response |
| `prediction` | Top label (null when no classification ran) |
| `confidence` | Softmax score for top candidate |
| `top_predictions` | Candidate label/score list |
| `low_confidence` | Whether top score is below threshold (0.70) |
| `threshold` | Current low-confidence threshold |
| `hand_detected`, `hand_box` | Detection/crop status when relevant |
| `diagnostics` | Device/input shape/latency details when requested; optional capture IDs for dev diagnostics |

The `/history` endpoint stores only a process-local bounded deque (maximum 200); it is diagnostic UI support, not account practice history. `/practice/session` is the durable learner attempt endpoint.

### Image crop reasoning

MediaPipe returns normalized hand landmarks. The detector converts extrema to pixel coordinates, computes a centered square window with a configurable margin (default crop padding 30% relative to the longer box side), clips the reported overlay box to the image and pads pixels outside the frame with edge values. The crop remains square before resize; directly stretching the hand bounding rectangle would distort shape. If no hand is required, the entire input frame can reach the classifier and the crop-specific fields are unset.

### Low confidence versus wrong prediction

These are different outcomes:

- **No hand/detector unavailable:** no label was classified; target correctness should not be computed from a null value.
- **Low confidence:** a candidate label exists, but the top score is below 0.70.
- **Stable candidate:** the browser has enough consistent votes to display/accept a letter.
- **Incorrect answer:** the accepted predicted letter differs from the current learner target.

Keeping those states distinct improves UI guidance and makes evaluation metrics meaningful.

## 24. Native video model mechanics and limitations

The current I3D preprocessor ports the official ASL Citizen video handling convention into `src/i3d_transfer/preprocessing.py`:

1. OpenCV decodes frames in its native BGR channel order; frames are intentionally not converted to RGB.
2. The sampler chooses a centered temporal window and adaptive frameskip (1, 2 or 3 based on source length) to produce up to 64 frames.
3. Short clips are padded to 64 by repeating an end frame; the current RNG seed makes the inference choice repeatable.
4. Spatial resize preserves aspect ratio, ensures the short side reaches at least 226 and caps long side at 256.
5. Pixel values are scaled from `[0,255]` to `[-1,1]` (not ImageNet normalization).
6. The center crop is 224 × 224. Tensor order becomes `(channels, time, height, width)` then gets a batch dimension for I3D.
7. I3D's `extract_features` output is temporally averaged; the linear head maps the 1,024-wide vector to classes; softmax and `topk` produce gloss scores.

The I3D path has different input conventions from ResNet. Reusing the alphabet normalization, RGB conversion, frame crop or tensor order would break compatibility with the video checkpoint. The current linear-head method collapses temporal feature maps by mean pooling; it is not a token-by-token sign-language sentence decoder. Video quality, signer coverage, camera framing and mismatch between dataset cameras and consumer webcams can materially affect results.

## 25. Admin/mentor management and audit behavior

There are two related admin surfaces:

- `backend/routers/admin.py` exposes read-oriented system and mentor aggregates.
- `backend/routers/admin_ops.py` exposes operational CRUD actions such as user management, catalog/achievement edits, dataset cleanup, model/prediction review and chatbot feedback inspection.

Management routes use `get_current_admin`. State-changing admin calls call the audit helper and commit the audit record in the same database transaction as the change. The front end has distinct tabs/components for users, content, models, dataset and audit events. Server authorization remains required even if a user manually invokes an endpoint outside the UI.

Operational caution for maintainers: user deletion, progress reset, data set deletion and role/active changes have persistent effects. The confirmation semantics are encoded in the endpoint request contract. Audit entries record actor, action and target metadata; do not put secrets, password values or raw video content into their detail field.

## 26. Development data collection and diagnostics

### Webcam collection

`ASL_QUEST_DEV_COLLECT=1` enables `/collect`. The flow creates a capture session with environment descriptors (room, lighting, background and glasses), analyzes candidate frame quality, saves labelled positive examples and optional negative/background examples, and supports deleting samples/sessions. Files go beneath `dataset/webcam` (the exact layout/metadata contract is in `backend/services/webcam_dataset.py`). The browser page is only included when Vite runs in development mode; the API feature flag must also be on.

Collected images can contain personally identifiable scene information such as faces and home/work interiors. Treat the collection as sensitive local training material. The collector is for controlled model development, and its samples should be split by room/signer/session when measuring generalization; random frame-level splits can leak nearly identical frames across train and test.

### Live diagnostics

`ASL_QUEST_DEV_DIAGNOSTICS=1` allows the backend's diagnostic service to save the original frame, hand crop and exact normalized model input for requests that ask for saved diagnostics. Metadata can include target, mirror status, frame size, device, tensor shape/dtype and request ID. This is useful for debugging mismatch between browser preview and model input, but is an intentional image write. It is off by default.

### Offline evaluation and scripts

Useful tools are grouped by task:

| Task | Common paths |
|---|---|
| Prepare/check dataset | `scripts/validate_dataset.py`, `scripts/audit_external_dataset.py`, manifest builders under `scripts/` |
| Train alphabet | `src/training/train.py`, root `train.py`, scripts for controlled experiment variants |
| Evaluate alphabet | `src/training/evaluate.py`, root `evaluate.py`, `scripts/evaluate_webcam.py`, `scripts/evaluate_webcam_heldout.py` |
| Investigate live failure | `scripts/diag_*.py`, `scripts/diagnose_*.py`, `scripts/investigate_*.py` |
| Compare/promote models | `scripts/compare_*.py`, `scripts/promote_checkpoint.py` |
| Native I3D experiment | `scripts/train_native_*.py`, `scripts/evaluate_native_*.py`, feature extraction/validation scripts |
| Database operations | Alembic CLI and `scripts/migrate_sqlite_to_postgres.py` |
| App validation | `scripts/smoke_test.py`, `scripts/validate_app.py`, `scripts/validate_app_extended.py` |

Training/evaluation utilities may have hardware, dataset-path and licensed-asset requirements that are not needed just to run the browser app. Read each command's arguments/help/source before running a long experiment.

## 27. Environment-specific configuration and startup order

### Backend startup order in practice

1. Python imports settings and derives the database URL, token settings, CORS and model choice from environment variables.
2. FastAPI object and routers are created.
3. On startup, `init_db()` ensures the data directory and compatibility schema; run Alembic separately for versioned migrations.
4. Achievement definitions are seeded.
5. A–Z checkpoint is resolved and loaded; missing/corrupt checkpoint leaves the recognition endpoint degraded/unavailable but should not necessarily prevent unrelated routes from being registered.
6. Hand detector is initialized. It may be unavailable if MediaPipe or its task asset is missing; `/health` reports this condition.
7. Optional landmark models are loaded only if configured.
8. Native I3D assets are loaded on first video prediction and retained in a module cache.

### Database URL examples

```text
SQLite:      sqlite:///data/asl_quest.db
PostgreSQL:  postgresql+psycopg://user:password@host:5432/asl_quest
```

The SQLite relative file location is resolved relative to the process working directory according to SQLAlchemy URL semantics; run commands from the repository root as documented. PostgreSQL deployments should use the `psycopg` v3 dialect prefix and a service-provided secret URL.

### Frontend API-base examples

```text
Local development: VITE_API_BASE_URL unset → /api → Vite proxy → localhost:8000
Production build:   VITE_API_BASE_URL=https://api.example.org → browser calls API origin directly
```

`BACKEND_NOT_CONFIGURED` is true on non-development builds where the API variable is absent; the frontend surfaces a configuration warning. This check prevents `/api` from silently pointing to the frontend host in production.

## 28. Troubleshooting map

| Symptom | First checks |
|---|---|
| Frontend says backend is unavailable | Is Uvicorn listening on 8000? Does `/health` respond? Is the frontend using `/api` proxy or a correct `VITE_API_BASE_URL`? |
| `/health` says model not loaded | Check configured/fallback A–Z checkpoint path, checkpoint integrity and Python/PyTorch import errors |
| Hand-required prediction returns detector unavailable | Check MediaPipe installation/API compatibility and `models/hand_landmarker.task` if Tasks API is used |
| Hand is found but letter feedback is poor | Check framing/crop diagnostics, lighting/orientation, confidence and stability settings; distinguish detector, crop and classifier issues |
| Login works until reload then fails | Check token persistence, backend secret consistency across restarts and current user `is_active` state |
| Learner progress appears reset across browsers | Progress is account scoped; verify login identity, API availability and whether server sync completed |
| Words list is empty or word API errors | Confirm Alembic migrations/word catalog seed are applied and database matches schema |
| Native sign API returns model unavailable | Check both I3D backbone and head assets at the exact paths in `src/i3d_transfer/predict.py`; inspect backend logs |
| Native video rejected | Check filename suffix against `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`; verify nonempty body and ≤100 MB |
| Chatbot reports provider unavailable | Check `/chatbot/status`, selected provider, backend environment and key/Ollama reachability; static fallback may still answer |
| Browser camera permission error | Use a secure context (localhost or HTTPS), grant site permission, check another app is not holding the device |
| CORS error on deployed frontend | Set exact deployed origin in `ASL_QUEST_CORS_ORIGINS`, redeploy backend and check scheme/host/port |
| Admin pages return 403 | Verify `/auth/me` reports admin role and that route dependency is satisfied; hiding UI controls is not authorization |

For recognition quality, inspect both input and output. A poor prediction can come from camera capture, JPEG encoding, hand detection, crop geometry, orientation, checkpoint version, or model generalization. The diagnostic tools are designed to isolate these stages.

## 29. Suggested feature-change workflow

1. Find the screen and its frontend API client. Prefer adding a typed API function rather than embedding endpoint strings across page components.
2. Find the FastAPI router that owns the endpoint. Confirm auth/admin dependency and request/response schemas.
3. Put reusable business rules in the service layer if multiple routes or tests need them.
4. If persistence changes, update `backend/models.py` **and** add a forward Alembic migration; consider startup compatibility behavior for existing local DBs.
5. Scope learner data by `current_user.id`; add indexes/unique constraints where query patterns or invariants require them.
6. For admin mutations, preserve the audit transaction and avoid recording secrets or raw media in the audit payload.
7. For model changes, record architecture, class ordering, preprocessing and checkpoint provenance together. Run held-out evaluation that reflects signer/room variation, not just shuffled adjacent frames.
8. Update frontend types and display empty/error/loading states for API errors and offline behavior.
9. Update focused backend/frontend tests and project docs. Run the relevant commands after implementation.
10. For database/model deployment, make migrations and asset provisioning part of the deployment procedure; the source code bundle alone may not contain large checkpoints.

## 30. What the project does not currently claim to provide

- Full continuous ASL-to-English translation, grammar analysis, sentence transcription or sign-language conversation understanding.
- Motion recognition for the alphabet letters J and Z through the static ResNet model.
- Direct recognition of arbitrary words through the Word Practice letter classifier.
- A large/open-ended native sign vocabulary beyond classes trained into the active checkpoint.
- Perfect webcam accuracy across lighting, signer, handedness, clothing, camera and background conditions.
- Durable all-user inference telemetry simply because `/history` exists; that endpoint is process-local, while user attempts are saved through learner practice endpoints.
- Automatic inclusion of ignored large model weights, ASL Citizen videos or local database files in a clean Git clone.
- A currently running production environment solely because Vercel/Render configuration or a deployment plan appears in the repository.

