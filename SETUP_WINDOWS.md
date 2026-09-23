# Setting up ASL-Quest on a new Windows laptop

This is the from-scratch guide for getting a **second** Windows laptop
running the full ASL-Quest app (A-Z recognition, Word Spelling, Native Signs,
gamification/progress, authentication, and the project-only chatbot). It
complements `README.md` (architecture/feature reference) and `DATABASE.md`
(schema/migration reference) rather than replacing them.

## What you're setting up

| Feature | Needs |
|---|---|
| A-Z alphabet recognition | `models/best_model.pth`, `models/hand_landmarker.task` |
| Word Spelling | Nothing extra — reuses A-Z + the seeded `words` catalog |
| Native ASL Sign recognition | `models/pretrained/asl_citizen_i3d/ASL_citizen_I3D_weights.pt`, `outputs/asl_citizen_native_10/experiments/i3d_frozen/checkpoints/head_best.pt` |
| Gamification / progress / achievements | Database only (auto-seeded) |
| SQLite database-backed features | `data/asl_quest.db` (fresh or migrated) |
| ASL-Quest project-only chatbot | Database tables (already in schema) + optional `OPENROUTER_API_KEY` |
| Authentication | `ASL_QUEST_SECRET_KEY` (any laptop can generate its own) |
| Frontend + backend | Node.js + Python 3.11, both on the same laptop |

None of this requires the ASL Citizen **video** dataset, Docker, WSL, or
Linux — everything here is native Windows/PowerShell.

## Step 1 — Get the code

```powershell
git clone https://github.com/GobsonJR/ASL-Quest "D:\path\you\choose\ASL-Quest"
cd "D:\path\you\choose\ASL-Quest"
```

(Any drive/folder works — nothing in the app depends on `D:\SIGN LANGUAGE`
specifically; see "Portability" below.)

## Step 2 — Run the bootstrap script

```powershell
.\scripts\windows\setup.ps1
```

This checks for Python 3.11 and Node.js, creates `venv311`, installs
`requirements.txt` and the frontend's `npm` packages, and finishes with a
read-only report of what's still missing. It does **not** touch the database
or download any model file — see Steps 3–4.

If Python 3.11 or Node.js aren't installed yet:

- Python 3.11: https://www.python.org/downloads/ (or `winget install Python.Python.3.11`)
- Node.js (LTS): https://nodejs.org/

## Step 3 — Copy the required model files

Four files are required at runtime and are **deliberately not in Git**
(`.gitignore`: `*.pth`, `*.pt`, `*.task` — see "Portability & licensing"
below for why). Copy them from an existing ASL-Quest checkout using
whatever means you have (USB drive, network share, cloud transfer, `scp`):

| Copy into `deployment\models\` | Goes to |
|---|---|
| `best_model.pth` | `models\best_model.pth` |
| `hand_landmarker.task` | `models\hand_landmarker.task` |
| `ASL_citizen_I3D_weights.pt` | `models\pretrained\asl_citizen_i3d\ASL_citizen_I3D_weights.pt` |
| `head_best.pt` | `outputs\asl_citizen_native_10\experiments\i3d_frozen\checkpoints\head_best.pt` |

Then run:

```powershell
.\scripts\windows\install_deployment_assets.ps1
```

It moves each file to its real destination and will **not** silently
overwrite anything already there. Full details, exact sizes, and where to
get each file if you don't have a copy: `deployment\README.md`.

## Step 4 — Set up the database

**Recommendation: start from a fresh, migrated + seeded database — do not
copy the existing `data/asl_quest.db` as-is.** Reasoning:

- The existing dev database currently contains real registered user accounts
  (usernames, emails, bcrypt password hashes). Copying it means handing a
  second person those accounts' data for no functional benefit.
- Every table that actually needs to be pre-populated for the app to work
  (`native_signs`, `achievements`, `words`) is reproduced **exactly** and
  deterministically by migrations + one seed script — nothing is "lost" by
  not copying, because no meaningful native-sign progress, sessions, or
  chatbot history exists in the dev database yet either.
- It matches the project's own documented "fresh install" path
  (`DATABASE.md`).

If your team specifically wants the existing data carried over anyway (e.g.
for demo continuity), that's a deliberate choice to make explicitly — copy
`data/asl_quest.db` into `deployment\database\` and run
`install_deployment_assets.ps1`, understanding that it also carries over the
existing user accounts.

**To create a fresh database** (recommended path):

```powershell
venv311\Scripts\python.exe -m alembic upgrade head
venv311\Scripts\python.exe scripts\seed_native_signs_native_10.py
```

(`setup.ps1 -RunMigrations -SeedNativeSigns` does both of these for you.)
Achievements seed themselves automatically the first time the backend
starts — no separate step needed.

> **Important ordering gotcha:** if you start the backend *before* running
> `alembic upgrade head`, the app will boot and A-Z/auth will work, but
> Native Signs and the chatbot will fail with "no such table" errors. Only
> the original ten tables are auto-created on startup; the rest
> (`native_signs`, `chatbot_*`, `words`, etc.) exist only after migrations.
> Always migrate first.

## Step 5 — Configure environment variables

```powershell
copy .env.example .env
```

Then edit `.env`:

| Variable | Required? | Notes |
|---|---|---|
| `DATABASE_URL` | No | Leave unset for the bundled SQLite database |
| `ASL_QUEST_SECRET_KEY` | Recommended | Any long random string; JWTs are signed with it |
| `ASL_QUEST_TOKEN_EXPIRE_MINUTES` | No | Defaults to 10080 (7 days) |
| `ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL` | No | Email that gets the admin role on first login |
| `ASL_QUEST_CORS_ORIGINS` | No | Defaults to the local Vite dev server origins |
| `OPENROUTER_API_KEY` | No | Without it, the Assistant page replies with a graceful setup message instead of answering |
| `CHATBOT_MODEL` | No | Defaults to `openrouter/free` |
| `ASL_CITIZEN_DATASET_ROOT` | No | Only for retraining/auditing against the licensed ASL Citizen dataset — not needed to run the app |

`.env` is git-ignored — never commit it. Run
`.\scripts\windows\validate_setup.ps1` any time to check which variables are
set (it never prints their values).

## Step 6 — Start the app

Two terminals:

```powershell
.\scripts\windows\start_backend.ps1     # http://127.0.0.1:8000
```

```powershell
.\scripts\windows\start_frontend.ps1    # http://localhost:5173
```

Open `http://localhost:5173`, register an account, and all features should
be available.

## Verifying everything is in place

```powershell
.\scripts\windows\validate_setup.ps1
```

Re-runnable at any time; checks model files, migration state, and env var
presence without modifying anything.

## Running tests

```powershell
venv311\Scripts\python.exe -m unittest discover -s tests
cd frontend
npm run build
```

Tests use their own isolated temporary SQLite databases — they never touch
`data\asl_quest.db`. Two caveats specific to a fresh transfer:

- Tests under `tests\test_asl_citizen.py` that need the licensed ASL Citizen
  video dataset **skip themselves** (not fail) if `ASL_CITIZEN_DATASET_ROOT`
  doesn't exist — expected and fine without that dataset.
- `tests\test_project.py::test_predictor_returns_one_image_prediction` and
  `::test_dataset_class_mapping_is_stable` read from `dataset\asl_alphabet_train\`
  / `dataset\asl_alphabet_test\`, which is **not** copied by this guide (it's
  only needed to retrain the A-Z model, not to run the app). Without it,
  those two specific tests fail rather than skip. Copy `dataset\` too if you
  want the full historical "148/148" test count to reproduce exactly.

## Portability & licensing

- Nothing in the application's source code hardcodes a machine-specific
  path, username, or drive letter — it can live anywhere on disk. (One
  training-only config, `src/asl_citizen/config.py`, has a default dataset
  path that's now overridable via `ASL_CITIZEN_DATASET_ROOT` rather than
  fixed.)
- The frontend talks to the backend over `http://127.0.0.1:8000` via Vite's
  dev proxy — both processes must run on the same laptop, which this guide
  already sets up.
- **Never** copy the ASL Citizen video dataset, or any file under a folder
  named for it, into this repository or `deployment/`. It is licensed by
  Microsoft Research separately from this project and is not required to
  run the app. See `deployment/README.md` for what to do if you genuinely
  need it for training/research.
