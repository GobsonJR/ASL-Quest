# Team handoff — running ASL-Quest before starting UI work

This is a checklist for a teammate setting up ASL-Quest on their own Windows
laptop for the first time, to get the **complete integrated system** running
(not just the frontend shell) before starting UI work. It doesn't replace
`SETUP_WINDOWS.md` (the detailed setup reference) or `deployment/README.md`
(the exact model-file manifest) — it sequences them into one path and adds
the verification checklist and Git workflow specific to joining this project.

You are cloning the stable branch, `main`. You will **not** get the Mentor
Dashboard or anything else from `david-dev` — that's separate, unmerged work
in progress. Everything below is what's on `main`.

---

## A. What you get from GitHub

Cloning `main` gives you all source code and configuration needed to run
the app: the FastAPI backend, the React/TypeScript frontend, Alembic
migrations, the native-sign catalog manifests (small JSON/CSV files, not
videos), `requirements.txt` / `package.json`, and the setup scripts and docs
referenced here. It does **not** give you the large/licensed runtime assets
below — those are deliberately excluded from Git (see `.gitignore`).

```powershell
git clone https://github.com/GobsonJR/ASL-Quest.git
cd ASL-Quest
```

## B. What must be copied manually

Four files, none of them in Git. Get them from an existing checkout (USB
drive, network share, cloud transfer — whatever's convenient):

| File | Real destination | Approx. size |
|---|---|---|
| `models/best_model.pth` | same path | ~43 MB |
| `models/hand_landmarker.task` | same path | ~7.5 MB |
| `models/pretrained/asl_citizen_i3d/ASL_citizen_I3D_weights.pt` | same path | ~58 MB |
| `outputs/asl_citizen_native_10/experiments/i3d_frozen/checkpoints/head_best.pt` | same path | ~43 KB |

Easiest path: drop all four (flat, no subfolders) into `deployment\models\`,
then run `.\scripts\windows\install_deployment_assets.ps1` — it moves each
one to its real destination and refuses to silently overwrite anything
already there. Full detail and where to get each one if you don't have a
copy (one is an official Google MediaPipe download, one is a Microsoft
Research checkpoint with documented provenance): `deployment\README.md`.

## C. What must NOT be copied

- The **A-Z training image dataset** (`dataset/asl_alphabet_train`,
  `dataset/asl_alphabet_test`, ~1.1 GB) — training-only, not needed to run
  the app.
- The **ASL Citizen video dataset** — licensed by Microsoft Research
  separately from this project; never store it in this repo or in
  `deployment/`. Not needed to run the app, only to retrain/re-evaluate
  Native Signs. See `deployment/README.md` if you genuinely need it.
- `.env` — machine-specific secrets, git-ignored, never shared.
- Any API key, password, or token, real or example — never paste one into
  a commit, a script, or this doc.
- `data/asl_quest.db` — see part D below; you create your own.
- `.venv` / `venv311` — recreated locally by the setup script, never
  transferred.
- `node_modules` — recreated locally by `npm install`, never transferred.

## D. Creating your own fresh SQLite database

**Do not ask for a copy of anyone else's `data/asl_quest.db`.** Create your
own — it's fast, deterministic, and starts you with a clean account:

```powershell
.\scripts\windows\setup.ps1                     # Python/Node checks, venv311, dependencies
.\scripts\windows\install_deployment_assets.ps1 # after copying the 4 model files into deployment\models\
venv311\Scripts\python.exe -m alembic upgrade head
venv311\Scripts\python.exe scripts\seed_native_signs_native_10.py
```

(`setup.ps1 -RunMigrations -SeedNativeSigns` does the last two for you in
one step.) Achievements seed themselves automatically the first time the
backend starts. **Migrate before your first backend start** — the app boots
without it, but Native Signs and the chatbot fail with "no such table"
errors until `alembic upgrade head` has run.

## E. Configuring `.env` safely

```powershell
copy .env.example .env
```

Then open `.env` and, at minimum, set `ASL_QUEST_SECRET_KEY` to any long
random string (it signs your JWTs — your own value, not anyone else's).
Everything else can stay blank for local dev; see the table in
`SETUP_WINDOWS.md` § Step 5 for what each variable does. Leave
`OPENROUTER_API_KEY` blank unless you have your own key — the assistant
degrades gracefully to a setup message without one, it does not crash.
**Never commit `.env`** — it's already in `.gitignore`; double-check
`git status` never lists it.

## F. Starting backend and frontend

Two terminals, from the repo root:

```powershell
.\scripts\windows\start_backend.ps1     # http://127.0.0.1:8000
```

```powershell
.\scripts\windows\start_frontend.ps1    # http://localhost:5173
```

## G. Verifying backend and frontend are connected

Open `http://localhost:5173`. The header shows a status dot and a line like
"Ready on cpu" or "Ready on cuda · <your GPU>" once the backend has loaded
the A-Z model — that line is the frontend successfully reaching the backend
through Vite's dev proxy. If it instead says "Backend unavailable," the
backend isn't running or isn't on port 8000 — check the backend terminal for
errors before going further.

You can also check the backend directly: `http://127.0.0.1:8000/health`
should return JSON with `"model_loaded": true`.

## H. Feature verification checklist

Register a fresh account in the UI, then confirm each of these actually
works end to end (not just that the page loads):

- [ ] **Login** — log out and back in with the account you just created.
- [ ] **A-Z** — practice a letter (webcam or image upload); you get a
      prediction with a confidence score.
- [ ] **Word Spelling** — start a word, spell it letter by letter; a
      completed attempt is recorded.
- [ ] **Native Signs** — the catalog shows 10 signs; practice one and get a
      real prediction result (not an error).
- [ ] **Progress** — the Progress page's XP/level/streak numbers change
      after an attempt above.
- [ ] **Gamification** — the Badges/Achievements page loads and reflects
      what you've unlocked so far.
- [ ] **Chatbot** — the Assistant page: ask something in-scope ("How is XP
      awarded?") and something off-topic ("What's the capital of France?")
      — the off-topic one must get the fixed refusal, never a real answer.

If any of these fails, re-run `.\scripts\windows\validate_setup.ps1` first —
it's a fast, read-only check of the model files, migration state, and which
env vars are set, and will usually point straight at the gap.

## I. Your database is separate from mine

The `data/asl_quest.db` you create in step D lives only on your laptop. It
is never committed (it's git-ignored) and has no connection to anyone
else's local database — your accounts, progress, and XP are entirely your
own and won't collide with or overwrite anyone else's.

## J. Your Git branch is UI changes only

Your branch should contain **UI/frontend work only** — building on the
stable `main` checkpoint, not on `david-dev` (which has unrelated,
in-progress backend work you don't need and shouldn't build on top of).

## K. Creating your branch from `main`

Run this yourself, on your own machine — this is not something anyone
else runs for you:

```powershell
git checkout main
git pull origin main
git checkout -b friend-ui
git push -u origin friend-ui
```

(`friend-ui` is just an example name — anything descriptive works, e.g.
`<yourname>-ui`.) Confirm you actually branched from the right place:

```powershell
git rev-parse HEAD
```

should print `3b25ceca257a801a6b7a87d3220b1c1263d54439` right after you
create the branch, before you've made any commits.

## L. Commit regularly

Small, frequent commits on your own branch — don't let a week of UI work
sit uncommitted. Before each commit, run the backend tests and frontend
build (see the checklist in part 4 below) so problems surface immediately
rather than after you've layered several more changes on top.

## M. Do not touch ML model files or training datasets for UI work

Normal frontend/UI work never needs to modify anything under `models/`,
`outputs/*/checkpoints/`, `dataset/`, or the training scripts under
`src/asl_citizen/`, `src/i3d_transfer/`, or `scripts/*train*`. If a UI task
ever seems to require changing one of those, stop and ask first — that's
almost certainly the wrong layer for a UI change to touch.

---

## 4. Baseline validation

Run these once your setup is complete, **before** starting UI work, so you
know your baseline is healthy and any later failure is something you
introduced, not something pre-existing.

### Runtime checks (required — do these; no large dataset needed)

```powershell
venv311\Scripts\python.exe -m unittest discover -s tests
cd frontend
npm run build
```

Also walk through the **Feature verification checklist** in part H above —
the automated tests cover the backend logic, but only clicking through the
UI confirms the frontend is actually wired up correctly end to end.

### Training/dataset tests (optional — these are research checks, not required to run the app)

You do **not** need the 1.1 GB A-Z training image set, or the licensed ASL
Citizen video dataset, just to verify the app runs. If you run the full
test suite without them:

- `tests/test_asl_citizen.py` — needs the ASL Citizen dataset at
  `ASL_CITIZEN_DATASET_ROOT`. **Skips itself cleanly** (not a failure)
  without it.
- `tests/test_project.py::test_predictor_returns_one_image_prediction` and
  `::test_dataset_class_mapping_is_stable` — need
  `dataset/asl_alphabet_train`/`asl_alphabet_test`. These **fail** (not
  skip) without that folder. That's expected and fine to ignore for UI
  work — they're validating the training pipeline's data, not anything the
  running app depends on. Only copy `dataset/` if you specifically want
  these two to pass too (e.g. you're going to retrain the A-Z model).

If every runtime check above passes and only those two dataset tests fail,
your baseline is healthy.

---

## 5. Your Git workflow, visually

```
main
  |
  v
friend-ui   (branched from the exact commit above, per part K)
  |
  v
UI changes  (small, focused commits)
  |
  v
tests       (unittest discover + npm run build — part 4)
  |
  v
commit
  |
  v
push        (to your own branch — never main)
```

Never commit directly to `main`. Never merge `david-dev` into your branch —
it's unrelated in-progress work.

---

## 6. Safety — do not do these

- Do not modify `main` directly, or push to it.
- Do not merge `david-dev` into anything.
- Do not modify or delete `data/asl_quest.db` other than by creating your
  own fresh one (part D) — never edit someone else's.
- Do not modify model weight files (`*.pth`, `*.pt`, `*.task`) — copy them
  as-is, never edit or regenerate them for UI work.
- Do not modify or add to the training datasets.
- Do not commit `.env`, or any real API key, password, or token — check
  `git status` and the diff before every commit.
- Do not force-push.
- Do not delete `outputs/experiments/` — it's a pre-existing, unrelated
  artifact; leave it alone even though it's untracked.
