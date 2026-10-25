# ASL Quest deployment runbook (Render backend + Render PostgreSQL + Vercel frontend)

Branch: `david-dev`. Nothing here deploys by itself: every step marked **[dashboard]** is done by you in Render/Vercel.

## Topology
| Part | Host | URL |
|---|---|---|
| Frontend (static Vite build) | Vercel | `https://asl-quest-1.vercel.app` |
| Backend (FastAPI) | Render web service | `https://asl-quest-1.onrender.com` |
| Database | Render PostgreSQL | linked via `DATABASE_URL` |

`VITE_API_BASE_URL` is baked into the frontend bundle at build time, so changing it needs a Vercel **redeploy**.

## Environment variables

### Render web service **[dashboard]**
| Variable | Value | Notes |
|---|---|---|
| `DATABASE_URL` | the Postgres service's *Internal* URL | `postgres://` and `postgresql://` are accepted; the code selects the psycopg 3 driver |
| `ASL_QUEST_SECRET_KEY` | long random string (32+ chars) | **Required.** On Render the app refuses to start without it. Generate: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ASL_QUEST_CORS_ORIGINS` | `https://asl-quest-1.vercel.app` | Replaces the defaults; comma-separate to add more (no paths, trailing slash ignored) |
| `ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL` | `models/experiments/<landmark model>.joblib` (the 24 KB L2 model) | Needed for letter recognition; see "Model files" |
| `ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL` | optional | Grants admin to that e-mail only while no admin exists |
| `ASL_QUEST_DEV_COLLECT` | leave **unset** | Dataset collection must stay off in production |
| `ASL_QUEST_DEV_DIAGNOSTICS` | leave **unset** | Saves camera frames to disk |

### Vercel **[dashboard]**
| Variable | Value |
|---|---|
| `VITE_API_BASE_URL` | `https://asl-quest-1.onrender.com` (no trailing slash) |

Set it for *Production* and redeploy (without the build cache) afterwards.

## Render service settings **[dashboard]**
- Branch: `david-dev`. Build: `pip install -r requirements.txt`.
- Pre-deploy command: `alembic upgrade head` **only after** `python scripts/inspect_deployment_db.py` reports `UP TO DATE`, `BEHIND`, or `EMPTY`. If it reports `AMBIGUOUS`, stop and follow `DATABASE.md`.
- Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
- Render's disk is ephemeral: models and local videos in the repo or downloaded at build time survive only until the next deploy.

## Model files
Weights are not committed (large, and some licences unstated).
- **Letters (Practice, Words):** need only the ~24 KB landmark `.joblib`; the browser downloads MediaPipe's hand model from Google itself. `/health` then reports `experimental_landmark.loaded = true` and the app counts as ready (the ResNet `best_model.pth` is optional fallback).
- **Native Signs scoring:** additionally needs the I3D backbone weights (~58 MB), `head_best.pt`, and roughly 1.5-2 GB RAM for torch. Skip on small plans.
- Provide files by a Render persistent disk, or a build-step download from storage you control. Verify with SHA-256 before use.

## Verifying (do not trust "Live" or HTTP 200 alone)
1. Run `python scripts/inspect_deployment_db.py` in each Render shell: it is read-only and prints no secrets.
2. `GET /health`: `experimental_landmark.loaded` should be `true`.
3. In a browser on the Vercel URL: DevTools Network shows requests to `asl-quest-1.onrender.com` with no CORS errors; sign up, practise one letter, and check Progress.
4. A CORS preflight from the Vercel origin must return 200, not 400.

## Safety rules
Never merge or push `main` from this flow; never force-push; back up Postgres before any migration; never enable dataset collection in production.
