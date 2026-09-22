from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.routers import achievements, analytics, auth, admin, challenges, ml, practice, progress, recommendations, users, words
from backend.services.progress import seed_achievements
from backend.settings import CORS_ORIGINS
from src.config import BEST_MODEL_PATH, LEGACY_MODEL_PATH
from src.inference.engine import Predictor
from src.inference.hands import HandDetector

app = FastAPI(
    title="ASL Quest",
    version="5.0.0",
    description="Gamified ASL alphabet and sequential word-spelling learning platform. The ML model recognizes static A–Z signs only.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml.router)
app.include_router(auth.router)
app.include_router(progress.router)
app.include_router(challenges.router)
app.include_router(achievements.router)
app.include_router(practice.router)
app.include_router(analytics.router)
app.include_router(users.router)
app.include_router(recommendations.router)
app.include_router(admin.router)
app.include_router(words.router)


@app.on_event("startup")
def startup() -> None:
    init_db()
    from backend.database import SessionLocal

    with SessionLocal() as db:
        seed_achievements(db)

    path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else LEGACY_MODEL_PATH
    predictor_instance = None
    hand_detector_instance = None
    error: str | None = None
    try:
        predictor_instance = Predictor(model_path=path if path.exists() else None)
        hand_detector_instance = HandDetector(static_image_mode=True)
        print(f"Loaded model from {predictor_instance.model_path} on {predictor_instance.device}")
        if not hand_detector_instance.available:
            print(f"Hand detector unavailable: {hand_detector_instance.error}")
    except Exception as exc:
        error = str(exc)
        print(f"Model not loaded: {error}")

    ml.set_runtime(predictor_instance, hand_detector_instance, error)


def health() -> dict:
    return ml.health()


async def predict(*args, **kwargs):
    return await ml.predict(*args, **kwargs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
