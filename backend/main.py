from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.routers import achievements, analytics, auth, admin, admin_ops, challenges, chatbot, collect, ml, native, practice, progress, recommendations, users, words
from backend.services.experimental_recognizer import load_experimental_landmarks
from backend.services.progress import seed_achievements
from backend.settings import CORS_ORIGINS, EXPERIMENTAL_LANDMARK_MODEL, HAND_DETECTION_CONFIDENCE, LANDMARK_MODEL_PATH, RECOGNIZER
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
app.include_router(admin_ops.router)
app.include_router(words.router)
app.include_router(native.router)
app.include_router(chatbot.router)
app.include_router(collect.router)  # development only: 404 unless ASL_QUEST_DEV_COLLECT=1


@app.on_event("startup")
def startup() -> None:
    from backend.startup_checks import enforce

    enforce()  # production only: refuses the public default JWT secret before anything else starts
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
        hand_detector_instance = HandDetector(static_image_mode=True, detection_confidence=HAND_DETECTION_CONFIDENCE)
        print(f"Loaded model from {predictor_instance.model_path} on {predictor_instance.device}")
        if not hand_detector_instance.available:
            print(f"Hand detector unavailable: {hand_detector_instance.error}")
    except Exception as exc:
        error = str(exc)
        print(f"Model not loaded: {error}")

    ml.set_runtime(predictor_instance, hand_detector_instance, error)

    # Optional alternative recognizer. Default stays "resnet"; "landmarks" must be chosen explicitly and fails loudly.
    if RECOGNIZER == "landmarks":
        try:
            from src.landmarks.classifier import LandmarkClassifier

            ml.set_recognizer("landmarks", LandmarkClassifier(LANDMARK_MODEL_PATH))
            print(f"Recognizer: landmarks ({LANDMARK_MODEL_PATH})")
        except Exception as exc:
            ml.set_recognizer("landmarks", None, f"Could not load the landmark model {LANDMARK_MODEL_PATH}: {exc}")
            print(ml.landmark_error)
    elif RECOGNIZER != "resnet":
        ml.set_recognizer("resnet", None, f"Unknown ASL_QUEST_RECOGNIZER={RECOGNIZER!r}; using resnet")
        print(ml.landmark_error)

    # Experimental landmark model for live side-by-side comparison (/predict?recognizer=landmarks). Off unless configured;
    # a failure to load only disables the comparison and never affects the ResNet.
    if EXPERIMENTAL_LANDMARK_MODEL:
        classifier, why_not = load_experimental_landmarks(EXPERIMENTAL_LANDMARK_MODEL)
        ml.set_experimental_landmarks(classifier, why_not)
        print(f"Experimental landmark recognizer: {classifier.path.name if classifier else 'NOT loaded - ' + str(why_not)}")


def health() -> dict:
    return ml.health()


async def predict(*args, **kwargs):
    return await ml.predict(*args, **kwargs)


def bind_address(environ: "os._Environ[str] | dict[str, str] | None" = None) -> tuple[str, int]:
    """Where `python -m backend.main` should listen.

    Hosting platforms (Render, etc.) hand the port to the app in $PORT and can only reach a server bound
    to 0.0.0.0; with no PORT set (local development) it stays on 127.0.0.1:8000 as before. The documented
    production start command is still `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
    """
    import os

    port = (environ if environ is not None else os.environ).get("PORT", "").strip()
    if port.isdigit() and 0 < int(port) < 65536:
        return "0.0.0.0", int(port)
    return "127.0.0.1", 8000


if __name__ == "__main__":
    import uvicorn

    _host, _port = bind_address()
    uvicorn.run("backend.main:app", host=_host, port=_port, reload=False)
