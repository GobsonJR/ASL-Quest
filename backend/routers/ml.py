from __future__ import annotations

import collections
import logging
import math
import time
from datetime import datetime, timezone
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from backend.services.experimental_recognizer import describe as describe_experimental
from backend.services.live_diagnostics import LiveDiagnostics
from backend.settings import DEV_DIAGNOSTICS, DEV_DIAGNOSTICS_DIR, HAND_DETECTION_CONFIDENCE
from src.config import CONFIDENCE_THRESHOLD
from src.inference.hands import HandObservation
from src.landmarks.features import N_LANDMARKS, LandmarkError

router = APIRouter(tags=["ml"])
logger = logging.getLogger("asl_quest.predict")

# Development-only capture of the exact classifier input (disabled unless ASL_QUEST_DEV_DIAGNOSTICS=1).
live_diag = LiveDiagnostics(DEV_DIAGNOSTICS_DIR, enabled=DEV_DIAGNOSTICS)

HISTORY_LIMIT = 200
history: collections.deque[dict] = collections.deque(maxlen=HISTORY_LIMIT)

predictor = None
hand_detector = None
load_error: str | None = None
# Which recognizer answers /predict by default (see backend/settings.py::RECOGNIZER). "resnet" is the default image classifier.
active_recognizer = "resnet"
landmark_classifier = None
landmark_error: str | None = None
# EXPERIMENTAL landmark model for per-request comparison (/predict?recognizer=landmarks); never the default.
experimental_landmarks = None
experimental_error: str | None = None


def set_runtime(predictor_instance, hand_detector_instance, error: str | None) -> None:
    global predictor, hand_detector, load_error
    predictor = predictor_instance
    hand_detector = hand_detector_instance
    load_error = error


def set_recognizer(name: str, classifier=None, error: str | None = None) -> None:
    global active_recognizer, landmark_classifier, landmark_error
    active_recognizer, landmark_classifier, landmark_error = name, classifier, error


def set_experimental_landmarks(classifier=None, error: str | None = None) -> None:
    global experimental_landmarks, experimental_error
    experimental_landmarks, experimental_error = classifier, error


@router.get("/health")
def health() -> dict:
    from src.utils.device import describe_device, get_device

    info = describe_device(get_device())
    return {
        "status": "ok" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
        "model_path": str(predictor.model_path) if predictor else None,
        "classes": predictor.class_names if predictor else [],
        "hand_detection_available": bool(hand_detector and hand_detector.available),
        "hand_detection_error": hand_detector.error if hand_detector else None,
        "error": load_error,
        "recognizer": active_recognizer,
        "landmark_model_loaded": landmark_classifier is not None,
        "landmark_model_error": landmark_error,
        "experimental_landmark": describe_experimental(experimental_landmarks, experimental_error),
        **info,
    }


@router.get("/history")
def get_history() -> dict:
    items = list(history)
    counts: dict[str, int] = {}
    for item in items:
        label = item["prediction"]
        counts[label] = counts.get(label, 0) + 1
    return {"count": len(items), "items": items[-50:], "letter_counts": counts}


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    require_hand: bool = False,
    debug: bool = False,
    top_k: int = 3,
    save_diagnostics: bool = False,
    target: str | None = None,
    preview_mirrored: bool | None = None,
    video_size: str | None = None,
    recognizer: str | None = None,
) -> JSONResponse:
    """Classify one still frame.

    `recognizer` (optional) picks the engine for THIS request: "resnet" (the production image classifier) or "landmarks"
    (the experimental MediaPipe-landmark model when one is configured, see ASL_QUEST_EXPERIMENTAL_LANDMARK_MODEL). Omitted,
    the server default applies (resnet unless ASL_QUEST_RECOGNIZER=landmarks), so existing clients are unaffected.

    Response `status` separates the stages so a weak prediction is never mistaken for "no hand":
    - hand_detector_unavailable / no_hand_detected: nothing was classified (prediction is null);
    - ok: a hand was found (or detection was not requested) and the model answered; check
      `low_confidence` for whether that answer cleared the confidence threshold.
    `debug=true` adds an image-free `diagnostics` object for development; `top_k` (1-10, default 3)
    sets how many candidate letters are returned in `top_predictions`. `save_diagnostics=true` (only when the
    server runs with ASL_QUEST_DEV_DIAGNOSTICS=1) writes this request's original frame, crop and model input to a
    local folder; `target`, `preview_mirrored` and `video_size` are recorded there as client context.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail=load_error or "Model is not loaded.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty upload. Choose an image file.")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image. Use JPG, PNG, or WEBP.") from None

    if recognizer not in (None, "resnet", "landmarks"):
        raise HTTPException(status_code=422, detail="recognizer must be 'resnet' or 'landmarks'.")
    choice = recognizer or active_recognizer
    if choice == "landmarks":
        # An explicit per-request choice prefers the experimental model; the server-wide landmark setting keeps its own.
        classifier, why_not = (experimental_landmarks or landmark_classifier, experimental_error or landmark_error) if recognizer == "landmarks" else (landmark_classifier, landmark_error)
        return _predict_landmarks(image, file.filename, top_k, debug, classifier, why_not)

    crop_box = None
    hand_detected = None
    source_image = image
    landmark_box = None
    if require_hand:
        if hand_detector is None or not hand_detector.available:
            return JSONResponse(
                {
                    "status": "hand_detector_unavailable",
                    "prediction": None,
                    "confidence": 0.0,
                    "top_predictions": [],
                    "low_confidence": True,
                    "threshold": CONFIDENCE_THRESHOLD,
                    "filename": file.filename,
                    "hand_detected": False,
                    "hand_box": None,
                    "hand_detection_available": False,
                    "message": hand_detector.error if hand_detector else "Hand detector is not initialized.",
                }
            )
        crop = hand_detector.crop_pil(image)
        hand_detected = crop.detected
        crop_box = crop.box
        landmark_box = crop.hand_box
        if not crop.detected:
            return JSONResponse(
                {
                    "status": "no_hand_detected",
                    "prediction": None,
                    "confidence": 0.0,
                    "top_predictions": [],
                    "low_confidence": True,
                    "threshold": CONFIDENCE_THRESHOLD,
                    "filename": file.filename,
                    "hand_detected": False,
                    "hand_box": None,
                    "hand_detection_available": True,
                    "message": "No hand detected.",
                }
            )
        source_image = crop.image

    want_capture = save_diagnostics and live_diag.enabled
    result = predictor.predict_pil(source_image, top_k=min(10, max(5 if want_capture else 1, top_k)))
    diagnostics = result.pop("diagnostics", None)
    captured = None
    if want_capture:
        captured = live_diag.save(
            payload=payload,
            frame=image,
            source_image=source_image,
            hand_box=landmark_box,
            crop_window=crop_box,
            predictor=predictor,
            result=result,
            meta={"target": target, "preview_mirrored": preview_mirrored, "video_size": video_size,
                  "require_hand": require_hand, "filename": file.filename},
        )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "predict frame=%s hand_detected=%s box=%s %s label=%s conf=%.3f",
            image.size, hand_detected, crop_box, diagnostics, result["prediction"], result["confidence"],
        )
    record = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "low_confidence": result["low_confidence"],
        "threshold": CONFIDENCE_THRESHOLD,
        "source": file.filename,
        "hand_detected": hand_detected,
        "hand_box": crop_box,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    history.append(record)
    return JSONResponse(
        {
            **result,
            "threshold": CONFIDENCE_THRESHOLD,
            "filename": file.filename,
            "status": "ok",
            "hand_detected": hand_detected,
            "hand_box": crop_box,
            "hand_detection_available": bool(hand_detector and hand_detector.available),
            "frame_size": [image.width, image.height],
            **(
                {"diagnostics": {**(diagnostics or {}), "hand_detection_confidence": HAND_DETECTION_CONFIDENCE,
                                  "capture_enabled": live_diag.enabled, **(captured or {})}}
                if debug
                else {}
            ),
        }
    )


class LandmarkPayload(BaseModel):
    """21 MediaPipe landmarks as the browser's HandLandmarker reports them (x, y normalised to the image, z relative depth).

    `handedness` is MediaPipe's RAW label ("Left"/"Right") for the image it was run on. `image_mirrored` says whether that
    image was a mirrored (selfie-flipped) frame; the app's models expect UNMIRRORED frames, so mirrored input is flipped back.
    """

    landmarks: list[list[float]]
    handedness: str | None = None
    image_mirrored: bool = False
    top_k: int = 3


def _validated_landmarks(payload: LandmarkPayload) -> list[tuple[float, float, float]]:
    if len(payload.landmarks) != N_LANDMARKS:
        raise HTTPException(status_code=422, detail=f"landmarks must contain exactly {N_LANDMARKS} points.")
    points = []
    for row in payload.landmarks:
        if len(row) != 3 or not all(math.isfinite(v) for v in row):
            raise HTTPException(status_code=422, detail="every landmark must be three finite numbers [x, y, z].")
        x, y, z = row
        if not (-0.5 <= x <= 1.5 and -0.5 <= y <= 1.5 and -5.0 <= z <= 5.0):
            raise HTTPException(status_code=422, detail="landmark coordinates are outside the plausible normalised range.")
        points.append((x, y, z))
    if payload.handedness not in (None, "Left", "Right"):
        raise HTTPException(status_code=422, detail="handedness must be 'Left', 'Right' or omitted.")
    return points


@router.post("/predict_landmarks")
def predict_landmarks(payload: LandmarkPayload) -> JSONResponse:
    """Classify landmarks the browser already detected (no image upload, no server-side detection).

    Uses the same LandmarkClassifier (canonicalisation + feature extraction + sklearn model) as /predict?recognizer=landmarks,
    so both paths give the same answer for the same landmarks. It never uses the ResNet and never changes the default.
    """
    started = time.perf_counter()
    classifier, why_not = (experimental_landmarks or landmark_classifier, experimental_error or landmark_error)
    if classifier is None:
        raise HTTPException(status_code=503, detail=why_not or "The landmark model is not loaded.")
    points = _validated_landmarks(payload)
    handedness = payload.handedness
    if payload.image_mirrored:
        points = [(1.0 - x, y, z) for x, y, z in points]
        handedness = {"Left": "Right", "Right": "Left"}.get(handedness or "")
    observation = HandObservation(points, handedness, None)
    t0 = time.perf_counter()
    try:
        result = classifier.predict_observation(observation, top_k=min(10, max(1, payload.top_k)))
    except LandmarkError:
        raise HTTPException(status_code=422, detail="The hand landmarks were unusable (degenerate hand).") from None
    except Exception:  # noqa: BLE001
        logger.exception("landmark classifier failed")
        raise HTTPException(status_code=503, detail="The landmark recognizer failed.") from None
    classify_ms = (time.perf_counter() - t0) * 1000
    return JSONResponse(
        {
            **result,
            "threshold": CONFIDENCE_THRESHOLD, "status": "ok", "recognizer": "landmarks", "actual_hand": observation.actual_hand,
            "timings_ms": {"classify": round(classify_ms, 2), "server_total": round((time.perf_counter() - started) * 1000, 2)},
            **({"landmark_model": classifier.path.name} if getattr(classifier, "path", None) else {}),
        }
    )


def _hand_missing(status: str, filename: str, available: bool, message: str) -> JSONResponse:
    return JSONResponse(
        {
            "status": status, "prediction": None, "confidence": 0.0, "top_predictions": [], "low_confidence": True,
            "threshold": CONFIDENCE_THRESHOLD, "filename": filename, "hand_detected": False, "hand_box": None,
            "hand_detection_available": available, "message": message, "recognizer": "landmarks",
        }
    )


def _predict_landmarks(image: Image.Image, filename: str, top_k: int, debug: bool, classifier=None, why_not: str | None = None) -> JSONResponse:
    """The landmark classifier path. It needs a detected hand by construction and never lets a failure crash the request:

    * model not loaded      -> 503 with the reason (the caller asked for an engine that does not exist)
    * detector missing/crashed -> the same "hand_detector_unavailable" reply the image path gives
    * no hand / unusable landmarks (degenerate hand, NaN) -> "no_hand_detected", nothing is guessed
    * the model itself fails -> 503 with a plain message (logged with its traceback)
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail=why_not or "The landmark model is not loaded.")
    if hand_detector is None or not hand_detector.available:
        return _hand_missing("hand_detector_unavailable", filename, False, hand_detector.error if hand_detector else "Hand detector is not initialized.")
    try:
        rgb = np.array(image.convert("RGB"))
        observation = hand_detector.observe(rgb)
        crop = hand_detector.crop_rgb(rgb, observation=observation) if observation is not None else None
    except Exception:  # noqa: BLE001 - MediaPipe can fail on odd frames; that must not take the app down
        logger.exception("hand detection failed on a frame")
        return _hand_missing("hand_detector_unavailable", filename, False, "Hand detection failed on this frame.")
    if observation is None:
        return _hand_missing("no_hand_detected", filename, True, "No hand detected.")
    try:
        result = classifier.predict_observation(observation, top_k=min(10, max(1, top_k)))
    except LandmarkError:
        return _hand_missing("no_hand_detected", filename, True, "The hand landmarks were unusable (degenerate or non-finite); nothing was classified.")
    except Exception:  # noqa: BLE001
        logger.exception("landmark classifier failed")
        raise HTTPException(status_code=503, detail="The landmark recognizer failed on this frame.") from None
    if not result:
        return _hand_missing("no_hand_detected", filename, True, "No hand detected.")
    box = crop.box if crop is not None else None
    history.append(
        {
            "prediction": result["prediction"], "confidence": result["confidence"], "low_confidence": result["low_confidence"],
            "threshold": CONFIDENCE_THRESHOLD, "source": filename, "hand_detected": True, "hand_box": box,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    model_file = getattr(getattr(classifier, "path", None), "name", None)
    return JSONResponse(
        {
            **result,
            "threshold": CONFIDENCE_THRESHOLD, "filename": filename, "status": "ok", "hand_detected": True, "hand_box": box,
            "hand_detection_available": True, "frame_size": [image.width, image.height], "recognizer": "landmarks",
            **({"landmark_model": model_file} if model_file else {}),
            **({"diagnostics": {"recognizer": "landmarks", "handedness_raw": observation.handedness,
                                "handedness_score": observation.handedness_score, "actual_hand": observation.actual_hand,
                                "hand_detection_confidence": HAND_DETECTION_CONFIDENCE}} if debug else {}),
        }
    )
