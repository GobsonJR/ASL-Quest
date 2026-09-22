from __future__ import annotations

import collections
from datetime import datetime, timezone
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from src.config import CONFIDENCE_THRESHOLD

router = APIRouter(tags=["ml"])

HISTORY_LIMIT = 200
history: collections.deque[dict] = collections.deque(maxlen=HISTORY_LIMIT)

predictor = None
hand_detector = None
load_error: str | None = None


def set_runtime(predictor_instance, hand_detector_instance, error: str | None) -> None:
    global predictor, hand_detector, load_error
    predictor = predictor_instance
    hand_detector = hand_detector_instance
    load_error = error


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
async def predict(file: UploadFile = File(...), require_hand: bool = False) -> JSONResponse:
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

    crop_box = None
    hand_detected = None
    source_image = image
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

    result = predictor.predict_pil(source_image)
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
        }
    )
