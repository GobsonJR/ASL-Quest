"""Development-only API for collecting LABELLED webcam frames (disabled unless ASL_QUEST_DEV_COLLECT=1).

Flow used by the Collect page:
    POST /collect/sessions            describe the room once (lighting, background, face/glasses...)
    POST /collect/analyze             quality-check a captured frame and preview the exact crop (saves nothing)
    POST /collect/samples             save it, only with the person's confirmed letter
    DELETE /collect/sessions/{id}/samples/{sample_id}   undo

The server never looks at a model prediction to decide a label.
"""

from __future__ import annotations

import base64
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from backend.routers import ml
from backend.services import webcam_dataset as wd
from backend.settings import DEV_COLLECT, WEBCAM_DATASET_DIR


def require_enabled() -> None:
    if not DEV_COLLECT:
        raise HTTPException(status_code=404, detail="Not found")


router = APIRouter(prefix="/collect", tags=["collect (development)"], dependencies=[Depends(require_enabled)])

dataset = wd.WebcamDataset(WEBCAM_DATASET_DIR)


class SessionIn(BaseModel):
    room: str = Field(default="room", max_length=80)
    lighting: str
    background: str
    background_notes: str = Field(default="", max_length=200)
    face_visible: bool = True
    glasses: bool = False
    camera: str = Field(default="", max_length=80)
    notes: str = Field(default="", max_length=300)


def _fail(error: wd.CollectError) -> HTTPException:
    return HTTPException(status_code=error.status, detail={"message": str(error), "reasons": error.reasons})


async def _read_image(file: UploadFile) -> tuple[bytes, Image.Image]:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail={"message": "empty upload", "reasons": []})
    try:
        return payload, wd.open_image(payload)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail={"message": "invalid image", "reasons": []}) from None


@router.get("/status")
def status() -> dict:
    return {
        "enabled": True,
        "letters": wd.LETTERS,
        "options": {"lighting": wd.LIGHTING, "background": wd.BACKGROUNDS, "hand": wd.HANDS, "distance": wd.DISTANCES},
        "target_per_letter": wd.DEFAULT_TARGET_PER_LETTER,
        "totals": dataset.overall_counts(),
        "thresholds": {
            "min_sharpness": wd.MIN_SHARPNESS,
            "min_hand_px": wd.MIN_HAND_PX,
            "brightness_range": [wd.MIN_BRIGHTNESS, wd.MAX_BRIGHTNESS],
        },
        "hand_detector_available": bool(ml.hand_detector and ml.hand_detector.available),
    }


@router.post("/sessions", status_code=201)
def create_session(body: SessionIn) -> dict:
    try:
        return dataset.create_session(body.model_dump())
    except wd.CollectError as error:
        raise _fail(error) from None


@router.get("/sessions")
def list_sessions() -> dict:
    return {"sessions": dataset.list_sessions(), "totals": dataset.overall_counts()}


@router.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    try:
        return dataset.summary(session_id)
    except wd.CollectError as error:
        raise _fail(error) from None


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    """Quality-check a frame and preview the exact crop the classifier would receive. Saves nothing."""
    payload, image = await _read_image(file)
    report, crop = wd.analyze_frame(image, ml.hand_detector)
    preview = None
    if crop is not None:
        buffer = BytesIO()
        crop.image.convert("RGB").resize((224, 224)).save(buffer, "PNG")
        preview = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return JSONResponse({**report.as_dict(), "hand_box": crop.box if crop else None, "crop_preview": preview})


@router.post("/samples", status_code=201)
async def save_sample(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    letter: str = Form(...),
    confirmed: bool = Form(False),
    hand: str = Form(...),
    distance: str = Form(...),
    mode: str = Form("single"),
    sequence_id: str | None = Form(None),
    frame_index: int | None = Form(None),
    video_size: str | None = Form(None),
) -> dict:
    payload, image = await _read_image(file)
    report, _ = wd.analyze_frame(image, ml.hand_detector)
    try:
        record = dataset.save_sample(
            session_id=session_id, letter=letter, confirmed=confirmed, payload=payload, image=image, report=report,
            hand=hand, distance=distance, mode=mode, sequence_id=sequence_id, frame_index=frame_index, video_size=video_size,
        )
    except wd.CollectError as error:
        raise _fail(error) from None
    return {"sample": record, "session": dataset.summary(session_id)}


@router.delete("/sessions/{session_id}/samples/{sample_id}")
def delete_sample(session_id: str, sample_id: str) -> dict:
    try:
        dataset.delete_sample(session_id, sample_id)
        return {"deleted": sample_id, "session": dataset.summary(session_id)}
    except wd.CollectError as error:
        raise _fail(error) from None
