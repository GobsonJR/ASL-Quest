"""Admin-only API for collecting LABELLED webcam data (alphabet frames and native-sign clips).

Two independent gates protect every endpoint here:
  1. the caller must be an authenticated ADMIN (role read from the database row of the token's user, never from the request);
  2. the operator must have switched the feature on with ASL_QUEST_DEV_COLLECT=1 (otherwise every route answers 404), because
     collected frames and clips can show identifiable people and their rooms.

Flow used by the Admin "Collection" workspace:
    POST /collect/sessions            describe the room once and record the contributor's consent (required)
    POST /collect/analyze             quality-check a captured frame and preview the exact crop (saves nothing)
    POST /collect/samples             save a frame, only with the person's confirmed letter
    POST /collect/negatives           save a no-hand / partial-hand frame (the recognizer must not answer on these)
    POST /collect/native-clips        save a recorded native-sign clip, only with the person's confirmed sign
    GET  /collect/sessions/{id}/samples | native-clips, and .../image | video   review gallery (admin-only file access)
    DELETE ...                        remove any sample / clip (audited)
    GET  /collect/export              labelled dataset as a zip with a manifest (audited)

The server never looks at a model prediction to decide a label. Collection and export are deliberately separate from
training, evaluation and model promotion: nothing here touches the training dataset or any model file.
"""

from __future__ import annotations

import base64
import os
import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from backend.database import get_db
from backend.models import NativeSign, User
from backend.routers import ml
from backend.security import get_current_admin
from backend.services import webcam_dataset as wd
from backend.services.audit import audit
from backend.settings import DEV_COLLECT, WEBCAM_DATASET_DIR


def require_enabled() -> None:
    if not DEV_COLLECT:
        raise HTTPException(status_code=404, detail="Not found")


# Authentication/authorization is evaluated first, so an anonymous caller cannot even learn whether collection is switched on.
router = APIRouter(
    prefix="/collect",
    tags=["collect (admin)"],
    dependencies=[Depends(get_current_admin), Depends(require_enabled)],
)

dataset = wd.WebcamDataset(WEBCAM_DATASET_DIR)

NO_STORE = {"Cache-Control": "private, no-store"}


class SessionIn(BaseModel):
    room: str = Field(default="room", max_length=80)
    lighting: str
    background: str
    background_notes: str = Field(default="", max_length=200)
    face_visible: bool = True
    glasses: bool = False
    person: str = Field(default="", max_length=32)
    camera: str = Field(default="", max_length=80)
    notes: str = Field(default="", max_length=300)
    # The person being recorded must agree before anything is captured; the session refuses to exist without it.
    consent_confirmed: bool = False


def _fail(error: wd.CollectError) -> HTTPException:
    return HTTPException(status_code=error.status, detail={"message": str(error), "reasons": error.reasons})


async def _read_limited(file: UploadFile, limit: int, what: str) -> bytes:
    payload = await file.read(limit + 1)  # never buffers more than one byte over the cap
    if not payload:
        raise HTTPException(status_code=400, detail={"message": f"empty {what} upload", "reasons": []})
    if len(payload) > limit:
        raise HTTPException(status_code=413, detail={"message": f"{what} is larger than {limit // (1024 * 1024)} MB", "reasons": []})
    return payload


async def _read_image(file: UploadFile) -> tuple[bytes, Image.Image]:
    payload = await _read_limited(file, wd.MAX_IMAGE_BYTES, "image")
    try:
        return payload, wd.validate_image_upload(payload)
    except wd.CollectError as error:
        raise _fail(error) from None


def _native_signs(db: Session) -> list[dict]:
    rows = db.query(NativeSign).filter(NativeSign.active.is_(True)).order_by(NativeSign.gloss).all()
    return [{"gloss": row.gloss, "display_name": row.display_name} for row in rows]


def _totals() -> tuple[dict[str, int], dict[str, int]]:
    letters = {letter: 0 for letter in wd.LETTERS}
    native: dict[str, int] = {}
    for session in dataset.list_sessions():
        for letter, n in session["counts"].items():
            letters[letter] += n
        for gloss, n in session.get("native_counts", {}).items():
            native[gloss] = native.get(gloss, 0) + n
    return letters, native


@router.get("/status")
def status(db: Session = Depends(get_db)) -> dict:
    letters, native = _totals()
    return {
        "enabled": True,
        "letters": wd.LETTERS,
        "native_signs": _native_signs(db),
        "options": {"lighting": wd.LIGHTING, "background": wd.BACKGROUNDS, "hand": wd.HANDS, "distance": wd.DISTANCES},
        "target_per_letter": wd.DEFAULT_TARGET_PER_LETTER,
        "target_per_native_sign": wd.DEFAULT_TARGET_PER_NATIVE_SIGN,
        "negative_kinds": list(wd.NEGATIVE_KINDS),
        "totals": letters,
        "native_totals": native,
        "limits": {
            "max_image_bytes": wd.MAX_IMAGE_BYTES,
            "max_clip_bytes": wd.MAX_CLIP_BYTES,
            "image_formats": ["JPEG", "PNG"],
            "clip_formats": list(wd.ALLOWED_CLIP_TYPES),
        },
        "consent_required": True,
        "thresholds": {
            "min_sharpness": wd.MIN_SHARPNESS,
            "min_hand_px": wd.MIN_HAND_PX,
            "brightness_range": [wd.MIN_BRIGHTNESS, wd.MAX_BRIGHTNESS],
        },
        "hand_detector_available": bool(ml.hand_detector and ml.hand_detector.available),
    }


@router.post("/sessions", status_code=201)
def create_session(body: SessionIn, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)) -> dict:
    try:
        record = dataset.create_session({**body.model_dump(), "recorded_by": admin.username})
    except wd.CollectError as error:
        raise _fail(error) from None
    audit(db, admin, "dataset.create_session", "dataset_session", record["session_id"], room=record.get("room"), contributor=record.get("person"))
    db.commit()
    return dataset.summary(record["session_id"])


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
    _, image = await _read_image(file)
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


@router.post("/negatives", status_code=201)
async def save_negative(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    kind: str = Form(...),
    confirmed: bool = Form(False),
    distance: str | None = Form(None),
) -> dict:
    """Save a frame with no usable hand (kind=no_hand) or a hand cut by the frame edge (kind=partial_hand). The label is the
    person's; what MediaPipe made of the frame is stored next to it as a measurement."""
    payload, image = await _read_image(file)
    report, crop = wd.analyze_frame(image, ml.hand_detector)
    try:
        record = dataset.save_negative(
            session_id=session_id, kind=kind, confirmed=confirmed, payload=payload, image=image,
            detector_found_hand=crop is not None, touches_edge=any("frame edge" in w for w in report.warnings), distance=distance,
        )
    except wd.CollectError as error:
        raise _fail(error) from None
    return {"negative": record, "session": dataset.summary(session_id)}


@router.post("/native-clips", status_code=201)
async def save_native_clip(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    gloss: str = Form(...),
    confirmed: bool = Form(False),
    hand: str | None = Form(None),
    duration_ms: int | None = Form(None),
    db: Session = Depends(get_db),
) -> dict:
    """Save a recorded clip of a native sign. The sign must be an active catalog sign, and the file type is judged from the
    file's own bytes (WebM or MP4), not from its name or declared MIME type."""
    gloss = (gloss or "").strip().upper()
    if db.query(NativeSign.id).filter(NativeSign.gloss == gloss, NativeSign.active.is_(True)).first() is None:
        raise _fail(wd.CollectError("unknown native sign", status=422))
    payload = await _read_limited(file, wd.MAX_CLIP_BYTES, "clip")
    kind = wd.sniff_clip_type(payload)
    if kind is None:
        raise _fail(wd.CollectError("unsupported video: use a WebM or MP4 recording", status=415))
    if duration_ms is not None and not (0 < duration_ms <= 120_000):
        raise _fail(wd.CollectError("duration_ms must be between 1 and 120000"))
    try:
        record = dataset.save_native_clip(
            session_id=session_id, gloss=gloss, confirmed=confirmed, payload=payload, kind=kind, duration_ms=duration_ms, hand=hand
        )
    except wd.CollectError as error:
        raise _fail(error) from None
    return {"clip": record, "session": dataset.summary(session_id)}


# ------------------------------------------------------------------ review gallery
@router.get("/sessions/{session_id}/samples")
def list_samples(session_id: str, label: str | None = Query(None, min_length=1, max_length=1, pattern="^[A-Za-z]$")) -> dict:
    try:
        samples = dataset.samples(session_id)
    except wd.CollectError as error:
        raise _fail(error) from None
    if label:
        samples = [s for s in samples if s["letter"] == label.upper()]
    return {"samples": samples}


@router.get("/sessions/{session_id}/samples/{sample_id}/image")
def sample_image(session_id: str, sample_id: str) -> FileResponse:
    try:
        path = dataset.sample_image_path(session_id, sample_id)
    except wd.CollectError as error:
        raise _fail(error) from None
    return FileResponse(path, headers=NO_STORE)


@router.get("/sessions/{session_id}/negatives/{sample_id}/image")
def negative_image(session_id: str, sample_id: str) -> FileResponse:
    try:
        path = dataset.negative_image_path(session_id, sample_id)
    except wd.CollectError as error:
        raise _fail(error) from None
    return FileResponse(path, headers=NO_STORE)


@router.get("/sessions/{session_id}/native-clips")
def list_native_clips(session_id: str, gloss: str | None = Query(None, pattern="^[A-Za-z0-9]{1,20}$")) -> dict:
    try:
        return {"clips": dataset.native_clips(session_id, gloss.upper() if gloss else None)}
    except wd.CollectError as error:
        raise _fail(error) from None


@router.get("/sessions/{session_id}/native-clips/{clip_id}/video")
def native_clip_video(session_id: str, clip_id: str) -> FileResponse:
    try:
        path = dataset.native_clip_path(session_id, clip_id)
    except wd.CollectError as error:
        raise _fail(error) from None
    media = "video/webm" if path.suffix == ".webm" else "video/mp4"
    return FileResponse(path, media_type=media, headers=NO_STORE)


# ------------------------------------------------------------------ deletion (every delete is audited)
@router.delete("/sessions/{session_id}/negatives/{sample_id}")
def delete_negative(session_id: str, sample_id: str, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)) -> dict:
    try:
        dataset.delete_negative(session_id, sample_id)
        audit(db, admin, "dataset.delete_negative", "dataset_sample", sample_id, session=session_id)
        db.commit()
        return {"deleted": sample_id, "session": dataset.summary(session_id)}
    except wd.CollectError as error:
        raise _fail(error) from None


@router.delete("/sessions/{session_id}/samples/{sample_id}")
def delete_sample(session_id: str, sample_id: str, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)) -> dict:
    try:
        dataset.delete_sample(session_id, sample_id)
        audit(db, admin, "dataset.delete_sample", "dataset_sample", sample_id, session=session_id)
        db.commit()
        return {"deleted": sample_id, "session": dataset.summary(session_id)}
    except wd.CollectError as error:
        raise _fail(error) from None


@router.delete("/sessions/{session_id}/native-clips/{clip_id}")
def delete_native_clip(session_id: str, clip_id: str, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)) -> dict:
    try:
        dataset.delete_native_clip(session_id, clip_id)
        audit(db, admin, "dataset.delete_clip", "dataset_sample", clip_id, session=session_id)
        db.commit()
        return {"deleted": clip_id, "session": dataset.summary(session_id)}
    except wd.CollectError as error:
        raise _fail(error) from None


# ------------------------------------------------------------------ export (collection only: no training, no model changes)
@router.get("/export")
def export_dataset(
    session_id: list[str] | None = Query(None, description="Limit the export to these sessions (default: every consented session)"),
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> FileResponse:
    for sid in session_id or []:
        try:
            dataset.session(sid)  # validates the id format and existence
        except wd.CollectError as error:
            raise _fail(error) from None
    handle = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    handle.close()
    try:
        result = dataset.export_zip(Path(handle.name), session_id)
    except Exception:
        os.unlink(handle.name)
        raise
    audit(
        db, admin, "dataset.export", "dataset", "all" if not session_id else ",".join(session_id),
        files=result["files"], sessions=len(result["sessions"]), skipped_without_consent=len(result["skipped_without_consent"]),
    )
    db.commit()
    return FileResponse(
        handle.name,
        media_type="application/zip",
        filename="asl_quest_dataset.zip",
        headers={**NO_STORE, "X-Export-Files": str(result["files"]), "X-Export-Skipped-Sessions": str(len(result["skipped_without_consent"]))},
        background=BackgroundTask(os.unlink, handle.name),
    )
