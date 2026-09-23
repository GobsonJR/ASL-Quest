from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import NativeSign, NativeSignProgress, NativeSignSession, SignPrediction, User
from backend.security import get_current_user
from src.i3d_transfer.predict import predict_video

router = APIRouter(prefix="/native", tags=["native"])

# No runtime model-version registration mechanism exists anywhere in this project yet
# (model_versions has no rows for any model, including A-Z). Per the integration audit,
# this is a plain constant rather than an invented admin/registration system.
MODEL_TYPE = "native_i3d"
MODEL_VERSION = "native_i3d_v1"

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB — no existing project convention to reuse; a reasonable local limit
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# A sign counts as "mastered" once its stored mastery reaches 100 — the same convention
# already used by Word Spelling (WORD_MASTERED_COMPLETIONS -> 100% mastery, see
# backend/services/words.py::word_mastery_percent) and A-Z letters
# (frontend/src/game/constants.ts::MASTERED_CORRECT). Not a new/invented threshold.
MASTERY_COMPLETE_THRESHOLD = 100.0


@router.get("/signs")
def list_native_signs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    signs = db.query(NativeSign).filter(NativeSign.active.is_(True)).order_by(NativeSign.gloss).all()
    return {
        "items": [
            {
                "id": sign.id,
                "gloss": sign.gloss,
                "display_name": sign.display_name,
                "meaning": sign.meaning,
                "category": sign.category,
                "difficulty": sign.difficulty,
                "description": sign.description,
                "example_text": sign.example_text,
                "dataset_available": sign.dataset_available,
                "model_available": sign.model_available,
                "active": sign.active,
            }
            for sign in signs
        ]
    }


def _serialize_sign_progress(sign: NativeSign, progress: NativeSignProgress | None) -> dict:
    return {
        "sign_id": sign.id,
        "gloss": sign.gloss,
        "display_name": sign.display_name,
        "attempts": progress.attempts if progress else 0,
        "correct": progress.correct_attempts if progress else 0,
        "mastery": progress.mastery if progress else 0.0,
        "last_practiced": progress.last_practiced.isoformat() if progress and progress.last_practiced else None,
    }


@router.get("/progress")
def get_native_progress(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    active_signs = db.query(NativeSign).filter(NativeSign.active.is_(True)).order_by(NativeSign.gloss).all()
    progress_rows = (
        db.query(NativeSignProgress).filter(NativeSignProgress.user_id == current_user.id).all()
    )
    progress_by_sign_id = {row.native_sign_id: row for row in progress_rows}

    signs_payload = []
    started_signs = 0
    mastered_signs = 0
    mastery_sum = 0.0
    for sign in active_signs:
        progress = progress_by_sign_id.get(sign.id)
        entry = _serialize_sign_progress(sign, progress)
        if entry["attempts"] > 0:
            started_signs += 1
        if entry["mastery"] >= MASTERY_COMPLETE_THRESHOLD:
            mastered_signs += 1
        mastery_sum += entry["mastery"]
        signs_payload.append(entry)

    total_signs = len(active_signs)
    overall_mastery = round(mastery_sum / total_signs, 1) if total_signs else 0.0

    return {
        "total_signs": total_signs,
        "started_signs": started_signs,
        "mastered_signs": mastered_signs,
        "overall_mastery": overall_mastery,
        "signs": signs_payload,
    }


@router.get("/progress/{sign_id}")
def get_native_sign_progress(
    sign_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    # Only active signs are ever exposed via GET /native/signs, so treat an inactive
    # or unknown sign_id identically here — a user can't have progress on a sign they
    # can never see in the catalog.
    sign = db.query(NativeSign).filter(NativeSign.id == sign_id, NativeSign.active.is_(True)).first()
    if sign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Native sign not found.")

    progress = (
        db.query(NativeSignProgress)
        .filter(NativeSignProgress.user_id == current_user.id, NativeSignProgress.native_sign_id == sign_id)
        .first()
    )
    return _serialize_sign_progress(sign, progress)


def _get_or_create_progress(db: Session, user_id: int, native_sign_id: int) -> NativeSignProgress:
    progress = (
        db.query(NativeSignProgress)
        .filter(NativeSignProgress.user_id == user_id, NativeSignProgress.native_sign_id == native_sign_id)
        .first()
    )
    if progress is None:
        # Only ever created here, on an actual practice attempt — never pre-created for
        # every sign at registration/catalog-view time.
        progress = NativeSignProgress(
            user_id=user_id,
            native_sign_id=native_sign_id,
            attempts=0,
            correct_attempts=0,
            accuracy=0.0,
            mastery=0.0,
        )
        db.add(progress)
    return progress


@router.post("/predict")
async def predict_native_sign(
    file: UploadFile = File(...),
    top_k: int = 5,
    expected_label: str | None = None,
    native_sign_id: int | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty upload. Choose a video file.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format. Use one of: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}.",
        )

    # native_sign_id drives the real practice flow (session + progress). Resolved from
    # the DB up front, before any inference work, so a bad/inactive sign_id fails fast
    # and never trusts a client-submitted gloss string as the practice target.
    expected_sign: NativeSign | None = None
    if native_sign_id is not None:
        expected_sign = (
            db.query(NativeSign).filter(NativeSign.id == native_sign_id, NativeSign.active.is_(True)).first()
        )
        if expected_sign is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Native sign not found.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit.",
        )

    tmp_path: Path | None = None
    started_at = datetime.now(timezone.utc)
    try:
        # Native inference (src/i3d_transfer/predict.py's decoder) needs a real file on
        # disk, not an in-memory buffer. Written to a private temp file and always
        # deleted below; the clip is never persisted anywhere.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)

        started = time.perf_counter()
        try:
            result = predict_video(tmp_path, top_k=top_k)
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Native sign model is not available right now.",
            ) from None
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not read or process the uploaded video.",
            ) from None
        latency_ms = int((time.perf_counter() - started) * 1000)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    completed_at = datetime.now(timezone.utc)

    resolved_native_sign_id = None
    matched_sign = db.query(NativeSign).filter(NativeSign.gloss == result["prediction"]).first()
    if matched_sign is not None:
        resolved_native_sign_id = matched_sign.id

    # Normalized comparison (case/whitespace-insensitive) against the real DB gloss —
    # never the other way around. expected_label always reflects the actual catalog
    # gloss, not whatever a client happened to send.
    correct: bool | None = None
    effective_expected_label = expected_label
    if expected_sign is not None:
        effective_expected_label = expected_sign.gloss
        correct = result["prediction"].strip().upper() == expected_sign.gloss.strip().upper()
    elif expected_label is not None:
        correct = result["prediction"].strip().upper() == expected_label.strip().upper()

    session_id = None
    if expected_sign is not None:
        practice_session = NativeSignSession(
            user_id=current_user.id,
            native_sign_id=expected_sign.id,
            model_version=MODEL_VERSION,
            started_at=started_at,
            completed_at=completed_at,
            result="correct" if correct else "incorrect",
            confidence=result["confidence"],
            response_time=latency_ms,
        )
        db.add(practice_session)
        db.flush()  # assign practice_session.id before it's referenced below
        session_id = practice_session.id

        # Progress is server-computed from this real attempt only — never fabricated,
        # never pre-created for signs the user hasn't actually practiced.
        progress = _get_or_create_progress(db, current_user.id, expected_sign.id)
        progress.attempts += 1
        if correct:
            progress.correct_attempts += 1
        progress.accuracy = round((progress.correct_attempts / progress.attempts) * 100, 1)
        # Native mastery has no separate formula/threshold anywhere in the existing
        # schema or project docs beyond the "mastery" float column itself, so — per
        # instruction not to invent a complicated algorithm — mastery is simply the
        # running accuracy. Documented here rather than left implicit.
        progress.mastery = progress.accuracy
        if progress.best_confidence is None or result["confidence"] > progress.best_confidence:
            progress.best_confidence = result["confidence"]
        if progress.best_response_time is None or latency_ms < progress.best_response_time:
            progress.best_response_time = latency_ms
        progress.last_practiced = completed_at

    prediction_row = SignPrediction(
        user_id=current_user.id,
        session_id=session_id,
        model_type=MODEL_TYPE,
        model_version=MODEL_VERSION,
        predicted_label=result["prediction"],
        expected_label=effective_expected_label,
        confidence=result["confidence"],
        latency_ms=latency_ms,
        correct=correct,
    )
    db.add(prediction_row)
    db.commit()

    response = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "top_k": result["top_k"],
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "latency_ms": latency_ms,
        "native_sign_id": resolved_native_sign_id,
        "correct": correct,
    }
    if expected_sign is not None:
        response["expected_sign"] = {
            "id": expected_sign.id,
            "gloss": expected_sign.gloss,
            "display_name": expected_sign.display_name,
        }
        response["session_id"] = session_id
        response["response_time"] = latency_ms
    return response
