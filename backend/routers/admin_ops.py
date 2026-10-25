"""Admin management actions: users, content, model/chatbot review, webcam dataset, audit log.

Read-only analytics live in ``admin.py``. Every state-changing endpoint here writes an ``AdminAuditLog`` row in the
same transaction as the change, so an action can never succeed without being recorded.
"""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import (
    AdminAuditLog,
    Achievement,
    ChallengeProgress,
    ChatbotConversation,
    ChatbotFeedback,
    ChatbotMessage,
    LetterProgress,
    ModelVersion,
    NativeSign,
    NativeSignProgress,
    NativeSignSession,
    PracticeSession,
    SignPrediction,
    User,
    UserAchievement,
    UserPreference,
    UserProgress,
    WordPracticeSession,
    WordProgress,
    XpEvent,
)
from backend.routers.collect import dataset
from backend.security import get_current_admin, hash_password
from backend.services.audit import audit
from backend.services.progress import ensure_user_progress
from backend.services import webcam_dataset as wd

router = APIRouter(prefix="/admin", tags=["admin (management)"])

ROLES = ("student", "admin")


def _get_user(db: Session, user_id: int) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return user


def _not_self(admin: User, user: User, verb: str) -> None:
    if admin.id == user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"You can't {verb} your own account.")


# ------------------------------------------------------------------ users
@router.get("/users/{user_id}")
def user_detail(user_id: int, _: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    user = _get_user(db, user_id)
    progress = db.query(UserProgress).filter(UserProgress.user_id == user.id).first()
    sessions = db.query(PracticeSession).filter(PracticeSession.user_id == user.id)
    total = sessions.count()
    correct = sessions.filter(PracticeSession.correct.is_(True)).count()
    recent = sessions.order_by(PracticeSession.created_at.desc()).limit(10).all()
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat(),
        "xp": progress.xp if progress else 0,
        "level": progress.level if progress else 1,
        "current_streak": progress.current_streak if progress else 0,
        "best_streak": progress.best_streak if progress else 0,
        "practice_sessions": total,
        "accuracy": round(correct / total * 100, 1) if total else None,
        "achievements": db.query(UserAchievement).filter(UserAchievement.user_id == user.id).count(),
        "word_sessions": db.query(WordPracticeSession).filter(WordPracticeSession.user_id == user.id).count(),
        "native_sessions": db.query(NativeSignSession).filter(NativeSignSession.user_id == user.id).count(),
        "recent_activity": [
            {"letter": row.letter, "correct": row.correct, "created_at": row.created_at.isoformat()} for row in recent
        ],
    }


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: Literal["student", "admin"] = "student"


@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(body: UserCreate, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Create an account with an explicit role. Only an authenticated admin can reach this (get_current_admin reads the
    role from the database row of the token's user, never from anything the browser sends), and it is audited."""
    username = body.username.strip()
    email = str(body.email).strip().lower()
    if db.query(User).filter((User.username == username) | (User.email == email)).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username or email already registered.")
    user = User(username=username, email=email, password_hash=hash_password(body.password), role=body.role)
    db.add(user)
    db.flush()
    audit(db, admin, "user.create", "user", user.id, username=username, role=body.role)
    db.commit()
    db.refresh(user)
    ensure_user_progress(db, user)
    return {"id": user.id, "username": user.username, "email": user.email, "role": user.role, "is_active": user.is_active}


class UserPatch(BaseModel):
    role: Literal["student", "admin"] | None = None
    is_active: bool | None = None


@router.patch("/users/{user_id}")
def update_user(
    user_id: int, body: UserPatch, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)
):
    user = _get_user(db, user_id)
    if body.role is None and body.is_active is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Nothing to change.")
    if body.role is not None and body.role != user.role:
        _not_self(admin, user, "change the role of")
        audit(db, admin, "user.role", "user", user.id, username=user.username, before=user.role, after=body.role)
        user.role = body.role
    if body.is_active is not None and body.is_active != user.is_active:
        _not_self(admin, user, "deactivate")
        audit(db, admin, "user.activate" if body.is_active else "user.deactivate", "user", user.id, username=user.username)
        user.is_active = body.is_active
    db.commit()
    return {"id": user.id, "role": user.role, "is_active": user.is_active}


@router.post("/users/{user_id}/reset-progress")
def reset_user_progress(user_id: int, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    user = _get_user(db, user_id)
    for model in (
        LetterProgress, ChallengeProgress, UserAchievement, PracticeSession, XpEvent, WordProgress,
        WordPracticeSession, NativeSignProgress, NativeSignSession,
    ):
        if model is NativeSignSession:  # predictions point at sessions; they are history, not progress
            db.query(SignPrediction).filter(SignPrediction.user_id == user.id).update({SignPrediction.session_id: None})
        db.query(model).filter(model.user_id == user.id).delete(synchronize_session=False)
    progress = db.query(UserProgress).filter(UserProgress.user_id == user.id).first()
    if progress:
        progress.xp = 0
        progress.level = 1
        progress.current_streak = progress.best_streak = progress.total_correct = 0
        progress.total_attempts = progress.sessions = 0
        progress.last_active_date = None
        progress.game_meta = "{}"
    audit(db, admin, "user.reset_progress", "user", user.id, username=user.username)
    db.commit()
    return {"reset": user.id}


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    confirm: str = Query(..., description="Must equal the username being deleted."),
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    user = _get_user(db, user_id)
    _not_self(admin, user, "delete")
    if confirm != user.username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Confirmation does not match the username.")
    # children declared with passive_deletes rely on DB-level cascades, which SQLite doesn't enforce by default
    db.query(SignPrediction).filter(SignPrediction.user_id == user.id).delete(synchronize_session=False)
    db.query(NativeSignProgress).filter(NativeSignProgress.user_id == user.id).delete(synchronize_session=False)
    db.query(NativeSignSession).filter(NativeSignSession.user_id == user.id).delete(synchronize_session=False)
    db.query(UserPreference).filter(UserPreference.user_id == user.id).delete(synchronize_session=False)
    for conversation in db.query(ChatbotConversation).filter(ChatbotConversation.user_id == user.id).all():
        db.delete(conversation)
    audit(db, admin, "user.delete", "user", user.id, username=user.username, email=user.email)
    db.delete(user)
    db.commit()
    return {"deleted": user_id}


# ------------------------------------------------------------------ content
class NativeSignPatch(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    meaning: str | None = Field(default=None, max_length=1000)
    description: str | None = Field(default=None, max_length=2000)
    category: str | None = Field(default=None, max_length=32)
    difficulty: str | None = Field(default=None, max_length=16)
    active: bool | None = None


def _sign_row(sign: NativeSign) -> dict:
    return {
        "id": sign.id, "gloss": sign.gloss, "display_name": sign.display_name, "meaning": sign.meaning,
        "description": sign.description, "category": sign.category, "difficulty": sign.difficulty,
        "active": sign.active, "dataset_available": sign.dataset_available, "model_available": sign.model_available,
    }


@router.get("/native-signs")
def list_native_signs(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    return {"items": [_sign_row(s) for s in db.query(NativeSign).order_by(NativeSign.gloss).all()]}


@router.patch("/native-signs/{sign_id}")
def update_native_sign(
    sign_id: int, body: NativeSignPatch, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)
):
    sign = db.get(NativeSign, sign_id)
    if sign is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sign not found.")
    changes = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None or k in ("meaning", "description", "category", "difficulty")}
    changed = {k: {"before": getattr(sign, k), "after": v} for k, v in changes.items() if getattr(sign, k) != v}
    for key, value in changes.items():
        setattr(sign, key, value)
    if changed:
        audit(db, admin, "native_sign.update", "native_sign", sign.id, gloss=sign.gloss, changes=changed)
    db.commit()
    return _sign_row(sign)


class AchievementPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = Field(default=None, min_length=1, max_length=1000)


@router.get("/achievements")
def list_achievements(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    unlocked = dict(
        db.query(UserAchievement.achievement_id, func.count(UserAchievement.id)).group_by(UserAchievement.achievement_id).all()
    )
    return {
        "items": [
            {"id": a.id, "slug": a.slug, "name": a.name, "description": a.description, "requirement": a.requirement,
             "unlocked_by": int(unlocked.get(a.id, 0))}
            for a in db.query(Achievement).order_by(Achievement.id).all()
        ]
    }


@router.patch("/achievements/{achievement_id}")
def update_achievement(
    achievement_id: int, body: AchievementPatch, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)
):
    row = db.get(Achievement, achievement_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Achievement not found.")
    changed = {k: {"before": getattr(row, k), "after": v} for k, v in body.model_dump(exclude_none=True).items() if getattr(row, k) != v}
    for key, change in changed.items():
        setattr(row, key, change["after"])
    if changed:
        audit(db, admin, "achievement.update", "achievement", row.id, slug=row.slug, changes=changed)
    db.commit()
    return {"id": row.id, "slug": row.slug, "name": row.name, "description": row.description}


# ------------------------------------------------------------------ models and chatbot (read-only)
@router.get("/models")
def model_versions(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    stats = {
        (row.model_type, row.model_version): row
        for row in db.query(
            SignPrediction.model_type,
            SignPrediction.model_version,
            func.count(SignPrediction.id).label("n"),
            func.sum(case((SignPrediction.correct.is_(True), 1), else_=0)).label("right"),
            func.count(SignPrediction.correct).label("graded"),
            func.avg(SignPrediction.latency_ms).label("latency"),
        ).group_by(SignPrediction.model_type, SignPrediction.model_version).all()
    }
    registered = []
    for mv in db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all():
        row = stats.get((mv.model_type, mv.version))
        registered.append(_model_row(mv.model_type, mv.version, row, mv))
    seen = {(r["model_type"], r["version"]) for r in registered}
    unregistered = [_model_row(t, v, row, None) for (t, v), row in stats.items() if (t, v) not in seen]
    return {"items": registered + unregistered}


def _model_row(model_type: str, version: str, row, mv: ModelVersion | None) -> dict:
    graded = int(row.graded) if row else 0
    return {
        "model_type": model_type,
        "version": version,
        "registered": mv is not None,
        "checkpoint_path": mv.checkpoint_path if mv else None,
        "num_classes": mv.num_classes if mv else None,
        "created_at": mv.created_at.isoformat() if mv else None,
        "predictions": int(row.n) if row else 0,
        "graded": graded,
        "accuracy": round(int(row.right or 0) / graded * 100, 1) if graded else None,
        "avg_latency_ms": round(float(row.latency), 1) if row and row.latency is not None else None,
    }


@router.get("/predictions")
def recent_predictions(_: User = Depends(get_current_admin), db: Session = Depends(get_db), limit: int = Query(30, ge=1, le=100)):
    rows = db.query(SignPrediction).order_by(SignPrediction.created_at.desc()).limit(limit).all()
    return {
        "items": [
            {"user_id": r.user_id, "model_version": r.model_version, "predicted": r.predicted_label,
             "expected": r.expected_label, "confidence": r.confidence, "correct": r.correct,
             "created_at": r.created_at.isoformat()}
            for r in rows
        ]
    }


@router.get("/chatbot/feedback")
def chatbot_feedback(
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
    negative_only: bool = True,
    limit: int = Query(30, ge=1, le=100),
):
    """Only messages a learner explicitly rated are exposed; full conversations stay private."""
    summary = {
        "conversations": db.query(ChatbotConversation).count(),
        "messages": db.query(ChatbotMessage).count(),
        "positive": db.query(ChatbotFeedback).filter(ChatbotFeedback.rating > 0).count(),
        "negative": db.query(ChatbotFeedback).filter(ChatbotFeedback.rating < 0).count(),
    }
    query = db.query(ChatbotFeedback, ChatbotMessage).join(ChatbotMessage, ChatbotFeedback.message_id == ChatbotMessage.id)
    if negative_only:
        query = query.filter(ChatbotFeedback.rating < 0)
    items = []
    for fb, msg in query.order_by(ChatbotFeedback.created_at.desc()).limit(limit).all():
        question = (
            db.query(ChatbotMessage)
            .filter(ChatbotMessage.conversation_id == msg.conversation_id, ChatbotMessage.role == "user", ChatbotMessage.id < msg.id)
            .order_by(ChatbotMessage.id.desc())
            .first()
        )
        items.append(
            {"rating": fb.rating, "comment": fb.feedback, "answer": msg.content,
             "question": question.content if question else None, "created_at": fb.created_at.isoformat()}
        )
    return {"summary": summary, "items": items}


# ------------------------------------------------------------------ webcam dataset
@router.get("/dataset/sessions")
def dataset_sessions(_: User = Depends(get_current_admin)):
    sessions = dataset.list_sessions()
    return {"sessions": sessions, "totals": dataset.overall_counts(), "target_per_letter": wd.DEFAULT_TARGET_PER_LETTER}


@router.delete("/dataset/sessions/{session_id}")
def delete_dataset_session(session_id: str, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    try:
        summary = dataset.summary(session_id)
        dataset.delete_session(session_id)
    except wd.CollectError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from None
    audit(db, admin, "dataset.delete_session", "dataset_session", session_id, frames=summary["total"], room=summary.get("room"))
    db.commit()
    return {"deleted": session_id}


# ------------------------------------------------------------------ audit log
@router.get("/audit-log")
def audit_log(
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    action: str | None = None,
):
    query = db.query(AdminAuditLog)
    if action:
        query = query.filter(AdminAuditLog.action == action)
    total = query.count()
    rows = query.order_by(AdminAuditLog.id.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {
        "page": page, "page_size": page_size, "total": total,
        "items": [
            {"id": r.id, "admin": r.admin_username, "action": r.action, "target_type": r.target_type,
             "target_id": r.target_id, "detail": json.loads(r.detail or "{}"), "created_at": r.created_at.isoformat()}
            for r in rows
        ],
    }
