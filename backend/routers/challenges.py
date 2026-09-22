from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import ChallengeProgress, User
from backend.schemas import ChallengeItemResponse, ChallengeProgressUpdate, GameStateSchema
from backend.security import get_current_user
from backend.services.progress import persist_game_state, serialize_game_state

router = APIRouter(prefix="/challenges", tags=["challenges"])


@router.get("", response_model=list[ChallengeItemResponse])
def get_challenges(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ChallengeItemResponse]:
    rows = db.query(ChallengeProgress).filter(ChallengeProgress.user_id == current_user.id).all()
    if not rows:
        state = serialize_game_state(db, current_user)
        persist_game_state(db, current_user, state)
        rows = db.query(ChallengeProgress).filter(ChallengeProgress.user_id == current_user.id).all()
    import json

    return [
        ChallengeItemResponse(
            challenge_type=row.challenge_type,
            progress=row.progress,
            completed=row.completed,
            completed_at=row.completed_at,
            meta=json.loads(row.meta or "{}"),
        )
        for row in rows
    ]


@router.post("/{challenge_type}/progress", response_model=ChallengeItemResponse)
def update_challenge_progress(
    challenge_type: str,
    payload: ChallengeProgressUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChallengeItemResponse:
    if challenge_type not in {"daily", "alphabet", "speed", "word"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown challenge type.")

    state = serialize_game_state(db, current_user)
    if challenge_type == "daily":
        state.dailyChallenge.progress = payload.progress
        state.dailyChallenge.completed = payload.completed
        if payload.meta:
            state.dailyChallenge.mistakes = payload.meta.get("mistakes", state.dailyChallenge.mistakes)
            state.dailyChallenge.perfectBonusAwarded = payload.meta.get(
                "perfectBonusAwarded", state.dailyChallenge.perfectBonusAwarded
            )
    elif challenge_type == "alphabet":
        state.challenges.alphabetCompleted = payload.completed
    elif challenge_type == "word":
        state.challenges.wordCount = payload.progress
        state.challenges.wordCompleted = payload.completed
    else:
        state.challenges.speedCount = payload.progress
        state.challenges.speedCompleted = payload.completed
        if payload.meta and "speedStartedAt" in payload.meta:
            state.challenges.speedStartedAt = payload.meta["speedStartedAt"]

    persist_game_state(db, current_user, state)
    row = (
        db.query(ChallengeProgress)
        .filter(ChallengeProgress.user_id == current_user.id, ChallengeProgress.challenge_type == challenge_type)
        .one()
    )
    import json

    return ChallengeItemResponse(
        challenge_type=row.challenge_type,
        progress=row.progress,
        completed=row.completed,
        completed_at=row.completed_at,
        meta=json.loads(row.meta or "{}"),
    )
