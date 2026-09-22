from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.schemas import GameStateSchema, LetterProgressResponse, LetterProgressUpdate, LetterStatsSchema, MigrationOfferResponse
from backend.security import get_current_user
from backend.services.progress import (
    has_meaningful_progress,
    local_progress_is_meaningful,
    persist_game_state,
    serialize_game_state,
)

router = APIRouter(prefix="/progress", tags=["progress"])


@router.get("", response_model=GameStateSchema)
def get_progress(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> GameStateSchema:
    return serialize_game_state(db, current_user)


@router.put("", response_model=GameStateSchema)
def update_progress(
    payload: GameStateSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> GameStateSchema:
    return persist_game_state(db, current_user, payload)


@router.get("/letters", response_model=list[LetterProgressResponse])
def get_letter_progress(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[LetterProgressResponse]:
    state = serialize_game_state(db, current_user)
    return [
        LetterProgressResponse(
            letter=letter,
            attempts=stats.attempts,
            correct_attempts=stats.correct,
            mastery=min(100.0, (stats.correct / 10) * 100.0),
            last_practiced=None,
        )
        for letter, stats in state.letterStats.items()
    ]


@router.put("/letters/{letter}", response_model=LetterProgressResponse)
def update_letter_progress(
    letter: str,
    payload: LetterProgressUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> LetterProgressResponse:
    letter = letter.upper()
    if len(letter) != 1 or letter < "A" or letter > "Z":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Letter must be A-Z.")

    state = serialize_game_state(db, current_user)
    state.letterStats[letter] = LetterStatsSchema(
        correct=payload.correct_attempts,
        attempts=payload.attempts,
    )
    persist_game_state(db, current_user, state)
    mastery = payload.mastery if payload.mastery is not None else min(100.0, (payload.correct_attempts / 10) * 100.0)
    return LetterProgressResponse(
        letter=letter,
        attempts=payload.attempts,
        correct_attempts=payload.correct_attempts,
        mastery=mastery,
        last_practiced=None,
    )


@router.post("/migrate-local", response_model=GameStateSchema)
def migrate_local_progress(
    payload: GameStateSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> GameStateSchema:
    if has_meaningful_progress(db, current_user):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Server progress already exists. Local progress was not imported.",
        )
    if not local_progress_is_meaningful(payload):
        return serialize_game_state(db, current_user)
    return persist_game_state(db, current_user, payload)


@router.get("/migration-offer", response_model=MigrationOfferResponse)
def migration_offer(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MigrationOfferResponse:
    if has_meaningful_progress(db, current_user):
        return MigrationOfferResponse(should_offer=False, reason="server_progress_exists")
    return MigrationOfferResponse(should_offer=True, reason="empty_server_progress")
