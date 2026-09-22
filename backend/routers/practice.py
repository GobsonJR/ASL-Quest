from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import PracticeSession, User, XpEvent
from backend.schemas import PracticeSessionCreate, PracticeSessionResponse
from backend.security import get_current_user

router = APIRouter(prefix="/practice", tags=["practice"])


@router.post("/session", response_model=PracticeSessionResponse, status_code=status.HTTP_201_CREATED)
def create_practice_session(
    payload: PracticeSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PracticeSessionResponse:
    letter = payload.letter.upper()
    prediction = payload.prediction.upper() if payload.prediction else None
    session = PracticeSession(
        user_id=current_user.id,
        letter=letter,
        prediction=prediction,
        correct=payload.correct,
        response_time=payload.response_time,
        xp_earned=payload.xp_earned,
        challenge_type=payload.challenge_type,
        confidence=payload.confidence,
    )
    db.add(session)
    if payload.xp_earned > 0:
        db.add(
            XpEvent(
                user_id=current_user.id,
                amount=payload.xp_earned,
                reason=payload.reason or ("correct_sign" if payload.correct else "practice"),
            )
        )
    db.commit()
    db.refresh(session)
    return PracticeSessionResponse(
        id=session.id,
        letter=session.letter,
        prediction=session.prediction,
        correct=session.correct,
        response_time=session.response_time,
        xp_earned=session.xp_earned,
        challenge_type=session.challenge_type,
        created_at=session.created_at,
    )
