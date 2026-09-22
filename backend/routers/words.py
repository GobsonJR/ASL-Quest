from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.schemas import WordPracticeSessionCreate
from backend.security import get_current_user
from backend.services.words import (
    challenge_word_ids,
    daily_word_ids,
    get_all_progress,
    get_word,
    get_word_analytics,
    get_word_progress_detail,
    get_word_recommendations,
    list_catalog_with_progress,
    record_word_session,
    words_for_ids,
)
from backend.words.catalog import WORD_CATEGORIES

router = APIRouter(prefix="/words", tags=["words"])


@router.get("")
def list_words_endpoint(
    category: str | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if category and category.lower() != "all" and category.title() not in WORD_CATEGORIES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown word category.")
    return {"items": list_catalog_with_progress(db, current_user, category), "categories": list(WORD_CATEGORIES)}


@router.get("/progress")
def list_word_progress(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return {"items": get_all_progress(db, current_user)}


@router.get("/recommendations")
def word_recommendations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_word_recommendations(db, current_user)


@router.get("/analytics")
def word_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_word_analytics(db, current_user)


@router.get("/challenge")
def word_challenge(
    mode: str = Query("word", pattern="^(word|daily)$"),
    current_user: User = Depends(get_current_user),
):
    if mode == "daily":
        ids = daily_word_ids(current_user.id)
    else:
        ids = challenge_word_ids(current_user.id)
    return {"mode": mode, "items": words_for_ids(ids)}


@router.get("/{word_id}")
def get_word_endpoint(
    word_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    detail = get_word_progress_detail(db, current_user, word_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Word not found.")
    return detail


@router.get("/{word_id}/progress")
def get_one_word_progress(
    word_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if get_word(word_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Word not found.")
    detail = get_word_progress_detail(db, current_user, word_id)
    return detail["progress"] if detail else None


@router.post("/practice/session", status_code=status.HTTP_201_CREATED)
def create_word_practice_session(
    payload: WordPracticeSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if get_word(payload.word_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Word not found.")
    try:
        return record_word_session(db, current_user, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
