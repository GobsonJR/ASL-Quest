from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.security import get_current_user
from backend.services.user_features import get_recommendations

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("")
def recommendations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_recommendations(db, current_user)
