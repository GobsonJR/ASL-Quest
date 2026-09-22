from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Achievement, User, UserAchievement
from backend.schemas import AchievementResponse
from backend.security import get_current_user
from backend.services.progress import seed_achievements

router = APIRouter(prefix="/achievements", tags=["achievements"])


@router.get("", response_model=list[AchievementResponse])
def list_achievements(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[AchievementResponse]:
    seed_achievements(db)
    earned = {
        link.achievement_id: link.earned_at
        for link in db.query(UserAchievement).filter(UserAchievement.user_id == current_user.id).all()
    }
    achievements = db.query(Achievement).order_by(Achievement.id.asc()).all()
    return [
        AchievementResponse(
            id=item.slug,
            name=item.name,
            description=item.description,
            requirement=item.requirement,
            earned=item.id in earned,
            earned_at=earned.get(item.id),
        )
        for item in achievements
    ]
