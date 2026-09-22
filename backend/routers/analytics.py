from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.security import get_current_user
from backend.services.analytics import (
    get_accuracy_trend,
    get_activity_trend,
    get_dashboard,
    get_heatmap,
    get_learning_funnel,
    get_letter_analytics,
    get_letter_detail,
    get_letter_insights,
    get_overview,
    get_practice_history,
    get_response_time_analytics,
    get_streak_analytics,
    get_weekly_summary,
    get_xp_analytics,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dashboard")
def analytics_dashboard(
    range: str = Query("30d", pattern="^(7d|30d|90d|12m|all)$"),
    heatmap_range: str = Query("12m", pattern="^(3m|6m|12m)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_dashboard(db, current_user, range, heatmap_range)  # type: ignore[arg-type]


@router.get("/overview")
def analytics_overview(
    range: str = Query("all", pattern="^(7d|30d|90d|12m|all)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_overview(db, current_user, range)  # type: ignore[arg-type]


@router.get("/activity")
def analytics_activity(
    range: str = Query("30d", pattern="^(7d|30d|90d|12m|all)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_activity_trend(db, current_user.id, range)  # type: ignore[arg-type]


@router.get("/accuracy")
def analytics_accuracy(
    range: str = Query("30d", pattern="^(7d|30d|90d|12m|all)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_accuracy_trend(db, current_user.id, range)  # type: ignore[arg-type]


@router.get("/letters")
def analytics_letters(
    range: str = Query("all", pattern="^(7d|30d|90d|12m|all)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_letter_analytics(db, current_user.id, range)  # type: ignore[arg-type]


@router.get("/letters/{letter}")
def analytics_letter_detail(
    letter: str,
    range: str = Query("all", pattern="^(7d|30d|90d|12m|all)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_letter_detail(db, current_user.id, letter, range)  # type: ignore[arg-type]


@router.get("/response-time")
def analytics_response_time(
    range: str = Query("all", pattern="^(7d|30d|90d|12m|all)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_response_time_analytics(db, current_user.id, range)  # type: ignore[arg-type]


@router.get("/history")
def analytics_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    letter: str | None = None,
    result: str | None = Query(None, pattern="^(correct|incorrect)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_practice_history(db, current_user.id, page=page, page_size=page_size, letter=letter, result=result)


@router.get("/heatmap")
def analytics_heatmap(
    range: str = Query("12m", pattern="^(3m|6m|12m)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_heatmap(db, current_user.id, range)  # type: ignore[arg-type]


@router.get("/streak")
def analytics_streak(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_streak_analytics(db, current_user)


@router.get("/xp")
def analytics_xp(
    range: str = Query("all", pattern="^(7d|30d|90d|12m|all)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_xp_analytics(db, current_user, range)  # type: ignore[arg-type]


@router.get("/insights")
def analytics_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_letter_insights(db, current_user.id)


@router.get("/funnel")
def analytics_funnel(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_learning_funnel(db, current_user.id)


@router.get("/weekly-summary")
def analytics_weekly_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_weekly_summary(db, current_user.id)
