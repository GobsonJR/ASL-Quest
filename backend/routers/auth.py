from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from backend.security import create_access_token, get_current_user, hash_password, verify_password
from backend.settings import get_bootstrap_admin_email
from backend.services.progress import ensure_user_progress, seed_achievements

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    username = payload.username.strip()
    email = str(payload.email).strip().lower()
    if db.query(User).filter((User.username == username) | (User.email == email)).first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username or email already registered.")

    user = User(
        username=username,
        email=email,
        password_hash=hash_password(payload.password),
        role="admin" if get_bootstrap_admin_email() and email == get_bootstrap_admin_email() else "student",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    ensure_user_progress(db, user)
    seed_achievements(db)
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    login_value = payload.login.strip().lower()
    user = (
        db.query(User)
        .filter((User.username == payload.login.strip()) | (User.email == login_value))
        .first()
    )
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username/email or password.")
    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        created_at=current_user.created_at,
    )
