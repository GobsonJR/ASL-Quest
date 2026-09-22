from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.schemas import UserResponse
from backend.security import get_current_user, hash_password, verify_password
from backend.services.user_features import build_profile, export_user_data

router = APIRouter(prefix="/users", tags=["users"])


class PreferencesUpdate(BaseModel):
    reduced_motion: bool | None = None


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)


class DeleteAccountRequest(BaseModel):
    password: str
    confirm: str = Field(description="Type DELETE to confirm")


@router.get("/me/profile")
def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return build_profile(db, current_user)


@router.get("/me/preferences")
def get_preferences(current_user: User = Depends(get_current_user)):
    return json.loads(current_user.preferences or "{}")


@router.put("/me/preferences")
def update_preferences(
    payload: PreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prefs = json.loads(current_user.preferences or "{}")
    if payload.reduced_motion is not None:
        prefs["reduced_motion"] = payload.reduced_motion
    current_user.preferences = json.dumps(prefs)
    db.commit()
    return prefs


@router.post("/me/change-password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    payload: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect.")
    current_user.password_hash = hash_password(payload.new_password)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(
    payload: DeleteAccountRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if payload.confirm.strip().upper() != "DELETE":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Confirmation text must be DELETE.")
    if not verify_password(payload.password, current_user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password is incorrect.")
    db.delete(current_user)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me/export")
def export_data(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return export_user_data(db, current_user)


@router.get("/me/export.json")
def export_data_download(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    payload = export_user_data(db, current_user)
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": 'attachment; filename="asl-quest-export.json"'},
    )
