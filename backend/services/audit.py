"""Admin audit-log helper shared by the admin and dataset-collection routers."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from backend.models import AdminAuditLog, User


def audit(db: Session, admin: User, action: str, target_type: str, target_id: object, **detail: object) -> None:
    """Stage an audit row; the caller's commit stores it together with the change itself."""
    db.add(
        AdminAuditLog(
            admin_id=admin.id,
            admin_username=admin.username,
            action=action,
            target_type=target_type,
            target_id=str(target_id),
            detail=json.dumps(detail, default=str),
        )
    )
