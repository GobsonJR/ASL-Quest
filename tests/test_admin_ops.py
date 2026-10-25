"""Tests for the admin management API (backend/routers/admin_ops.py): permissions, audit trail, safety guards."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["ASL_QUEST_DATABASE_URL"] = f"sqlite:///{TEST_DB.name}"
os.environ["ASL_QUEST_SECRET_KEY"] = "admin-ops-test-secret"

from backend import models  # noqa: E402,F401
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import AdminAuditLog, NativeSign, PracticeSession, User, UserProgress  # noqa: E402


class AdminOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.admin_headers = self.make("boss", admin=True)
        self.student_headers = self.make("student1")

    def make(self, username: str, admin: bool = False) -> dict:
        response = self.client.post(
            "/auth/register", json={"username": username, "email": f"{username}@example.com", "password": "Password123!"}
        )
        self.assertEqual(response.status_code, 201, response.text)
        if admin:
            with SessionLocal() as db:
                db.query(User).filter(User.username == username).one().role = "admin"
                db.commit()
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    def uid(self, username: str) -> int:
        with SessionLocal() as db:
            return db.query(User).filter(User.username == username).one().id

    def audit_actions(self) -> list[str]:
        with SessionLocal() as db:
            return [row.action for row in db.query(AdminAuditLog).order_by(AdminAuditLog.id)]

    def test_non_admin_is_refused_everywhere(self) -> None:
        sid = self.uid("student1")
        for method, path in [
            ("get", f"/admin/users/{sid}"), ("patch", f"/admin/users/{sid}"), ("post", f"/admin/users/{sid}/reset-progress"),
            ("delete", f"/admin/users/{sid}?confirm=student1"), ("get", "/admin/audit-log"), ("get", "/admin/models"),
            ("get", "/admin/native-signs"), ("get", "/admin/dataset/sessions"), ("get", "/admin/chatbot/feedback"),
        ]:
            extra = {"json": {}} if method == "patch" else {}
            self.assertEqual(getattr(self.client, method)(path, headers=self.student_headers, **extra).status_code, 403, f"{method} {path}")
            self.assertEqual(getattr(self.client, method)(path, **extra).status_code, 401, f"{method} {path}")

    def test_role_change_and_deactivation_are_audited_and_enforced(self) -> None:
        sid = self.uid("student1")
        self.assertEqual(self.client.patch(f"/admin/users/{sid}", json={"is_active": False}, headers=self.admin_headers).status_code, 200)
        self.assertEqual(self.client.get("/auth/me", headers=self.student_headers).status_code, 403)
        login = self.client.post("/auth/login", json={"login": "student1", "password": "Password123!"})
        self.assertEqual(login.status_code, 403)
        self.client.patch(f"/admin/users/{sid}", json={"is_active": True, "role": "admin"}, headers=self.admin_headers)
        self.assertEqual(self.client.get("/auth/me", headers=self.student_headers).json()["role"], "admin")
        self.assertEqual(self.audit_actions(), ["user.deactivate", "user.role", "user.activate"])

    def test_admin_cannot_demote_deactivate_or_delete_self(self) -> None:
        me = self.uid("boss")
        self.assertEqual(self.client.patch(f"/admin/users/{me}", json={"role": "student"}, headers=self.admin_headers).status_code, 400)
        self.assertEqual(self.client.patch(f"/admin/users/{me}", json={"is_active": False}, headers=self.admin_headers).status_code, 400)
        self.assertEqual(self.client.delete(f"/admin/users/{me}?confirm=boss", headers=self.admin_headers).status_code, 400)
        self.assertEqual(self.audit_actions(), [])

    def test_invalid_role_rejected(self) -> None:
        sid = self.uid("student1")
        self.assertEqual(self.client.patch(f"/admin/users/{sid}", json={"role": "superuser"}, headers=self.admin_headers).status_code, 422)

    def test_reset_progress_keeps_account(self) -> None:
        sid = self.uid("student1")
        with SessionLocal() as db:
            db.query(UserProgress).filter(UserProgress.user_id == sid).one().xp = 500
            db.add(PracticeSession(user_id=sid, letter="A", correct=True, xp_earned=10))
            db.commit()
        self.assertEqual(self.client.post(f"/admin/users/{sid}/reset-progress", headers=self.admin_headers).status_code, 200)
        detail = self.client.get(f"/admin/users/{sid}", headers=self.admin_headers).json()
        self.assertEqual((detail["xp"], detail["level"], detail["practice_sessions"]), (0, 1, 0))
        self.assertEqual(self.audit_actions(), ["user.reset_progress"])

    def test_delete_requires_matching_username(self) -> None:
        sid = self.uid("student1")
        self.assertEqual(self.client.delete(f"/admin/users/{sid}?confirm=wrong", headers=self.admin_headers).status_code, 400)
        self.assertEqual(self.client.delete(f"/admin/users/{sid}", headers=self.admin_headers).status_code, 422)
        self.assertEqual(self.client.delete(f"/admin/users/{sid}?confirm=student1", headers=self.admin_headers).status_code, 200)
        self.assertEqual(self.client.get(f"/admin/users/{sid}", headers=self.admin_headers).status_code, 404)
        self.assertEqual(self.audit_actions(), ["user.delete"])

    def test_native_sign_edit_is_audited(self) -> None:
        with SessionLocal() as db:
            sign = NativeSign(gloss="hello", display_name="Hello", active=True)
            db.add(sign)
            db.commit()
            sign_id = sign.id
        response = self.client.patch(
            f"/admin/native-signs/{sign_id}", json={"active": False, "display_name": "Hi"}, headers=self.admin_headers
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["active"])
        listing = self.client.get("/admin/native-signs", headers=self.admin_headers).json()["items"]
        self.assertEqual(listing[0]["display_name"], "Hi")
        log = self.client.get("/admin/audit-log", headers=self.admin_headers).json()
        self.assertEqual(log["items"][0]["action"], "native_sign.update")
        self.assertIn("active", log["items"][0]["detail"]["changes"])

    def test_audit_log_filter(self) -> None:
        sid = self.uid("student1")
        self.client.patch(f"/admin/users/{sid}", json={"role": "admin"}, headers=self.admin_headers)
        self.client.patch(f"/admin/users/{sid}", json={"is_active": False}, headers=self.admin_headers)
        log = self.client.get("/admin/audit-log?action=user.role", headers=self.admin_headers).json()
        self.assertEqual((log["total"], log["items"][0]["admin"]), (1, "boss"))

    def test_read_only_panels_work_on_empty_database(self) -> None:
        for path in ("/admin/models", "/admin/predictions", "/admin/chatbot/feedback", "/admin/achievements", "/admin/dataset/sessions"):
            self.assertEqual(self.client.get(path, headers=self.admin_headers).status_code, 200, path)


if __name__ == "__main__":
    unittest.main()
