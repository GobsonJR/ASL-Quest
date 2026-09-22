from __future__ import annotations

import json
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
os.environ["ASL_QUEST_SECRET_KEY"] = "test-secret-key-phase4"
os.environ["ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL"] = "admin@example.com"

from backend.database import Base, SessionLocal, engine, init_db  # noqa: E402
from backend.main import app  # noqa: E402
from backend.services.progress import seed_achievements  # noqa: E402


class Phase4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Base.metadata.drop_all(bind=engine)
        init_db()
        with SessionLocal() as db:
            seed_achievements(db)
        cls.client = TestClient(app)

    def auth_headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def register_user(self, username: str, email: str, password: str = "Password123!") -> str:
        response = self.client.post(
            "/auth/register",
            json={"username": username, "email": email, "password": password},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["access_token"]

    def test_profile_requires_auth(self) -> None:
        response = self.client.get("/users/me/profile")
        self.assertEqual(response.status_code, 401)

    def test_profile_returns_real_fields(self) -> None:
        token = self.register_user("profile_user", "profile@example.com")
        response = self.client.get("/users/me/profile", headers=self.auth_headers(token))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["username"], "profile_user")
        self.assertEqual(payload["email"], "profile@example.com")
        self.assertIn("avatar_initial", payload)
        self.assertIn("total_days_active", payload)
        self.assertNotIn("password", payload)
        self.assertNotIn("password_hash", payload)

    def test_recommendations_empty_user(self) -> None:
        token = self.register_user("reco_user", "reco@example.com")
        response = self.client.get("/recommendations", headers=self.auth_headers(token))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["has_data"])
        self.assertEqual(payload["recommended"], [])

    def test_export_contains_only_user_data(self) -> None:
        token = self.register_user("export_user", "export@example.com")
        response = self.client.get("/users/me/export", headers=self.auth_headers(token))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("profile", payload)
        self.assertIn("practice_history", payload)
        self.assertIn("achievements", payload)
        dumped = json.dumps(payload)
        self.assertNotIn("password", dumped.lower())

    def test_change_password(self) -> None:
        token = self.register_user("pass_user", "pass@example.com")
        response = self.client.post(
            "/users/me/change-password",
            headers=self.auth_headers(token),
            json={"current_password": "Password123!", "new_password": "NewPassword456!"},
        )
        self.assertEqual(response.status_code, 204)
        login = self.client.post(
            "/auth/login",
            json={"login": "pass_user", "password": "NewPassword456!"},
        )
        self.assertEqual(login.status_code, 200)

    def test_delete_account(self) -> None:
        token = self.register_user("delete_user", "delete@example.com")
        response = self.client.request(
            "DELETE",
            "/users/me",
            headers=self.auth_headers(token),
            json={"password": "Password123!", "confirm": "DELETE"},
        )
        self.assertEqual(response.status_code, 204)
        me = self.client.get("/auth/me", headers=self.auth_headers(token))
        self.assertEqual(me.status_code, 401)

    def test_student_cannot_access_admin(self) -> None:
        token = self.register_user("student_user", "student@example.com")
        response = self.client.get("/admin/overview", headers=self.auth_headers(token))
        self.assertEqual(response.status_code, 403)

    def test_admin_bootstrap_and_access(self) -> None:
        token = self.register_user("admin_user", "admin@example.com")
        me = self.client.get("/auth/me", headers=self.auth_headers(token))
        self.assertEqual(me.json()["role"], "admin")
        overview = self.client.get("/admin/overview", headers=self.auth_headers(token))
        self.assertEqual(overview.status_code, 200)
        self.assertIn("total_users", overview.json())

    def test_admin_users_search(self) -> None:
        response = self.client.post(
            "/auth/register",
            json={"username": "admin_search", "email": "admin@example.com", "password": "Password123!"},
        )
        if response.status_code == 409:
            login = self.client.post(
                "/auth/login",
                json={"login": "admin@example.com", "password": "Password123!"},
            )
            self.assertEqual(login.status_code, 200)
            admin_token = login.json()["access_token"]
        else:
            admin_token = response.json()["access_token"]
        self.register_user("findme", "findme@example.com")
        response = self.client.get(
            "/admin/users?search=findme",
            headers=self.auth_headers(admin_token),
        )
        self.assertEqual(response.status_code, 200)
        items = response.json()["items"]
        self.assertTrue(any(item["username"] == "findme" for item in items))

    def test_preferences_update(self) -> None:
        token = self.register_user("prefs_user", "prefs@example.com")
        response = self.client.put(
            "/users/me/preferences",
            headers=self.auth_headers(token),
            json={"reduced_motion": True},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["reduced_motion"])

    def test_practice_session_user_isolation(self) -> None:
        token_a = self.register_user("user_a", "user_a@example.com")
        token_b = self.register_user("user_b", "user_b@example.com")
        self.client.post(
            "/practice/session",
            headers=self.auth_headers(token_a),
            json={"letter": "A", "prediction": "A", "correct": True, "response_time": 1200},
        )
        export_b = self.client.get("/users/me/export", headers=self.auth_headers(token_b))
        self.assertEqual(len(export_b.json()["practice_history"]), 0)


if __name__ == "__main__":
    unittest.main()
