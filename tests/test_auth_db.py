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
os.environ["ASL_QUEST_SECRET_KEY"] = "test-secret-key"

from backend.database import Base, SessionLocal, engine, init_db  # noqa: E402
from backend.main import app  # noqa: E402
from backend.services.progress import seed_achievements  # noqa: E402


class AuthProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Base.metadata.drop_all(bind=engine)
        init_db()
        with SessionLocal() as db:
            seed_achievements(db)
        cls.client = TestClient(app)

    def register_user(self, username: str, email: str, password: str = "Password123!") -> str:
        response = self.client.post(
            "/auth/register",
            json={"username": username, "email": email, "password": password},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["access_token"]

    def auth_headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_user_registration(self) -> None:
        token = self.register_user("alice", "alice@example.com")
        self.assertTrue(token)

    def test_duplicate_registration_rejected(self) -> None:
        self.register_user("bob", "bob@example.com")
        response = self.client.post(
            "/auth/register",
            json={"username": "bob", "email": "bob@example.com", "password": "Password123!"},
        )
        self.assertEqual(response.status_code, 409)

    def test_login(self) -> None:
        self.register_user("carol", "carol@example.com")
        response = self.client.post(
            "/auth/login",
            json={"login": "carol", "password": "Password123!"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())

    def test_invalid_password(self) -> None:
        self.register_user("dave", "dave@example.com")
        response = self.client.post(
            "/auth/login",
            json={"login": "dave", "password": "WrongPassword!"},
        )
        self.assertEqual(response.status_code, 401)

    def test_authentication_required(self) -> None:
        response = self.client.get("/progress")
        self.assertEqual(response.status_code, 401)

    def test_me_endpoint(self) -> None:
        token = self.register_user("erin", "erin@example.com")
        response = self.client.get("/auth/me", headers=self.auth_headers(token))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["username"], "erin")

    def test_progress_retrieval_and_update(self) -> None:
        token = self.register_user("frank", "frank@example.com")
        headers = self.auth_headers(token)

        get_response = self.client.get("/progress", headers=headers)
        self.assertEqual(get_response.status_code, 200)
        state = get_response.json()
        self.assertEqual(state["xp"], 0)

        state["xp"] = 120
        state["totalCorrect"] = 3
        state["streak"] = 2
        state["letterStats"]["A"]["correct"] = 2
        state["letterStats"]["A"]["attempts"] = 3

        put_response = self.client.put("/progress", headers=headers, json=state)
        self.assertEqual(put_response.status_code, 200)
        updated = put_response.json()
        self.assertEqual(updated["xp"], 120)
        self.assertEqual(updated["letterStats"]["A"]["correct"], 2)

    def test_letter_progress_endpoint(self) -> None:
        token = self.register_user("gina", "gina@example.com")
        headers = self.auth_headers(token)
        response = self.client.put(
            "/progress/letters/B",
            headers=headers,
            json={"attempts": 4, "correct_attempts": 3},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["letter"], "B")
        self.assertEqual(response.json()["correct_attempts"], 3)

    def test_challenge_progress(self) -> None:
        token = self.register_user("hank", "hank@example.com")
        headers = self.auth_headers(token)
        response = self.client.post(
            "/challenges/daily/progress",
            headers=headers,
            json={"progress": 4, "completed": False, "meta": {"mistakes": 1}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["progress"], 4)

    def test_achievements_retrieval(self) -> None:
        token = self.register_user("iris", "iris@example.com")
        response = self.client.get("/achievements", headers=self.auth_headers(token))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(len(payload), 5)
        self.assertFalse(payload[0]["earned"])

    def test_practice_session_creation(self) -> None:
        token = self.register_user("jake", "jake@example.com")
        response = self.client.post(
            "/practice/session",
            headers=self.auth_headers(token),
            json={"letter": "C", "prediction": "C", "correct": True, "response_time": 3200},
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["correct"])

    def test_user_isolation(self) -> None:
        token_a = self.register_user("user_a", "user_a@example.com")
        token_b = self.register_user("user_b", "user_b@example.com")

        state = self.client.get("/progress", headers=self.auth_headers(token_a)).json()
        state["xp"] = 500
        self.client.put("/progress", headers=self.auth_headers(token_a), json=state)

        other_state = self.client.get("/progress", headers=self.auth_headers(token_b)).json()
        self.assertEqual(other_state["xp"], 0)

    def test_local_migration_prefers_empty_server(self) -> None:
        token = self.register_user("migrator", "migrator@example.com")
        headers = self.auth_headers(token)
        local_state = self.client.get("/progress", headers=headers).json()
        local_state["xp"] = 80
        local_state["totalCorrect"] = 2

        migrated = self.client.post("/progress/migrate-local", headers=headers, json=local_state)
        self.assertEqual(migrated.status_code, 200)
        self.assertEqual(migrated.json()["xp"], 80)

        conflict = self.client.post("/progress/migrate-local", headers=headers, json=local_state)
        self.assertEqual(conflict.status_code, 409)


if __name__ == "__main__":
    unittest.main()
