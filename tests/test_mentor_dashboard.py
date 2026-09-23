"""Focused tests for the Mentor Analytics Dashboard
(backend/services/mentor_dashboard.py, GET /admin/mentor-dashboard).

Independent of the existing app test suite: uses its own temporary SQLite
file and creates every table via Base.metadata.create_all, matching the
pattern in test_native.py / test_chatbot.py. Nothing here touches
data/asl_quest.db.

Also regression-covers GET /admin/overview and GET /admin/letters, since
this feature moved their implementations out of backend/routers/admin.py
and into backend/services/admin_stats.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["ASL_QUEST_DATABASE_URL"] = f"sqlite:///{TEST_DB.name}"
os.environ["ASL_QUEST_SECRET_KEY"] = "mentor-dashboard-test-secret"

from backend import models  # noqa: E402,F401  (registers all ORM tables on Base.metadata)
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import (  # noqa: E402
    ChatbotConversation,
    ChatbotFeedback,
    ChatbotMessage,
    NativeSign,
    NativeSignProgress,
    NativeSignSession,
    User,
    UserProgress,
)


class MentorDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    def register(self, username: str, email: str | None = None) -> str:
        response = self.client.post(
            "/auth/register",
            json={"username": username, "email": email or f"{username}@example.com", "password": "Password123!"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["access_token"]

    def auth(self, token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def admin_token(self) -> str:
        # Promoted directly via the ORM rather than ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL:
        # that env var is process-global, and unittest discover imports every test
        # module (running each file's module-level os.environ[...] = ...) before
        # any test runs, so whichever test file happens to import last would win
        # and silently break every other file relying on a different bootstrap
        # email. Setting the role directly is deterministic regardless of
        # discovery/import order.
        token = self.register("mentor_admin", "mentor_admin@example.com")
        with SessionLocal() as db:
            user = db.query(User).filter(User.username == "mentor_admin").first()
            user.role = "admin"
            db.commit()
        return token

    # --- Access control ---------------------------------------------------------

    def test_requires_auth(self) -> None:
        response = self.client.get("/admin/mentor-dashboard")
        self.assertEqual(response.status_code, 401)

    def test_requires_admin(self) -> None:
        token = self.register("plain_student")
        response = self.client.get("/admin/mentor-dashboard", headers=self.auth(token))
        self.assertEqual(response.status_code, 403)

    def test_admin_can_access(self) -> None:
        token = self.admin_token()
        response = self.client.get("/admin/mentor-dashboard", headers=self.auth(token))
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        for key in ("overview", "students", "az", "words", "native", "chatbot"):
            self.assertIn(key, payload)

    # --- Honest empty state (no invented data) -----------------------------------

    def test_empty_database_reports_honest_zeros(self) -> None:
        token = self.admin_token()
        payload = self.client.get("/admin/mentor-dashboard", headers=self.auth(token)).json()
        self.assertEqual(payload["overview"]["total_letter_attempts"], 0)
        self.assertEqual(payload["overview"]["total_native_sign_attempts"], 0)
        self.assertEqual(payload["overview"]["total_word_practice_sessions"], 0)
        self.assertEqual(payload["native"]["total_attempts"], 0)
        self.assertEqual(payload["native"]["category_distribution"], [])
        self.assertEqual(payload["chatbot"]["total_conversations"], 0)
        self.assertEqual(payload["chatbot"]["positive_feedback"], 0)
        self.assertEqual(payload["chatbot"]["negative_feedback"], 0)
        # The only student is the admin themself; a fresh account has no mastery
        # yet, and that must show as null (unknown), never a fabricated 0%.
        student = payload["students"][0]
        self.assertIsNone(student["native_mastery"])
        self.assertIsNone(student["accuracy"])

    # --- Overview reflects real data ----------------------------------------------

    def test_overview_reflects_real_practice_sessions(self) -> None:
        token = self.register("az_learner")
        self.client.post(
            "/practice/session",
            json={"letter": "A", "prediction": "A", "correct": True, "response_time": 500},
            headers=self.auth(token),
        )
        self.client.post(
            "/practice/session",
            json={"letter": "B", "prediction": "C", "correct": False, "response_time": 700},
            headers=self.auth(token),
        )
        admin = self.admin_token()
        payload = self.client.get("/admin/mentor-dashboard", headers=self.auth(admin)).json()
        self.assertEqual(payload["overview"]["total_letter_attempts"], 2)
        self.assertEqual(payload["overview"]["total_users"], 2)
        letter_a = next(item for item in payload["az"]["popular"] if item["letter"] == "A")
        self.assertEqual(letter_a["attempts"], 1)
        self.assertEqual(letter_a["accuracy"], 100.0)

    # --- Student progress row -----------------------------------------------------

    def test_student_row_reflects_progress_and_streak(self) -> None:
        token = self.register("progress_student")
        me = self.client.get("/auth/me", headers=self.auth(token)).json()
        with SessionLocal() as db:
            progress = db.query(UserProgress).filter(UserProgress.user_id == me["id"]).first()
            progress.xp = 340
            progress.level = 2
            progress.current_streak = 5
            progress.total_correct = 8
            progress.total_attempts = 10
            db.commit()

        admin = self.admin_token()
        payload = self.client.get("/admin/mentor-dashboard", headers=self.auth(admin)).json()
        row = next(item for item in payload["students"] if item["username"] == "progress_student")
        self.assertEqual(row["xp"], 340)
        self.assertEqual(row["level"], 2)
        self.assertEqual(row["current_streak"], 5)
        self.assertEqual(row["accuracy"], 80.0)

    # --- Native signs section -------------------------------------------------------

    def test_native_stats_reflect_sessions_and_mastery(self) -> None:
        token = self.register("native_learner")
        me = self.client.get("/auth/me", headers=self.auth(token)).json()
        with SessionLocal() as db:
            sign = NativeSign(
                gloss="HELLO", display_name="Hello", meaning="A greeting.", category="Greetings",
                difficulty="Easy", description="d", example_text=None,
                dataset_available=True, model_available=True, active=True,
            )
            db.add(sign)
            db.flush()
            db.add(
                NativeSignProgress(
                    user_id=me["id"], native_sign_id=sign.id, attempts=4, correct_attempts=3,
                    accuracy=75.0, mastery=75.0, last_practiced=datetime.now(timezone.utc),
                )
            )
            for _ in range(4):
                db.add(
                    NativeSignSession(
                        user_id=me["id"], native_sign_id=sign.id, model_version="native_i3d_v1",
                        started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
                        result="correct", confidence=0.9, response_time=500,
                    )
                )
            db.commit()

        admin = self.admin_token()
        payload = self.client.get("/admin/mentor-dashboard", headers=self.auth(admin)).json()
        self.assertEqual(payload["overview"]["total_native_sign_attempts"], 4)
        self.assertEqual(payload["native"]["total_attempts"], 4)
        sign_row = next(item for item in payload["native"]["per_sign"] if item["gloss"] == "HELLO")
        self.assertEqual(sign_row["attempts"], 4)
        self.assertEqual(sign_row["mastery"], 75.0)
        category = next(item for item in payload["native"]["category_distribution"] if item["category"] == "Greetings")
        self.assertEqual(category["attempts"], 4)

        student_row = next(item for item in payload["students"] if item["username"] == "native_learner")
        self.assertEqual(student_row["native_mastery"], 75.0)

    # --- Chatbot section --------------------------------------------------------------

    def test_chatbot_stats_reflect_conversations_and_feedback(self) -> None:
        token = self.register("chatbot_user")
        me = self.client.get("/auth/me", headers=self.auth(token)).json()
        with SessionLocal() as db:
            conv = ChatbotConversation(user_id=me["id"], title="How is XP awarded?")
            db.add(conv)
            db.flush()
            msg1 = ChatbotMessage(conversation_id=conv.id, role="user", content="How is XP awarded?")
            msg2 = ChatbotMessage(conversation_id=conv.id, role="assistant", content="Server-side, per attempt.")
            db.add_all([msg1, msg2])
            db.flush()
            db.add(ChatbotFeedback(message_id=msg2.id, rating=1))
            db.add(ChatbotFeedback(message_id=msg2.id, rating=0))
            db.commit()

        admin = self.admin_token()
        payload = self.client.get("/admin/mentor-dashboard", headers=self.auth(admin)).json()
        self.assertEqual(payload["chatbot"]["total_conversations"], 1)
        self.assertEqual(payload["chatbot"]["total_messages"], 2)
        self.assertEqual(payload["chatbot"]["positive_feedback"], 1)
        self.assertEqual(payload["chatbot"]["negative_feedback"], 1)

    # --- Regression: refactored admin endpoints still work ------------------------------

    def test_admin_overview_endpoint_still_works(self) -> None:
        token = self.admin_token()
        response = self.client.get("/admin/overview", headers=self.auth(token))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("total_users", response.json())

    def test_admin_letters_endpoint_still_works(self) -> None:
        token = self.admin_token()
        response = self.client.get("/admin/letters", headers=self.auth(token))
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("popular", payload)
        self.assertIn("difficult", payload)
        self.assertEqual(len(payload["popular"]), 10)


if __name__ == "__main__":
    unittest.main()
