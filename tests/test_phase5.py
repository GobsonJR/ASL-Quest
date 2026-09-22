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
os.environ["ASL_QUEST_SECRET_KEY"] = "test-secret-key-phase5"
os.environ["ASL_QUEST_BOOTSTRAP_ADMIN_EMAIL"] = "admin@example.com"

from backend.database import Base, SessionLocal, engine, init_db  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import PracticeSession, XpEvent  # noqa: E402
from backend.services.progress import seed_achievements  # noqa: E402
from backend.services.words import XP_WORD_CHALLENGE, XP_WORD_COMPLETED, daily_word_ids  # noqa: E402
from backend.words.catalog import WORD_CATALOG, validate_catalog  # noqa: E402


class Phase5WordTests(unittest.TestCase):
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

    def test_catalog_is_a_to_z_only(self) -> None:
        validate_catalog()
        self.assertGreaterEqual(len(WORD_CATALOG), 20)
        for item in WORD_CATALOG:
            self.assertTrue(item["word"].isalpha())
            self.assertEqual(item["word"], item["word"].upper())
            self.assertEqual(item["letters"], list(item["word"]))

    def test_catalog_includes_core_demo_words(self) -> None:
        words = {item["word"] for item in WORD_CATALOG}
        for word in ("CAT", "DOG", "MOM", "DAD", "BOOK", "HOME", "SCHOOL", "HELLO", "THANKS", "FRIEND"):
            self.assertIn(word, words)

    def test_words_require_auth(self) -> None:
        self.assertEqual(self.client.get("/words").status_code, 401)
        self.assertEqual(self.client.get("/words/progress").status_code, 401)
        self.assertEqual(self.client.post("/words/practice/session", json={"word_id": "beginner-cat"}).status_code, 401)

    def test_list_and_get_word(self) -> None:
        token = self.register_user("word_list", "word_list@example.com")
        listing = self.client.get("/words", headers=self.auth_headers(token))
        self.assertEqual(listing.status_code, 200)
        items = listing.json()["items"]
        self.assertTrue(any(item["id"] == "beginner-cat" for item in items))
        detail = self.client.get("/words/beginner-cat", headers=self.auth_headers(token))
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["word"], "CAT")
        self.assertEqual(detail.json()["letters"], ["C", "A", "T"])
        school = self.client.get("/words/school-school", headers=self.auth_headers(token))
        self.assertEqual(school.status_code, 200)
        self.assertEqual(school.json()["word"], "SCHOOL")

    def test_unknown_word_404(self) -> None:
        token = self.register_user("missing_word", "missing_word@example.com")
        response = self.client.get("/words/not-a-word", headers=self.auth_headers(token))
        self.assertEqual(response.status_code, 404)

    def test_word_session_completion_and_xp(self) -> None:
        token = self.register_user("word_xp", "word_xp@example.com")
        headers = self.auth_headers(token)
        before = self.client.get("/progress", headers=headers).json()["xp"]
        response = self.client.post(
            "/words/practice/session",
            headers=headers,
            json={
                "word_id": "beginner-cat",
                "completed": True,
                "duration_ms": 20000,
                "mistakes": 1,
                "letters": [
                    {"letter": "C", "prediction": "C", "correct": True, "response_time": 1200},
                    {"letter": "A", "prediction": "A", "correct": True, "response_time": 1400},
                    {"letter": "T", "prediction": "T", "correct": True, "response_time": 1100},
                ],
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        payload = response.json()
        self.assertTrue(payload["completed"])
        self.assertEqual(payload["xp_earned"], XP_WORD_COMPLETED)
        self.assertEqual(payload["progress"]["completions"], 1)
        reasons = [event["reason"] for event in payload["xp_events"]]
        self.assertIn("word_completed", reasons)
        self.assertNotIn("word_letter_correct", reasons)
        after = self.client.get("/progress", headers=headers).json()["xp"]
        self.assertEqual(after, before + XP_WORD_COMPLETED)

        with SessionLocal() as db:
            user_events = db.query(XpEvent).filter(XpEvent.reason == "word_completed").all()
            self.assertTrue(any(event.amount == XP_WORD_COMPLETED for event in user_events))
            letter_rows = db.query(PracticeSession).all()
            self.assertEqual(len(letter_rows), 0)

    def test_incomplete_word_session_awards_no_xp(self) -> None:
        token = self.register_user("word_incomplete", "word_incomplete@example.com")
        headers = self.auth_headers(token)
        before = self.client.get("/progress", headers=headers).json()["xp"]
        response = self.client.post(
            "/words/practice/session",
            headers=headers,
            json={
                "word_id": "beginner-cat",
                "completed": False,
                "resume_index": 1,
                "letters": [{"letter": "C", "prediction": "C", "correct": True}],
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["xp_earned"], 0)
        after = self.client.get("/progress", headers=headers).json()["xp"]
        self.assertEqual(after, before)
        progress = self.client.get("/words/beginner-cat/progress", headers=headers).json()
        self.assertEqual(progress["completions"], 0)
        self.assertEqual(progress["resume_index"], 1)

    def test_perfect_and_challenge_xp_reasons(self) -> None:
        token = self.register_user("word_perfect", "word_perfect@example.com")
        response = self.client.post(
            "/words/practice/session",
            headers=self.auth_headers(token),
            json={
                "word_id": "beginner-dog",
                "completed": True,
                "duration_ms": 3000,
                "mistakes": 0,
                "challenge_type": "word",
                "challenge_finished": True,
                "letters": [
                    {"letter": "D", "prediction": "D", "correct": True},
                    {"letter": "O", "prediction": "O", "correct": True},
                    {"letter": "G", "prediction": "G", "correct": True},
                ],
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        reasons = {event["reason"]: event["amount"] for event in response.json()["xp_events"]}
        self.assertIn("word_completed", reasons)
        self.assertIn("word_perfect", reasons)
        self.assertIn("word_fast", reasons)
        self.assertEqual(reasons["word_challenge_completed"], XP_WORD_CHALLENGE)

    def test_duplicate_completion_is_separate_attempt_not_letter_xp(self) -> None:
        token = self.register_user("word_dup", "word_dup@example.com")
        headers = self.auth_headers(token)
        body = {
            "word_id": "beginner-sun",
            "completed": True,
            "duration_ms": 9000,
            "mistakes": 0,
            "letters": [
                {"letter": "S", "prediction": "S", "correct": True},
                {"letter": "U", "prediction": "U", "correct": True},
                {"letter": "N", "prediction": "N", "correct": True},
            ],
        }
        first = self.client.post("/words/practice/session", headers=headers, json=body)
        second = self.client.post("/words/practice/session", headers=headers, json=body)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 201)
        progress = self.client.get("/words/beginner-sun/progress", headers=headers).json()
        self.assertEqual(progress["completions"], 2)
        self.assertEqual(progress["attempts"], 2)

    def test_user_isolation(self) -> None:
        token_a = self.register_user("word_iso_a", "word_iso_a@example.com")
        token_b = self.register_user("word_iso_b", "word_iso_b@example.com")
        self.client.post(
            "/words/practice/session",
            headers=self.auth_headers(token_a),
            json={
                "word_id": "beginner-hat",
                "completed": True,
                "letters": [
                    {"letter": "H", "prediction": "H", "correct": True},
                    {"letter": "A", "prediction": "A", "correct": True},
                    {"letter": "T", "prediction": "T", "correct": True},
                ],
            },
        )
        progress_b = self.client.get("/words/progress", headers=self.auth_headers(token_b)).json()["items"]
        self.assertFalse(any(item["word_id"] == "beginner-hat" and item["completions"] > 0 for item in progress_b))
        progress_a = self.client.get("/words/beginner-hat/progress", headers=self.auth_headers(token_a)).json()
        self.assertEqual(progress_a["completions"], 1)

    def test_recommendations_deterministic(self) -> None:
        token = self.register_user("word_reco", "word_reco@example.com")
        first = self.client.get("/words/recommendations", headers=self.auth_headers(token)).json()
        second = self.client.get("/words/recommendations", headers=self.auth_headers(token)).json()
        self.assertEqual(first["recommended"], second["recommended"])
        self.assertTrue(first["recommended"])
        self.assertIn("message", first["recommended"][0])

    def test_word_analytics_from_database(self) -> None:
        token = self.register_user("word_an", "word_an@example.com")
        headers = self.auth_headers(token)
        self.client.post(
            "/words/practice/session",
            headers=headers,
            json={
                "word_id": "everyday-home",
                "completed": True,
                "letters": [
                    {"letter": "H", "prediction": "H", "correct": True},
                    {"letter": "O", "prediction": "O", "correct": True},
                    {"letter": "M", "prediction": "M", "correct": True},
                    {"letter": "E", "prediction": "E", "correct": True},
                ],
            },
        )
        analytics = self.client.get("/words/analytics", headers=headers).json()
        self.assertEqual(analytics["words_learned"], 1)
        self.assertEqual(analytics["words_completed"], 1)
        self.assertTrue(analytics["has_data"])
        dashboard = self.client.get("/analytics/dashboard", headers=headers).json()
        self.assertIn("words", dashboard)
        self.assertEqual(dashboard["words"]["words_completed"], 1)

    def test_word_challenge_and_daily_words(self) -> None:
        token = self.register_user("word_ch", "word_ch@example.com")
        headers = self.auth_headers(token)
        challenge = self.client.get("/words/challenge?mode=word", headers=headers).json()
        self.assertEqual(len(challenge["items"]), 5)
        daily = self.client.get("/words/challenge?mode=daily", headers=headers).json()
        self.assertEqual(len(daily["items"]), 3)
        again = self.client.get("/words/challenge?mode=daily", headers=headers).json()
        self.assertEqual([item["id"] for item in daily["items"]], [item["id"] for item in again["items"]])

    def test_word_starter_achievement(self) -> None:
        token = self.register_user("word_badge", "word_badge@example.com")
        headers = self.auth_headers(token)
        response = self.client.post(
            "/words/practice/session",
            headers=headers,
            json={
                "word_id": "beginner-bed",
                "completed": True,
                "letters": [
                    {"letter": "B", "prediction": "B", "correct": True},
                    {"letter": "E", "prediction": "E", "correct": True},
                    {"letter": "D", "prediction": "D", "correct": True},
                ],
            },
        )
        names = [item["name"] for item in response.json()["new_achievements"]]
        self.assertIn("Word Starter", names)
        self.assertNotIn("word_starter", names)
        achievements = self.client.get("/achievements", headers=headers).json()
        starter = next(item for item in achievements if item["id"] == "word_starter")
        self.assertTrue(starter["earned"])
        self.assertEqual(starter["name"], "Word Starter")
        again = self.client.post(
            "/words/practice/session",
            headers=headers,
            json={
                "word_id": "beginner-sad",
                "completed": True,
                "letters": [
                    {"letter": "S", "prediction": "S", "correct": True},
                    {"letter": "A", "prediction": "A", "correct": True},
                    {"letter": "D", "prediction": "D", "correct": True},
                ],
            },
        )
        again_names = [item["id"] for item in again.json()["new_achievements"]]
        self.assertNotIn("word_starter", again_names)

    def test_export_includes_word_data_without_secrets(self) -> None:
        token = self.register_user("word_export", "word_export@example.com")
        headers = self.auth_headers(token)
        self.client.post(
            "/words/practice/session",
            headers=headers,
            json={"word_id": "greetings-hi", "completed": True, "letters": [{"letter": "H", "correct": True}, {"letter": "I", "correct": True}]},
        )
        export = self.client.get("/users/me/export", headers=headers).json()
        self.assertIn("word_progress", export)
        self.assertIn("word_practice_sessions", export)
        dumped = str(export).lower()
        self.assertNotIn("password_hash", dumped)
        self.assertNotIn("access_token", dumped)

    def test_admin_word_aggregates(self) -> None:
        admin_response = self.client.post(
            "/auth/register",
            json={"username": "admin_words", "email": "admin@example.com", "password": "Password123!"},
        )
        if admin_response.status_code == 409:
            login = self.client.post("/auth/login", json={"login": "admin@example.com", "password": "Password123!"})
            self.assertEqual(login.status_code, 200)
            admin = login.json()["access_token"]
        else:
            self.assertEqual(admin_response.status_code, 201, admin_response.text)
            admin = admin_response.json()["access_token"]
        student = self.register_user("student_words", "student_words@example.com")
        self.client.post(
            "/words/practice/session",
            headers=self.auth_headers(student),
            json={
                "word_id": "greetings-hello",
                "completed": True,
                "letters": [{"letter": letter, "correct": True} for letter in "HELLO"],
            },
        )
        forbidden = self.client.get("/admin/words", headers=self.auth_headers(student))
        self.assertEqual(forbidden.status_code, 403)
        overview = self.client.get("/admin/overview", headers=self.auth_headers(admin))
        self.assertEqual(overview.status_code, 200)
        self.assertIn("total_word_practice_sessions", overview.json())
        words = self.client.get("/admin/words", headers=self.auth_headers(admin)).json()
        self.assertIn("most_practiced", words)
        self.assertFalse(any("@" in str(item) for item in words["most_practiced"]))

    def test_daily_words_are_user_deterministic(self) -> None:
        ids_one = daily_word_ids(42, "2026-09-14")
        ids_two = daily_word_ids(42, "2026-09-14")
        ids_other_user = daily_word_ids(43, "2026-09-14")
        self.assertEqual(ids_one, ids_two)
        self.assertEqual(len(ids_one), 3)
        self.assertNotEqual(ids_one, ids_other_user)


if __name__ == "__main__":
    unittest.main()
