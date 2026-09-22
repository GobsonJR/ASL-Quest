from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["ASL_QUEST_DATABASE_URL"] = f"sqlite:///{TEST_DB.name}"
os.environ["ASL_QUEST_SECRET_KEY"] = "analytics-test-secret"

from backend.database import Base, SessionLocal, engine, init_db  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import PracticeSession, User, XpEvent  # noqa: E402
from backend.services.analytics import (  # noqa: E402
    get_accuracy_trend,
    get_daily_activity,
    get_heatmap,
    get_learning_funnel,
    get_overview,
    get_practice_history,
    get_response_time_analytics,
)
from backend.services.progress import seed_achievements  # noqa: E402


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Base.metadata.drop_all(bind=engine)
        init_db()
        with SessionLocal() as db:
            seed_achievements(db)
        cls.client = TestClient(app)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        init_db()
        with SessionLocal() as db:
            seed_achievements(db)

    def register(self, username: str) -> tuple[str, int]:
        response = self.client.post(
            "/auth/register",
            json={"username": username, "email": f"{username}@example.com", "password": "Password123!"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        token = response.json()["access_token"]
        me = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        return token, me.json()["id"]

    def seed_practice(self, user_id: int) -> None:
        now = datetime.now(timezone.utc)
        with SessionLocal() as db:
            db.add(
                PracticeSession(
                    user_id=user_id,
                    letter="A",
                    prediction="A",
                    correct=True,
                    response_time=2400,
                    xp_earned=20,
                    created_at=now,
                )
            )
            db.add(
                PracticeSession(
                    user_id=user_id,
                    letter="J",
                    prediction="P",
                    correct=False,
                    response_time=4100,
                    xp_earned=0,
                    created_at=now - timedelta(days=1),
                )
            )
            db.add(XpEvent(user_id=user_id, amount=20, reason="correct_sign", created_at=now))
            db.commit()

    def test_overview_statistics(self) -> None:
        token, user_id = self.register("analytics_a")
        self.seed_practice(user_id)
        with SessionLocal() as db:
            user = db.get(User, user_id)
            overview = get_overview(db, user, "all")
        self.assertEqual(overview["total_attempts"], 2)
        self.assertEqual(overview["total_correct"], 1)
        self.assertEqual(overview["overall_accuracy"], 50.0)

    def test_daily_activity_aggregation(self) -> None:
        token, user_id = self.register("analytics_b")
        self.seed_practice(user_id)
        with SessionLocal() as db:
            daily = get_daily_activity(db, user_id)
        self.assertEqual(len(daily), 2)
        self.assertEqual(sum(item["attempts"] for item in daily), 2)

    def test_accuracy_calculation(self) -> None:
        token, user_id = self.register("analytics_c")
        self.seed_practice(user_id)
        with SessionLocal() as db:
            trend = get_accuracy_trend(db, user_id, "30d")
        self.assertTrue(trend["has_data"])

    def test_heatmap_aggregation(self) -> None:
        token, user_id = self.register("analytics_d")
        self.seed_practice(user_id)
        with SessionLocal() as db:
            heatmap = get_heatmap(db, user_id, "3m")
        self.assertGreater(len(heatmap["cells"]), 0)

    def test_letter_statistics(self) -> None:
        token, user_id = self.register("analytics_e")
        self.seed_practice(user_id)
        response = self.client.get("/analytics/letters", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        letters = {item["letter"]: item for item in response.json()}
        self.assertEqual(letters["A"]["correct"], 1)
        self.assertFalse(letters["B"]["practiced"])

    def test_mastery_funnel(self) -> None:
        token, user_id = self.register("analytics_f")
        self.seed_practice(user_id)
        with SessionLocal() as db:
            funnel = get_learning_funnel(db, user_id)
        self.assertEqual(funnel["total_letters"], 26)

    def test_response_time_analytics(self) -> None:
        token, user_id = self.register("analytics_g")
        self.seed_practice(user_id)
        with SessionLocal() as db:
            stats = get_response_time_analytics(db, user_id, "all")
        self.assertTrue(stats["has_data"])
        self.assertEqual(stats["average_sec"], 2.4)

    def test_practice_history_pagination(self) -> None:
        token, user_id = self.register("analytics_h")
        self.seed_practice(user_id)
        with SessionLocal() as db:
            page = get_practice_history(db, user_id, page=1, page_size=1)
        self.assertEqual(page["total"], 2)
        self.assertEqual(len(page["items"]), 1)

    def test_user_isolation(self) -> None:
        token_a, user_a = self.register("analytics_i")
        token_b, user_b = self.register("analytics_j")
        self.seed_practice(user_a)
        response = self.client.get("/analytics/overview", headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_attempts"], 0)

    def test_empty_user(self) -> None:
        token, _ = self.register("analytics_k")
        response = self.client.get("/analytics/dashboard", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["overview"]["has_data"])

    def test_dashboard_endpoint(self) -> None:
        token, user_id = self.register("analytics_l")
        self.seed_practice(user_id)
        response = self.client.get("/analytics/dashboard?range=30d", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("heatmap", payload)
        self.assertIn("weekly_summary", payload)


if __name__ == "__main__":
    unittest.main()
