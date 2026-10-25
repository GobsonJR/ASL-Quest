"""Activity heatmap: real data only, counted by the learner's local day, with consistent summaries."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["ASL_QUEST_DATABASE_URL"] = f"sqlite:///{TEST_DB.name}"
os.environ["ASL_QUEST_SECRET_KEY"] = "heatmap-test-secret"

from backend import models  # noqa: E402,F401
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import NativeSign, NativeSignSession, PracticeSession, User, WordPracticeSession  # noqa: E402
from backend.services import analytics  # noqa: E402

# "Now" for every test: Sunday 25 Oct 2026, 12:00 UTC (17:30 in India, 08:00 in New York).
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


class HeatmapTests(unittest.TestCase):
    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        patcher = patch.object(analytics, "utcnow", return_value=NOW)
        patcher.start()
        self.addCleanup(patcher.stop)
        with SessionLocal() as db:
            for name in ("ann", "bob"):
                db.add(User(username=name, email=f"{name}@example.com", password_hash="x"))
            db.add(NativeSign(gloss="HELLO", display_name="Hello", meaning="m", category="c", difficulty="Easy", description="d",
                              example_text="e", dataset_available=True, model_available=False, active=True))
            db.commit()
            self.ann = db.query(User).filter(User.username == "ann").one().id
            self.bob = db.query(User).filter(User.username == "bob").one().id
            self.sign = db.query(NativeSign).one().id

    def letter(self, user: int, when: datetime, correct: bool = True, xp: int = 20) -> None:
        with SessionLocal() as db:
            db.add(PracticeSession(user_id=user, letter="A", correct=correct, xp_earned=xp, created_at=when))
            db.commit()

    def word(self, user: int, when: datetime) -> None:
        with SessionLocal() as db:
            db.add(WordPracticeSession(user_id=user, word_id="beginner-cat", completed=True, created_at=when))
            db.commit()

    def native(self, user: int, when: datetime) -> None:
        with SessionLocal() as db:
            db.add(NativeSignSession(user_id=user, native_sign_id=self.sign, result="correct", created_at=when))
            db.commit()

    def heatmap(self, user: int | None = None, range_key: str = "3m", tz: str | None = None) -> dict:
        with SessionLocal() as db:
            return analytics.get_heatmap(db, user or self.ann, range_key, tz)  # type: ignore[arg-type]

    @staticmethod
    def cell(data: dict, day: str) -> dict:
        return next(c for c in data["cells"] if c["date"] == day)

    # ---------------------------------------------------------------- empty and little history
    def test_no_history_gives_an_honest_empty_grid(self) -> None:
        data = self.heatmap()
        self.assertGreaterEqual(len(data["cells"]), 90)
        self.assertEqual(data["end"], "2026-10-25")
        self.assertEqual(sum(c["count"] for c in data["cells"]), 0)
        self.assertEqual({c["intensity"] for c in data["cells"]}, {0})
        self.assertEqual(data["summary"], {"total_activity": 0, "active_days": 0, "current_streak": 0, "longest_streak": 0,
                                           "letter_accuracy": None, "letter_attempts": 0})

    def test_one_day_of_activity_leaves_every_other_day_at_zero_nothing_is_invented(self) -> None:
        self.letter(self.ann, utc(2026, 10, 20, 9, 0))
        data = self.heatmap()
        self.assertEqual([c["date"] for c in data["cells"] if c["count"]], ["2026-10-20"])
        self.assertEqual(data["summary"]["active_days"], 1)

    def test_cells_are_one_per_day_in_order_with_no_gaps(self) -> None:
        dates = [c["date"] for c in self.heatmap(range_key="6m")["cells"]]
        parsed = [datetime.fromisoformat(d).date() for d in dates]
        self.assertEqual(parsed, sorted(set(parsed)))
        self.assertTrue(all((b - a).days == 1 for a, b in zip(parsed, parsed[1:])))

    # ---------------------------------------------------------------- what counts as activity
    def test_letters_words_and_native_attempts_all_count_and_other_users_do_not(self) -> None:
        day = utc(2026, 10, 22, 10, 0)
        self.letter(self.ann, day, correct=True)
        self.letter(self.ann, day + timedelta(minutes=1), correct=False)
        self.word(self.ann, day)
        self.native(self.ann, day)
        self.letter(self.bob, day)  # someone else's practice
        cell = self.cell(self.heatmap(), "2026-10-22")
        self.assertEqual((cell["letters"], cell["words"], cell["native"], cell["count"]), (2, 1, 1, 4))
        # the original A-Z fields keep their meaning for existing consumers
        self.assertEqual((cell["attempts"], cell["correct"], cell["accuracy"], cell["xp"]), (2, 1, 50.0, 40))
        self.assertEqual(self.cell(self.heatmap(self.bob), "2026-10-22")["count"], 1)

    def test_totals_match_the_database_rows_in_range(self) -> None:
        for offset in range(0, 12):
            self.letter(self.ann, utc(2026, 10, 25 - offset, 8, 0))
        self.word(self.ann, utc(2026, 10, 3, 8, 0))
        self.native(self.ann, utc(2026, 10, 4, 8, 0))
        self.letter(self.ann, utc(2025, 1, 1, 8, 0))  # far outside the 3-month window
        data = self.heatmap()
        with SessionLocal() as db:
            in_range = (
                db.query(PracticeSession).filter(PracticeSession.user_id == self.ann, PracticeSession.created_at >= utc(2026, 7, 1)).count()
                + db.query(WordPracticeSession).filter(WordPracticeSession.user_id == self.ann).count()
                + db.query(NativeSignSession).filter(NativeSignSession.user_id == self.ann).count()
            )
        self.assertEqual(data["summary"]["total_activity"], in_range)
        self.assertEqual(data["summary"]["letter_attempts"], 12)

    def test_intensity_levels(self) -> None:
        for n, want in ((0, 0), (1, 1), (4, 1), (5, 2), (9, 2), (10, 3), (19, 3), (20, 4), (80, 4)):
            self.assertEqual(analytics.heatmap_intensity(n), want, n)

    # ---------------------------------------------------------------- date and timezone boundaries
    def test_days_follow_the_learners_timezone_not_utc(self) -> None:
        late_utc = utc(2026, 10, 24, 20, 0)  # 24 Oct 20:00 UTC = 25 Oct 01:30 in India, 24 Oct 16:00 in New York
        self.letter(self.ann, late_utc)
        self.assertEqual(self.cell(self.heatmap(tz="UTC"), "2026-10-24")["count"], 1)
        self.assertEqual(self.cell(self.heatmap(tz="Asia/Kolkata"), "2026-10-25")["count"], 1)
        self.assertEqual(self.cell(self.heatmap(tz="Asia/Kolkata"), "2026-10-24")["count"], 0)
        self.assertEqual(self.cell(self.heatmap(tz="America/New_York"), "2026-10-24")["count"], 1)

    def test_the_last_second_and_first_second_of_a_local_day(self) -> None:
        # India is UTC+5:30: local midnight of 21 Oct is 20 Oct 18:30 UTC
        self.letter(self.ann, utc(2026, 10, 20, 18, 29, 59))  # 20 Oct 23:59:59 local
        self.letter(self.ann, utc(2026, 10, 20, 18, 30, 0))  # 21 Oct 00:00:00 local
        data = self.heatmap(tz="Asia/Kolkata")
        self.assertEqual(self.cell(data, "2026-10-20")["count"], 1)
        self.assertEqual(self.cell(data, "2026-10-21")["count"], 1)

    def test_the_grid_ends_on_the_learners_today_even_when_utc_is_still_yesterday(self) -> None:
        with patch.object(analytics, "utcnow", return_value=utc(2026, 10, 24, 22, 0)):  # already 25 Oct 03:30 in India
            self.assertEqual(self.heatmap(tz="Asia/Kolkata")["end"], "2026-10-25")
            self.assertEqual(self.heatmap(tz="UTC")["end"], "2026-10-24")

    def test_naive_datetimes_from_sqlite_are_treated_as_utc(self) -> None:
        with SessionLocal() as db:
            db.add(PracticeSession(user_id=self.ann, letter="B", correct=True, xp_earned=0, created_at=datetime(2026, 10, 24, 20, 0)))
            db.commit()
        self.assertEqual(self.cell(self.heatmap(tz="Asia/Kolkata"), "2026-10-25")["count"], 1)

    def test_unknown_timezone_is_rejected(self) -> None:
        with self.assertRaises(analytics.UnknownTimezone):
            self.heatmap(tz="Mars/Olympus")

    # ---------------------------------------------------------------- streaks come from the same cells as the grid
    def test_streaks(self) -> None:
        for day in (25, 24, 23, 21, 20, 19, 18):
            self.letter(self.ann, utc(2026, 10, day, 9, 0))
        summary = self.heatmap()["summary"]
        self.assertEqual((summary["current_streak"], summary["longest_streak"], summary["active_days"]), (3, 4, 7))

    def test_a_streak_stays_alive_until_the_end_of_today(self) -> None:
        for day in (24, 23, 22):  # nothing yet today (25 Oct)
            self.letter(self.ann, utc(2026, 10, day, 9, 0))
        self.assertEqual(self.heatmap()["summary"]["current_streak"], 3)
        self.letter(self.ann, utc(2026, 10, 20, 9, 0))
        self.assertEqual(self.heatmap()["summary"]["current_streak"], 3)

    def test_a_missed_day_breaks_the_streak(self) -> None:
        self.letter(self.ann, utc(2026, 10, 22, 9, 0))
        self.assertEqual(self.heatmap()["summary"]["current_streak"], 0)


class HeatmapEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    def token(self) -> str:
        reply = self.client.post("/auth/register", json={"username": "cara", "email": "cara@example.com", "password": "correct-horse-1"})
        return reply.json()["access_token"]

    def test_requires_authentication(self) -> None:
        self.assertEqual(self.client.get("/analytics/heatmap").status_code, 401)

    def test_returns_the_authenticated_users_grid_and_validates_inputs(self) -> None:
        headers = {"Authorization": f"Bearer {self.token()}"}
        ok = self.client.get("/analytics/heatmap", params={"range": "3m", "tz": "Asia/Kolkata"}, headers=headers)
        self.assertEqual(ok.status_code, 200, ok.text)
        body = ok.json()
        self.assertEqual(body["timezone"], "Asia/Kolkata")
        self.assertEqual(body["summary"]["total_activity"], 0)
        self.assertEqual(self.client.get("/analytics/heatmap", params={"tz": "Mars/Olympus"}, headers=headers).status_code, 422)
        self.assertEqual(self.client.get("/analytics/heatmap", params={"range": "9y"}, headers=headers).status_code, 422)


if __name__ == "__main__":
    unittest.main()
