"""Regression tests for the XP-overwrite race between PUT /progress (the
client's full-GameState push, used by A-Z) and Native / Word Spelling's
direct, atomic server-side XP increments (backend/routers/native.py,
backend/services/words.py::record_word_session).

Root cause: backend/services/progress.py::persist_game_state used to do
`progress.xp = state.xp` unconditionally. A client holding a GameState
snapshot from *before* a Native/Word XP gain (most realistically: another
browser tab/session on the same account) would then silently erase that
already-earned, already-persisted XP on its next sync. See the Native
Integration Regression Audit for the original live reproduction against a
running server.

Independent of the existing app test suite: uses its own temporary SQLite
file and creates every table via Base.metadata.create_all (matching
test_native.py's pattern, since this needs both Native and Word tables).
Nothing here touches data/asl_quest.db. Native inference is mocked
(predict_video is patched); Word Spelling's catalog is static data, no
seeding needed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["ASL_QUEST_DATABASE_URL"] = f"sqlite:///{TEST_DB.name}"
os.environ["ASL_QUEST_SECRET_KEY"] = "progress-sync-test-secret"

from backend import models  # noqa: E402,F401  (registers all ORM tables on Base.metadata)
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import NativeSign  # noqa: E402
from backend.services.progress import seed_achievements  # noqa: E402
from backend.services.progress import level_from_xp  # noqa: E402


def _seed_one_native_sign() -> int:
    with SessionLocal() as db:
        sign = NativeSign(
            gloss="HELLO",
            display_name="Hello",
            meaning="A greeting.",
            category="Greetings",
            difficulty="Easy",
            description="The sign for HELLO.",
            example_text="Hello there.",
            dataset_available=True,
            model_available=True,
            active=True,
        )
        db.add(sign)
        db.commit()
        db.refresh(sign)
        return sign.id


class ProgressSyncRaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        with SessionLocal() as db:
            seed_achievements(db)
        self.hello_id = _seed_one_native_sign()

    def register(self, username: str) -> str:
        response = self.client.post(
            "/auth/register",
            json={"username": username, "email": f"{username}@example.com", "password": "Password123!"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["access_token"]

    def auth(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def get_xp(self, token: str) -> int:
        return self.client.get("/progress", headers=self.auth(token)).json()["xp"]

    def native_correct(self, token: str, confidence: float = 0.9):
        with patch(
            "backend.routers.native.predict_video",
            return_value={
                "prediction": "HELLO",
                "confidence": confidence,
                "top_k": [{"gloss": "HELLO", "confidence": confidence}],
                "video": "irrelevant.mp4",
                "num_classes": 1,
            },
        ):
            return self.client.post(
                f"/native/predict?native_sign_id={self.hello_id}",
                headers=self.auth(token),
                files={"file": ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")},
            )

    def word_complete(self, token: str, word_id: str = "beginner-cat"):
        return self.client.post(
            "/words/practice/session",
            headers=self.auth(token),
            json={
                "word_id": word_id,
                "completed": True,
                "duration_ms": 20000,
                "mistakes": 0,
                "letters": [
                    {"letter": letter, "prediction": letter, "correct": True, "response_time": 1000}
                    for letter in {"beginner-cat": "CAT", "beginner-dog": "DOG"}[word_id]
                ],
            },
        )

    # --- The exact scenario from the audit -------------------------------------

    def test_stale_progress_put_does_not_erase_native_xp(self) -> None:
        token = self.register("race_user")
        self.assertEqual(self.get_xp(token), 0)

        # Native correct attempt #1: server XP 0 -> 20.
        first = self.native_correct(token)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["xp_earned"], 20)
        self.assertEqual(self.get_xp(token), 20)

        # Client captures a snapshot right here -- this is what a second
        # tab/session, or just a slow client, would still be holding.
        stale_snapshot = self.client.get("/progress", headers=self.auth(token)).json()
        self.assertEqual(stale_snapshot["xp"], 20)

        # Native correct attempt #2: server XP 20 -> 40, independent of the
        # client above, which never sees this happen.
        second = self.native_correct(token)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["xp_earned"], 20)
        self.assertEqual(self.get_xp(token), 40)

        # The stale client now syncs its full (outdated) state -- exactly
        # what scheduleProgressSync sends after *any* local change, even one
        # that never touched xp (e.g. a single wrong A-Z attempt).
        put_response = self.client.put("/progress", headers=self.auth(token), json=stale_snapshot)
        self.assertEqual(put_response.status_code, 200, put_response.text)

        # The 20 XP from native attempt #2 must survive.
        self.assertEqual(put_response.json()["xp"], 40)
        self.assertEqual(self.get_xp(token), 40)

    def test_level_stays_consistent_with_final_xp_after_the_race(self) -> None:
        token = self.register("race_user_level")
        stale_snapshot = self.client.get("/progress", headers=self.auth(token)).json()
        for _ in range(60):  # 60 * 20 = 1200 xp, comfortably past one level boundary (1000)
            self.native_correct(token)
        final = self.client.put("/progress", headers=self.auth(token), json=stale_snapshot).json()
        self.assertEqual(final["xp"], 1200)
        server_state = self.client.get("/progress", headers=self.auth(token)).json()
        self.assertEqual(server_state["xp"], 1200)
        # level isn't part of GameStateSchema, so check it via the level_from_xp
        # helper directly against the persisted value -- this is exactly what
        # persist_game_state itself must have set progress.level to.
        self.assertEqual(level_from_xp(server_state["xp"]), level_from_xp(1200))

    def test_word_completion_xp_also_survives_a_stale_progress_put(self) -> None:
        """Word Spelling shares the exact same direct-write pattern as Native
        (backend/services/words.py::record_word_session), so it needs the
        same protection."""
        token = self.register("race_user_word")
        stale_snapshot = self.client.get("/progress", headers=self.auth(token)).json()
        self.assertEqual(stale_snapshot["xp"], 0)

        response = self.word_complete(token, "beginner-cat")
        self.assertEqual(response.status_code, 201, response.text)
        xp_earned = response.json()["xp_earned"]
        self.assertGreater(xp_earned, 0)
        self.assertEqual(self.get_xp(token), xp_earned)

        put_response = self.client.put("/progress", headers=self.auth(token), json=stale_snapshot)
        self.assertEqual(put_response.json()["xp"], xp_earned)

    def test_native_and_word_xp_both_survive_the_same_stale_put(self) -> None:
        """Both independent server-side writers stack correctly, and neither
        is lost to a single stale client push captured before either ran."""
        token = self.register("race_user_both")
        stale_snapshot = self.client.get("/progress", headers=self.auth(token)).json()
        self.assertEqual(stale_snapshot["xp"], 0)

        native_earned = self.native_correct(token).json()["xp_earned"]
        word_earned = self.word_complete(token, "beginner-dog").json()["xp_earned"]
        expected_total = native_earned + word_earned
        self.assertEqual(self.get_xp(token), expected_total)

        put_response = self.client.put("/progress", headers=self.auth(token), json=stale_snapshot)
        self.assertEqual(put_response.json()["xp"], expected_total)
        self.assertEqual(self.get_xp(token), expected_total)

    # --- Legitimate, non-stale syncs must keep working exactly as before -------

    def test_legitimate_az_progress_sync_still_raises_xp_normally(self) -> None:
        """The common case (an A-Z client that IS ahead of the server, the
        normal way XP has always reached the server) must be unaffected."""
        token = self.register("az_user")
        state = self.client.get("/progress", headers=self.auth(token)).json()
        self.assertEqual(state["xp"], 0)

        state["xp"] = 120
        state["totalCorrect"] = 3
        state["totalAttempts"] = 4
        state["streak"] = 2
        state["letterStats"]["A"]["correct"] = 2
        state["letterStats"]["A"]["attempts"] = 3

        response = self.client.put("/progress", headers=self.auth(token), json=state)
        self.assertEqual(response.status_code, 200, response.text)
        updated = response.json()
        self.assertEqual(updated["xp"], 120)
        self.assertEqual(updated["totalCorrect"], 3)
        self.assertEqual(updated["totalAttempts"], 4)
        self.assertEqual(updated["streak"], 2)
        self.assertEqual(updated["letterStats"]["A"]["correct"], 2)
        self.assertEqual(self.get_xp(token), 120)

    def test_stale_xp_does_not_block_other_fields_in_the_same_put(self) -> None:
        """Only xp (and the level derived from it) is protected -- letterStats,
        totalCorrect/totalAttempts, streak and badges are untouched by this
        fix and must still apply exactly as the client sent them, even in
        the same request whose xp gets clamped."""
        token = self.register("race_user_fields")
        self.native_correct(token)  # server xp: 0 -> 20
        stale_snapshot = self.client.get("/progress", headers=self.auth(token)).json()
        second = self.native_correct(token)  # server xp: 20 -> 40, stale_snapshot still says 20
        self.assertEqual(second.json()["xp_earned"], 20)

        stale_snapshot["totalCorrect"] = 9
        stale_snapshot["totalAttempts"] = 11
        stale_snapshot["streak"] = 3
        stale_snapshot["letterStats"]["B"]["correct"] = 5
        stale_snapshot["letterStats"]["B"]["attempts"] = 6

        response = self.client.put("/progress", headers=self.auth(token), json=stale_snapshot)
        updated = response.json()
        self.assertEqual(updated["xp"], 40)  # protected: did not fall back to the stale 20
        self.assertEqual(updated["totalCorrect"], 9)  # unprotected fields still fully client-authoritative
        self.assertEqual(updated["totalAttempts"], 11)
        self.assertEqual(updated["streak"], 3)
        self.assertEqual(updated["letterStats"]["B"]["correct"], 5)


if __name__ == "__main__":
    unittest.main()
