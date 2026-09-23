"""Focused tests for backend/routers/native.py.

Independent of the existing app test suite: uses its own temporary SQLite
file and creates every table directly via Base.metadata.create_all (not just
the legacy ten init_db() creates), matching the pattern in
test_database_phase2.py. Nothing here touches data/asl_quest.db.

Inference itself is mocked (predict_video is patched) -- this suite does not
require GPU/model execution, matching the project's existing convention of
keeping ML-heavy work out of the fast backend test suite.
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
os.environ["ASL_QUEST_SECRET_KEY"] = "native-test-secret"

from backend import models  # noqa: E402,F401  (registers all ORM tables on Base.metadata)
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import NativeSign, NativeSignProgress, NativeSignSession, SignPrediction  # noqa: E402
from backend.services.progress import seed_achievements  # noqa: E402


def _seed_native_signs() -> None:
    """Minimal, self-contained seed (avoids depending on the manifest file
    scripts/seed_native_signs_native_10.py reads, so this test can't be
    broken by unrelated changes to data/asl_citizen_native_10/). Includes one
    inactive sign so inactive-handling can be tested consistently with
    GET /native/signs (which already filters to active-only)."""
    with SessionLocal() as db:
        for gloss in ("BOOK", "HELLO", "WATER"):
            db.add(
                NativeSign(
                    gloss=gloss,
                    display_name=gloss.title(),
                    meaning=f"Meaning of {gloss}.",
                    category="Everyday",
                    difficulty="Easy",
                    description=f"The sign for {gloss}.",
                    example_text=f"Example with {gloss}.",
                    dataset_available=True,
                    model_available=True,
                    active=True,
                )
            )
        db.add(
            NativeSign(
                gloss="RETIRED",
                display_name="Retired",
                meaning="An inactive test sign.",
                category="Everyday",
                difficulty="Easy",
                description="Not shown in the catalog.",
                example_text=None,
                dataset_available=True,
                model_available=False,
                active=False,
            )
        )
        db.commit()


def _sign_id(gloss: str) -> int:
    with SessionLocal() as db:
        sign = db.query(NativeSign).filter(NativeSign.gloss == gloss).one()
        return sign.id


class NativeRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)  # all tables, not just the legacy ten
        with SessionLocal() as db:
            seed_achievements(db)
        _seed_native_signs()
        cls.client = TestClient(app)

    def setUp(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        with SessionLocal() as db:
            seed_achievements(db)
        _seed_native_signs()

    def register(self, username: str) -> str:
        response = self.client.post(
            "/auth/register",
            json={"username": username, "email": f"{username}@example.com", "password": "Password123!"},
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["access_token"]

    # --- GET /native/signs ---------------------------------------------------

    def test_signs_requires_auth(self) -> None:
        response = self.client.get("/native/signs")
        self.assertEqual(response.status_code, 401)

    def test_signs_returns_seeded_catalog(self) -> None:
        token = self.register("native_signs_user")
        response = self.client.get("/native/signs", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        glosses = {item["gloss"] for item in payload["items"]}
        self.assertEqual(glosses, {"BOOK", "HELLO", "WATER"})
        item = next(item for item in payload["items"] if item["gloss"] == "BOOK")
        for field in (
            "id", "gloss", "display_name", "meaning", "category", "difficulty",
            "description", "example_text", "dataset_available", "model_available", "active",
        ):
            self.assertIn(field, item)
        self.assertTrue(item["active"])

    # --- POST /native/predict -------------------------------------------------

    def test_predict_requires_auth(self) -> None:
        response = self.client.post(
            "/native/predict",
            files={"file": ("clip.mp4", BytesIO(b"fake-bytes"), "video/mp4")},
        )
        self.assertEqual(response.status_code, 401)

    def test_predict_rejects_missing_upload(self) -> None:
        token = self.register("native_predict_missing")
        response = self.client.post("/native/predict", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 422)  # FastAPI's own required-file validation

    def test_predict_rejects_empty_file(self) -> None:
        token = self.register("native_predict_empty")
        response = self.client.post(
            "/native/predict",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("clip.mp4", BytesIO(b""), "video/mp4")},
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_rejects_unsupported_extension(self) -> None:
        token = self.register("native_predict_badext")
        response = self.client.post(
            "/native/predict",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("clip.txt", BytesIO(b"not a video"), "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_success_with_mocked_inference(self) -> None:
        token = self.register("native_predict_ok")
        fake_result = {
            "prediction": "BOOK",
            "confidence": 0.87,
            "top_k": [
                {"gloss": "BOOK", "confidence": 0.87},
                {"gloss": "HELLO", "confidence": 0.08},
            ],
            "video": "irrelevant-in-mock.mp4",
            "num_classes": 10,
        }
        with patch("backend.routers.native.predict_video", return_value=fake_result) as mocked:
            response = self.client.post(
                "/native/predict",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(mocked.called)

        payload = response.json()
        self.assertEqual(payload["prediction"], "BOOK")
        self.assertEqual(payload["confidence"], 0.87)
        self.assertEqual(payload["top_k"], fake_result["top_k"])
        self.assertEqual(payload["model_type"], "native_i3d")
        self.assertEqual(payload["model_version"], "native_i3d_v1")
        self.assertIsInstance(payload["latency_ms"], int)
        self.assertGreaterEqual(payload["latency_ms"], 0)
        self.assertIsNotNone(payload["native_sign_id"])  # BOOK exists in the seeded catalog

        with SessionLocal() as db:
            rows = db.query(SignPrediction).all()
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row.model_type, "native_i3d")
            self.assertEqual(row.model_version, "native_i3d_v1")
            self.assertEqual(row.predicted_label, "BOOK")
            self.assertEqual(row.confidence, 0.87)
            self.assertIsNone(row.expected_label)
            self.assertIsNone(row.correct)

    def test_predict_with_expected_label_logs_correctness(self) -> None:
        token = self.register("native_predict_expected")
        fake_result = {
            "prediction": "HELLO",
            "confidence": 0.5,
            "top_k": [{"gloss": "HELLO", "confidence": 0.5}],
            "video": "irrelevant.mp4",
            "num_classes": 10,
        }
        with patch("backend.routers.native.predict_video", return_value=fake_result):
            response = self.client.post(
                "/native/predict?expected_label=HELLO",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        with SessionLocal() as db:
            row = db.query(SignPrediction).order_by(SignPrediction.id.desc()).first()
            self.assertEqual(row.expected_label, "HELLO")
            self.assertTrue(row.correct)


    # --- GET /native/progress -------------------------------------------------

    def test_progress_requires_auth(self) -> None:
        response = self.client.get("/native/progress")
        self.assertEqual(response.status_code, 401)

    def test_progress_new_user_is_honestly_zero(self) -> None:
        token = self.register("native_progress_new")
        response = self.client.get("/native/progress", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["total_signs"], 3)  # only the 3 active seeded signs
        self.assertEqual(payload["started_signs"], 0)
        self.assertEqual(payload["mastered_signs"], 0)
        self.assertEqual(payload["overall_mastery"], 0.0)
        self.assertEqual(len(payload["signs"]), 3)
        for entry in payload["signs"]:
            self.assertEqual(entry["attempts"], 0)
            self.assertEqual(entry["correct"], 0)
            self.assertEqual(entry["mastery"], 0.0)
            self.assertIsNone(entry["last_practiced"])

    def test_progress_aggregates_real_rows(self) -> None:
        token = self.register("native_progress_real")
        response = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        user_id = response.json()["id"]

        book_id = _sign_id("BOOK")
        hello_id = _sign_id("HELLO")
        with SessionLocal() as db:
            db.add(
                NativeSignProgress(
                    user_id=user_id, native_sign_id=book_id,
                    attempts=5, correct_attempts=5, accuracy=100.0, mastery=100.0,
                )
            )
            db.add(
                NativeSignProgress(
                    user_id=user_id, native_sign_id=hello_id,
                    attempts=2, correct_attempts=1, accuracy=50.0, mastery=40.0,
                )
            )
            db.commit()

        response = self.client.get("/native/progress", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["total_signs"], 3)
        self.assertEqual(payload["started_signs"], 2)  # BOOK + HELLO; WATER untouched
        self.assertEqual(payload["mastered_signs"], 1)  # only BOOK reached 100
        self.assertAlmostEqual(payload["overall_mastery"], (100.0 + 40.0 + 0.0) / 3, places=1)

        book_entry = next(e for e in payload["signs"] if e["gloss"] == "BOOK")
        self.assertEqual(book_entry["attempts"], 5)
        self.assertEqual(book_entry["correct"], 5)
        self.assertEqual(book_entry["mastery"], 100.0)

    def test_progress_is_isolated_per_user(self) -> None:
        token_a = self.register("native_progress_user_a")
        token_b = self.register("native_progress_user_b")
        user_a_id = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token_a}"}).json()["id"]

        with SessionLocal() as db:
            db.add(
                NativeSignProgress(
                    user_id=user_a_id, native_sign_id=_sign_id("BOOK"),
                    attempts=10, correct_attempts=10, accuracy=100.0, mastery=100.0,
                )
            )
            db.commit()

        response_b = self.client.get("/native/progress", headers={"Authorization": f"Bearer {token_b}"})
        payload_b = response_b.json()
        self.assertEqual(payload_b["started_signs"], 0)
        self.assertEqual(payload_b["mastered_signs"], 0)
        self.assertEqual(payload_b["overall_mastery"], 0.0)

        response_a = self.client.get("/native/progress", headers={"Authorization": f"Bearer {token_a}"})
        payload_a = response_a.json()
        self.assertEqual(payload_a["started_signs"], 1)
        self.assertEqual(payload_a["mastered_signs"], 1)

    # --- GET /native/progress/{sign_id} ----------------------------------------

    def test_single_sign_progress_works(self) -> None:
        token = self.register("native_progress_single")
        book_id = _sign_id("BOOK")
        response = self.client.get(f"/native/progress/{book_id}", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["sign_id"], book_id)
        self.assertEqual(payload["gloss"], "BOOK")
        self.assertEqual(payload["attempts"], 0)
        self.assertIsNone(payload["last_practiced"])

    def test_single_sign_progress_unknown_id_404(self) -> None:
        token = self.register("native_progress_unknown")
        response = self.client.get("/native/progress/999999", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 404)

    def test_single_sign_progress_inactive_sign_404(self) -> None:
        token = self.register("native_progress_inactive")
        retired_id = _sign_id("RETIRED")
        response = self.client.get(f"/native/progress/{retired_id}", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 404)

    # --- POST /native/predict?native_sign_id=... (practice flow) --------------

    def _fake_result(self, prediction: str, confidence: float = 0.8) -> dict:
        return {
            "prediction": prediction,
            "confidence": confidence,
            "top_k": [{"gloss": prediction, "confidence": confidence}],
            "video": "irrelevant.mp4",
            "num_classes": 10,
        }

    def _practice(self, token: str, native_sign_id: int, prediction: str, confidence: float = 0.8):
        with patch(
            "backend.routers.native.predict_video", return_value=self._fake_result(prediction, confidence)
        ):
            return self.client.post(
                f"/native/predict?native_sign_id={native_sign_id}",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")},
            )

    def test_practice_requires_auth(self) -> None:
        book_id = _sign_id("BOOK")
        response = self.client.post(
            f"/native/predict?native_sign_id={book_id}",
            files={"file": ("clip.mp4", BytesIO(b"fake-bytes"), "video/mp4")},
        )
        self.assertEqual(response.status_code, 401)

    def test_practice_unknown_sign_404(self) -> None:
        token = self.register("native_practice_unknown")
        with patch("backend.routers.native.predict_video", return_value=self._fake_result("BOOK")):
            response = self.client.post(
                "/native/predict?native_sign_id=999999",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("clip.mp4", BytesIO(b"fake-bytes"), "video/mp4")},
            )
        self.assertEqual(response.status_code, 404)

    def test_practice_inactive_sign_404(self) -> None:
        token = self.register("native_practice_inactive")
        retired_id = _sign_id("RETIRED")
        with patch("backend.routers.native.predict_video", return_value=self._fake_result("BOOK")):
            response = self.client.post(
                f"/native/predict?native_sign_id={retired_id}",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("clip.mp4", BytesIO(b"fake-bytes"), "video/mp4")},
            )
        self.assertEqual(response.status_code, 404)

    def test_practice_correct_prediction(self) -> None:
        token = self.register("native_practice_correct")
        book_id = _sign_id("BOOK")
        response = self._practice(token, book_id, "BOOK", confidence=0.91)
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["correct"])
        self.assertEqual(payload["expected_sign"]["gloss"], "BOOK")
        self.assertEqual(payload["prediction"], "BOOK")
        self.assertIsNotNone(payload["session_id"])
        self.assertIsInstance(payload["response_time"], int)

        with SessionLocal() as db:
            sessions = db.query(NativeSignSession).all()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].result, "correct")
            self.assertEqual(sessions[0].model_version, "native_i3d_v1")
            self.assertEqual(sessions[0].native_sign_id, book_id)

    def test_practice_incorrect_prediction(self) -> None:
        token = self.register("native_practice_incorrect")
        book_id = _sign_id("BOOK")
        response = self._practice(token, book_id, "WATER")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["correct"])
        with SessionLocal() as db:
            session_row = db.query(NativeSignSession).order_by(NativeSignSession.id.desc()).first()
            self.assertEqual(session_row.result, "incorrect")

    def test_practice_creates_progress_only_after_attempt(self) -> None:
        token = self.register("native_practice_progress_create")
        response = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        user_id = response.json()["id"]
        book_id = _sign_id("BOOK")

        with SessionLocal() as db:
            self.assertIsNone(
                db.query(NativeSignProgress)
                .filter(NativeSignProgress.user_id == user_id, NativeSignProgress.native_sign_id == book_id)
                .first()
            )

        self._practice(token, book_id, "BOOK")

        with SessionLocal() as db:
            progress = (
                db.query(NativeSignProgress)
                .filter(NativeSignProgress.user_id == user_id, NativeSignProgress.native_sign_id == book_id)
                .first()
            )
            self.assertIsNotNone(progress)
            self.assertEqual(progress.attempts, 1)
            self.assertEqual(progress.correct_attempts, 1)
            self.assertEqual(progress.accuracy, 100.0)
            self.assertEqual(progress.mastery, 100.0)
            self.assertIsNotNone(progress.last_practiced)

    def test_practice_accumulates_attempts_and_accuracy(self) -> None:
        token = self.register("native_practice_accumulate")
        book_id = _sign_id("BOOK")

        self._practice(token, book_id, "BOOK", confidence=0.6)   # correct
        self._practice(token, book_id, "WATER", confidence=0.95)  # incorrect, higher confidence
        self._practice(token, book_id, "BOOK", confidence=0.5)   # correct

        progress_response = self.client.get(
            f"/native/progress/{book_id}", headers={"Authorization": f"Bearer {token}"}
        )
        payload = progress_response.json()
        self.assertEqual(payload["attempts"], 3)
        self.assertEqual(payload["correct"], 2)
        self.assertAlmostEqual(payload["mastery"], round(2 / 3 * 100, 1), places=1)

        with SessionLocal() as db:
            user_id = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["id"]
            progress = (
                db.query(NativeSignProgress)
                .filter(NativeSignProgress.user_id == user_id, NativeSignProgress.native_sign_id == book_id)
                .first()
            )
            self.assertEqual(progress.best_confidence, 0.95)  # best confidence regardless of correctness

    def test_practice_does_not_leak_progress_to_other_user(self) -> None:
        token_a = self.register("native_practice_user_a")
        token_b = self.register("native_practice_user_b")
        book_id = _sign_id("BOOK")

        self._practice(token_a, book_id, "BOOK")

        response_b = self.client.get(
            f"/native/progress/{book_id}", headers={"Authorization": f"Bearer {token_b}"}
        )
        payload_b = response_b.json()
        self.assertEqual(payload_b["attempts"], 0)  # user B untouched by user A's practice

    def test_practice_invalid_upload_still_validated(self) -> None:
        token = self.register("native_practice_badupload")
        book_id = _sign_id("BOOK")
        response = self.client.post(
            f"/native/predict?native_sign_id={book_id}",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("clip.txt", BytesIO(b"not a video"), "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        with SessionLocal() as db:
            self.assertEqual(db.query(NativeSignSession).count(), 0)


if __name__ == "__main__":
    unittest.main()
