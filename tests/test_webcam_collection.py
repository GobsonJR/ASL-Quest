"""Tests for the labelled webcam data collector (backend/services/webcam_dataset.py + routers/collect.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.routers import collect, ml  # noqa: E402
from backend.services import webcam_dataset as wd  # noqa: E402
from src.inference.hands import HandCrop  # noqa: E402


def textured(seed: int, size=(640, 480), blur: float = 0.7) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(40, 215, size=(size[1], size[0], 3), dtype=np.uint8))
    return image.filter(ImageFilter.GaussianBlur(blur))


def jpeg(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return buffer.getvalue()


class FakeDetector:
    """Always 'finds' a hand in a fixed box (or none), like MediaPipe would."""

    available = True
    error = None

    def __init__(self, box=(150.0, 100.0, 400.0, 360.0), detected: bool = True) -> None:
        self.box, self.detected = box, detected

    def crop_pil(self, image, padding: float = 0.30) -> HandCrop:
        if not self.detected:
            return HandCrop(image=image, box=None, detected=False)
        x1, y1, x2, y2 = self.box
        window = (int(x1), int(y1), int(x2), int(y2))
        return HandCrop(image=image.crop(window), box=window, detected=True, hand_box=self.box)


class ExplodingPredictor:
    """The collector must never ask a model for a label."""

    model_path = Path("none")
    class_names = wd.LETTERS

    def predict_pil(self, *a, **k):
        raise AssertionError("the collector consulted the model for a label")


SESSION = {"room": "Bedroom A", "lighting": "lamp", "background": "shelves", "face_visible": True, "glasses": True,
           "consent_confirmed": True}


class ServiceTests(unittest.TestCase):
    def test_sharpness_separates_blurred_from_sharp(self) -> None:
        sharp = wd.sharpness(textured(1))
        blurred = wd.sharpness(textured(1).filter(ImageFilter.GaussianBlur(9)))
        self.assertGreater(sharp, wd.MIN_SHARPNESS)
        self.assertLess(blurred, wd.MIN_SHARPNESS)

    def test_sharpness_threshold_sits_in_the_gap_measured_on_real_webcam_crops(self) -> None:
        # Measured with wd.sharpness on 60 real captured crops: motion-blurred frames 82-100, skin close-ups 33-38, and the
        # sharpest-of-the-rest started at 126. Moving the threshold outside this gap would pass blur or reject good frames.
        self.assertGreater(wd.MIN_SHARPNESS, 100.0)
        self.assertLess(wd.MIN_SHARPNESS, 126.0)

    def test_dhash_collides_for_the_same_picture_not_for_different_ones(self) -> None:
        a, b = textured(1), textured(2)
        self.assertEqual(wd.hamming(wd.dhash(a), wd.dhash(a)), 0)
        self.assertGreater(wd.hamming(wd.dhash(a), wd.dhash(b)), wd.NEAR_DUPLICATE_HAMMING)

    def test_quality_reasons(self) -> None:
        ok, crop = wd.analyze_frame(textured(3), FakeDetector())
        self.assertTrue(ok.usable and crop is not None)
        self.assertFalse(wd.analyze_frame(textured(3), FakeDetector(detected=False))[0].usable)
        small = wd.analyze_frame(textured(3), FakeDetector(box=(100.0, 100.0, 150.0, 150.0)))[0]
        self.assertIn("hand too small", " ".join(small.reasons))
        blurry = wd.analyze_frame(textured(3).filter(ImageFilter.GaussianBlur(9)), FakeDetector())[0]
        self.assertIn("blurry", " ".join(blurry.reasons))
        dark = wd.analyze_frame(Image.new("RGB", (640, 480), (5, 5, 5)), FakeDetector())[0]
        self.assertIn("too dark", " ".join(dark.reasons))
        edge = wd.analyze_frame(textured(3), FakeDetector(box=(0.0, 100.0, 250.0, 360.0)))[0]
        self.assertTrue(edge.usable and any("frame edge" in w for w in edge.warnings))


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "webcam"
        self.original = (collect.DEV_COLLECT, collect.dataset)
        collect.DEV_COLLECT = True
        collect.dataset = wd.WebcamDataset(self.root)
        ml.set_runtime(ExplodingPredictor(), FakeDetector(), None)
        # The router now requires an admin and a database (for the audit log). The authorization rules themselves are
        # tested in tests/test_roles_and_collection_security.py; here we only need an admin and a scratch database.
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from backend import models
        from backend.database import Base, get_db
        from backend.security import get_current_admin

        engine = create_engine(f"sqlite:///{Path(self.tmp.name) / 'test.db'}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        self.addCleanup(engine.dispose)  # Windows cannot delete the temp folder while the SQLite file is still open
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        with factory() as setup_db:
            admin = models.User(username="root", email="root@example.com", password_hash="x", role="admin")
            setup_db.add(admin)
            setup_db.commit()

        def override_db():
            with factory() as db:
                yield db

        app = FastAPI()
        app.include_router(collect.router)
        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_current_admin] = lambda: admin
        self.client = TestClient(app)
        self.session = self.client.post("/collect/sessions", json=SESSION).json()["session_id"]

    def tearDown(self) -> None:
        collect.DEV_COLLECT, collect.dataset = self.original
        ml.set_runtime(None, None, None)

    def post_sample(self, image=None, **fields):
        data = {"session_id": self.session, "letter": "C", "confirmed": "true", "hand": "right", "distance": "medium"}
        data.update(fields)
        files = {"file": ("frame.jpg", jpeg(image or textured(10)), "image/jpeg")}
        return self.client.post("/collect/samples", data=data, files=files)

    def test_disabled_unless_explicitly_enabled(self) -> None:
        collect.DEV_COLLECT = False
        self.assertEqual(self.client.get("/collect/status").status_code, 404)
        self.assertEqual(self.client.post("/collect/sessions", json=SESSION).status_code, 404)

    def test_session_options_are_validated(self) -> None:
        bad = self.client.post("/collect/sessions", json={**SESSION, "lighting": "disco"})
        self.assertEqual(bad.status_code, 422)

    def test_analyze_previews_the_crop_and_saves_nothing(self) -> None:
        reply = self.client.post("/collect/analyze", files={"file": ("f.jpg", jpeg(textured(4)), "image/jpeg")}).json()
        self.assertTrue(reply["usable"])
        self.assertTrue(reply["crop_preview"].startswith("data:image/png;base64,"))
        self.assertEqual(self.client.get(f"/collect/sessions/{self.session}").json()["total"], 0)

    def test_a_confirmed_sample_is_saved_with_a_person_label_and_full_metadata(self) -> None:
        reply = self.post_sample()
        self.assertEqual(reply.status_code, 201)
        sample = reply.json()["sample"]
        folder = self.root / self.session / "C"
        meta = json.loads((folder / f"{sample['sample_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["letter"], meta["label_source"]), ("C", "person_confirmed"))
        self.assertEqual((meta["hand"], meta["distance"], meta["lighting"], meta["background"]), ("right", "medium", "lamp", "shelves"))
        self.assertTrue(meta["face_visible"] and meta["glasses"])
        self.assertTrue((folder / meta["image"]).exists())
        self.assertGreater(meta["quality"]["metrics"]["sharpness"], wd.MIN_SHARPNESS)
        self.assertEqual(reply.json()["session"]["counts"]["C"], 1)

    def test_unconfirmed_or_invalid_labels_are_refused(self) -> None:
        self.assertEqual(self.post_sample(confirmed="false").status_code, 422)
        self.assertEqual(self.post_sample(letter="CC").status_code, 422)
        self.assertEqual(self.post_sample(letter="7").status_code, 422)
        self.assertEqual(self.post_sample(hand="foot").status_code, 422)
        self.assertEqual(self.client.get(f"/collect/sessions/{self.session}").json()["total"], 0)

    def test_unusable_frames_are_rejected_with_reasons_and_not_stored(self) -> None:
        blurry = self.post_sample(textured(11).filter(ImageFilter.GaussianBlur(9)))
        self.assertEqual(blurry.status_code, 422)
        self.assertTrue(any("blurry" in r for r in blurry.json()["detail"]["reasons"]))
        ml.set_runtime(ExplodingPredictor(), FakeDetector(detected=False), None)
        self.assertEqual(self.post_sample(textured(12)).status_code, 422)
        self.assertEqual(self.client.get(f"/collect/sessions/{self.session}").json()["total"], 0)
        self.assertEqual(list((self.root / self.session).glob("*/*.jpg")), [])

    def test_near_duplicate_singles_are_rejected_but_hold_sequences_are_not(self) -> None:
        image = textured(20)
        self.assertEqual(self.post_sample(image).status_code, 201)
        duplicate = self.post_sample(image)
        self.assertEqual(duplicate.status_code, 422)
        self.assertEqual(duplicate.json()["detail"]["reasons"], ["near-duplicate"])
        for index in range(3):  # a deliberate hold: same pose, kept for stability measurements
            reply = self.post_sample(image, mode="hold", sequence_id="seq1", frame_index=str(index))
            self.assertEqual(reply.status_code, 201)
        self.assertEqual(self.post_sample(image, mode="hold").status_code, 422)  # hold needs a sequence id

    def test_the_collector_never_asks_a_model_for_a_label(self) -> None:
        # ExplodingPredictor raises if consulted; every call above and below must succeed without it.
        self.assertEqual(self.post_sample(textured(30), letter="W").status_code, 201)

    def test_delete_removes_the_files(self) -> None:
        sample = self.post_sample().json()["sample"]
        reply = self.client.delete(f"/collect/sessions/{self.session}/samples/{sample['sample_id']}")
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(list((self.root / self.session).glob("*/*")), [])

    def test_session_and_sample_ids_cannot_escape_the_dataset_folder(self) -> None:
        self.assertEqual(self.client.get("/collect/sessions/..%2F..%2Fetc").status_code, 404)
        self.assertEqual(self.client.delete(f"/collect/sessions/{self.session}/samples/..%2F..").status_code, 404)
        self.assertEqual(self.client.get("/collect/sessions/s20990101_000000_x/../..").status_code, 404)

    def test_status_reports_totals_and_thresholds(self) -> None:
        self.post_sample()
        status = self.client.get("/collect/status").json()
        self.assertEqual(status["totals"]["C"], 1)
        self.assertEqual(status["thresholds"]["min_sharpness"], wd.MIN_SHARPNESS)
        self.assertEqual(len(status["letters"]), 26)


class NegativeAndPersonTests(unittest.TestCase):
    setUp = ApiTests.setUp  # same fixture, without re-running ApiTests' own tests
    tearDown = ApiTests.tearDown
    """Held-out-evaluation support: who the person was, and frames the recognizer must not answer on."""

    def post_negative(self, image=None, **fields):
        data = {"session_id": self.session, "kind": "no_hand", "confirmed": "true"}
        data.update(fields)
        return self.client.post("/collect/negatives", data=data, files={"file": ("frame.jpg", jpeg(image or textured(50)), "image/jpeg")})

    def test_session_records_a_slugged_person(self) -> None:
        reply = self.client.post("/collect/sessions", json={**SESSION, "room": "Other room", "person": "  Asha K. "})
        self.assertEqual(reply.json()["person"], "asha-k")
        self.assertEqual(self.client.get(f"/collect/sessions/{self.session}").json()["person"], "")  # older/unnamed sessions stay unnamed

    def test_a_negative_is_stored_outside_the_letter_folders_and_never_counted_as_a_sample(self) -> None:
        ml.set_runtime(ExplodingPredictor(), FakeDetector(detected=False), None)
        record = self.post_negative().json()["negative"]
        self.assertEqual((record["kind"], record["expected"], record["detector_found_hand"]), ("no_hand", "no_answer", False))
        folder = self.root / self.session / "negatives" / "no_hand"
        self.assertTrue((folder / record["image"]).exists() and (folder / f"{record['sample_id']}.json").exists())
        summary = self.client.get(f"/collect/sessions/{self.session}").json()
        self.assertEqual((summary["total"], summary["negatives"]), (0, {"no_hand": 1, "partial_hand": 0}))
        self.assertEqual(collect.dataset.samples(self.session), [])  # exporters enumerate */*.json and must not see negatives

    def test_the_detectors_verdict_is_recorded_not_used_as_the_label(self) -> None:
        # MediaPipe "finds" a hand in a frame the person says has none: stored as no_hand, detection flagged as a false positive
        record = self.post_negative(textured(51)).json()["negative"]
        self.assertEqual((record["kind"], record["detector_found_hand"]), ("no_hand", True))

    def test_negatives_need_confirmation_and_a_known_kind(self) -> None:
        self.assertEqual(self.post_negative(confirmed="false").status_code, 422)
        self.assertEqual(self.post_negative(kind="fist").status_code, 422)
        self.assertEqual(self.client.get(f"/collect/sessions/{self.session}").json()["negatives"], {"no_hand": 0, "partial_hand": 0})

    def test_near_duplicate_negatives_are_refused_per_kind(self) -> None:
        self.assertEqual(self.post_negative(textured(52)).status_code, 201)
        self.assertEqual(self.post_negative(textured(52)).status_code, 422)
        self.assertEqual(self.post_negative(textured(52), kind="partial_hand").status_code, 201)

    def test_negatives_can_be_undone_and_ids_cannot_escape(self) -> None:
        record = self.post_negative().json()["negative"]
        self.assertEqual(self.client.delete(f"/collect/sessions/{self.session}/negatives/{record['sample_id']}").status_code, 200)
        self.assertEqual(list((self.root / self.session / "negatives").glob("*/*")), [])
        self.assertEqual(self.client.delete(f"/collect/sessions/{self.session}/negatives/..%2F..").status_code, 404)


if __name__ == "__main__":
    unittest.main()
