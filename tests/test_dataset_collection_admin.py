"""Admin dataset collection: consent, upload validation, labelling, native-sign clips, gallery, deletion and export."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import models  # noqa: E402
from backend.database import Base, get_db  # noqa: E402
from backend.routers import collect, ml  # noqa: E402
from backend.security import get_current_admin  # noqa: E402
from backend.services import webcam_dataset as wd  # noqa: E402
from src.inference.hands import HandCrop  # noqa: E402

SESSION = {"room": "Studio", "lighting": "lamp", "background": "shelves", "person": "person-1", "consent_confirmed": True}
WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 600
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 600


def textured(seed: int, size=(640, 480)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(40, 215, size=(size[1], size[0], 3), dtype=np.uint8)).filter(ImageFilter.GaussianBlur(0.7))


def encode(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


class FakeDetector:
    available = True
    error = None

    def crop_pil(self, image, padding: float = 0.30) -> HandCrop:
        window = (150, 100, 400, 360)
        return HandCrop(image=image.crop(window), box=window, detected=True, hand_box=(150.0, 100.0, 400.0, 360.0))


class CollectionAdminTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.original = (collect.DEV_COLLECT, collect.dataset)
        collect.DEV_COLLECT = True
        collect.dataset = wd.WebcamDataset(Path(self.tmp.name) / "webcam")
        self.addCleanup(lambda: (setattr(collect, "DEV_COLLECT", self.original[0]), setattr(collect, "dataset", self.original[1])))
        ml.set_runtime(None, FakeDetector(), None)
        self.addCleanup(lambda: ml.set_runtime(None, None, None))

        engine = create_engine(f"sqlite:///{Path(self.tmp.name) / 'db.sqlite'}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        self.addCleanup(engine.dispose)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        self.factory = factory
        with factory() as db:
            self.admin = models.User(username="root", email="root@example.com", password_hash="x", role="admin")
            db.add(self.admin)
            for gloss in ("HELLO", "BOOK"):
                db.add(models.NativeSign(gloss=gloss, display_name=gloss.title(), meaning="m", category="c", difficulty="Easy",
                                         description="d", example_text="e", dataset_available=True, model_available=False, active=True))
            db.add(models.NativeSign(gloss="RETIRED", display_name="Retired", meaning="m", category="c", difficulty="Easy",
                                     description="d", example_text="e", dataset_available=True, model_available=False, active=False))
            db.commit()

        def override_db():
            with factory() as db:
                yield db

        app = FastAPI()
        app.include_router(collect.router)
        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_current_admin] = lambda: self.admin
        self.client = TestClient(app)
        self.session = self.client.post("/collect/sessions", json=SESSION).json()["session_id"]

    # ---------------------------------------------------------------- helpers
    def sample(self, image: bytes | None = None, name: str = "f.jpg", mime: str = "image/jpeg", **fields):
        data = {"session_id": self.session, "letter": "C", "confirmed": "true", "hand": "right", "distance": "medium", **fields}
        return self.client.post("/collect/samples", data=data, files={"file": (name, image or encode(textured(1)), mime)})

    def clip(self, payload: bytes = WEBM, name: str = "clip.webm", mime: str = "video/webm", **fields):
        data = {"session_id": self.session, "gloss": "HELLO", "confirmed": "true", "hand": "right", "duration_ms": "4200", **fields}
        return self.client.post("/collect/native-clips", data=data, files={"file": (name, payload, mime)})

    def audit_actions(self) -> list[str]:
        with self.factory() as db:
            return [row.action for row in db.query(models.AdminAuditLog).order_by(models.AdminAuditLog.id).all()]

    # ---------------------------------------------------------------- consent
    def test_a_session_cannot_be_created_without_consent(self) -> None:
        for body in ({**SESSION, "consent_confirmed": False}, {k: v for k, v in SESSION.items() if k != "consent_confirmed"}):
            reply = self.client.post("/collect/sessions", json=body)
            self.assertEqual(reply.status_code, 422)
            self.assertIn("consent", reply.json()["detail"]["message"])

    def test_consent_is_recorded_with_who_and_when_and_no_secrets(self) -> None:
        session = self.client.get(f"/collect/sessions/{self.session}").json()
        self.assertTrue(session["consent"]["confirmed"])
        self.assertEqual(session["consent"]["recorded_by"], "root")
        self.assertTrue(session["consent"]["recorded_at"])
        blob = json.dumps(session).lower()
        for forbidden in ("token", "password", "authorization", "secret", "@example.com"):
            self.assertNotIn(forbidden, blob)

    def test_nothing_is_saved_into_a_session_that_has_no_consent_record(self) -> None:
        legacy = "s20200101_000000_old"
        folder = Path(self.tmp.name) / "webcam" / legacy
        folder.mkdir(parents=True)
        (folder / "session.json").write_text(json.dumps({"session_id": legacy, "lighting": "lamp", "background": "shelves",
                                                         "face_visible": True, "glasses": False}), encoding="utf-8")
        self.assertEqual(self.sample(session_id=legacy).status_code, 409)
        self.assertEqual(self.clip(session_id=legacy).status_code, 409)

    # ---------------------------------------------------------------- upload validation (images)
    def test_a_confirmed_letter_sample_is_saved_with_label_and_metadata(self) -> None:
        reply = self.sample()
        self.assertEqual(reply.status_code, 201, reply.text)
        record = reply.json()["sample"]
        self.assertEqual(record["letter"], "C")
        self.assertEqual(record["label_source"], "person_confirmed")
        self.assertTrue(record["captured_at"])
        self.assertEqual(record["session_id"], self.session)
        self.assertRegex(record["image"], r"^C_\d+_[0-9a-f]{4}\.jpg$")  # server-generated name, nothing from the client
        self.assertEqual(reply.json()["session"]["counts"]["C"], 1)

    def test_invalid_labels_and_missing_confirmation_are_refused(self) -> None:
        self.assertEqual(self.sample(confirmed="false").status_code, 422)
        for bad in ("", "AB", "1", "../C", "c1"):
            self.assertEqual(self.sample(letter=bad).status_code, 422, bad)

    def test_unsupported_empty_corrupt_and_oversized_images_are_refused(self) -> None:
        self.assertEqual(self.sample(encode(textured(2), "WEBP"), "f.webp", "image/webp").status_code, 415)
        self.assertEqual(self.sample(encode(textured(2), "GIF"), "f.gif", "image/gif").status_code, 415)
        self.assertEqual(self.sample(b"definitely not an image", "f.jpg").status_code, 400)
        self.assertEqual(self.client.post("/collect/samples", data={"session_id": self.session, "letter": "C", "confirmed": "true",
                         "hand": "right", "distance": "medium"}, files={"file": ("f.jpg", b"", "image/jpeg")}).status_code, 400)
        too_big = b"\xff\xd8\xff" + b"0" * (wd.MAX_IMAGE_BYTES + 10)
        self.assertEqual(self.sample(too_big).status_code, 413)
        tiny = Image.new("RGB", (16, 16), (120, 120, 120))
        self.assertEqual(self.sample(encode(tiny)).status_code, 422)

    def test_a_png_keeps_a_png_extension_and_a_wrong_extension_in_the_name_is_ignored(self) -> None:
        reply = self.sample(encode(textured(3), "PNG"), "pretend.jpg", "image/jpeg")
        self.assertEqual(reply.status_code, 201, reply.text)
        self.assertTrue(reply.json()["sample"]["image"].endswith(".png"))

    # ---------------------------------------------------------------- native-sign clips
    def test_a_native_clip_is_saved_under_its_sign_with_metadata(self) -> None:
        reply = self.clip()
        self.assertEqual(reply.status_code, 201, reply.text)
        record = reply.json()["clip"]
        self.assertEqual(record["gloss"], "HELLO")
        self.assertEqual(record["label_source"], "person_confirmed")
        self.assertRegex(record["file"], r"^C_\d+_[0-9a-f]{4}\.webm$")
        self.assertEqual(reply.json()["session"]["native_counts"], {"HELLO": 1})
        self.assertEqual(self.clip(MP4, "c.mp4", "video/mp4", gloss="BOOK").json()["clip"]["file"][-4:], ".mp4")

    def test_native_clip_validation(self) -> None:
        self.assertEqual(self.clip(confirmed="false").status_code, 422)
        self.assertEqual(self.clip(gloss="NOPE").status_code, 422)
        self.assertEqual(self.clip(gloss="RETIRED").status_code, 422)  # inactive signs are not collectable
        self.assertEqual(self.clip(gloss="../HELLO").status_code, 422)
        self.assertEqual(self.clip(b"GIF89a....", "c.webm", "video/webm").status_code, 415)  # type comes from the bytes, not the name
        self.assertEqual(self.clip(b"", "c.webm").status_code, 400)
        self.assertEqual(self.clip(duration_ms="999999").status_code, 422)
        self.assertEqual(self.clip(hand="both").status_code, 422)

    def test_oversized_clips_are_refused_without_being_stored(self) -> None:
        big = WEBM + b"0" * wd.MAX_CLIP_BYTES
        self.assertEqual(self.clip(big).status_code, 413)
        self.assertEqual(self.client.get(f"/collect/sessions/{self.session}").json()["native_total"], 0)

    # ---------------------------------------------------------------- gallery and counts
    def test_gallery_lists_samples_and_serves_each_image_and_clip(self) -> None:
        sample_id = self.sample().json()["sample"]["sample_id"]
        self.sample(encode(textured(7)), letter="D")
        clip_id = self.clip().json()["clip"]["sample_id"]

        listing = self.client.get(f"/collect/sessions/{self.session}/samples").json()["samples"]
        self.assertEqual({s["letter"] for s in listing}, {"C", "D"})
        only_c = self.client.get(f"/collect/sessions/{self.session}/samples", params={"label": "c"}).json()["samples"]
        self.assertEqual([s["letter"] for s in only_c], ["C"])

        image = self.client.get(f"/collect/sessions/{self.session}/samples/{sample_id}/image")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(Image.open(BytesIO(image.content)).size, (640, 480))
        self.assertIn("no-store", image.headers["cache-control"])

        clips = self.client.get(f"/collect/sessions/{self.session}/native-clips").json()["clips"]
        self.assertEqual([c["sample_id"] for c in clips], [clip_id])
        video = self.client.get(f"/collect/sessions/{self.session}/native-clips/{clip_id}/video")
        self.assertEqual((video.status_code, video.headers["content-type"], video.content), (200, "video/webm", WEBM))

    def test_visible_counts_per_class_across_letters_and_native_signs(self) -> None:
        self.sample()
        self.sample(encode(textured(8)))
        self.clip()
        status = self.client.get("/collect/status").json()
        self.assertEqual(status["totals"]["C"], 2)
        self.assertEqual(status["native_totals"], {"HELLO": 1})
        self.assertEqual([s["gloss"] for s in status["native_signs"]], ["BOOK", "HELLO"])  # active signs only
        self.assertTrue(status["consent_required"])
        self.assertEqual(status["limits"]["clip_formats"], ["webm", "mp4"])

    def test_ids_cannot_escape_the_dataset_folder(self) -> None:
        for sid in ("..", "../x", "s20200101_000000_x/../../y", "%2e%2e"):
            self.assertIn(self.client.get(f"/collect/sessions/{sid}/samples").status_code, (404, 405, 422), sid)
        for bad in ("../../etc/passwd", "C_1_zzzz", "C_1_0000/../../x", "c_1_0000"):
            self.assertEqual(self.client.get(f"/collect/sessions/{self.session}/samples/{bad}/image").status_code, 404, bad)
            self.assertEqual(self.client.get(f"/collect/sessions/{self.session}/native-clips/{bad}/video").status_code, 404, bad)
            self.assertEqual(self.client.delete(f"/collect/sessions/{self.session}/samples/{bad}").status_code, 404, bad)

    # ---------------------------------------------------------------- deletion
    def test_deleting_a_sample_or_clip_removes_the_files_updates_counts_and_is_audited(self) -> None:
        sample_id = self.sample().json()["sample"]["sample_id"]
        clip_id = self.clip().json()["clip"]["sample_id"]
        root = Path(self.tmp.name) / "webcam" / self.session

        self.assertEqual(self.client.delete(f"/collect/sessions/{self.session}/samples/{sample_id}").json()["session"]["counts"]["C"], 0)
        self.assertEqual(list(root.glob("C/*")), [])
        self.assertEqual(self.client.get(f"/collect/sessions/{self.session}/samples/{sample_id}/image").status_code, 404)

        self.assertEqual(self.client.delete(f"/collect/sessions/{self.session}/native-clips/{clip_id}").json()["session"]["native_total"], 0)
        self.assertEqual(list(root.glob("native/HELLO/*.*")), [])
        self.assertEqual(self.client.delete(f"/collect/sessions/{self.session}/native-clips/{clip_id}").status_code, 404)  # already gone
        self.assertEqual(self.audit_actions(), ["dataset.create_session", "dataset.delete_sample", "dataset.delete_clip"])

    # ---------------------------------------------------------------- export
    def test_export_is_a_documented_labelled_zip_with_a_manifest(self) -> None:
        self.sample()
        self.sample(encode(textured(9)), letter="D", hand="left")
        self.clip()
        reply = self.client.get("/collect/export")
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.headers["content-type"], "application/zip")
        archive = zipfile.ZipFile(BytesIO(reply.content))
        names = set(archive.namelist())
        self.assertTrue({"README.txt", "manifest.csv", "manifest.jsonl"} <= names)
        self.assertEqual(sum(n.startswith("images/C/") for n in names), 1)
        self.assertEqual(sum(n.startswith("images/D/") for n in names), 1)
        self.assertEqual(sum(n.startswith("native_clips/HELLO/") for n in names), 1)

        rows = list(csv.DictReader(StringIO(archive.read("manifest.csv").decode())))
        self.assertEqual(sorted((r["label"], r["label_type"]) for r in rows), [("C", "letter"), ("D", "letter"), ("HELLO", "native_sign")])
        for row in rows:
            self.assertIn(row["file"], names)  # every manifest entry points at a real file in the archive
            self.assertEqual(row["consent_confirmed"], "True")
            self.assertEqual(row["contributor"], "person-1")
        readme = archive.read("README.txt").decode()
        self.assertIn("identifiable", readme)
        self.assertIn("NOT TRAINING", readme)
        blob = b"".join(archive.read(n) for n in names if n.endswith((".csv", ".jsonl", ".txt"))).lower()
        for forbidden in (b"token", b"password", b"authorization", b"@example.com", b"root"):
            self.assertNotIn(forbidden, blob)
        self.assertIn("dataset.export", self.audit_actions())

    def test_export_skips_sessions_without_consent_and_can_be_limited_to_one_session(self) -> None:
        self.sample()
        legacy = "s20200101_000000_old"
        folder = Path(self.tmp.name) / "webcam" / legacy / "A"
        folder.mkdir(parents=True)
        (folder.parent / "session.json").write_text(json.dumps({"session_id": legacy, "lighting": "lamp", "background": "shelves",
                                                                "face_visible": True, "glasses": False}), encoding="utf-8")
        (folder / "A_1_0000.jpg").write_bytes(encode(textured(5)))
        (folder / "A_1_0000.json").write_text(json.dumps({"sample_id": "A_1_0000", "letter": "A", "image": "A_1_0000.jpg",
                                                          "captured_at": "x", "mode": "single"}), encoding="utf-8")
        reply = self.client.get("/collect/export")
        self.assertEqual(reply.headers["x-export-skipped-sessions"], "1")
        self.assertFalse(any(n.startswith("images/A/") for n in zipfile.ZipFile(BytesIO(reply.content)).namelist()))
        self.assertEqual(self.client.get("/collect/export", params={"session_id": "../etc"}).status_code, 404)

    def test_empty_export_is_still_a_valid_documented_archive(self) -> None:
        reply = self.client.get("/collect/export")
        self.assertEqual(reply.status_code, 200)
        archive = zipfile.ZipFile(BytesIO(reply.content))
        self.assertEqual(archive.read("manifest.jsonl"), b"")
        self.assertIn("Files: 0", archive.read("README.txt").decode())


if __name__ == "__main__":
    unittest.main()
