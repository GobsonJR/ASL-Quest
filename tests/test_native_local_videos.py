"""Locally stored native-sign reference videos: safe path resolution, streaming with Range, and the reference payload."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["ASL_QUEST_DATABASE_URL"] = f"sqlite:///{TEST_DB.name}"
os.environ["ASL_QUEST_SECRET_KEY"] = "native-local-video-test-secret"

from backend import models  # noqa: E402,F401
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402
from backend.models import NativeSign, NativeSignReference  # noqa: E402
from backend.routers.native import _serialize_reference  # noqa: E402
from backend.services import native_local_videos as nlv  # noqa: E402

FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 8  # content is irrelevant, only bytes are streamed


class LocalVideoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / "hello.mp4").write_bytes(FAKE_MP4)
        (self.dir / "thank-you.mp4").write_bytes(FAKE_MP4)
        (self.dir.parent / "secret.mp4").write_bytes(b"outside")
        self.addCleanup(lambda: (self.dir.parent / "secret.mp4").unlink(missing_ok=True))
        patcher = patch.object(nlv, "LOCAL_VIDEO_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(app)

    def test_path_resolution_default_and_override_names(self) -> None:
        self.assertEqual(nlv.local_video_path("HELLO"), (self.dir / "hello.mp4").resolve())
        self.assertEqual(nlv.local_video_path("thankyou"), (self.dir / "thank-you.mp4").resolve())  # override + case-insensitive
        self.assertIsNone(nlv.local_video_path("WATER"))  # no such file
        self.assertIsNone(nlv.local_video_path(""))

    def test_traversal_and_odd_names_never_resolve(self) -> None:
        for bad in ["../secret", "..", "hello/../hello", "HELLO.MP4", "hel lo", "a" * 80, "..\\secret", "%2e%2e"]:
            with self.subTest(bad):
                self.assertIsNone(nlv.local_video_path(bad))

    def test_endpoint_streams_the_file(self) -> None:
        response = self.client.get("/native/reference-video/HELLO")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "video/mp4")
        self.assertEqual(response.content, FAKE_MP4)

    def test_endpoint_supports_range_requests_for_seeking(self) -> None:
        response = self.client.get("/native/reference-video/HELLO", headers={"Range": "bytes=0-9"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, FAKE_MP4[:10])
        self.assertEqual(response.headers["content-range"], f"bytes 0-9/{len(FAKE_MP4)}")

    def test_endpoint_404_for_missing_unknown_or_malicious(self) -> None:
        for gloss in ["WATER", "NOPE", "..%2fsecret", "secret"]:
            with self.subTest(gloss):
                self.assertEqual(self.client.get(f"/native/reference-video/{gloss}").status_code, 404)

    def test_reference_payload_exposes_local_url_only_when_a_file_exists(self) -> None:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        with SessionLocal() as db:
            for gloss in ("HELLO", "WATER"):
                sign = NativeSign(gloss=gloss, display_name=gloss.title(), meaning="m", category="c", difficulty="Easy",
                                  description="d", example_text="e", dataset_available=True, model_available=False, active=True)
                db.add(sign)
                db.flush()
                if gloss == "HELLO":
                    db.add(NativeSignReference(native_sign_id=sign.id, source_type="vimeo", source_identifier="https://vimeo.com/1/abc",
                                               license_note="n", is_primary=True))
            db.commit()
            hello = _serialize_reference(db.query(NativeSign).filter_by(gloss="HELLO").one())
            water = _serialize_reference(db.query(NativeSign).filter_by(gloss="WATER").one())
        self.assertEqual(hello["local_video_url"], "/native/reference-video/HELLO")
        self.assertEqual(hello["video_url"], "https://vimeo.com/1/abc")  # the online reference is still there as the fallback
        self.assertTrue(hello["available"])
        self.assertIsNone(water["local_video_url"])
        self.assertFalse(water["available"])  # neither a link nor a local file


if __name__ == "__main__":
    unittest.main()
