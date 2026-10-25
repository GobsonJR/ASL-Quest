"""Attaching reference-video links to native signs: validation, upsert, all-or-nothing, and that the API then serves them."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["ASL_QUEST_DATABASE_URL"] = f"sqlite:///{TEST_DB.name}"
os.environ["ASL_QUEST_SECRET_KEY"] = "native-ref-test-secret"

from backend import models  # noqa: E402,F401
from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.models import NativeSign, NativeSignReference  # noqa: E402
from backend.routers.native import _serialize_reference  # noqa: E402
from backend.services.native_references import ReferenceError_, attach_references, source_type_for  # noqa: E402

GOOD = {"url": "https://www.youtube.com/watch?v=abcdefghijk", "license_note": "Own recording, linking allowed"}


class NativeReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        self.db = SessionLocal()
        for gloss in ("HELLO", "BOOK"):
            self.db.add(NativeSign(gloss=gloss, display_name=gloss.title(), meaning="m", category="c", difficulty="Easy",
                                   description="d", example_text="e", dataset_available=True, model_available=False, active=True))
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()

    def test_source_types(self) -> None:
        self.assertEqual(source_type_for("https://youtu.be/abcdefghijk"), "youtube")
        self.assertEqual(source_type_for("https://vimeo.com/123"), "vimeo")
        self.assertEqual(source_type_for("https://example.org/a.mp4"), "url")

    def test_attached_reference_is_served_as_available_and_updates_in_place(self) -> None:
        attach_references(self.db, {"HELLO": GOOD})
        sign = self.db.query(NativeSign).filter_by(gloss="HELLO").one()
        ref = _serialize_reference(sign)
        self.assertTrue(ref["available"])
        self.assertEqual(ref["source_type"], "youtube")
        self.assertEqual(ref["license_note"], GOOD["license_note"])

        attach_references(self.db, {"HELLO": {**GOOD, "url": "https://vimeo.com/123"}})
        self.assertEqual(self.db.query(NativeSignReference).count(), 1)  # updated, not duplicated

    def test_invalid_entries_are_refused_and_change_nothing(self) -> None:
        bad = [
            {"HELLO": {"url": "http://example.org/a.mp4", "license_note": "x"}},  # not https
            {"HELLO": {"url": "https://example.org/a.mp4", "license_note": ""}},  # no licence note
            {"NOPE": GOOD},  # unknown sign
            {"HELLO": GOOD, "BOOK": {"url": "ftp://x", "license_note": "x"}},  # one bad entry blocks the good one too
        ]
        for mapping in bad:
            with self.subTest(mapping=list(mapping)):
                with self.assertRaises(ReferenceError_):
                    attach_references(self.db, mapping)
        self.assertEqual(self.db.query(NativeSignReference).count(), 0)


if __name__ == "__main__":
    unittest.main()
