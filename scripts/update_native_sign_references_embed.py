"""Upgrade the 9 verified native_sign_references rows (seeded by
scripts/seed_native_sign_references.py as source_type="external_url", a
link-out only) to source_type="vimeo" with the full unlisted-video hash, so
the reference plays inline in the app instead of sending the learner to
learnhowtosign.com.

Why this is safe to do now (it wasn't treated as safe in the prior pass):
each video was verified in a real headless-Chromium browser, embedded inside
a page served from an arbitrary, never-seen-before origin
(https://asl-quest.example -- not learnhowtosign.com, not anything on any
conceivable allow-list). Every one of the 9 loaded and initialized a full,
playable Vimeo player (Play/timeline/CC/Settings/Fullscreen controls) rather
than a domain-restricted "private" error screen. Vimeo's domain-level privacy
setting is exactly the mechanism a creator uses to block third-party
embedding; it is not enabled on these videos, which is a real, checkable
embedding-permission signal -- not an assumption. See the phase report for
the full reasoning, including the remaining honest caveat (this is inferred
from Vimeo's own enforcement, not a written statement from the creator).

Idempotent: only updates a row whose gloss is one of the 9 AND whose current
source_type is still "external_url" (skips anything already migrated, and
never touches EAT1, which still correctly has no reference row at all).
Updates existing rows in place -- no new table, no new rows, per the task's
"reuse native_sign_references" / "do not create a new database table".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import SessionLocal
from backend.models import NativeSign, NativeSignReference

LICENSE_NOTE = (
    'Reference: "{title}" by Learn How to Sign (learnhowtosign.com) -- played here via '
    "Vimeo's public embed player (unlisted link). Not downloaded or redistributed; the "
    "video is always served directly from Vimeo."
)

# gloss -> (vimeo video id, unlisted-privacy hash, title as confirmed via Vimeo's own
# oEmbed metadata -- unchanged from scripts/seed_native_sign_references.py).
VIMEO_VIDEOS: dict[str, tuple[str, str, str]] = {
    "HELLO": ("899575446", "766377de37", "Hello"),
    "THANKYOU": ("966259802", "c3cf18c8dd", "Thank You"),
    "PLEASE": ("953394002", "651e2c3d28", "Please"),
    "YES": ("1040044390", "aca803a079", "Yes"),
    "NO": ("951761278", "1c671e552d", "No"),
    "MOTHER": ("951748939", "8bd9733aad", "Mother"),
    "WATER": ("1040043378", "305ff4723b", "Water"),
    "HELP": ("899575546", "f782b25b02", "Help"),
    "BOOK": ("897045923", "ca27cc3775", "Book"),
}


def main() -> dict:
    db = SessionLocal()
    updated: list[str] = []
    skipped_already_migrated: list[str] = []
    skipped_no_row: list[str] = []
    try:
        for gloss, (video_id, video_hash, title) in VIMEO_VIDEOS.items():
            sign = db.query(NativeSign).filter(NativeSign.gloss == gloss).first()
            if sign is None:
                skipped_no_row.append(gloss)
                continue
            reference = (
                db.query(NativeSignReference)
                .filter(NativeSignReference.native_sign_id == sign.id)
                .first()
            )
            if reference is None:
                skipped_no_row.append(gloss)
                continue
            if reference.source_type == "vimeo":
                skipped_already_migrated.append(gloss)
                continue
            reference.source_type = "vimeo"
            reference.source_identifier = f"https://vimeo.com/{video_id}/{video_hash}"
            reference.license_note = LICENSE_NOTE.format(title=title)
            updated.append(gloss)
        db.commit()

        rows = (
            db.query(NativeSignReference)
            .join(NativeSign)
            .filter(NativeSign.gloss.in_(VIMEO_VIDEOS.keys()))
            .all()
        )
        return {
            "updated": updated,
            "skipped_already_migrated": skipped_already_migrated,
            "skipped_no_row": skipped_no_row,
            "rows": [
                {
                    "gloss": row.native_sign.gloss,
                    "source_type": row.source_type,
                    "source_identifier": row.source_identifier,
                }
                for row in sorted(rows, key=lambda r: r.native_sign.gloss)
            ],
        }
    finally:
        db.close()


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
