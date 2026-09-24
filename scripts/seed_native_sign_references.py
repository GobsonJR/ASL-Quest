"""Insert native_sign_references rows for verified public reference videos.

Idempotent: skips any gloss that already has a reference row. Only inserts into
native_sign_references (never downloads/redistributes any video); touches no other
table.

Source policy: every entry below points at a public, freely-viewable dictionary page on
learnhowtosign.com, run by
Meredith Rathbone (a state-certified ASL teacher and certified educational interpreter
with a Master's in Deaf Education). Each entry was verified before insertion by fetching
the actual page, extracting its embedded Vimeo video ID, and independently confirming via
Vimeo's own oEmbed metadata (not the site's page text) that the video's title is an exact
match for the target word -- e.g. gloss HELLO's video's own Vimeo title is literally
"Hello", author "Learn How to Sign". This caught one real mismatch during verification:
the only candidate page found for EAT1 ("Eat, Food") has an embedded video whose Vimeo
title is "December" -- clearly a content/CMS mismatch on the source site -- so EAT1 is
deliberately left with no reference row rather than a guessed/fabricated one.

source_type is "external_url", not "vimeo": these are unlisted (hash-protected) Vimeo
videos, embeddable mechanically, but no explicit third-party embedding permission was
confirmed for this teacher's own paid-adjacent instructional content. source_identifier
therefore points at the public dictionary PAGE (which anyone can already view for free,
no login), not the raw video asset -- the frontend renders this as a "Watch on ... ->"
link-out, never an inline iframe of their video. This is the conservative reading of the
task's source policy ("public/approved external reference pages") and also satisfies its
explicit "Open source fallback link" UI requirement directly, rather than as a fallback.
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

ATTRIBUTION = (
    "Reference: \"{title}\" by Learn How to Sign (learnhowtosign.com), a public ASL "
    "sign dictionary maintained by a certified ASL teacher. Not hosted or embedded here "
    "-- opens their page in a new tab."
)

# gloss -> (dictionary page URL, video title as independently confirmed via Vimeo's own
# oEmbed metadata -- see module docstring).
REFERENCES: dict[str, tuple[str, str]] = {
    "HELLO": ("https://learnhowtosign.com/dictionary/hello/", "Hello"),
    "THANKYOU": ("https://learnhowtosign.com/dictionary/thank-you/", "Thank You"),
    "PLEASE": ("https://learnhowtosign.com/dictionary/please/", "Please"),
    "YES": ("https://learnhowtosign.com/dictionary/yes/", "Yes"),
    "NO": ("https://learnhowtosign.com/dictionary/no/", "No"),
    "MOTHER": ("https://learnhowtosign.com/dictionary/mother/", "Mother"),
    "WATER": ("https://learnhowtosign.com/dictionary/water/", "Water"),
    "HELP": ("https://learnhowtosign.com/dictionary/help/", "Help"),
    "BOOK": ("https://learnhowtosign.com/dictionary/book/", "Book"),
    # EAT1 intentionally omitted -- see module docstring.
}


def main() -> dict:
    db = SessionLocal()
    created: list[str] = []
    skipped_existing: list[str] = []
    skipped_no_sign: list[str] = []
    try:
        for gloss, (url, title) in REFERENCES.items():
            sign = db.query(NativeSign).filter(NativeSign.gloss == gloss).first()
            if sign is None:
                skipped_no_sign.append(gloss)
                continue
            existing = (
                db.query(NativeSignReference)
                .filter(NativeSignReference.native_sign_id == sign.id)
                .first()
            )
            if existing is not None:
                skipped_existing.append(gloss)
                continue
            db.add(
                NativeSignReference(
                    native_sign_id=sign.id,
                    source_type="external_url",
                    source_identifier=url,
                    thumbnail_path=None,
                    license_note=ATTRIBUTION.format(title=title),
                    is_primary=True,
                )
            )
            created.append(gloss)
        db.commit()

        rows = (
            db.query(NativeSignReference)
            .join(NativeSign)
            .filter(NativeSign.gloss.in_(REFERENCES.keys()))
            .all()
        )
        result = {
            "created": created,
            "skipped_existing": skipped_existing,
            "skipped_no_sign_row": skipped_no_sign,
            "intentionally_not_included": ["EAT1"],
            "rows": [
                {
                    "gloss": row.native_sign.gloss,
                    "source_type": row.source_type,
                    "source_identifier": row.source_identifier,
                }
                for row in sorted(rows, key=lambda r: r.native_sign.gloss)
            ],
        }
        return result
    finally:
        db.close()


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
