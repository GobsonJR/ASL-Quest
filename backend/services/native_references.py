"""Attach approved, externally hosted reference videos to native signs.

Nothing is downloaded or copied: a reference row only stores the link. Dataset clips (ASL Citizen) must never be added here,
their licence forbids redistribution. The links must come from a source whose terms allow linking/embedding.
"""

from __future__ import annotations

from urllib.parse import urlparse

from sqlalchemy.orm import Session

from backend.models import NativeSign, NativeSignReference


class ReferenceError_(ValueError):
    pass


def source_type_for(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if host in ("youtube.com", "m.youtube.com", "youtu.be"):
        return "youtube"
    if host in ("vimeo.com", "player.vimeo.com"):
        return "vimeo"
    return "url"


def validate_entry(gloss: str, entry: dict) -> tuple[str, str]:
    url = (entry.get("url") or "").strip()
    note = (entry.get("license_note") or "").strip()
    if urlparse(url).scheme != "https" or not urlparse(url).hostname:
        raise ReferenceError_(f"{gloss}: url must be an https link, got {url!r}")
    if len(url) > 255:
        raise ReferenceError_(f"{gloss}: url is longer than 255 characters")
    if not note:
        raise ReferenceError_(f"{gloss}: license_note is required (say who owns the video and why linking is allowed)")
    return url, note


def attach_references(db: Session, mapping: dict[str, dict]) -> dict[str, str]:
    """Create or update the primary reference of each gloss. All-or-nothing: any invalid entry changes nothing."""
    plan: list[tuple[NativeSign, str, str]] = []
    for gloss, entry in mapping.items():
        url, note = validate_entry(gloss, entry)
        sign = db.query(NativeSign).filter(NativeSign.gloss == gloss).first()
        if sign is None:
            raise ReferenceError_(f"{gloss}: no such native sign in the catalog")
        plan.append((sign, url, note))
    result: dict[str, str] = {}
    for sign, url, note in plan:
        row = (
            db.query(NativeSignReference)
            .filter(NativeSignReference.native_sign_id == sign.id, NativeSignReference.is_primary.is_(True))
            .first()
        )
        if row is None:
            row = NativeSignReference(native_sign_id=sign.id, is_primary=True)
            db.add(row)
        row.source_type, row.source_identifier, row.license_note = source_type_for(url), url, note
        result[sign.gloss] = row.source_type
    db.commit()
    return result
